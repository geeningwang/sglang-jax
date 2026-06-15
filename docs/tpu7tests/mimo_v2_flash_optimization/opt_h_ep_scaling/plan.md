# Opt H — Expert Parallelism (EP) Scaling

**Date**: 2026-06-15
**Status**: H-1 complete; H-2 pending
**Baseline**: 534 tok/s @ conc=16 (page-size=32, cps=2048, 1-node tp=8)

---

## Decomposition Overview

| Sub-problem | Goal | Risk | Dependency | Status |
|-------------|------|------|------------|--------|
| **H-1** | Identify model dims + actual MoE backend | None | None | ✅ Complete |
| **H-2a** | Benchmark FusedEPMoE backend on 1 node | Low | H-1 | Pending |
| **H-2b** | Add tuned block configs for FusedEPMoE | Low | H-2a positive | Pending |
| **H-3** | Profile TPOT: weight bandwidth fraction + a2a cost | Low | H-2a | Pending |
| **H-4** | 2-node EP experiment (ep_size=16) | High | H-3 passes gate | Pending |

Each sub-problem has a single deliverable, an explicit exit criterion, and a stop
condition that ends the chain if it fails.

---

## 1. Current EP Architecture (Confirmed by H-1)

> Full details: [`h1_model_dims.md`](h1_model_dims.md)

### MiMo-V2-Flash model dimensions (confirmed from config.json)

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| moe_intermediate_size | **2048** |
| num_experts (n_routed_experts) | 256 |
| num_experts_per_tok (top-k) | 8 |
| num_hidden_layers | 48 |
| MoE layers | 47 (layers 1–47; layer 0 is dense) |
| weight dtype | float8_e4m3fn (FP8) |
| quant_block_size | [128, 128] |

### Actual backend: EPMoE with ep_size=1 (TP-style)

The production server runs **EPMoE** with **ep_size=1**, NOT FusedEPMoE as originally assumed.

How this is determined (code path):
```
server_args.moe_backend = "epmoe"          # default: server_args.py:138
ModelConfig.moe_backend = MoEBackend.EPMOE # model_config.py:77
model_runner.py:336 → hf_config.moe_backend = "epmoe"
model_runner.py:332 → hf_config.ep_size    = server_args.ep_size = 1

mimo_v2_flash.py:100  getattr(config, "moe_backend", "epmoe") → "epmoe"
mimo_v2_flash.py:101  use_fused = ("epmoe" == "fused") → False
mimo_v2_flash.py:125  EPMoE(ep_size=1) instantiated
```

### What ep_size=1 means in EPMoE (layers/moe.py)

```python
world_size = 1 × 8 = 8          # dp=1, tp=8
tp_size    = 8 // 1 = 8         # all devices are tensor-parallel
experts_per_device = 256 // 1 = 256   # all 256 experts on EVERY device

# Mesh reshaped to (expert=1, tensor=8)
# Expert weights: P("expert", None, "tensor")
#   "expert" axis size=1  → NOT sharded across EP groups
#   "tensor" axis size=8  → intermediate dim split 8-way
```

**Every device holds all 256 experts, with intermediate dim = 2048/8 = 256 per device.**

### Communication pattern with ep_size=1

- **No expert routing all-to-all** (`if self.ep_size > 1:` at moe.py:563 is skipped)
- **psum allreduce after each MoE layer** (`if self.tp_size > 1:` at moe.py:557)
- Allreduce volume per layer: `num_tokens × 4096 × BF16` (e.g., 16 tokens → 131 KB)

### Per-device expert weight bandwidth (current vs EP-scaled)

For decode at conc=16 (16 tokens × top-k=8 = 128 expert activations):

| Config | Experts/device | Weight cols/expert | Weight bytes/activation | Communication |
|--------|---------------|---------------------|-------------------------|---------------|
| EPMoE, ep=1 **(current)** | 256 | inter/8 = 256 | 4096×256 FP8 | psum allreduce |
| EPMoE, ep=8 (1-node) | 32 | inter = 2048 | 4096×2048 FP8 | ICI a2a (ragged) |
| FusedEPMoE (1-node) | 32 | inter = 2048 | 4096×2048 FP8 | ICI a2a (Pallas) |
| EPMoE, ep=16 (2-node) | 16 | inter = 2048 | 4096×2048 FP8 | DCN a2a |

**Key insight (from H-1)**: On a single 8-device node, EPMoE(ep=1) and EPMoE(ep=8)
have the **same total per-device expert weight bandwidth** (256 × 256 cols = 32 × 2048 cols).
Changing ep_size on the same node does NOT reduce weight reads — it only trades
psum allreduce for a2a. The weight bandwidth benefit only materialises with 2-node EP
(ep=16 → 16 experts × 2048 cols, genuinely half the current per-device weight reads).

---

## 2. Two Paths to "Increasing EP"

### Path A — Switch to FusedEPMoE backend (1-node, low risk)

```bash
--json-model-override-args '{"moe_backend": "fused"}'
```

- FusedEPMoE's Pallas kernel derives ep_size from the JAX mesh: `dp × tp = 1 × 8 = 8`
- Result: 32 local experts per device, ICI a2a INSIDE the Pallas kernel
- Expert weight per device: same as current (see table above)
- Communication: ICI a2a (within-node, fast) instead of psum allreduce
- Requires tuned block configs for MiMo-V2-Flash shape (hidden=4096, inter=2048, ep=8)
  — currently falls back to `DEFAULT_FUSED_MOE_BLOCK_CONFIG`

Path A does not need any code changes — only a launch flag. This is the first test.

### Path B — Keep EPMoE, increase --ep-size (same node or multi-node)

```bash
--ep-size 8    # same 1-node: trades psum for a2a, same weight bandwidth
--ep-size 16   # 2-node: halves local expert count + adds DCN a2a
```

- `server_args.ep_size` is wired to `hf_config.ep_size` at `model_runner.py:332`
- EPMoE already has ep_size > 1 code path using `jax.lax.ragged_all_to_all`
- For 1-node (ep=8): same weight bandwidth, different communication → uncertain benefit
- For 2-node (ep=16): fewer local experts, genuinely less weight bandwidth, but DCN overhead

Path B for 2-node is the high-risk, high-reward option — tested only if Path A (H-2a)
and profiling (H-3) show the approach is viable.

---

## 3. Benefit/Cost Analysis (updated for correct architecture)

### 1-node EP (Path A or Path B ep=8): communication pattern change only

Estimated impact on TPOT=30ms (current):
- Weight bandwidth per device: unchanged
- psum allreduce → ICI a2a: latency difference unknown; ICI a2a is generally
  slightly more expensive than psum for small tensors (harder to fuse, more routing)
- Potential gain: if ICI a2a is cheaper or the Pallas kernel fuses better → marginal gain
- Potential regression: if a2a overhead > allreduce overhead → negative

**Verdict**: uncertain; measure via H-2a benchmark.

### 2-node EP (Path B ep=16): genuine weight bandwidth reduction

With ep=16 (2-node, 16 devices), local experts drop from 32 → 16 per device.

Weight bandwidth per device: halved. If weight reads are X% of TPOT:
- X=80%: TPOT = 0.2 × 30 + 0.5 × 0.8 × 30 = 6 + 12 = 18ms → ~888 tok/s (+66%)
- X=60%: TPOT = 0.4 × 30 + 0.5 × 0.6 × 30 = 12 + 9 = 21ms → ~762 tok/s (+43%)
- X=40%: TPOT = 0.6 × 30 + 0.5 × 0.4 × 30 = 18 + 6 = 24ms → ~667 tok/s (+25%)

DCN a2a overhead (94 calls per step = 47 layers × 2):

| DCN latency/call | Added TPOT | Breakeven X% | Net tok/s (X=80%) |
|-----------------|-----------|-------------|------------------|
| 0.05 ms | +4.7 ms | 13% | ~700 (+31%) |
| 0.10 ms | +9.4 ms | 25% | ~547 (+2%) |
| 0.15 ms | +14.1 ms | 36% | ~461 (-14%) |
| 0.20 ms | +18.8 ms | 47% | ~396 (-26%) |

Breakeven DCN latency (for X=80%): ~0.12 ms/call.
Breakeven DCN latency (for X=60%): ~0.09 ms/call.

**The weight bandwidth fraction (X) and DCN latency are both unknown — measured by H-3.**

### Historical evidence

From Maxtext (different hardware, different backend):
- **Opt 9 (ragged a2a, 1-node)**: 316 vs 576 tok/s → -45%
  - Added a2a WITHOUT reducing expert weights — analogous to Path B ep=8 (1-node)
  - Confirms 1-node EP is likely neutral or negative on MoE with small batch

---

## Sub-problem H-1: Identify Model Dimensions ✅ COMPLETE

**Result**: See [`h1_model_dims.md`](h1_model_dims.md).

Key findings:
- moe_intermediate_size = **2048** (was estimated as ~768 — wrong)
- Backend is **EPMoE** with **ep_size=1** (was assumed FusedEPMoE with ep=8 — wrong)
- No expert routing a2a exists today; experts are TP-sharded
- Original H-2 (tuned block configs) was predicated on FusedEPMoE — needs revision
- Two EP paths identified: Path A (switch to FusedEPMoE) and Path B (--ep-size flag)

---

## Sub-problem H-2a: Benchmark FusedEPMoE on 1-Node

**Goal**: Determine whether switching to FusedEPMoE backend improves throughput on the
existing 1-node tp=8 setup, compared to the current EPMoE(ep=1) baseline.

**What to do**:

1. Run the standard Phase 2 sweep with:
   ```
   --json-model-override-args '{"moe_backend": "fused"}'
   ```
   All other flags unchanged (tp=8, page-size=32, cps=2048, max-running-requests=32).

2. Create a new bench job YAML `scripts/mimo_v2_flash_1node_opt_h2a_bench_job.yaml`
   copying the Opt G YAML structure, adding the override arg.

3. Run Phase 2 (conc sweep, input=512, output=256) and record tok/s vs. 534 tok/s baseline.

**Deliverable**: `h2a_results.md` with tok/s at each concurrency vs. EPMoE(ep=1) baseline.

**Exit criteria**:
- ✅ PASS (FusedEPMoE ≥ EPMoE): proceed to H-2b (add tuned block configs to further improve)
- ✅ PASS-flat (within ±3%): tuned block configs (H-2b) may still help; proceed
- ❌ STOP (FusedEPMoE clearly worse, >5% regression): document a2a overhead; Path A is negative.
  Evaluate whether to proceed to H-4 (2-node EPMoE) or close Opt H.

**Effort**: 1 DWS slot (~45 min).

---

## Sub-problem H-2b: Add Tuned Block Configs for FusedEPMoE

**Only run if H-2a shows FusedEPMoE is viable (PASS or PASS-flat).**

**Goal**: Replace the DEFAULT block config with tuned configs for the MiMo-V2-Flash shape
at ep_size=8 in `python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py`.

The current lookup key for MiMo-V2-Flash at ep=8 is:
```python
('bfloat16', 'float8_e4m3fn', num_tokens, 256, 8, 4096, 2048, 8, False, False)
```
This key has NO match in the table → falls back to `DEFAULT_FUSED_MOE_BLOCK_CONFIG`.

**What to do**:
1. Run a block config micro-sweep: vary `(bt, bf, bd1, bd2)` at the relevant token counts
   (typically 8, 16, 32 for decode-batch sizes)
2. Add the best entries to `tuned_block_configs.py` under `"TPU v7"`
3. Re-run Phase 2 sweep with tuned configs enabled to measure gain over H-2a

**Deliverable**: A commit to `tuned_block_configs.py` + `h2b_results.md` with before/after.

**Exit criteria**:
- ✅ PASS (≥3% gain over H-2a): commit configs, record gain, continue to H-3
- ✅ PASS-flat (<3%): DEFAULT was already near-optimal; commit anyway (documents intent)
- ❌ STOP: Sweep crashes or all configs regress → debug before continuing

**Effort**: 1 DWS slot for sweep + 2-4h to analyze and add entries.

---

## Sub-problem H-3: Profile TPOT Breakdown

**Goal**: Measure what fraction of TPOT is spent on expert weight reads vs. a2a vs.
attention. This is the go/no-go gate for 2-node EP.

**What to do**:
1. Use the `/start_profile` endpoint on the best 1-node server (EPMoE or FusedEPMoE,
   whichever is faster from H-2a)
2. Capture ~50-step decode trace at conc=16 (optimal concurrency)
3. Download from GCS, analyze in Perfetto:
   - Expert weight HBM read time per layer
   - a2a scatter + gather time per layer
   - Attention compute time
   - Other overhead (layernorm, routing topk, KV, etc.)

**Deliverable**: `h3_profile/results.md` with:
- TPOT breakdown: weight / a2a / attention / other (ms and %)
- Current a2a latency per call (µs)
- Estimated breakeven DCN latency for 2-node EP to be profitable (from benefit/cost table)

**Exit criteria**:
- ✅ PASS: Weight bandwidth ≥ 50% of TPOT AND estimated breakeven DCN > 0.1ms
  → weight-bandwidth-bound enough that 2-node EP could help; proceed to H-4
- ❌ STOP (weight not dominant, <40%): 2-node EP cannot halve enough of TPOT; **close Opt H**
- ❌ STOP (a2a already costly, >5ms/step): adding DCN a2a will worsen it; **close Opt H**

**Effort**: 1 slot for trace capture + 2-4h analysis.

---

## Sub-problem H-4: 2-node EP Experiment (ep_size=16)

**Only run if H-3 passes both gates above.**

### H-4a: Weight loading validation

**Goal**: Verify that EPMoE with ep_size=16 loads correctly on 2 nodes (Orbax resharding
from 8 → 16 devices, expert mesh changes from (1,8) to (16,1)).

**What to do**:
- Create `scripts/mimo_v2_flash_2node_validate_job.yaml`: 2 TPU nodes, server with
  `--nnodes 2 --tp-size 16 --ep-size 16`, load weights, print expert weight shapes, exit.
- Confirm: each device holds 16 experts with full inter=2048.

**Exit criteria**:
- ✅ PASS: Weights load, shapes correct → proceed to H-4b
- ❌ STOP: Crash or shape mismatch → root-cause before proceeding

### H-4b: Add tuned block configs for ep_size=16

**Goal**: Add tuned block configs for EPMoE ep=16 shape (or FusedEPMoE ep=16 if using
that backend for 2-node).

Config lookup key for ep=16:
```python
('bfloat16', 'float8_e4m3fn', num_tokens, 256, 8, 4096, 2048, 16, False, False)
```

### H-4c: 2-node performance benchmark

**Goal**: Measure 2-node throughput vs. 1-node baseline.

- Run Phase 2 + Phase 5 sweeps (same as Opt G YAML pattern but 2-node)
- Record peak tok/s

**Exit criteria**:
- ✅ SUCCESS (>534 tok/s): adopt 2-node, update production config
- ❌ CLOSE NEGATIVE (≤534 tok/s): record DCN overhead, close Opt H as negative

---

## 4. Stopping Rules (Summary)

| Condition | Stop at | Action |
|-----------|---------|--------|
| FusedEPMoE clearly worse than EPMoE (H-2a) | H-2a | Document; evaluate H-4 direct |
| Weight bandwidth < 40% of TPOT (H-3) | H-3 | Close Opt H negative |
| ICI a2a already > 5ms/step (H-3) | H-3 | Close Opt H negative |
| Orbax resharding fails (H-4a) | H-4a | Debug or close |
| 2-node peak ≤ 534 tok/s (H-4c) | H-4c | Close Opt H negative |

---

## 5. Risk Matrix

| Risk | Severity | Likelihood | Mitigated by |
|------|----------|------------|-------------|
| ICI a2a > psum allreduce for small tensors | Medium | Medium | H-2a measurement |
| No tuned configs for ep_size=8 (FusedEPMoE) | Medium | Certain | H-2b adds them |
| DCN latency > breakeven (~0.12ms/call) | High | High | H-3 gate before H-4 |
| Orbax reshape fails on 16-device mesh | High (crash) | Medium | H-4a validation job |
| 2-node YAML pod coordination bugs | Operational | Medium | H-4a catches early |
| Weight bandwidth fraction < 50% | High | Unknown | H-3 measures this |

---

## 6. Files per Sub-problem

| Sub-problem | Files |
|-------------|-------|
| H-1 ✅ | `opt_h_ep_scaling/h1_model_dims.md` |
| H-2a | `scripts/mimo_v2_flash_1node_opt_h2a_bench_job.yaml`, `h2a_results.md` |
| H-2b | `tuned_block_configs.py` (edit), sweep script, `h2b_results.md` |
| H-3 | `scripts/profile_flash.py` (reuse), `h3_profile/results.md` |
| H-4a | `scripts/mimo_v2_flash_2node_validate_job.yaml` |
| H-4b | `tuned_block_configs.py` (edit for ep=16) |
| H-4c | `scripts/mimo_v2_flash_2node_opt_h_bench_job.yaml`, `h4_2node/results.md` |
