# Opt H — Expert Parallelism (EP) Scaling

**Date**: 2026-06-15
**Status**: PLAN — Approved, implementation not yet started
**Baseline**: 534 tok/s @ conc=16 (page-size=32, cps=2048, 1-node tp=8)

---

## Decomposition Overview

The EP scaling problem is split into four independent sub-problems, each with its own
exit criterion. Later sub-problems only run if earlier ones pass.

| Sub-problem | Goal | Risk | Dependency |
|-------------|------|------|------------|
| **H-1** | Find exact model dims + understand current block config | None | None |
| **H-2** | Add tuned block configs for 1-node (ep_size=8) | Low | H-1 |
| **H-3** | Profile TPOT: measure how weight-bandwidth-bound decode is | Low | H-2 (or parallel) |
| **H-4** | 2-node EP experiment (ep_size=16) | High | H-3 passes gate |

Each sub-problem has:
- A **single deliverable** (one job, one file, one measurement)
- A **pass/fail exit criterion** defined up front
- A **stop condition** that ends the chain if it fails

---

## 1. Current EP Architecture (What We Have Today)

### How ep_size actually works in the FusedEPMoE kernel

There are two "ep_size" concepts that are easy to confuse:

| Concept | Value | Source |
|---------|-------|--------|
| `model_config.ep_size` | **1** | Hardcoded in `python/sgl_jax/srt/configs/model_config.py:74` |
| `hf_config.ep_size` | **1** | From the MiMo-V2-Flash `config.json` |
| Kernel `ep_size` | **8** | Derived at runtime: `get_ep_size(mesh) = dp_size × tp_size = 1 × 8` |

The `fused_ep_moe()` kernel ignores `ep_size` from hf_config entirely. It computes its own ep_size from the JAX mesh (`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:480-484`):
```python
def get_ep_size(mesh, dp_axis_name, tp_axis_name):
    dp_size = mesh.shape[dp_axis_name]   # = 1 (1 node)
    tp_size = mesh.shape[tp_axis_name]   # = 8 (tp=8)
    return dp_size * tp_size              # = 8 (current)
```

### Current production state (1-node, tp=8)

```
Mesh:                  (data=1, tensor=8)
Kernel ep_size:        8
Local experts/device:  256 / 8 = 32
Expert sharding:       P(("data", "tensor"), None, None)
All-to-all (a2a):      ENABLED — intra-node ICI (fast)
                       disable_a2a=False; MiMo-V2-Flash does NOT disable it
Tuned block configs:   NONE for MiMo-V2-Flash shape → DEFAULT_FUSED_MOE_BLOCK_CONFIG
```

The kernel routes tokens between the 8 devices via ICI all-to-all today. ICI is intra-chip-interconnect, very fast compared to DCN (between-node).

### MiMo-V2-Flash model dimensions

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| moe_intermediate_size | ~768 (check config.json for exact value) |
| num_experts | 256 |
| num_experts_per_tok (top-k) | 8 |
| num_layers | 48 (47 MoE + 1 dense layer-0) |
| weight dtype | float8_e4m3fn (FP8) |
| activation dtype | bfloat16 |

---

## 2. What "Increasing EP" Means

"Increasing EP" means **adding more nodes** to increase the total device count:

| Config | Nodes | Total devices | Kernel ep_size | Local experts/device |
|--------|-------|--------------|----------------|---------------------|
| **Current** | 1 | 8 | 8 | 32 |
| 2-node | 2 | 16 | 16 | 16 |
| 4-node | 4 | 32 | 32 | 8 |

For 2-node EP (nnodes=2, tp=8 per node):
```
Mesh: (data=2, tensor=8)
Kernel ep_size: 16
Local experts per device: 16
ICI a2a: within each node (fast, already working)
DCN a2a: between nodes (NEW — the critical unknown)
```

---

## 3. Code Architecture Analysis

### What needs to change for 2-node EP

**Item 1: Mesh — likely works already**

`python/sgl_jax/srt/managers/scheduler.py:272-276` creates the mesh with:
```python
ici_parallelism=[self.dp_size, self.tp_size // self.dp_size],
dcn_parallelism=[1, 1],
```

With `--nnodes 2 --tp-size 16`, multi-node mesh construction should be handled automatically. Whether `dcn_parallelism` needs explicit adjustment is an open question.

**Item 2: Tuned block configs — MISSING (required)**

`python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py` has NO entries for MiMo-V2-Flash. The existing entries are for:
- 128-expert models (hidden=2048, inter=768, ep_size=8)
- 256-expert models (hidden=8192, inter=2048, ep_size=32)

MiMo-V2-Flash (256 experts, hidden=4096, inter=~768) falls through to `DEFAULT_FUSED_MOE_BLOCK_CONFIG` for all token counts and ep sizes. This is suboptimal even on 1-node today.

**Item 3: `--ep-size` flag — not blocking**

`model_config.ep_size` is hardcoded to 1 (`python/sgl_jax/srt/configs/model_config.py:74`). For FusedEPMoE, the kernel derives ep_size from the mesh, so `--ep-size` being unwired does NOT block 2-node testing.

**Item 4: Orbax checkpoint resharding — needs verification**

Expert weights use `P(("data", "tensor"), None, None)` sharding. With 16 devices instead of 8, Orbax will reshard from (data=1, tensor=8) to (data=2, tensor=8) at load time. This should work in principle but has not been tested for this shape change.

**Item 5: 2-node YAML — most complex change**

Multi-node server launch requires:
- DWS ProvisioningRequest for 2× 2x2x1 TPU nodes
- Head/worker pod coordination across nodes
- `--nnodes 2 --node-rank {0,1}` flags
- Distributed init (TCP rendezvous or MXLA)

This is significantly more complex than the existing 1-node YAML pattern.

---

## 4. Benefit/Cost Analysis

### Weight bandwidth reduction (upside)

Decode is weight-bandwidth-bound. With ep_size=16 (2-node):
- Local experts: 32 → 16 per device
- Weight bytes per decode step: ~halved
- Theoretical peak speedup: up to **2×**

At current TPOT=30ms with 534 tok/s, assuming weight reads are 80% of TPOT:
- Weight-bound: 24ms → 12ms (halved)
- Fixed (attention, routing, etc.): 6ms
- Optimistic new TPOT: ~18ms → ~888 tok/s (+66%)

### DCN all-to-all overhead (downside)

Each decode step requires DCN a2a to route tokens between nodes.

| Parameter | Value |
|-----------|-------|
| A2a calls per step | 47 layers × 2 (scatter + gather) = 94 |
| Token payload | conc=16 × hidden=4096 × 2B ≈ 131 KB |
| Raw DCN transfer at 50 GB/s | ~2.6 µs per call |
| **Actual DCN latency (incl. SW overhead)** | **0.05–1 ms per call** |

Sensitivity to DCN overhead:

| DCN latency per call | Added TPOT | New TPOT | New tok/s | vs. baseline |
|---------------------|-----------|---------|---------|-------------|
| 0.05 ms | 4.7 ms | 22.7 ms | ~704 | **+32%** |
| 0.10 ms | 9.4 ms | 27.4 ms | ~583 | **+9%** |
| 0.20 ms | 18.8 ms | 36.8 ms | ~433 | -19% |
| 0.50 ms | 47.0 ms | 65.0 ms | ~246 | -54% |

Breakeven point: ~0.12 ms per DCN a2a call.

**The DCN latency is the critical unknown for this cluster configuration.**

### Historical evidence

From Maxtext experiments (ref: `plan.md`):
- **Opt 9 (ragged a2a, 1-node)**: 316 tok/s vs 576 tok/s = -45% regression
  - Root cause: a2a overhead added WITHOUT reducing expert weight reads (sparse routing)
  - Different from 2-node EP which DOES reduce expert weights per device

Maxtext did not test true multi-node EP. No direct evidence either way.

### Verdict

**High uncertainty.** Breakeven at ~0.12ms/call DCN latency is tight. Risk of regression is high without first profiling to measure:
1. What fraction of TPOT is currently weight bandwidth (vs. a2a, attention, other)
2. Whether the existing ICI a2a already has measurable overhead

---

## Sub-problem H-1: Identify Model Dimensions

**Goal**: Determine the exact `moe_intermediate_size` and confirm the kernel block config
lookup key for MiMo-V2-Flash, so subsequent sub-problems use the correct parameters.

**What to do**:
1. Add a one-time log statement at server startup to print the full HF config (or read
   the `config.json` directly from the GCS weights path)
2. Record: `moe_intermediate_size`, `n_routed_experts`, `num_experts_per_tok`,
   `hidden_size`, `moe_backend`, `ep_size` (from hf_config)
3. Verify the tuned block config lookup key that the kernel will use:
   `('bfloat16', 'float8_e4m3fn', num_tokens, 256, 8, hidden_size, moe_inter, 8, False, False)`
4. Confirm the DEFAULT block config is indeed being used (no matching entry in table)

**Deliverable**: A single file `docs/tpu7tests/mimo_v2_flash_optimization/opt_h_ep_scaling/h1_model_dims.md`
recording the confirmed model dimensions and block config lookup key.

**Exit criteria**:
- ✅ PASS: Dimensions confirmed; key identified; no tuned config exists → proceed to H-2
- ❌ STOP: Tuned config already exists (somehow missed) → re-check current performance first

**Effort**: 1 hour (read config.json, write findings).

---

## Sub-problem H-2: Tuned Block Configs for 1-node (ep_size=8)

**Goal**: Replace the DEFAULT block config with a tuned config for MiMo-V2-Flash at ep_size=8.
This is a standalone improvement independent of multi-node EP.

**What to do**:
1. Write a block config sweep script (`scripts/block_config_sweep_flash.py`) that:
   - Launches the existing 1-node server
   - Runs Phase 2 sweep with different `block_config` overrides (bt, bf, bd1, bd2 variants)
   - Records throughput per config
2. Find the best config for each `num_tokens` value in the concurrency sweep
3. Add the winning entries to `tuned_block_configs.py` under `"TPU v7"`
4. Re-run the standard Phase 2 sweep to confirm improvement

**Deliverable**: A commit adding tuned config entries to `tuned_block_configs.py` + a
benchmark result showing the improvement (or confirming DEFAULT is already near-optimal).

**Exit criteria**:
- ✅ PASS (improvement): New configs yield ≥3% gain → commit, record in
  `h2_block_configs/results.md`, continue to H-3
- ✅ PASS (flat): New configs show <3% gain → DEFAULT is near-optimal; commit anyway
  (good to have explicit configs), continue to H-3
- ❌ STOP: Sweep crashes / configs are invalid → debug before continuing

**Effort**: 1 benchmark job slot (~2 hours runtime, mainly waiting for DWS).

---

## Sub-problem H-3: Profile TPOT Breakdown (1-node)

**Goal**: Measure what fraction of the 30ms TPOT is spent on:
- Expert weight HBM reads (w1/w2/w3 for 47 layers)
- ICI a2a (scatter + gather within the node)
- Attention computation
- Other (routing, layernorm, etc.)

This is the go/no-go gate for 2-node EP. If weight bandwidth is not dominant, EP scaling
cannot overcome DCN overhead.

**What to do**:
1. Use the existing `/start_profile` HTTP endpoint on the production 1-node server
2. Capture a ~50-step decode trace at conc=16 (the optimal concurrency)
3. Download from GCS and open in Perfetto
4. Annotate the trace to estimate per-layer time budgets

**Deliverable**: A file `h3_profile/results.md` with:
- Time breakdown pie: weight reads / a2a / attention / other (% of TPOT)
- Current ICI a2a overhead estimate (µs per call)
- Estimated breakeven DCN latency for 2-node EP to be profitable

**Exit criteria**:
- ✅ PASS: Weight bandwidth ≥ 50% of TPOT AND estimated breakeven DCN latency > 0.1ms
  → proceed to H-4
- ❌ STOP (weight not dominant): Weight bandwidth < 40% of TPOT → 2-node EP cannot
  halve enough of the step to overcome DCN; **close Opt H as negative**
- ❌ STOP (a2a already hurts): Current ICI a2a > 5ms/step → the kernel is already
  a2a-bottlenecked; adding DCN a2a will certainly make it worse; **close Opt H as negative**

**Effort**: 1 benchmark slot for profiling + 2-4 hours trace analysis.

---

## Sub-problem H-4: 2-node EP Experiment (ep_size=16)

**Only run this if H-3 passes both gates above.**

This sub-problem is itself divided into three steps to fail fast on infrastructure
issues before spending a full benchmark slot.

### H-4a: Weight loading validation (fast fail)

**Goal**: Verify Orbax can reshard expert weights from 8 → 16 devices without crashing.

**What to do**:
1. Create a minimal job YAML (`scripts/mimo_v2_flash_2node_validate_job.yaml`) that:
   - Provisions 2 TPU nodes
   - Launches the server with `--nnodes 2 --tp-size 16`
   - Waits for weight loading to complete and prints the expert weight shapes per device
   - Does NOT run any inference — exits immediately after load
2. Confirm: `w1.shape = (16, hidden, moe_inter)` on each of the 16 devices

**Exit criteria**:
- ✅ PASS: Server loads weights, shapes correct → proceed to H-4b
- ❌ STOP: Orbax resharding fails / shape mismatch / crash → root-cause before proceeding

**Effort**: 1 small job slot (~30 min).

### H-4b: Add tuned block configs for ep_size=16

**Goal**: Add tuned block config entries for the MiMo-V2-Flash shape at ep_size=16
(from H-1 dimensions). Without these, the 2-node kernel falls back to DEFAULT config
which may be significantly suboptimal at ep_size=16.

**What to do**:
1. Run a block config sweep at ep_size=16 (2-node job, sweep a small grid of bt values)
2. Add the best entries to `tuned_block_configs.py`

Key lookup key:
```python
('bfloat16', 'float8_e4m3fn', num_tokens, 256, 8, hidden_size, moe_inter, 16, False, False)
```

**Exit criteria**:
- ✅ PASS: Configs found and added → proceed to H-4c
- ❌ STOP: Kernel validation errors for all configs → investigate block constraint violations

**Effort**: 1 benchmark slot.

### H-4c: 2-node performance benchmark

**Goal**: Measure 2-node throughput and compare to 1-node baseline.

**What to do**:
1. Run standard benchmark job (`scripts/mimo_v2_flash_2node_opt_h_bench_job.yaml`)
   - Phase 2: concurrency sweep (input=512, output=256)
   - Phase 5: long-input sweep (input=2048, output=64)
2. Record results in `h4_2node/results.md`
3. Compare: 2-node peak tok/s vs. 1-node 534 tok/s

**Exit criteria**:
- ✅ SUCCESS: 2-node peak > 534 tok/s → adopt 2-node config, update production config,
  update plan.md status to ✅
- ❌ CLOSE NEGATIVE: 2-node peak ≤ 534 tok/s → record DCN overhead findings, close
  Opt H as negative (DCN a2a cost exceeds weight bandwidth savings)

**Effort**: 1 benchmark slot (~90 min runtime).

---

## 5. Stopping Rules (Summary)

Stop the chain immediately and record findings if:

| Condition | Stop at | Action |
|-----------|---------|--------|
| Tuned config already exists (H-1) | H-1 | Re-benchmark current state |
| Weight bandwidth < 40% of TPOT (H-3) | H-3 | Close Opt H negative |
| ICI a2a already > 5ms/step (H-3) | H-3 | Close Opt H negative |
| Orbax resharding crashes (H-4a) | H-4a | Debug or close |
| Block config validation errors (H-4b) | H-4b | Debug or close |
| 2-node peak ≤ 534 tok/s (H-4c) | H-4c | Close Opt H negative |

---

## 6. Risk Matrix

| Risk | Severity | Likelihood | Mitigated by |
|------|----------|------------|-------------|
| DCN latency > breakeven (~0.12ms/call) | High | High | H-3 gate (profile first) |
| No tuned configs for ep_size=16 | Medium | Certain | H-4b (sweep before benchmark) |
| Orbax reshape fails on 16 devices | High (crash) | Medium | H-4a (validation-only job) |
| 2-node YAML pod coordination bugs | Operational | Medium | H-4a catches this early |
| DEFAULT config suboptimal today | Medium | High | H-2 addresses independently |

---

## 7. Files per Sub-problem

| Sub-problem | Files |
|-------------|-------|
| H-1 | `opt_h_ep_scaling/h1_model_dims.md` |
| H-2 | `scripts/block_config_sweep_flash.py`, `tuned_block_configs.py` (edit), `h2_block_configs/results.md` |
| H-3 | `scripts/profile_flash.py` (reuse if exists), `h3_profile/results.md` |
| H-4a | `scripts/mimo_v2_flash_2node_validate_job.yaml` |
| H-4b | `tuned_block_configs.py` (edit), sweep job YAML |
| H-4c | `scripts/mimo_v2_flash_2node_opt_h_bench_job.yaml`, `h4_2node/results.md` |
