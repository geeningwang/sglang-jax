# HBM Usage Investigation — MiMo-V2.5-Pro on TPU v7x

**Status**: Phase 1–4 complete (2026-06-05). Phase 5 (reduction strategies) pending.  
**Goal**: Identify the source of unknown HBM overhead, map the complete HBM timeline,
and find opportunities to enable 2-node (tp-16) inference.

---

## Confirmed Findings (2026-06-05)

### Corrected HBM baseline

The actual JAX-visible HBM per TensorCore on TPU v7x is **101.73 GB**, not 96 GB.
At `mem_fraction_static=0.75`:
- XLA temp pool: 101.73 × 0.25 = **25.43 GB**
- Static pool: 101.73 × 0.75 = **76.30 GB**

### Correct HBM breakdown at tp-32 (fast-restore, measured)

```
101.73 GB  HBM per TC (JAX-visible limit)
─── Static pool (75%) = 76.30 GB ────────────────────────────
   11.07 GB  apply_moe_quantization FP32 scales  [REAL HBM, not abstract]
   53.61 GB  checkpoint weights + restore overhead
             (33.75 GB actual weights + 19.86 GB restore overhead)
   11.62 GB  KV cache (SWA pool: 286.64 GB + 59.72 GB total / 32 TCs)
─── XLA temp pool (25%) = 25.43 GB ──────────────────────────
   25.43 GB  EPMoE EXTEND/DECODE XLA compilation buffers
─────────────────────────────────────────────────────────────
   101.73 GB  total accounted for ✓
```

### Hypotheses resolved

| Hypothesis | Test | Result |
|-----------|------|--------|
| H7: `nnx.split()` copies weight arrays | T7→T8a delta in timeline | **RULED OUT** — delta = 0.00 GB |
| H8: Overhead is unreleased intermediates | GC effect test | **RULED OUT** — GC frees < 10 MB |
| JAX runtime overhead ('J' term) | T0 baseline | **ZERO** — T0 = 0.00 GB |
| apply_moe_quantization allocates real HBM | T4c timeline snap | **CONFIRMED** — 11.07 GB at tp-32, 11.72 GB at tp-16 (nearly fixed) |
| init_attention_backend allocates HBM | T3 timeline snap | **ZERO** — FlashAttention backend is pure Python metadata |
| nnx.eval_shape allocates HBM | T4b timeline snap | **ZERO** — abstract shapes, no device allocation |

### TP-scaling of overhead

| Component | tp-32 | tp-16 | Scales? |
|-----------|-------|-------|---------|
| apply_moe_quantization (FP32 scales) | 11.07 GB | 11.72 GB | **Nearly fixed** (+0.65 GB) |
| Checkpoint weights + restore | 53.61 GB | ~90 GB (OOM) | **~1.68× scales** |
| Total model footprint | 64.68 GB | ~102 GB | OOM at tp-16 |
| XLA temp requirement (EXTEND compile) | 25.43 GB | 25.43 GB | **Fixed** |

### EPMoE XLA temp pool minimum

EPMoE EXTEND precompile (384 experts, block-wise FP8 GEMM) requires **~20 GB XLA temp**.
At `mem_fraction_static=0.85` (14.4 GB temp), EXTEND OOMs with: "Exceeded hbm capacity by 5.54G".
→ Minimum viable `mem_fraction_static` ≤ (101.73 − 20) / 101.73 ≈ **0.803**.
→ Tested values 0.85 / 0.90 / 0.93 / 0.95 / 0.97 all fail.

### 2-node feasibility conclusion

At tp-16:
- Model footprint after restore ≈ **102 GB** (exceeds 101.73 GB HBM — OOM during restore itself)
- Even without the restore OOM: available after model = ~−0.3 GB → no room for KV
- EPMoE XLA temp floor (~20 GB) is fixed; reducing static fraction does not help
- **Option B (increase mem_fraction_static) is infeasible.**
- **Only viable path to 2-node: EP > 1** (split 384 experts across 2 nodes → 192 experts/TC → weights halve)

---

## 1. Motivation (original)

## 2. What We Know

### Measurement methodology
- `bytes_in_use` from `jax.local_devices()[i].memory_stats()` after garbage collection
- `total_device_memory` measured **before** `load_model()` (includes JAX runtime overhead)
- `available_kv_cache` = `bytes_free_after_load` − `total_device_memory × 0.25`

### Code flow (timing of HBM changes)
```
__init__()
  [A] total_device_memory = get_available_device_memory()   ← before loading
  [B] init_attention_backend()                              ← FlashAttention metadata only, no HBM
  [C] load_model()
    [C1] nnx.eval_shape(model_class)                       ← abstract shapes, no HBM
    [C2] apply_moe_quantization()                          ← abstract shapes, no HBM
    [C3] _load_checkpoint() / load_weights()               ← actual HBM allocation
    [C4] nnx.update(model, state)                          ← assigns checkpoint arrays to model
  [D] initialize_jit()
    [D1] nnx.split(model)                                  ← reference-only, no copy
    [D2] jax.tree_util.tree_flatten(model_state)           ← reference-only
    [D3] define JIT closures                               ← no HBM
  [E] init_memory_pool()
    [E1] available = get_available_device_memory()         ← measures HERE
    [E2] KV_available = available - total_before × 0.25
    → logs "available_kv_cache=10.8GB"
```

### Key numbers (tp-32, measured 2026-06-05)

| Metric | Value | Source |
|--------|-------|--------|
| HBM per TC | 96 GB | Hardware spec |
| `total_device_memory` (before load) | ~96 GB | Estimated (JAX startup overhead small) |
| Checkpoint data per TC | 33.75 GB | Checkpoint `_METADATA` tensor sizes |
| `available_kv_cache` (after load) | 10.80 GB | Log: `TPU Memory profiling` |
| Implied `bytes_in_use` after load | 61.2 GB | 96 − (10.8 + 24) |
| **Unexplained overhead** | **27.45 GB** | 61.2 − 33.75 |

### Hypotheses for the 27.45 GB overhead

| # | Hypothesis | Est. Size | TP-scaling | Priority |
|---|-----------|-----------|-----------|---------|
| H1 | EPMoE GMM kernel workspace (pre-allocated routing buffers) | ~10–20 GB | Scales with experts/TC → likely doubles at tp-16 | High |
| H2 | `_maybe_convert_epmoe_scale_for_kernel()` intermediate arrays | ~1–3 GB | Freed after conversion? | Medium |
| H3 | XLA compiled code stored on-device (HLO constants) | ~2–5 GB | Fixed | Medium |
| H4 | FlashAttention Pallas kernel workspace | ~1–5 GB | Scales with heads/TC | Medium |
| H5 | JAX PRNG / runtime metadata | ~0.5 GB | Fixed | Low |
| H6 | `RoutedExpertsCapturer` buffers | Proportional to num_tokens | After KV alloc | Low |
| H7 | `nnx.split()` creating full copies of parameters | ~33.75 GB | Doubles at tp-16 | High (if true, explains everything) |
| H8 | Intermediate arrays from checkpoint restore not yet GC'd | ~10–30 GB | Scales with weights | High |

---

## 3. Investigation Plan

### Phase 1 — HBM timeline instrumentation (1–2 days)

**Goal**: Measure `bytes_in_use` at every step in the startup sequence.

**Tool**: `python/sgl_jax/tools/hbm/snapshot.py` — a lightweight utility that logs
`bytes_in_use` across all local devices, with optional `jax.live_arrays()` snapshot.

**Measurement points** (to be added to `model_runner.py` behind env var
`SGLANG_HBM_TRACE=1`):
```
[T0] Before JAX distributed init
[T1] After JAX distributed init (jax.distributed.initialize)
[T2] After init_attention_backend()
[T3] After nnx.eval_shape(model_class)
[T4] After apply_moe_quantization()
[T5] After _load_checkpoint() / load_weights()
[T6] After nnx.update(model, state)
[T7] After gc.collect() post-restore
[T8] After initialize_jit() → nnx.split()
[T9] After initialize_jit() returns
[T10] In _profile_available_bytes() — this is "available_kv_cache" point
[T11] After KV cache allocation
[T12] After XLA EXTEND precompile
[T13] After XLA DECODE precompile
[T14] During first inference forward pass (peak)
```

**Expected output** (one line per checkpoint):
```
[HBM T3] nnx.eval_shape done:       bytes_in_use=  1.2 GB  delta= +0.1 GB
[HBM T4] apply_moe_quantization:    bytes_in_use=  1.4 GB  delta= +0.2 GB
[HBM T5] _load_checkpoint:          bytes_in_use= 35.1 GB  delta=+33.7 GB  ← weights
[HBM T6] nnx.update:                bytes_in_use= 35.1 GB  delta= +0.0 GB
[HBM T7] gc.collect:                bytes_in_use= 35.1 GB  delta= +0.0 GB
[HBM T8] nnx.split:                 bytes_in_use= 61.2 GB  delta=+26.1 GB  ← mystery HERE?
[HBM T9] initialize_jit done:       bytes_in_use= 61.2 GB  delta= +0.0 GB
[HBM T10] KV profiler:              bytes_in_use= 61.2 GB  delta= +0.0 GB  → 10.8 GB available
```

If the delta at T8 is large → `nnx.split()` copies arrays (H7 confirmed).
If it happens at T5 → checkpoint restore keeps intermediate buffers (H8 confirmed).

**Deliverable**: `python/sgl_jax/tools/hbm/snapshot.py` + PR adding
`SGLANG_HBM_TRACE=1` instrumentation to `model_runner.py`.

### Phase 2 — Live array attribution (1–2 days)

**Goal**: For each large delta, enumerate which specific arrays are in HBM.

**Tool**: `python/sgl_jax/tools/hbm/attribution.py` — wraps `jax.live_arrays()`,
groups by shape+dtype, prints the top-N allocations and their total size.

**Sample output**:
```
=== HBM attribution after nnx.split() ===
  (384, 6144, 2048)  float8_e4m3fn  × 69 layers = 10.4 GB  ← wi_0 COPIES?
  (384, 6144, 2048)  float8_e4m3fn  × 69 layers = 10.4 GB  ← wi_0 originals
  ...
  Total live: 61.2 GB
```

This directly answers whether H7 (nnx.split copies) is true.

**Standalone test**: `scripts/hbm/test_split_overhead.yaml` — a minimal 1-node job
that:
1. Loads a small dummy model with known weight shapes
2. Calls `nnx.split()` and measures HBM delta
3. Reports whether it copies or shares

### Phase 3 — Isolate each hypothesis (2–3 days)

For each confirmed hypothesis, a standalone K8s job that tests it in isolation:

| Job | Tests | Hypothesis | Expected result |
|-----|-------|-----------|----------------|
| `test_hbm_baseline.yaml` | Bare JAX, no model, 1 node | baseline | JAX runtime overhead (J term) < 2 GB |
| `test_split_overhead.yaml` | `nnx.split()` on dummy model | H7 | delta ≈ 0 GB (no copy) |
| `test_epmoe_min_temp.yaml` | Binary search on XLA temp pool | H-temp | Minimum EPMoE temp requirement |
| `test_gc_effect.yaml` | GC+clear_caches before profiler | H8 | If delta > 5 GB: intermediates are the source |
| `hbm_timeline_tp32_slowpath.yaml` | Full trace, NFS slow path | H8 | Compare overhead vs fast path |
| `hbm_timeline_tp16.yaml` | Full trace, tp-16 fast restore | H-scale | Overhead at tp-16; TP-scaling factor |

Each job reports per-step `bytes_in_use` deltas and optional `live_arrays()` attribution.

### Phase 4 — TP-scaling characterization (1 day)

Run the HBM timeline on both tp-32 and tp-16 (if 2-node doesn't OOM before the
measurement point) and compare deltas at each checkpoint.

This determines whether the overhead is:
- **Fixed**: same absolute GB regardless of TP (can be amortized at larger TP)
- **TC-proportional**: scales 1/TP_size (doubles at tp-16)
- **Expert-proportional**: scales with experts_per_TC

### Phase 5 — Reduction strategies (updated with measurements)

Strategies are now ranked by feasibility given Phase 1–4 findings.

#### S7: EP > 1 — Only reliable path to 2-node (Recommended)

With `ep_size=2`, each TC handles 192 of 384 experts. MoE weight per TC halves,
reducing the total model footprint from ~102 GB to ~34 GB at tp-16.

#### XLA EXTEND OOM Analysis (from EPMoE min-temp test)

The error `"Used 100.28G of 94.75G hbm. Exceeded hbm capacity by 5.54G"` during
EXTEND precompile at `mem_fraction_static=0.85` (14.4 GB XLA temp) means:

- 94.75 GB = XLA's HBM budget for the compilation step
- 100.28 GB = total HBM XLA tried to allocate: model params (64.68 GB) + KV cache
  (20.3 GB) + peak intermediate activations during a single EXTEND step
- 5.54 GB over — only ~5.5% excess, very close to fitting

Because the overage is small, the following tweaks could push it under the limit
without requiring EP > 1:

**XLA compiler flags (no code changes):**

```bash
# Force XLA to minimize peak HBM over execution speed
export XLA_FLAGS="--xla_tpu_rematerialization_algo=PEAK_PRIORITY"

# Trick compiler into believing less HBM is available → hyper-aggressive opts
export XLA_FLAGS="$XLA_FLAGS --xla_tpu_max_hbm_size_mib=90000"

# Reduce parallel loop prefetch if model uses while_loops
export XLA_FLAGS="$XLA_FLAGS --maximum_parallel_iterations=1"
```

**JAX rematerialization (code change, high impact):**

Wrap EPMoE forward pass in `jax.remat()` to discard intermediate activations
during compilation and recompute them on-the-fly:

```python
@functools.partial(jax.remat,
                   policy=jax.checkpoint_policies.nothing_saveable)
def epmoe_forward(...): ...
```
Trades ~10–30% slower compute for 5–20 GB reduction in peak compilation HBM.

**Reduce prefill batch size:**

EXTEND activations scale with `--max-prefill-tokens` (currently 16384). Halving
to 8192 reduces attention intermediates proportionally.

**Buffer donation (verify):**

`jitted_run_model` already uses `donate_argnums=["memory_pools"]`. Verify all
large donatable inputs are listed — un-donated inputs inflate peak HBM by forcing
XLA to hold old + new copies simultaneously.

#### Updated strategy table

| Strategy | Savings | Risk | Status |
|----------|---------|------|--------|
| **S3**: Delete intermediate arrays early after restore | 0 GB | — | **RULED OUT** (H8: GC frees < 10 MB) |
| **S4**: `jax.clear_caches()` before profiler | 0 GB | — | **RULED OUT** (delta = 0.00 GB measured) |
| **S6**: Reduce `mem_fraction_static` | Blocked | EPMoE needs ~20 GB XLA min | **RULED OUT** (all 0.85–0.97 fail) |
| **S1**: Offload apply_moe_quantization scales to host RAM | ~11 GB | Each-step latency | Feasible; requires pinned-memory transfer |
| **S5**: `--xla_tpu_rematerialization_algo=PEAK_PRIORITY` XLA flag | ~5 GB? | Slower compile | **TO TEST** — low-risk, might fix 0.85 |
| **S5b**: `jax.remat()` on EPMoE forward | ~5–20 GB? | ~10–30% slower | **TO TEST** — higher impact, needs code |
| **S5c**: Reduce `--max-prefill-tokens` 16384→8192 | ~2–5 GB? | Prefill throughput | **TO TEST** — easy flag change |
| **S2**: Move BF16 attention weights to host RAM | ~2.76 GB (tp-16) | Per-layer latency | Feasible; small savings |
| **S7**: EP > 1 (ep_size=2, tp_size=8) | ~30 GB weights | Architecture change | **RECOMMENDED — most reliable** |

---

## 4. File Structure

```
docs/tpu7tests/hbm_investigation/
  plan.md                      ← this document
  results/                     ← measurement outputs from each experiment
    tp32_hbm_timeline.txt      ← Phase 1 result at tp-32
    tp16_hbm_timeline.txt      ← Phase 4 result at tp-16
    attribution_T8.txt         ← Phase 2 live array dump at T8

python/sgl_jax/tools/hbm/
  __init__.py
  snapshot.py                  ← HBM snapshot utility (bytes_in_use + delta)
  attribution.py               ← live_arrays() grouping and attribution

scripts/hbm/
  hbm_timeline_tp32.yaml       ← Full startup trace job (tp-32, 4-node)
  hbm_timeline_tp16.yaml       ← Full startup trace job (tp-16, 2-node)
  test_split_overhead.yaml     ← nnx.split() copy test (1-node)
  test_epmoe_init.yaml         ← EPMoE init overhead (1-node)
  test_checkpoint_restore.yaml ← Orbax restore alone (1-node)
  test_pallas_workspace.yaml   ← FlashAttention backend init (1-node)
  test_hbm_baseline.yaml       ← Bare JAX distributed (1-node)
```

---

## 5. Success Criteria

| Milestone | Criterion |
|-----------|----------|
| Phase 1 complete | Full HBM timeline with per-step deltas at tp-32 |
| Phase 2 complete | Each delta attributed to specific array types/sources |
| Phase 3 complete | Each hypothesis confirmed or ruled out with standalone test |
| Phase 4 complete | TP-scaling characterized (fixed vs TC-proportional vs expert-proportional) |
| Phase 5 complete | At least one strategy reduces overhead enough to enable 2-node, OR clear statement that 2-node is fundamentally infeasible |

---

## 6. Related Documents

- [gke_tpu7x_resource_allocation.md](../gke_tpu7x_resource_allocation.md) — current measured HBM tables
- [mimo_v25_pro_weight_checkpoint.md](../mimo_v25_pro_weight_checkpoint.md) — checkpoint structure and timing
- [mimo_v25_pro_progress.md](../mimo_v25_pro_progress.md) — overall project status
