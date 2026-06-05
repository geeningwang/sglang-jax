# HBM Usage Investigation — MiMo-V2.5-Pro on TPU v7x

**Status**: Investigation in progress  
**Goal**: Identify the source of ~27.45 GB/TC unexplained HBM overhead, map the
complete HBM timeline during inference startup, and find opportunities to reduce
HBM pressure to enable 2-node (tp-16) inference.

---

## 1. Motivation

At tp-32 (4-node, 32 TCs), HBM breaks down as:

```
96 GB total per TC
  24 GB  XLA temp pool (25% of 96 GB; required for EPMoE 384-expert GEMM)
  33.75 GB  model weights (from checkpoint metadata: wi_0/wi_1/wo FP8 + BF16 attn)
  10.80 GB  KV cache (measured: available_kv_cache from profiler log)
  ─────────
  27.45 GB  UNKNOWN (the gap — not accounted for by any of the above)
```

At tp-16 (2-node), weights double to ~67.5 GB/TC:

```
96 GB total
  24 GB  XLA temp pool (unchanged — EPMoE still needs it)
  67.5 GB  model weights (doubled)
  27+ GB  unknown overhead (unknown scaling)
  ─────────
  < 0 GB  available for KV → likely OOM
```

The 2-node case is currently infeasible. To unlock it, we need to:
1. Identify what the 27.45 GB is
2. Determine if it scales with TP size (if not: 4.5 GB left for KV at tp-16)
3. Find ways to move or reduce it

---

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

| Job | Tests | Expected result |
|-----|-------|----------------|
| `test_epmoe_init.yaml` | EPMoE layer init (no model) | Measures EPMoE-specific overhead |
| `test_checkpoint_restore.yaml` | Orbax restore alone, no model | Measures restore intermediate buffers |
| `test_jit_overhead.yaml` | `initialize_jit()` with dummy model | Measures JIT closure overhead |
| `test_pallas_workspace.yaml` | FlashAttention backend init alone | Measures FA workspace allocation |
| `test_hbm_baseline.yaml` | Bare JAX distributed init, no model | Measures JAX runtime baseline |

Each job reports: `bytes_in_use` before and after the tested operation, and
`live_arrays()` diff showing what was allocated.

### Phase 4 — TP-scaling characterization (1 day)

Run the HBM timeline on both tp-32 and tp-16 (if 2-node doesn't OOM before the
measurement point) and compare deltas at each checkpoint.

This determines whether the overhead is:
- **Fixed**: same absolute GB regardless of TP (can be amortized at larger TP)
- **TC-proportional**: scales 1/TP_size (doubles at tp-16)
- **Expert-proportional**: scales with experts_per_TC

### Phase 5 — Reduction strategies (1–3 days, after Phase 1–4)

Based on findings, implement and test reduction strategies:

| Strategy | Savings potential | Risk | Condition |
|----------|-----------------|------|-----------|
| **S1**: Offload weight scales to host RAM (pinned) | ~1 GB | Latency on dereference | If scales are accessed once per forward |
| **S2**: Move BF16 attention weights to host RAM | ~2.76 GB (tp-16) | Latency per layer | If attention compute is latency-tolerant |
| **S3**: Delete intermediate arrays early after restore | ? | Complexity | If H8 confirmed |
| **S4**: Use `jax.clear_caches()` before KV profiling | ~5–10 GB? | Cache miss cost | If H3 (XLA code) confirmed |
| **S5**: Reduce `max_prefill_tokens` or `chunked_prefill_size` | ? | Prefill throughput | If prefill buffers are the source |
| **S6**: Reduce `mem_fraction_static` just enough to get XLA temp from 24→18 GB | +6 GB static | EPMoE may OOM | Benchmark EPMoE temp requirement precisely |
| **S7**: EP > 1 to distribute experts across nodes | Reduces wi_0/wi_1/wo per TC | Architecture change | Separate optimization track |

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
