# MiMo-V2.5-Pro on TPU v7x — Progress Summary

**Last updated**: 2026-06-05 (2-node confirmed infeasible, next-step actions complete)  
**Branch**: `tpu7` (`geeningwang/sglang-jax`)  
**Cluster**: `jingnw-tpu7-cluster`, zone `us-central1-c`

---

## Completed Work

### 1. Smoke test — 4-node inference (gcsfuse) ✅

First successful end-to-end MiMo-V2.5-Pro inference on GKE TPU v7x (2x2x4, tp-size=32).

| Metric | Value |
|--------|-------|
| Weight loading (gcsfuse) | ~2h25m |
| XLA warmup (warm cache) | ~55s |
| Decode throughput | 10.81 tok/s |
| Total startup | ~2h26m |

Script: `scripts/mimo_v25_pro_demo_job.yaml` | Doc: [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) Test 4.

---

### 2. Smoke test — 4-node inference (NFS RAM) ✅

Same inference via 3 × n2-highmem-48 RAM-backed NFS servers (jingnw-nfs-weights-1/2/3).

| Metric | Value |
|--------|-------|
| MoE loading rate | ~5–7 s/group (vs gcsfuse ~14–17) |
| Total MoE load time | **~42 min** (vs gcsfuse ~2h25m) |
| XLA warmup | ~55s (warm cache) |
| Decode throughput | 10.80 tok/s |
| Total startup | ~57 min |

Script: `scripts/mimo_v25_pro_nfs_demo_job.yaml` | Container: `jax0.9.0-rev1`

NFS servers:
| VM | Internal IP | Files | RAM used |
|----|------------|-------|----------|
| `jingnw-nfs-weights-1` | 10.128.0.92 | 12 safetensors | 322 GB |
| `jingnw-nfs-weights-2` | 10.128.15.231 | 12 safetensors | 350 GB |
| `jingnw-nfs-weights-3` | 10.128.0.45 | 10 safetensors | 292 GB |

---

### 3. Performance benchmark ✅

Full sweep: concurrent requests, prefill lengths, output lengths.
Full results: [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md).

| Metric | Value | Notes |
|--------|-------|-------|
| TP | 32 | Only viable config — TP=16 has no KV cache headroom |
| EP | 1 | Fixed by model config (`ep_size=1`) |
| Peak decode throughput | **20.2 tok/s** | Saturates at concurrency=2 |
| Peak prefill throughput | **3,850 tok/s** | At 2048-token inputs |
| Scheduler ceiling | `#running-req: 2` | Root cause: structural (see Opt-1d) |

Flags tested (none broke the ceiling): `--precompile-bs-paddings`, `--disable-overlap-schedule`, `--chunked-prefill-size 4096`.

---

### 4. 2-node feasibility test ✅

**Confirmed infeasible**: TP=16 weights fill ~60 GB/TC leaving no HBM for KV cache.
`RuntimeError: Not enough memory` during KV cache profiling.
See: [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md)

---

### 5. Orbax checkpoint — save/load ✅ COMPLETE (2026-06-05)

**Goal**: Replace ~42 min NFS loading with ~90s GCS checkpoint restore.

**Result**: End-to-end checkpoint save and restore validated on 4-node 2x2x4 (tp-size=32).
Inference confirmed working after restore.

| Metric | Value |
|--------|-------|
| Checkpoint save time | ~155s (~2.6 min) |
| Checkpoint restore time | **98.6s (~1m38s)** |
| GCS read throughput | 5.3 GiB/s per host |
| Total startup (restore + KV cache + XLA warmup) | **~395s (~6.6 min)** |
| vs NFS slow-path | ~57 min |
| **Startup reduction** | **~50 min saved per run** |

**Checkpoint location** (live, in use):
```
gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/
  95dc2640/
    tp32_bfloat16/                      ← Orbax OCDBT checkpoint (sharded across 32 TCs)
    tp32_bfloat16_abstract_state.pkl    ← abstract state metadata for restore
```

**Implementation**: `python/sgl_jax/srt/model_loader/loader.py`

Enabled via env var: `SGLANG_CHECKPOINT_DIR=gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint`

---

#### Bugs fixed during integration (2026-06-03 to 2026-06-05)

**Bug 1 — FP8 host-to-device allocation (JAX/libtpu)**
`jax.device_put(numpy_float8, tpu)` silently returns `ShapeDtypeStruct` on TPU v7x.
Fix: monkey-patch `jax.device_put` — transfer FP8 as `uint8`, `bitcast_convert_type`
back to `float8_e4m3fn` on-device. All 4 validation tests passed
(see `fp8_restore_workaround/README.md`).

**Bug 2 — PartitionSpec → NamedSharding upgrade crashes on MoE sub-mesh axes**
MoE params have `PartitionSpec('expert', None, 'tensor')` but the main mesh only has
`('data', 'tensor')`. Fix: skip the upgrade for PartitionSpec axes not in `self.mesh`.

**Bug 3 — apply_linear_quantization creates weight_q mismatch with checkpoint**
In the fast-restore path, `apply_linear_quantization` converts attention/MLP layers
to `QuantizedLinear` (adding `weight_q`). The checkpoint stores BF16 `weight` for
those layers (they remain BF16 by design — only MoE experts are FP8 via
`apply_moe_quantization`).
Fix: skip `apply_linear_quantization` when `checkpoint_ready=True` so the model
structure (BF16 `weight` for linear layers, `wi_0/wi_1/wo` FP8 for MoE experts)
matches the checkpoint exactly.

**Bug 4 — allow_narrow_n_blockwise post-restore kernel error**
After checkpoint restore, FP8 layers restored from `apply_moe_quantization` have
`allow_narrow_n_blockwise=False`, causing `RuntimeError: Block-wise kernel does not
support out_dim=128` during KV cache profiling (v_proj, q_proj with 128 output/TC).
Fix: `_patch_narrow_blockwise()` sets `allow_narrow_n_blockwise=True` on all FP8
layers after restore.

#### Checkpoint tensor structure

| Layer type | Format in checkpoint | Notes |
|-----------|---------------------|-------|
| MoE expert weights | `wi_0`, `wi_1`, `wo` (FP8) | via `apply_moe_quantization` |
| MoE expert scales | `wi_0_scale`, `wi_1_scale`, `wo_scale` | blockwise float32 |
| Attention q/k/v/o_proj | `weight` (BF16) | NOT quantized at linear level |
| Dense MLP gate/up/down_proj | `weight` (BF16) | NOT quantized at linear level |
| Layer norms | `scale` (BF16) | unchanged |
| Embeddings | `embedding` (BF16) | unchanged |

---

### 6. HBM investigation — 2-node feasibility ✅ Complete (2026-06-05)

**Goal**: Understand whether 2-node (tp-16) inference is feasible and what the
27 GB unexplained HBM overhead was.

**Conclusion**: 2-node with ep_size=1 is **infeasible**. Only viable path: **EP > 1**.

**Investigation**: 7 measurement tests across two TPU configurations. All results
in `docs/tpu7tests/hbm_investigation/`.

Key confirmed facts:

| Finding | Value | Notes |
|---------|-------|-------|
| JAX-visible HBM per TC | **101.73 GB** | Not 96 GB as assumed |
| JAX runtime overhead | **0 GB** | T0 = 0.00 GB |
| `apply_moe_quantization` HBM | **11.07 GB** (tp-32) / 11.72 GB (tp-16) | Real FP32 scale arrays; nearly TP-independent |
| `nnx.split()` overhead | **0 GB** | No copies (H7 ruled out) |
| GC effect | **0 GB** | Overhead is permanent (H8 ruled out) |
| Total model footprint at tp-32 | **64.68 GB/TC** | scales + weights + restore overhead |
| KV cache at tp-32 | **11.62 GB/TC** (346 GB total) | precisely measured |
| EPMoE min XLA temp | **~20 GB** | 0.85 mem_fraction fails; min = 0.803 frac |
| tp-16 model footprint | **~102 GB** | OOM during restore (layer 42/70) |
| 2-node (ep-1) | **INFEASIBLE** | OOM during restore; no KV headroom |

**Tools built**: `SGLANG_HBM_TRACE=1` + `SGLANG_HBM_ATTRIBUTE=1` + `SGLANG_HBM_GC_BEFORE_PROFILER=1`
instrumentation in `model_runner.py`, `model_runner_kv_cache_mixin.py`, `loader.py`.
Snapshot/attribution tools: `python/sgl_jax/tools/hbm/`.

---

### 7. Documentation ✅

| Doc | Contents |
|-----|----------|
| [gke_tpu7x_env_setup.md](gke_tpu7x_env_setup.md) | DWS node pool setup, resubmit workflow, pitfalls |
| [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) | HBM/RAM/GCS — corrected measurements (2026-06-05) |
| [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) | Test 1–6 runbooks |
| [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) | Module-by-module pipeline |
| [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md) | Benchmark results + optimization roadmap |
| [mimo_v25_pro_weight_checkpoint.md](mimo_v25_pro_weight_checkpoint.md) | Checkpoint design, timing, memory, data dependencies |
| [fp8_restore_workaround/README.md](fp8_restore_workaround/README.md) | FP8 restore bug + workaround (all tests passed) |
| [hbm_investigation/plan.md](hbm_investigation/plan.md) | HBM investigation — findings, hypotheses, test results |
| [mimo_v25_pro_progress.md](mimo_v25_pro_progress.md) | This document |

---

## Pending / Planned

### 8. Scheduler throughput (Opt-1d) ⬜

The `#running-req: 2` ceiling is structural — all flag-level fixes failed. Next:
add debug logging to `get_new_batch_prefill()` in `managers/scheduler.py` to trace
`batch_is_full` state and `add_one_req` return codes per queued request.

### 9. 2-node inference — CONFIRMED INFEASIBLE (2026-06-05) ✅

Tested ep=1 tp=16, ep=2 tp=8 — both OOM identically.

**Root cause**: `per-TC weight = total_weight / total_TCs`. EP factoring does not
reduce per-TC weight — only total TC count does. At 16 TCs (2 nodes), per-TC
wi_0 shard = 302 MB regardless of ep_size. Total model footprint ~112 GB >> 101.73 GB limit.

EP > 1 is fully implemented in `moe.py` (auto-creates moe_mesh, psum combine)
but provides no HBM benefit at the same TC count. EP > 1 on 4 nodes would
provide throughput improvements (better expert balancing) without memory savings.

**2-node is not achievable for MiMo-V2.5-Pro with 96 GB HBM per chip.**
Would require: smaller model, different quantization, or future TPU with more HBM.

### 9b. XLA rematerialization flag — BLOCKED (2026-06-05) ❌

Attempted `mem_fraction_static=0.85` (XLA temp 14.4 GB, KV 20.3 GB) with:
- `--xla_tpu_rematerialization_algo=PEAK_PRIORITY` → **Unknown flag** in jax0.9.0-rev1
- `--xla_enable_hlo_rematerialization=true` → **Unknown flag** in jax0.9.0-rev1
- `--max-prefill-tokens 8192` → No effect on EXTEND compile peak

EXTEND precompile OOMs by exactly 5.54 GB at 0.85 regardless of all approaches.
The rematerialization flags required exist only in newer XLA versions (jax0.10+).

**Status**: Blocked by JAX 0.9.0 XLA version. Would require upgrading to
jax0.10.x-rev1 container when available. Reverted to `mem_fraction_static=0.75`.

### 10. EP > 1 at 4 nodes for throughput (Opt-2) ⬜

EP > 1 is fully implemented in `moe.py` (auto-creates moe_mesh, psum combine,
`--ep-size` flag already plumbed through). EP > 1 at 4 nodes provides:
- Better expert load balancing
- Reduced per-TC token processing (routing only to assigned experts)
- No HBM benefit (per-TC weight = total_weight / total_TCs, unchanged)

Try `--ep-size 2 --tp-size 16 --nnodes 4` (32 TCs, ep=2 groups × tp=16 each).

### 11. FP8 GMM kernel tuning (Opt-3) ⬜

Sweep `tm`, `tn`, `tk` block sizes for EPMoE GEMM to improve per-step efficiency.

---

## Key Infrastructure State (as of 2026-06-05)

| Resource | Status | Notes |
|----------|--------|-------|
| `jingnw-nfs-weights-1/2/3` | **RUNNING** | 3 × n2-highmem-48, ~$15–18/hr — weights in RAM |
| TPU DWS nodes | ✅ 0 nodes | Pool empty, no active job |
| GCS checkpoint (tp-32 ep-1) | **LIVE ✅** | `95dc2640/tp32_bfloat16/` — validated, ~98s restore |
| GCS checkpoint (tp-16 ep-1) | SAVED | `95dc2640/tp16_bfloat16/` — saved but unusable (OOM) |
| GCS checkpoint (tp-16 ep-2) | NOT SAVED | ep=2 OOMs during restore, no checkpoint created |
| XLA compilation cache | WARM | `gs://.../jax-compilation-cache/` (tp-size=32) |
| Container image (current) | `jax0.9.0-rev1` | FP8 restore via monkey-patch + structure fix |

---

## Recent Key Commits

| Commit | Description |
|--------|-------------|
| `cc89152` | docs+test: 2-node confirmed infeasible — EP > 1 does not help HBM |
| `63fd964` | feat(ep2): add 2-node ep=2 tp=16 demo job (EP already implemented in moe.py) |
| `5e7c740` | docs: complete HBM investigation — all 7 tests done, conclusions recorded |
| `ef0101d` | docs(hbm): slow-path timeline — fast vs slow path identical footprint |
| `3a9c163` | docs(hbm): XLA OOM analysis and reduction strategies for EXTEND compile |
| `4601d8e` | docs(hbm): EPMoE min temp test — ~20 GB XLA required for EXTEND compile |
| `4f238a2` | fix(checkpoint): skip apply_linear_quantization when restoring from checkpoint |
| `c6588a7` | docs: checkpoint restore validated — ~98s restore, ~6.6 min total startup |
| `4458f86` | feat: NFS weight loading demo job |
