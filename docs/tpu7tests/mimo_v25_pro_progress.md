# MiMo-V2.5-Pro on TPU v7x — Progress Summary

**Last updated**: 2026-06-03  
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

### 5. Orbax checkpoint — save/load investigation ✅ (workaround validated)

**Goal**: Replace ~42 min NFS loading with ~90s GCS checkpoint restore.

**Checkpoint mechanics work**: Orbax saves at ~4.5 GiB/s, restores at ~5.5 GiB/s in 88–94s. The checkpoint IO is fast and correct.

**Root cause**: JAX's libtpu cannot create `float8_e4m3fn` arrays via
`jax.device_put(numpy_float8, tpu_device)`. Since MiMo-V2.5-Pro is 100% FP8,
all 1038 tensors fail — returning `ShapeDtypeStruct` instead of `jax.Array`.

**Workaround validated** (2026-06-03): Monkey-patch `jax.device_put` to transfer
FP8 as `uint8` then `bitcast_convert_type` back to `float8_e4m3fn` on-device.
All 4 validation tests passed:

| Test | Result |
|------|--------|
| `bitcast_convert_type(uint8→float8)` on TPU v7x | ✅ PASS |
| Patch under ~14 MB free HBM (post-model-load) | ✅ PASS |
| Orbax shard concurrency = 1 (no semaphore needed) | ✅ PASS |
| 4-node 32-TC multi-host `device_put` intercepted | ✅ PASS |

**Next step**: Integrate the monkey-patch into `loader.py _load_checkpoint()`.
See `fp8_restore_workaround/README.md` for implementation code.

**Approaches that failed**:

| Attempt | Outcome |
|---------|---------|
| Orbax 0.11.28, float8 direct | ShapeDtypeStruct (FP8 restore fails) |
| Orbax 0.12.0, float8 direct | ShapeDtypeStruct (same) |
| uint8 workaround (save as uint8, restore+bitcast) | HBM OOM during bitcast (only 14 MB free) |
| Per-shard addressable_shards bitcast | HBM OOM (still needs temp buffer) |
| Option B: hybrid (non-FP8 from ckpt, FP8 from NFS) | N/A — model is 100% FP8 |
| JAX 0.9.0 container (`jax0.9.0-rev1`) | ShapeDtypeStruct (libtpu still blocks FP8) |

**Path forward**: Requires a container where libtpu natively supports float8 buffer
allocation. No `jax0.10.x-rev1` container exists yet in the public registry.
The JAX Python changelog (0.9.x, 0.10.x) contains no float8 TPU mentions.

**Checkpoint location** (saved, not yet usable):
```
gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/
  95dc2640/
    tp32_bfloat16/           ← Orbax checkpoint with FP8 arrays
    tp32_bfloat16_abstract_state.pkl
```

Implementation: `python/sgl_jax/srt/model_loader/loader.py`

---

### 6. Documentation ✅

| Doc | Contents |
|-----|----------|
| [gke_tpu7x_env_setup.md](gke_tpu7x_env_setup.md) | DWS node pool setup, resubmit workflow, pitfalls |
| [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) | HBM/RAM/GCS for 4-node and 2-node |
| [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) | Test 1–6 runbooks |
| [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) | Module-by-module pipeline |
| [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md) | Benchmark results + optimization roadmap |
| [mimo_v25_pro_weight_checkpoint.md](mimo_v25_pro_weight_checkpoint.md) | Checkpoint analysis (blocked — see above) |
| [mimo_v25_pro_progress.md](mimo_v25_pro_progress.md) | This document |

---

## Pending / Planned

### 7. Scheduler throughput (Opt-1d) ⬜

The `#running-req: 2` ceiling is structural — all flag-level fixes failed. Next:
add debug logging to `get_new_batch_prefill()` in `managers/scheduler.py` to trace
`batch_is_full` state and `add_one_req` return codes per queued request.

### 8. EP > 1 (Opt-2) ⬜

Change `ep_size > 1` in model config + update MoE sub-mesh wiring in `weight_utils.py`.
Expected: proportional MoE throughput gain (up to 8× with EP=8).

### 9. FP8 checkpoint unblock ⬜

Wait for a TPU container image where libtpu supports float8 device buffer allocation
(e.g., `jax0.10.x-rev1` when available), then retry the checkpoint restore.

### 10. FP8 GMM kernel tuning (Opt-3) ⬜

Sweep `tm`, `tn`, `tk` block sizes for EPMoE GEMM to improve per-step efficiency.

---

## Key Infrastructure State (as of 2026-06-03)

| Resource | Status | Notes |
|----------|--------|-------|
| `jingnw-nfs-weights-1/2/3` | **RUNNING** | 3 × n2-highmem-48, ~$15–18/hr — weights in RAM |
| TPU DWS nodes | ✅ 0 nodes | Pool empty, no active job |
| GCS checkpoint | SAVED ✅ | `95dc2640/tp32_bfloat16/` — usable with monkey-patch workaround |
| XLA compilation cache | WARM | `gs://.../jax-compilation-cache/` (tp-size=32) |
| Container image (current) | `jax0.9.0-rev1` | FP8 restore unblocked via monkey-patch |

---

## Recent Key Commits

| Commit | Description |
|--------|-------------|
| `90ba338` | feat: upgrade container to jax0.9.0-rev1 (FP8 still blocked) |
| `ed35748` | docs: JAX 0.8.1+0.9.0 FP8 restore confirmed blocked at libtpu |
| `f00a4c0` | feat: Option C — Orbax 0.12, revert uint8 (also blocked) |
| `9c400b1` | fix(checkpoint): patch allow_narrow_n_blockwise post-restore |
| `7519803` | fix(checkpoint): save FP8 as uint8 for Orbax (HBM OOM on restore) |
| `3004a85` | feat: orbax checkpoint save/load initial implementation |
| `4458f86` | feat: NFS weight loading demo job |
