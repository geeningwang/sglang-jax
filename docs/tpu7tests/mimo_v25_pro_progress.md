# MiMo-V2.5-Pro on TPU v7x — Progress Summary

**Last updated**: 2026-06-03  
**Branch**: `tpu7` (`geeningwang/sglang-jax`)  
**Cluster**: `jingnw-tpu7-cluster`, zone `us-central1-c`

---

## Completed Work

### 1. Smoke test — 4-node inference ✅

First successful end-to-end MiMo-V2.5-Pro inference on GKE TPU v7x (2x2x4,
tp-size=32). Key results from `scripts/mimo_v25_pro_demo_job.yaml`:

| Metric | Value |
|--------|-------|
| Weight loading (gcsfuse) | ~2h25m |
| XLA warmup (warm cache) | ~55s |
| Decode throughput | 10.81 tok/s |
| Total startup | ~2h26m |

See: [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) Test 4.

---

### 2. Performance benchmark ✅

Full sweep across concurrent requests, prefill lengths, and output lengths.
Results in [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md).

| Metric | Value | Notes |
|--------|-------|-------|
| TP | 32 | Only viable config (TP=16 has no KV cache headroom) |
| EP | 1 | Fixed by model config |
| Peak decode throughput | 20.2 tok/s | Saturates at concurrency=2 |
| Peak prefill throughput | 3,850 tok/s | At 2048-token inputs |
| Scheduler ceiling | `#running-req: 2` | Root cause under investigation |

**Throughput ceiling root cause**: scheduler always runs 2 requests at a time
despite higher concurrency. Attempts to fix via `--precompile-bs-paddings`,
`--disable-overlap-schedule`, and `--chunked-prefill-size` all had no effect.
Root cause is in the scheduler's batch admission logic (Opt-1d pending).

---

### 3. 2-node feasibility test ✅

Confirmed: MiMo-V2.5-Pro **cannot run on 2 TPU v7x nodes** (tp-size=16).

- Weights consume ~60 GB/TC → no HBM left for KV cache after XLA scratch
- KV cache profiler OOMs: `RuntimeError: Not enough memory`
- See [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md)

---

### 4. RAM-backed NFS weight servers ✅

Three `n2-highmem-48` VMs with 962 GB of weights in RAM-backed tmpfs, served
via NFS. Weight loading 2–3× faster than gcsfuse.

| VM | Internal IP | Files | Size |
|----|------------|-------|------|
| `jingnw-nfs-weights-1` | 10.128.0.92 | 12 safetensors | 322 GB |
| `jingnw-nfs-weights-2` | 10.128.15.231 | 12 safetensors | 350 GB |
| `jingnw-nfs-weights-3` | 10.128.0.45 | 10 safetensors | 292 GB |

NFS demo job: `scripts/mimo_v25_pro_nfs_demo_job.yaml`  
MoE loading rate via NFS: **~5–7 s/group** vs gcsfuse ~14–17 s/group.  
Total MoE load time: **~42 min** vs gcsfuse ~2h25m.

---

### 5. Documentation ✅

| Doc | Contents |
|-----|----------|
| [gke_tpu7x_env_setup.md](gke_tpu7x_env_setup.md) | DWS node pool setup, resubmit workflow, known pitfalls |
| [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) | HBM/RAM/GCS allocation for 4-node and 2-node |
| [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) | Test 1–5 runbooks |
| [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) | Module-by-module pipeline walkthrough |
| [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md) | Benchmark plan + full results |
| [mimo_v25_pro_weight_checkpoint.md](mimo_v25_pro_weight_checkpoint.md) | Checkpoint analysis, plan, implementation |
| [mimo_v25_pro_progress.md](mimo_v25_pro_progress.md) | This document |

---

## In Progress

### 6. Orbax checkpoint save/load 🔄

**Goal**: Replace ~42 min weight loading with ~90s checkpoint restore from GCS.

**Approach**: After first full load, save sharded Orbax checkpoint to GCS. Each
TC saves/loads only its 30 GB shard (vs 240 GB full node read). 8× less I/O
per node.

**Checkpoint location**:
```
gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/
  95dc2640/
    tp32_bfloat16/           ← Orbax sharded checkpoint (~962 GB, 32 shards)
    tp32_bfloat16_abstract_state.pkl  ← pytree structure metadata (~KB)
```

**Status**: Checkpoint saved. Restore loads in **88–94 seconds at ~5.5 GiB/s**
per host. Currently debugging a post-restore issue:

| Attempt | Error | Fix |
|---------|-------|-----|
| Restore v1 | `tree structures do not match` (weight vs weight_q) | Save abstract_state.pkl alongside checkpoint |
| Restore v2 | `Block-wise kernel: out_dim=128` | Set `allow_narrow_n_blockwise=True` post-restore |
| Restore v3 (in progress) | `ShapeDtypeStruct is not valid JAX type` | Revert pre-restore flag; patch layers post-restore |

Current fix: `_patch_narrow_blockwise()` — after `nnx.update(model, state)`,
iterate all FP8 linear layers and set `allow_narrow_n_blockwise=True`. This
replicates what `load_weights()` did at checkpoint save time.

**Latest commit**: `9c400b1` — running now.

**Implementation files**:
- `python/sgl_jax/srt/model_loader/loader.py` — checkpoint save/load logic
- `scripts/mimo_v25_pro_nfs_demo_job.yaml` — sets `SGLANG_CHECKPOINT_DIR`

---

## Pending / Planned

### 7. Checkpoint restore validation ⬜

Once restore v3 succeeds, measure:
- Total startup time with checkpoint (target: <10 min vs ~42 min NFS load)
- Inference quality (same output as non-checkpoint run)
- Record in [mimo_v25_pro_weight_checkpoint.md](mimo_v25_pro_weight_checkpoint.md)

### 8. Scheduler throughput investigation (Opt-1d) ⬜

The `#running-req: 2` ceiling has resisted all flag-level fixes. Next step:
add debug logging to `get_new_batch_prefill()` in `managers/scheduler.py` to
trace `batch_is_full` state and `add_one_req` return codes per request.

### 9. EP > 1 (Opt-2) ⬜

Enable expert parallelism by setting `ep_size > 1` in the model config and
updating the MoE sub-mesh wiring in `weight_utils.py`. Expected: proportional
MoE throughput improvement (up to 8× with EP=8).

### 10. FP8 GMM kernel tuning (Opt-3) ⬜

Block size sweep for the EPMoE GEMM (`tm`, `tn`, `tk`) to improve per-step
efficiency. Currently uses default block sizes logged as
`[GMM kernel] using default block sizes`.

---

## Key Infrastructure State

| Resource | Status | Notes |
|----------|--------|-------|
| `jingnw-nfs-weights-1/2/3` | RUNNING | 3 × n2-highmem-48, ~$15–18/hr total |
| `mimo-v25-pro-nfs-demo` job | RUNNING | Checkpoint restore v3 in progress |
| GCS checkpoint | SAVED | `95dc2640/tp32_bfloat16/` |
| XLA compilation cache | WARM | `gs://.../jax-compilation-cache/` |

---

## Related Commits (recent)

| Commit | Description |
|--------|-------------|
| `9c400b1` | fix(checkpoint): patch allow_narrow_n_blockwise post-restore |
| `f2b1262` | fix(checkpoint): set flag before apply_linear_quantization (reverted) |
| `ed43d80` | fix(checkpoint): set allow_narrow_n_blockwise=True on restore path |
| `b091c36` | fix(checkpoint): save abstract state structure alongside checkpoint |
| `fe13a9a` | fix(checkpoint): clean path — dtype.__name__ not str() |
| `4ec6db6` | fix(checkpoint): check commit_success.txt |
| `3004a85` | feat(tpu7x): orbax checkpoint save/load implementation |
| `aa4e34d` | feat(tpu7x): timing instrumentation + new question |
| `4458f86` | feat(tpu7x): NFS weight loading demo job |
