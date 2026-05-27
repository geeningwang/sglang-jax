# GKE TPU v7x Resource Allocation — MiMo-V2.5-Pro Demo

Measured resource allocations for the 4-node 2x2x4 MiMo-V2.5-Pro inference demo
(`scripts/mimo_v25_pro_demo_job.yaml`). Numbers come from the pod spec and from
Cloud Logging profiling output captured during a successful run (2026-05-27).

---

## Hardware topology

| Unit | Count | Notes |
|------|-------|-------|
| Nodes | 4 | `tpu7x-standard-4t`, 2x2x4 DWS slice |
| Chips per node | 4 | Each chip has 2 TensorCores |
| TensorCores total | 32 | tp-size=32 |
| HBM per TensorCore | 96 GB | Independent JAX device |
| **Total HBM** | **3072 GB** | 32 × 96 GB |

---

## GCS storage

| Resource | Size | Notes |
|----------|------|-------|
| Model weights (`hf-weights/`, 34 safetensors, FP8) | ~962 GB | `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` |
| JAX compilation cache | ~85 MB | `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/`; accumulates incrementally across restarts |

The compilation cache key encodes model hash, TPU topology, and XLA version. Changing
`--tp-size` or the container image invalidates cached entries.

---

## Hard disk (per node)

| Resource | Per node | Total (4 nodes) |
|----------|----------|-----------------|
| hyperdisk-balanced | 100 GB | 400 GB |

Used only for OS and the container image. Model weights are not stored on disk —
they are streamed from GCS via gcsfuse into a RAM-backed file cache.

---

## Main memory / RAM

| Allocation | Per pod | 4-pod total |
|------------|---------|-------------|
| Pod request / limit | 900 Gi | 3600 Gi |
| gcsfuse file cache (`emptyDir medium: Memory`) | up to 850 Gi | up to 3400 Gi |
| OS + Python process + gcsfuse daemon | ~50 Gi | ~200 Gi |

The gcsfuse cache (`--file-cache-max-size-mb=800000`, 800 GB LRU limit) holds
recently-accessed weight chunks in RAM. The 34 safetensors files (~962 GB total)
do not fully fit, so LRU eviction keeps hot MoE expert files resident. First access
of each file downloads it from GCS; subsequent accesses are served from RAM (~10×
faster than GCS FUSE).

---

## HBM (TPU High Bandwidth Memory)

### Top-level split

`--mem-fraction-static 0.75` divides each TensorCore's 96 GB into two pools:

| Pool | Per TensorCore | Per node (8 TCs) | Total (32 TCs) |
|------|---------------|-----------------|----------------|
| Static (weights + KV cache) | 72 GB | 576 GB | 2304 GB |
| XLA temporaries (remaining 25%) | 24 GB | 192 GB | 768 GB |

The XLA temporary pool covers scratch buffers during the forward pass (e.g. MoE
expert GEMM intermediates). At `mem-fraction-static=0.92` this pool was only 8%
(~7.7 GB per TensorCore), insufficient for the 384-expert MoE forward pass —
reducing to 0.75 provides 24 GB per TensorCore for temporaries.

### Static pool breakdown (from profiling logs)

Per node (8 TensorCores, 768 GB total HBM):

| Use | Per node | Per TensorCore | Notes |
|-----|----------|----------------|-------|
| Model weights (FP8, sharded across 32 TCs) | ~240 GB | ~30 GB | 962 GB ÷ 4 nodes |
| KV cache — attention layers | 286.20 GB | ~35.8 GB | 156,288 tokens; bfloat16 |
| KV cache — MLA/linear layers | 59.62 GB | ~7.5 GB | 195,360 tokens; bfloat16 |
| **Total static used** | **~586 GB** | **~73 GB** | |

### KV cache capacity

| Metric | Value |
|--------|-------|
| Max KV tokens (attention) | 156,288 per node |
| Max KV tokens (MLA/linear) | 195,360 per node |
| KV cache dtype | bfloat16 |
| Page size | 16 tokens |
| Context length | 1,048,576 |

The profiling step also reports `available_kv_cache=10.8 GB` per TensorCore — this
is the residual measured mid-profiling while the max-batch forward pass temporarily
occupies scratch buffers. The final allocated KV cache (~43 GB per TensorCore) is
larger because scratch memory is released after the profiling pass completes.

---

## Why 4 nodes instead of 2

At tp-size=16 (2 nodes, 16 TensorCores), model weights fill ~93% of the 72 GB static
pool per TensorCore, leaving only ~5 GB for KV cache — too small for useful context.
Doubling to tp-size=32 halves the per-TensorCore weight footprint to ~30 GB, freeing
~42 GB per TensorCore for KV cache.

---

## Summary

| Resource | Allocated |
|----------|-----------|
| GCS weights | ~962 GB |
| GCS compilation cache | ~85 MB |
| Hard disk (total) | 400 GB (OS/image only) |
| RAM (total, 4 pods) | 3600 Gi requested; up to 3400 Gi as gcsfuse cache |
| HBM (total) | 3072 GB |
| HBM — static pool | 2304 GB (75%) |
| HBM — XLA temporaries | 768 GB (25%) |
| HBM — weights | ~960 GB |
| HBM — KV cache | ~1384 GB (all nodes combined) |
