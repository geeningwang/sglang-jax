# GKE TPU v7x Resource Allocation — MiMo-V2.5-Pro Demo

Resource allocations for the MiMo-V2.5-Pro inference demo in two configurations:
- **4-node** (`scripts/mimo_v25_pro_demo_job.yaml`): tp-size=32, 2x2x4 DWS slice
- **2-node** (`scripts/mimo_v25_pro_2node_demo_job.yaml`): tp-size=16, 2x2x2 DWS slice

Numbers come from the pod spec and Cloud Logging profiling output captured during
successful runs.

---

## Configuration comparison

| Parameter | 4-node | 2-node |
|-----------|--------|--------|
| Script | `mimo_v25_pro_demo_job.yaml` | `mimo_v25_pro_2node_demo_job.yaml` |
| Node pool | `jingnw-dws-tpu7-16ch` | `jingnw-dws-tpu7-8ch` |
| TPU topology | 2x2x4 | 2x2x2 |
| Nodes | 4 | 2 |
| TensorCores | 32 | 16 |
| Total HBM | 3072 GB | 1536 GB |
| `--tp-size` | 32 | 16 |
| `--nnodes` | 4 | 2 |
| `--mem-fraction-static` | 0.75 | 0.75 |
| `--max-running-requests` | 2 | 1 |
| Weights per TC | ~30 GB | ~60 GB |
| KV cache per TC | ~43 GB | ~12 GB (minimal) |
| XLA temp per TC | 24 GB | 24 GB |

---

## Hardware topology

### 4-node (2x2x4)

| Unit | Count | Notes |
|------|-------|-------|
| Nodes | 4 | `tpu7x-standard-4t`, 2x2x4 DWS slice |
| Chips per node | 4 | Each chip has 2 TensorCores |
| TensorCores total | 32 | `--tp-size 32` |
| HBM per TensorCore | 96 GB | Independent JAX device |
| **Total HBM** | **3072 GB** | 32 × 96 GB |

### 2-node (2x2x2)

| Unit | Count | Notes |
|------|-------|-------|
| Nodes | 2 | `tpu7x-standard-4t`, 2x2x2 DWS slice |
| Chips per node | 4 | Each chip has 2 TensorCores |
| TensorCores total | 16 | `--tp-size 16` |
| HBM per TensorCore | 96 GB | Independent JAX device |
| **Total HBM** | **1536 GB** | 16 × 96 GB |

---

## GCS storage (shared by both configurations)

| Resource | Size | Notes |
|----------|------|-------|
| Model weights (`hf-weights/`, 34 safetensors, FP8) | ~962 GB | `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` |
| JAX compilation cache | ~85 MB | `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/` |

The compilation cache key encodes model hash, TPU topology, and XLA version.
**Changing `--tp-size` invalidates the cache** — the 4-node and 2-node configs each
build their own cache entry. With a warm cache, XLA warmup takes ~55 s instead of 15+ h.

---

## Hard disk (per node / total)

| Config | Per node | Total |
|--------|----------|-------|
| 4-node | 100 GB | 400 GB |
| 2-node | 100 GB | 200 GB |

Used only for OS and container image. Model weights are not stored on disk —
they are streamed from GCS via gcsfuse into a RAM-backed file cache.

---

## Main memory / RAM

The RAM profile depends on the weight-loading mode. Two modes have been measured:

---

### Mode A — NFS + Orbax checkpoint restore (current production path)

**TPU worker nodes** (4×, pod limit 900 Gi each):

| Allocation | Per node | 4-node total | Notes |
|------------|----------|-------------|-------|
| Pod request / limit | 900 Gi | 3600 Gi | Hard limit set in pod spec |
| OS + kernel | ~2 GB | ~8 GB | Linux system |
| Container runtime (containerd, kubelet, GKE agents) | ~1 GB | ~4 GB | |
| Python process (JAX runtime, NNX model graph, sglang-jax, tokenizer) | ~8–15 GB | ~32–60 GB | Steady-state after restore |
| **Orbax I/O buffer** (GCS OCDBT streaming read window) | **up to 89.4 GB** | **up to 358 GB** | `restore_concurrent_bytes=96,000,000,000`; released after restore |
| JAX host-side staging (FP8 monkey-patch, uint8 shard buffers) | ~3–6 GB peak | ~12–24 GB peak | Each shard ~3 MB; serial, transient |
| XLA compilation cache (HLO modules in RAM) | ~2–5 GB | ~8–20 GB | Warm after first run |
| NFS client page cache (tokenizer + config files) | <0.1 GB | <0.5 GB | Weights come from GCS, not NFS in fast path |
| **Peak (during restore)** | **~100–110 GB** | **~400–440 GB** | Dominated by Orbax buffer |
| **Steady-state (serving)** | **~15–20 GB** | **~60–80 GB** | Buffer released; model is in HBM |
| **Headroom** | **~780–785 GB** | | Of 900 Gi pod limit |

> `enable_pinned_host_transfer=False` confirmed in logs — Orbax does **not** use pinned (page-locked) host memory. The 89.4 GiB is a soft concurrency cap on in-flight GCS reads, not a reserved allocation.

**NFS weight servers** (3×, always-on, not accessed during fast-restore):

| VM | Machine type | Total RAM | tmpfs used | Safetensors files |
|----|-------------|-----------|-----------|-------------------|
| `jingnw-nfs-weights-1` | n2-highmem-48 | **384 GB** | ~322 GB | 12 files |
| `jingnw-nfs-weights-2` | n2-highmem-48 | **384 GB** | ~350 GB | 12 files |
| `jingnw-nfs-weights-3` | n2-highmem-48 | **384 GB** | ~292 GB | 10 files |
| **Total NFS RAM** | | **1,152 GB** | **~964 GB** | 34 files |

NFS servers are used **only** for the slow-path (first-run weight load, ~42 min).
On checkpoint-restore runs, NFS is mounted read-only but only tokenizer/config
files (KBs) are accessed.

---

### Mode B — gcsfuse direct (legacy, slow path)

| Allocation | Per pod | 4-pod total |
|------------|---------|-------------|
| Pod request / limit | 900 Gi | 3600 Gi |
| gcsfuse file cache (`emptyDir medium: Memory`) | up to 850 Gi | up to 3400 Gi |
| OS + Python process + gcsfuse daemon | ~50 Gi | ~200 Gi |

The gcsfuse cache (`--file-cache-max-size-mb=800000`, 800 GB LRU limit) held
recently-accessed weight chunks in RAM. The 34 safetensors files (~962 GB) did
not fully fit, so hot MoE experts stayed resident. Load time: ~2h25m.
**This mode is no longer used.**

---

## HBM (TPU High Bandwidth Memory)

`--mem-fraction-static 0.75` divides each TensorCore's 96 GB into two pools
for **both** configurations:

| Pool | Per TensorCore | Notes |
|------|---------------|-------|
| Static (weights + KV cache) | 72 GB | 75% |
| XLA temporaries | 24 GB | 25%; required for 384-expert MoE GEMM |

### 4-node static pool breakdown (per node, 8 TensorCores, 768 GB total)

| Use | Per node | Per TensorCore | Notes |
|-----|----------|----------------|-------|
| Model weights (FP8 MoE + BF16 attn, sharded ÷ 32 TCs) | ~240 GB | ~30 GB | 962 GB ÷ 32 |
| KV cache — full layers (60 SWA layers) | 286.64 GB | ~35.8 GB | 156,528 tokens; bfloat16 (measured) |
| KV cache — SWA layers (10 layers) | 59.72 GB | ~7.5 GB | 195,664 tokens; bfloat16 (measured) |
| **Total static used** | **~586 GB** | **~73 GB** | |

### 2-node static pool breakdown (per node, 8 TensorCores, 768 GB total)

| Use | Per node | Per TensorCore | Notes |
|-----|----------|----------------|-------|
| Model weights (FP8, sharded ÷ 16 TCs) | ~480 GB | ~60 GB | 962 GB ÷ 16 |
| KV cache (minimal) | ~96 GB | ~12 GB | weights double per TC vs 4-node |
| **Total static used** | **~576 GB** | **~72 GB** | |

### KV cache capacity

| Metric | 4-node | 2-node |
|--------|--------|--------|
| KV cache per TC | ~43 GB | ~12 GB |
| Total KV cache | ~1384 GB | ~192 GB |
| KV cache dtype | bfloat16 | bfloat16 |
| Page size | 16 tokens | 16 tokens |
| Max running requests | 2 | 1 (minimal) |

The 2-node KV cache is intentionally minimal — just enough for a single-request
smoke test. The tight budget (~12 GB per TC) leaves no room for concurrent requests.

---

## Launch settings

### 4-node

```bash
python3 -m sgl_jax.launch_server \
  --model-path /mnt/gcs/hf-weights \
  --trust-remote-code \
  --tp-size 32 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --page-size 16 \
  --chunked-prefill-size 512 \
  --max-running-requests 2 \
  --host 0.0.0.0 \
  --port 8080 \
  --nnodes 4 \
  --node-rank <rank> \
  --dist-init-addr <coordinator>:6006
```

### 2-node

```bash
python3 -m sgl_jax.launch_server \
  --model-path /mnt/gcs/hf-weights \
  --trust-remote-code \
  --tp-size 16 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --page-size 16 \
  --chunked-prefill-size 512 \
  --max-running-requests 1 \
  --host 0.0.0.0 \
  --port 8080 \
  --nnodes 2 \
  --node-rank <rank> \
  --dist-init-addr <coordinator>:6006
```

---

## Why 4 nodes vs 2 nodes

At tp-size=16 (2 nodes, 16 TensorCores), model weights occupy ~60 GB per TensorCore
out of the 72 GB static pool, leaving only ~12 GB for KV cache. This is sufficient
for a single-request smoke test but too small for production use.

At tp-size=32 (4 nodes, 32 TensorCores), weights halve to ~30 GB per TensorCore,
freeing ~43 GB for KV cache (~3.5× more context capacity).

| Config | Weights/TC | KV cache/TC | Usable for |
|--------|-----------|-------------|------------|
| 2-node (tp-16) | ~60 GB | ~12 GB | Smoke test, single request |
| 4-node (tp-32) | ~30 GB | ~43 GB | Production, concurrent requests |

---

## Summary

| Resource | 4-node | 2-node |
|----------|--------|--------|
| GCS checkpoint | ~482 GB logical | — |
| GCS weights (HF safetensors) | ~962 GB | ~962 GB (shared) |
| GCS compilation cache | ~85 MB | ~85 MB (separate cache key) |
| Hard disk total | 400 GB | 200 GB |
| **RAM — TPU nodes (pods, limit)** | **3600 Gi** | **1800 Gi** |
| RAM — TPU nodes, peak (restore) | ~400–440 GB | — |
| RAM — TPU nodes, steady-state | ~60–80 GB | — |
| RAM — NFS weight servers (tmpfs) | **~964 GB** (3 VMs) | ~964 GB (slow path only) |
| HBM total | 3072 GB | 1536 GB |
| HBM static pool (75%) | 2304 GB | 1152 GB |
| HBM XLA temp (25%) | 768 GB | 384 GB |
| HBM weights | ~962 GB | ~962 GB |
| HBM KV cache (measured) | **346 GB** (286.6 + 59.7) | ~192 GB |
