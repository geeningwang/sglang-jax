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

> **Corrected baseline (measured 2026-06-05)**: JAX-visible HBM per TensorCore on
> TPU v7x is **101.73 GB**, not 96 GB. All pool sizes below use this figure.

`--mem-fraction-static 0.75` divides each TensorCore's 101.73 GB into two pools:

| Pool | Per TensorCore | Notes |
|------|---------------|-------|
| Static (weights + KV cache) | **76.30 GB** | 75% of 101.73 GB |
| XLA temporaries | **25.43 GB** | 25%; EPMoE EXTEND compile needs ~20 GB minimum |

### EPMoE XLA temp pool requirement (measured)

EPMoE with 384 experts, block-wise FP8 GEMM requires **~20 GB XLA temp** for EXTEND
(prefill) kernel compilation. Tested at tp-32:

| `mem_fraction_static` | XLA temp | EXTEND compile | Result |
|---|---|---|---|
| 0.75 | 25.43 GB | ✅ PASS | **baseline** |
| 0.85 | 14.40 GB | ❌ OOM +5.54 GB | FAIL |
| 0.90–0.97 | < 10 GB | ❌ OOM | FAIL |

Minimum viable `mem_fraction_static` ≤ 0.803 (leaves ≥ 20 GB for XLA).
**Do not reduce `mem_fraction_static` below 0.75** without profiling EPMoE compilation.

Attempted `mem_fraction_static=0.85` with XLA rematerialization flags to close
the 5.54 GB gap — blocked: flags `--xla_tpu_rematerialization_algo=PEAK_PRIORITY`
and `--xla_enable_hlo_rematerialization` are **not available in jax0.9.0-rev1**.
Would require jax0.10.x+ container. Reverted to baseline 0.75.

### 4-node HBM breakdown per TensorCore (measured, fast-restore path)

| Allocation | Per TC | 32-TC total | How allocated |
|-----------|--------|-------------|---------------|
| `apply_moe_quantization` FP32 scales | **11.07 GB** | 354 GB | Real HBM (not abstract) — created at model init |
| Checkpoint weights + restore overhead | **53.61 GB** | 1,716 GB | 33.75 GB actual + 19.86 GB restore double-buffering |
| KV cache — 60 SWA layers | **8.96 GB** | 286.64 GB | 156,528 tokens, bfloat16 |
| KV cache — 10 full layers | **1.87 GB** | 59.72 GB | 195,664 tokens, bfloat16 |
| XLA temp pool | **25.43 GB** | 813 GB | 25% reservation |
| **Total** | **101.34 GB** | **3,243 GB** | ≤ 101.73 GB ✓ |

Key notes:
- `apply_moe_quantization(is_static_input=True)` allocates **real** FP32 scale arrays
  even though weight arrays remain abstract until checkpoint restore. This is 11 GB
  at tp-32 and **nearly identical at tp-16** (11.72 GB — nearly TP-independent).
- `nnx.split(model)` delta = 0 GB — no weight copies.
- `gc.collect() + jax.clear_caches()` frees < 10 MB — overhead is permanent.
- Restore overhead (19.86 GB) comes from **FP8 monkey-patch uint8/float8 double-buffering**:
  each of 1,038 FP8 tensor shards is first placed as uint8 (same size), then bitcast to
  float8. Python `del arr_u8` defers the actual HBM release to JAX's async GC, so uint8
  buffers accumulate across all 1,038 restores. At tp-16, shards are 2× larger → overhead
  ≈ 39 GB (OOM during restore). At tp-32, steady-state footprint is correct (GC clears by T10).
- Slow path (NFS `load_weights`) and fast path (checkpoint restore) produce **identical** HBM
  footprint: T4f delta = +0.02 GB (noise). Scale conversion and dequantization intermediates
  are correctly freed before the KV profiler runs.

### 2-node HBM analysis (tp-16) — INFEASIBLE with EP=1

> **Status**: 2-node with ep_size=1 is infeasible. OOM occurs during checkpoint
> restore before the KV profiler even runs. The only viable path is **EP > 1**.

At tp-16, weight shards double per TC. Measured/estimated breakdown:

| Allocation | Per TC | Notes |
|-----------|--------|-------|
| `apply_moe_quantization` FP32 scales | **11.72 GB** | Nearly same as tp-32 (TP-independent) |
| Checkpoint weights at tp-16 | **~90 GB** | OOM at layer 42/70 during restore |
| XLA temp (required for EPMoE) | **25.43 GB** | Same as tp-32 |
| **Available for KV** | **< 0 GB** | OOM before KV profiler runs |

Root cause: `apply_moe_quantization` (11.72 GB) + MoE weight shards (~67.5 GB) +
restore double-buffering overhead = **~102 GB**, which exceeds the 101.73 GB HBM limit.
The OOM occurs mid-restore (at layer 42 of 70 `wi_0` tensors).

Reducing `mem_fraction_static` does not help — EPMoE compilation needs ~20 GB XLA temp,
which is already close to the 25.43 GB available.

### KV cache capacity (4-node only, measured)

| Metric | 4-node (tp-32) | 2-node (tp-16) |
|--------|---------------|----------------|
| KV cache per TC | **11.62 GB** (measured) | N/A — OOM before KV alloc |
| Total KV cache | **346.36 GB** (286.64 + 59.72) | N/A |
| Max tokens | 195,664 (SWA) / 156,528 (full) | N/A |
| KV cache dtype | bfloat16 | — |
| Page size | 16 tokens | — |
| Max running requests | 2 | — |

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

## Why 4 nodes vs 2 nodes (updated 2026-06-05)

At tp-size=16 (2 nodes, 16 TensorCores), the MoE weight shards per TC double.
The full model footprint (~102 GB) exceeds the 101.73 GB HBM limit, causing OOM
**during checkpoint restore** before the server even starts.

This is not a KV cache headroom issue — the model simply cannot fit in HBM at tp-16
with ep_size=1 (all 384 experts on every TC).

The only viable path to 2-node is **EP > 1** (expert parallelism):
- With ep_size=2: each TC handles 192 experts → MoE weight per TC halves
- Combined with tp=8 within each node → same footprint per TC as 4-node tp-32

**Key formula**: `per-TC weight = total_weight / total_TCs`
EP factoring (ep × tp) does NOT change per-TC weight. Only total TC count matters.

| Config | Total TCs | wi_0/TC | Model footprint | KV/TC | Status |
|--------|-----------|---------|----------------|-------|--------|
| 4-node ep=1 tp=32 | 32 | 151 MB | 64.68 GB | 11.62 GB | ✅ Production |
| 4-node ep=2 tp=16 | 32 | 151 MB | ~64 GB (same) | ~11 GB | ⬜ Throughput opt |
| 2-node ep=1 tp=16 | 16 | 302 MB | ~112 GB | OOM | ❌ Infeasible |
| 2-node ep=2 tp=8 | 16 | 302 MB | ~112 GB (same!) | OOM | ❌ Infeasible (tested) |

---

## Summary (corrected 2026-06-05)

| Resource | 4-node | 2-node (ep-1) |
|----------|--------|---------------|
| HBM per TC (JAX-visible) | **101.73 GB** | **101.73 GB** |
| HBM limit reported in logs | 101.73 GB | 101.73 GB |
| HBM static pool (75%) | **76.30 GB/TC** | 76.30 GB/TC |
| HBM XLA temp (25%) | **25.43 GB/TC** | 25.43 GB/TC |
| apply_moe_quantization scales | **11.07 GB/TC** | 11.72 GB/TC |
| Checkpoint weights + overhead | **53.61 GB/TC** | ~90 GB/TC (OOM) |
| HBM KV cache (measured) | **11.62 GB/TC** (346 GB total) | N/A |
| EPMoE EXTEND min XLA temp | **~20 GB** | same |
| GCS checkpoint | ~482 GB logical | ~962 GB logical (tp-16 saved) |
| GCS weights (HF safetensors) | ~962 GB | ~962 GB (shared) |
| GCS compilation cache | ~85 MB | ~85 MB (separate key) |
| RAM — TPU pods (limit) | 900 Gi × 4 | 900 Gi × 2 |
| RAM — peak during restore | ~400–440 GB | — |
| RAM — steady-state serving | ~60–80 GB | — |
| RAM — NFS servers (tmpfs) | ~964 GB (3 VMs) | ~964 GB (slow path) |
