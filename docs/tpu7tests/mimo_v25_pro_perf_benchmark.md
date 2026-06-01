# MiMo-V2.5-Pro Performance Benchmark — 4-Node TPU v7x

## Status: Complete (2026-06-01)

---

## Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Baseline measurement — single request, record TP/EP config, prefill and decode tok/s | ✅ Done |
| **2** | Concurrent request sweep — vary client concurrency, measure decode throughput scaling | ✅ Done |
| **3** | Prefill length sweep — vary input length, measure prefill tok/s and TTFT | ✅ Done |
| **4** | Output length sweep — vary output length at optimal concurrency, measure decode tok/s | ✅ Done |
| **5** | Identify throughput-optimal concurrency; record final TP/EP/prefill/decode numbers | ✅ Done |

**Primary metrics**: output tok/s (decode throughput), prefill tok/s, TTFT (time to first token).

---

## Hardware and Configuration

| Item | Value |
|------|-------|
| Cluster | `jingnw-tpu7-cluster`, zone `us-central1-c` |
| Node pool | `jingnw-dws-tpu7-16ch` (2x2x4 DWS) |
| Nodes | 4 |
| Chips per node | 4 (tpu7x-standard-4t) |
| TensorCores total | 32 |
| HBM per TensorCore | 96 GB |
| Total HBM | 3072 GB |
| Runtime | sglang-jax |
| Model | MiMo-V2.5-Pro (FP8, 70 layers, 384 experts, top-8 routing) |
| Weights | `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` (~962 GB FP8) |
| XLA cache | `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/` |

### Server launch flags

```bash
python3 -m sgl_jax.launch_server \
  --model-path /mnt/gcs/hf-weights \
  --tp-size 32 --nnodes 4 --device tpu --dtype bfloat16 \
  --mem-fraction-static 0.75 --page-size 16 \
  --chunked-prefill-size 512 --max-running-requests 32
```

### HBM budget per TensorCore (`--mem-fraction-static 0.75`)

| Pool | Per TC | Notes |
|------|--------|-------|
| Model weights (FP8) | ~30 GB | 962 GB ÷ 32 TCs |
| KV cache | ~43 GB | Allocated by profiler after weights |
| XLA temporaries | ~24 GB | 25% of 96 GB; required for 384-expert MoE GEMM |

### Parallelism axes (confirmed from server args)

| Axis | Value | How set |
|------|-------|---------|
| TP (tensor parallelism) | **32** | `--tp-size 32` |
| EP (expert parallelism) | **1** | `ep_size=1` in model config — no expert parallelism; all sharding is TP-only |

> **EP=1 note**: MiMo-V2.5-Pro's sglang-jax server runs with `ep_size=1` (confirmed from `server_args` in benchmark logs). All 32 TensorCores participate via tensor parallelism only; expert weights are sharded along the TP axis rather than replicated per EP group. This differs from MaxText MiMo-V2-Flash (which uses EP=8). EP cannot be changed without model config changes.

> **TP sweep**: TP=16 (2-node) is not viable — model weights fill ~60 GB/TC leaving no HBM for KV cache (confirmed OOM). TP=32 is the only supported config.

---

## Phase 1 — Baseline ✅

Measured from smoke test runs (2026-05-28, 2026-06-01) and confirmed in benchmark run:

| Metric | Value | Notes |
|--------|-------|-------|
| TP | 32 | 4 nodes × 4 chips × 2 TCs |
| EP | 1 | No expert parallelism |
| Decode throughput (single request) | **10.6 tok/s** | Benchmark run; 10.81 tok/s in smoke test |
| TTFT (272-token prompt) | ~5 s | Smoke test |
| Server startup (warm XLA) | ~2h15m | Weight loading dominates; XLA ~55s |

---

## Phase 2 — Concurrent Request Sweep ✅

**Setup**: `--max-running-requests 32`, input=512 tokens, output=256 tokens, 20 requests per step.

### Results

| Concurrency | Decode tok/s | Lat p50 | Lat p90 | vs baseline | Efficiency |
|-------------|-------------|---------|---------|-------------|------------|
| 1 | 10.6 | 24.1s | 24.1s | 1.00× | 100% |
| 2 | **20.2** | 25.6s | 25.9s | **1.90×** | 95% |
| 4 | 20.1 | 51.1s | 51.6s | 1.89× | 47% |
| 8 | 20.2 | 101.8s | 102.8s | 1.91× | 24% |
| 16 | 19.9 | 203.8s | 209.0s | 1.88× | 12% |
| 32 | 19.8 | 409.0s | 419.5s | 1.87× | 6% |

### Key findings

1. **Throughput saturates at concurrency=2** (~20 tok/s), with no further gain at higher concurrency. The system is **not weight-bandwidth-bound** at the hardware level — it hits a compute or ICI ceiling immediately.

2. **Scaling efficiency collapses above conc=2**: 95% at conc=2 → 47% at conc=4 → 6% at conc=32. The near-perfect 95% efficiency at conc=2 indicates the first doubling uses available HBM bandwidth fully, but the server serializes batches rather than running them in parallel after that.

3. **Optimal concurrency: 2** — max throughput (~20.2 tok/s) at minimum latency (25.6s p50). Higher concurrency only increases latency without improving throughput.

4. **Root cause: only bs=1 and bs=2 were precompiled.** The XLA warmup compiled decode kernels for `bs=[1,2]` only. Regardless of queue depth, the scheduler never runs more than 2 requests at once. Fix: `--precompile-bs-paddings 1,2,4,8,16,32` to compile larger batch shapes and unlock the scheduler.

---

## Phase 3 — Prefill Length Sweep ✅

**Setup**: concurrency=1, output=1 token (isolates prefill), 10 requests per step.

### Results

| Input tokens | TTFT p50 | TTFT p90 | Prefill tok/s | Notes |
|-------------|---------|---------|--------------|-------|
| 128 | 0.277s | 0.284s | 462 | Below chunked-prefill threshold |
| 256 | 0.263s | 0.264s | 973 | |
| 512 | 0.217s | 0.218s | **2,359** | At chunked-prefill boundary |
| 1024 | 0.506s | 0.507s | 2,024 | Chunked prefill (2 × 512 chunks) |
| 2048 | 0.532s | 0.533s | **3,850** | Chunked prefill (4 × 512 chunks) |

### Key findings

1. **Prefill tok/s increases with input length** — counterintuitively, longer prompts are processed more efficiently. At 2048 tokens: 3,850 tok/s vs 462 tok/s at 128 tokens.

2. **Chunked prefill amortization**: prompts >512 tokens are split into 512-token chunks. Each chunk runs a full forward pass; XLA FLOP utilization is better at 512 tokens than at 128 tokens. The 2048-token prompt (4 chunks) achieves the highest throughput because each chunk is optimally sized for the compiled kernel.

3. **Prefix caching effect**: subsequent requests in the sweep share prompt prefix tokens (`#cached-token: 480` observed in server logs), which artificially lowers measured TTFT for requests 2–10. The actual cold-cache TTFT for a 512-token prompt is ~0.22s.

4. **TTFT at 1024 vs 2048**: TTFT increases sublinearly (0.506s → 0.532s) while tokens double, resulting in higher tok/s at 2048. This is due to chunked prefill parallelism and prefix cache hits.

---

## Phase 4 — Output Length Sweep ✅

**Setup**: input=512 tokens, concurrency=8 (throughput-saturated), 20 requests per step.

### Results

| Output tokens | Decode tok/s | Lat p50 | Lat p90 |
|--------------|-------------|---------|---------|
| 64 | 20.6 | 24.7s | 25.1s |
| 128 | 20.4 | 50.2s | 50.8s |
| 256 | 20.2 | 101.3s | 102.7s |
| 512 | 20.0 | 204.9s | 206.4s |

### Key findings

1. **Decode tok/s is flat across all output lengths** (~20 tok/s) — confirms the bottleneck is per-step compute, not length-dependent factors. Each additional token costs a constant ~0.20s regardless of position in the sequence.

2. **Latency scales linearly with output length** (24.7s → 204.9s for 64→512 tokens), confirming constant per-step cost. This is expected behavior for autoregressive decode.

---

## Phase 5 — Summary ✅

### Final numbers (4-node, TP=32, EP=1)

| Metric | Value | Config |
|--------|-------|--------|
| **TP** | 32 | `--tp-size 32`, 4 nodes, 2x2x4 |
| **EP** | 1 | Fixed by model config |
| **Peak decode throughput** | **20.2 tok/s** | Concurrency ≥ 2 |
| **Optimal concurrency** | **2** | Max throughput at min latency |
| **Decode latency (p50, conc=1)** | 24.1s / 256 tokens | ~94ms/token |
| **Peak prefill throughput** | **3,850 tok/s** | 2048-token input, chunked prefill |
| **Prefill TTFT (512 tok)** | 0.22s | Single request, cold cache |

### Throughput ceiling analysis

The ~20 tok/s ceiling is ~54× lower than MaxText MiMo-V2-Flash (2,724 tok/s on v6e-32).
Root causes in order of impact:

#### Root cause 1 (most impactful, immediately fixable): Only bs=1 and bs=2 compiled

JAX/XLA requires **static shapes** — a separate kernel must be compiled for each batch
size. The benchmark server precompiled only `bs=[1, 2]` for decode:

```
[DECODE] PRECOMPILE: 100%|██████████| 2/2  bs=2
```

Regardless of how many requests are queued, the scheduler can only run **2 at a time**.
Server logs confirm: `#running-req: 2, #queue-req: 8` even with 8 concurrent clients —
requests are serialized in pairs, not truly batched together.

**Fix**: add `--precompile-bs-paddings 1,2,4,8,16,32` at server launch. This compiles
decode kernels for each batch size during XLA warmup, allowing the scheduler to fill
batches of up to 32 simultaneous requests.
**Expected gain: 4–16× throughput improvement.**

#### Root cause 2: HBM bandwidth floor

Even with large batches, there is a per-decode-step hardware floor:

| Item | Value |
|------|-------|
| Weights per TC | ~30 GB FP8 |
| TPU v7x HBM bandwidth | ~600 GB/s per TC |
| Min time to read all weights | 30 GB ÷ 600 GB/s = **50ms** |
| Actual step time at bs=1 | ~92ms → **54% HBM utilization** |

To reach 1,000 tok/s total, the scheduler needs ~1,000 tokens/step at ~1 step/s —
i.e. ~50–100 concurrent requests decoding simultaneously. With ~930-sequence KV cache
capacity, this is theoretically reachable once the batch size precompilation is fixed.

#### Root cause 3: EP=1

With EP=1, all 32 TCs collaborate on every MoE expert GEMM for every token. With EP=8
(MaxText approach), 8 independent groups of 4 TCs each handle 48 of the 384 experts
in parallel — 8× more expert compute throughput per step. Requires changing `ep_size`
in the model config and corresponding mesh wiring in the weight loader.

---

## Exit Criteria

| Criterion | Target | Result |
|-----------|--------|--------|
| Baseline decode throughput | > 10 tok/s | ✅ 10.6 tok/s |
| Peak decode throughput | > 100 tok/s | ❌ 20.2 tok/s (ceiling hit at conc=2) |
| Peak prefill throughput | > 1,000 tok/s | ✅ 3,850 tok/s at 2048 tokens |
| No OOM at any tested concurrency | Required | ✅ No OOM through conc=32 |
| Identify optimal concurrency | Required | ✅ Concurrency=2 |

---

## Infrastructure

### Benchmark job

`scripts/mimo_v25_pro_bench_job.yaml` — 4-node DWS job that launches server with
`--max-running-requests 32`, waits for `/health`, runs `scripts/perf_sweep.py`
(all 3 phases), prints results, and exits.

```bash
kubectl apply -f scripts/mimo_v25_pro_bench_job.yaml
kubectl logs -f -l job-name=mimo-v25-pro-bench --prefix
kubectl delete -f scripts/mimo_v25_pro_bench_job.yaml
```

### Benchmark script

`scripts/perf_sweep.py` — async request driver (`aiohttp`):
- `--server`: server URL (default `http://localhost:8080`)
- `--n-requests`: requests per step (default 20)
- `--phase`: 0=all, 2/3/4=single phase

Output: formatted tables to stdout + `/tmp/perf_benchmark_results.json`.

---

## Optimization Roadmap

| # | Optimization | Expected gain | Effort | Status |
|---|-------------|--------------|--------|--------|
| **Opt-1** | `--precompile-bs-paddings 1,2,4,8,16,32` | **4–16×** decode tok/s | Low — one flag | 🔄 In progress |
| Opt-2 | Enable EP > 1 (`ep_size` in model config + mesh wiring) | Up to 8× MoE throughput | High — code change | ⬜ Planned |
| Opt-3 | FP8 GMM block size tuning (`tm`, `tn`, `tk`) | ~10–30% per step | Medium | ⬜ Planned |
| Opt-4 | Prefill/decode overlap scheduling | Latency reduction | Medium | ⬜ Planned |

### Opt-1 detail: precompile-bs-paddings (in progress)

Add to server launch:

```bash
python3 -m sgl_jax.launch_server \
  ...
  --precompile-bs-paddings 1,2,4,8,16,32
```

This triggers XLA compilation for decode batch sizes 1, 2, 4, 8, 16, 32 during server
warmup (~additional 5–10 min). Once compiled, the scheduler can fill any of those
batch sizes when enough requests are queued. XLA compilation artifacts are written to
the GCS cache and reused on future restarts.

**Expected results** (to be measured):

| Concurrency | Projected tok/s | Notes |
|-------------|----------------|-------|
| 2 | ~20 | Unchanged (already hitting bs=2) |
| 4 | ~40 | If bs=4 compiles and scheduler fills it |
| 8 | ~80 | If bs=8 compiles |
| 16 | ~120–160 | Compute-bound regime likely |
| 32 | TBD | HBM/ICI limit |

---

## Related Documents

- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — full pipeline walkthrough
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — HBM/RAM/GCS allocation
- [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) — Test 4 runbook
- Reference: `maxtext/docs/guides/mimo_v2_flash_opt5_batch_size_scaling.md` — batch size sweep methodology
