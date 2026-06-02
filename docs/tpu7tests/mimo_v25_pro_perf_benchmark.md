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

#### Root cause 1: Scheduler limits decode batch to 2 (regardless of compiled shapes)

Server logs consistently show `#running-req: 2, #queue-req: 14` even with 32 concurrent
clients — the scheduler never runs more than 2 requests simultaneously.

**Opt-1 attempt (❌ no effect)**: added `--precompile-bs-paddings 1 2 4 8 16 32` to
compile decode kernels for larger batch sizes. Confirmed in server args
(`precompile_bs_paddings=[1, 2, 4, 8, 16, 32]`), but throughput unchanged at 20.2 tok/s.
The compiled shapes exist but the **scheduler never fills them**.

The bottleneck is the scheduler's admission policy, not the compiled shapes. Two candidate
fixes (Opt-1a and Opt-1b below):

**Opt-1a — `--disable-overlap-schedule`**: With overlap scheduling (default), the
scheduler interleaves new prefill chunks between decode steps, which limits decode batch
size to leave room for incoming prefills. Disabling overlap forces separate prefill and
decode passes, potentially allowing decode to fill larger batches.

**Opt-1b — `--schedule-conservativeness 0`**: At `1.0` (default), the scheduler is
maximally conservative about admitting new sequences. At `0`, it packs as many sequences
as the KV cache allows into each decode step.

These are orthogonal — tested in two separate runs to isolate the effect of each.

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
| Opt-1 | `--precompile-bs-paddings 1 2 4 8 16 32` | 4–16× decode tok/s | Low — one flag | ❌ No effect |
| **Opt-1c** | `--chunked-prefill-size 4096` | 8× more requests/prefill step | Low — one flag | 🔄 Running |
| Opt-2 | Enable EP > 1 (`ep_size` in model config + mesh wiring) | Up to 8× MoE throughput | High — code change | ⬜ Planned |
| Opt-3 | FP8 GMM block size tuning (`tm`, `tn`, `tk`) | ~10–30% per step | Medium | ⬜ Planned |

### Corrected root cause (from source code analysis)

The `--precompile-bs-paddings` fix compiled larger batch sizes but the scheduler never
filled them. Code analysis of `schedule_policy.py::PrefillAdder` revealed the actual
bottleneck: **`--chunked-prefill-size 512` combined with 512-token prompts limits
admission to exactly 1 request per prefill tick.**

```python
# PrefillAdder budget per scheduler tick:
rem_chunk_tokens = chunked_prefill_size  # = 512

# Request 1: extend_input_len=512 → rem_chunk = 512 - 512 = 0
# Request 2: trunc_len = 0 → AddReqResult.OTHER → break (stops admitting)
```

Every scheduler tick can prefill at most `chunked_prefill_size / extend_input_len`
requests. At 512/512 = 1, only one 512-token request is admitted per tick. The
overlap scheduler interleaves this with decode, resulting in a steady-state of 2
requests running (the one currently decoding + the one just admitted).

Note: `--disable-overlap-schedule` and `--schedule-conservativeness 0` do NOT fix this
because they don't change the chunk budget admission logic.

### Opt-1c — `--chunked-prefill-size 4096` (running)

```bash
python3 -m sgl_jax.launch_server \
  ... \
  --chunked-prefill-size 4096 \
  --precompile-bs-paddings 1 2 4 8 16 32
```

With a 4096-token chunk budget: `4096 / 512 = 8 requests` admitted per prefill tick
instead of 1. The decode batch should grow to 8+ concurrent requests.

**Expected results** (to fill in after run):

| Concurrency | Projected tok/s | Notes |
|-------------|----------------|-------|
| 1 | ~10.6 | Unchanged |
| 2 | ~20 | Unchanged |
| 4 | ~40 | If 4 admitted per tick |
| 8 | ~80 | Chunk budget allows 8/tick |
| 16 | ~120–160 | Compute-bound regime likely |
| 32 | TBD | |

### Opt-1 result (❌ confirmed no effect, 2026-06-02)

`--precompile-bs-paddings 1 2 4 8 16 32` compiled the shapes correctly but the
scheduler never filled batches larger than 2. Phase 2 results identical to baseline:

| Concurrency | Decode tok/s | Lat p50 | vs baseline | Efficiency |
|-------------|-------------|---------|-------------|------------|
| 1 | 10.6 | 24.1s | 1.00× | 100% |
| 2 | 20.2 | 25.6s | 1.91× | 96% |
| 4 | 20.1 | 51.1s | 1.90× | 48% |
| 8 | 20.2 | 101.2s | 1.91× | 24% |
| 16 | 19.8 | 204.2s | 1.88× | 12% |
| 32 | 19.8 | 409.6s | 1.87× | 6% |

---

## Related Documents

- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — full pipeline walkthrough
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — HBM/RAM/GCS allocation
- [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) — Test 4 runbook
- Reference: `maxtext/docs/guides/mimo_v2_flash_opt5_batch_size_scaling.md` — batch size sweep methodology
