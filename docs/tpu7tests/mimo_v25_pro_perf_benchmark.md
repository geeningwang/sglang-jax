# MiMo-V2.5-Pro Performance Benchmark — 4-Node TPU v7x

## Status: Running (2026-06-01)

---

## Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Baseline measurement — single request, record TP/EP config, prefill and decode tok/s | ✅ Done |
| **2** | Concurrent request sweep — vary client concurrency, measure decode throughput scaling | 🔄 Running |
| **3** | Prefill length sweep — vary input length, measure prefill tok/s and TTFT | 🔄 Running |
| **4** | Output length sweep — vary output length at optimal concurrency, measure decode tok/s | 🔄 Running |
| **5** | Identify throughput-optimal concurrency; record final TP/EP/prefill/decode numbers | ⬜ Pending results |

**Primary metrics**: output tok/s (decode throughput), prefill tok/s, TTFT (time to first token).

**Parallelism context**: TP=32 is the only viable configuration for this model on 4 nodes
(TP=16 is infeasible — model weights consume ~60 GB/TC, leaving no HBM for KV cache).
EP (expert parallelism) is derived from the MoE sub-mesh within the TP-32 mesh.

---

## Hardware and Configuration

| Item | Value |
|------|-------|
| Cluster | `jingnw-tpu7-cluster`, zone `us-central1-c` |
| Node pool | `jingnw-dws-tpu7-16ch` (2x2x4 DWS) |
| Nodes | 4 |
| Chips per node | 4 (tpu7x-standard-4t) |
| TensorCores total | 32 (tp-size=32) |
| HBM per TensorCore | 96 GB |
| Total HBM | 3072 GB |
| Runtime | sglang-jax |
| Model | MiMo-V2.5-Pro (FP8, 70 layers, 384 experts, top-8 routing) |
| Weights | `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` (~962 GB FP8) |
| XLA cache | `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/` |

### HBM budget per TensorCore (at `--mem-fraction-static 0.75`)

| Pool | Per TC | Notes |
|------|--------|-------|
| Model weights (FP8) | ~30 GB | 962 GB ÷ 32 TCs |
| KV cache | ~43 GB | Allocated by profiler after weights |
| XLA temporaries | ~24 GB | 25% of 96 GB; required for 384-expert MoE GEMM |

### Parallelism axes

| Axis | Value | How set |
|------|-------|---------|
| TP (tensor parallelism) | 32 | `--tp-size 32` |
| EP (expert parallelism) | derived from `ep_size` in model config | Sub-mesh `("expert", "tensor")` inside weight loader |

> **Note on TP sweep**: TP=16 (2-node) is not viable — model weights fill ~60 GB/TC leaving no KV cache headroom (confirmed OOM in earlier test). TP=32 is the only supported config for this model on TPU v7x.

---

## Baseline (Phase 1) ✅

Measured from smoke test runs (2026-05-28, 2026-06-01):

| Metric | Measured value | Notes |
|--------|---------------|-------|
| TP | 32 | `--tp-size 32`, 4 nodes × 4 chips × 2 TCs |
| EP | implicit (from model `ep_size`) | Sub-mesh `("expert", "tensor")` |
| `--max-running-requests` | 2 | Smoke test config |
| Concurrent requests | 1 (single demo request) | Load level during measurement |
| Prefill tokens | 272 | Demo prompt length |
| Prefill throughput | — | Not isolated in smoke test |
| Decode throughput | **10.81 tok/s** | Steady-state, single request |
| Time to first token | ~5 s | Prefill 272 tokens |
| Max output tokens | 256 | Demo cap |
| Server startup (warm XLA) | ~2h15m | Weight loading dominates; XLA ~55s |

The 10.81 tok/s baseline is a **single-sequence** result. sglang-jax batches multiple
concurrent requests, so throughput is expected to scale with concurrency up to the
hardware saturation point (weight-bandwidth-bound → compute-bound transition).

---

## Phase 2 — Concurrent Request Sweep 🔄

### Hypothesis

sglang-jax decode is weight-bandwidth-bound at low concurrency: each decode step reads
~30 GB of FP8 expert weights per TensorCore regardless of how many tokens are being
decoded. Throughput should scale near-linearly with concurrency until:
1. **Compute bound**: FLOPS/TC saturated (unlikely before batch > 32 for these dimensions)
2. **HBM capacity**: KV cache exhaustion (~43 GB available per TC)
3. **ICI bandwidth**: all-reduce data volume grows with batch

### Sweep plan

The benchmark server is launched with `--max-running-requests 32`. Client-side
concurrency is varied from 1 to 32 via `perf_sweep.py`, which drives N parallel
async requests for each step.

| Step | Client concurrency | Fixed input | Fixed output | Notes |
|------|--------------------|-------------|--------------|-------|
| A | 1 | 512 tok | 256 tok | Single-request baseline |
| B | 2 | 512 tok | 256 tok | |
| C | 4 | 512 tok | 256 tok | First scaling point |
| D | 8 | 512 tok | 256 tok | Monitor step latency growth |
| E | 16 | 512 tok | 256 tok | May enter mixed regime |
| F | 32 | 512 tok | 256 tok | Run only if E is still near-linear |

### KV cache capacity estimate

With tp-size=32, 70 attention layers, 8 KV heads, head_dim=128, BF16:

$$\text{KV per TC per sequence} \approx 2 \times 70 \times \frac{8}{32} \times L \times 128 \times 2 \approx 89.6 \text{ KB} \times L$$

where L is the sequence length. At L=512 tokens: ~46 MB per sequence per TC.
With ~43 GB KV cache per TC, this supports ~930 concurrent sequences at 512 tokens.
KV cache is not the limiting factor in this sweep — compute and ICI bandwidth are
more likely to bind first.

### Metrics recorded per step

1. **Decode throughput** (tok/s): total output tokens / wall time across all requests
2. **Latency p50 / p90** (s): per-request end-to-end latency percentiles
3. **Scaling efficiency**: `throughput_at_N / (baseline_throughput × N)`

---

## Phase 3 — Prefill Length Sweep 🔄

Concurrency=1, output=1 token (isolates prefill). Varies input length to measure
TTFT and prefill tok/s across prompt sizes.

| Step | Input tokens | Notes |
|------|-------------|-------|
| A | 128 | Short prompt |
| B | 256 | |
| C | 512 | |
| D | 1024 | Chunked prefill active (`--chunked-prefill-size 512`) |
| E | 2048 | 4 prefill chunks |

**Prefill tok/s** = `input_tokens / TTFT_p50`. Chunked prefill (>512 tokens) may
reduce throughput slightly due to chunk coordination overhead.

**TTFT** = time from request submission to first output token (includes prefill + first decode step).

---

## Phase 4 — Output Length Sweep 🔄

Client concurrency fixed at throughput-optimal level from Phase 2. Varies max output
tokens to characterize decode tok/s across generation budgets.

| Step | Output tokens | Notes |
|------|--------------|-------|
| A | 64 | Short generation |
| B | 128 | |
| C | 256 | Smoke test setting |
| D | 512 | |

Records per step: decode tok/s, latency p50/p90, total request latency.

---

## Expected Results Table

Fill in as sweep runs complete.

### Phase 2 — Concurrent request scaling

| `--max-running-requests` | Decode tok/s | TTFT median | Step latency | Throughput vs baseline | Scaling efficiency | Status |
|--------------------------|-------------|-------------|--------------|----------------------|-------------------|--------|
| 1 | 10.81 | ~5 s | ~92 ms | 1.00× | 100% | ✅ Baseline |
| 2 | — | — | — | — | — | ⬜ |
| 4 | — | — | — | — | — | ⬜ |
| 8 | — | — | — | — | — | ⬜ |
| 16 | — | — | — | — | — | ⬜ |
| 32 | — | — | — | — | — | ⬜ |

### Phase 3 — Prefill length sweep

| Input tokens | Prefill tok/s | TTFT median | Notes | Status |
|-------------|--------------|-------------|-------|--------|
| 128 | — | — | | ⬜ |
| 256 | — | — | | ⬜ |
| 512 | — | — | | ⬜ |
| 1024 | — | — | Chunked prefill | ⬜ |
| 2048 | — | — | Chunked prefill | ⬜ |
| 4096 | — | — | Chunked prefill | ⬜ |

### Phase 4 — Output length sweep (at optimal concurrency)

| Output tokens | Decode tok/s | TTFT | Total latency | Status |
|--------------|-------------|------|---------------|--------|
| 64 | — | — | — | ⬜ |
| 128 | — | — | — | ⬜ |
| 256 | — | — | — | ⬜ |
| 512 | — | — | — | ⬜ |
| 1024 | — | — | — | ⬜ |

---

## Infrastructure

### Benchmark job

`scripts/mimo_v25_pro_bench_job.yaml` — 4-node DWS job that:
1. Mounts weights via gcsfuse, launches server with `--max-running-requests 32`
2. Waits for `/health`
3. Runs `scripts/perf_sweep.py` (all 3 phases automatically)
4. Prints results to stdout and writes `/tmp/perf_benchmark_results.json`

```bash
# Submit
kubectl apply -f scripts/mimo_v25_pro_bench_job.yaml

# Watch progress
kubectl logs -f -l job-name=mimo-v25-pro-bench --prefix

# Clean up
kubectl delete -f scripts/mimo_v25_pro_bench_job.yaml
```

### Benchmark script

`scripts/perf_sweep.py` — async request driver using `aiohttp`:

```
--server        server URL (default: http://localhost:8080)
--n-requests    requests per sweep step (default: 20)
--phase         0=all phases, 2/3/4=single phase
```

Output: formatted tables to stdout + `/tmp/perf_benchmark_results.json`.

Prompt generation: synthetic English technical text padded to ~target token count
(~3.5 chars/token), varied by request index to avoid cache hits.

---

## Parallelism Configuration Reference

### Tensor Parallelism (TP)

| Config | TP | TCs | Weights/TC | KV cache/TC | Feasible |
|--------|----|-----|-----------|-------------|---------|
| 2-node | 16 | 16 | ~60 GB | ~0 GB | ❌ No KV cache headroom |
| 4-node | **32** | **32** | **~30 GB** | **~43 GB** | **✅ Production config** |

TP=32 is the only viable setting for MiMo-V2.5-Pro on TPU v7x with this HBM budget.

### Expert Parallelism (EP)

EP in sglang-jax is implicitly derived from `ep_size` in the HuggingFace model
config and the JAX mesh shape. For MiMo-V2.5-Pro:

- The MoE sub-mesh `("expert", "tensor")` is built inside the weight loader
- `ep_size × tp_size = world_size = 32`
- The split between EP and TP within the MoE experts depends on `config.ep_size`

EP is not a runtime flag in sglang-jax (unlike MaxText's `ici_expert_parallelism`).
It is fixed by the model config and cannot be swept without code changes.

---

## Exit Criteria

| Criterion | Target | Pass if |
|-----------|--------|---------|
| Baseline decode throughput | > 10 tok/s | ✅ Already 10.81 tok/s |
| Peak decode throughput (concurrent) | > 100 tok/s | TBD |
| Prefill throughput | > 1,000 tok/s | TBD |
| No OOM at optimal concurrency | Required | TBD |
| Scaling efficiency at 4× concurrency | > 50% | TBD |

---

## Next Steps

1. ✅ `scripts/perf_sweep.py` — created (commit `02e2980`)
2. ✅ `scripts/mimo_v25_pro_bench_job.yaml` — created (commit `fcc418e`)
3. 🔄 Benchmark job `mimo-v25-pro-bench` submitted (2026-06-01); awaiting results
4. ⬜ Fill in results tables (Phases 2–4) from job logs once complete
5. ⬜ Identify throughput-optimal concurrency and record final numbers in Phase 5

---

## Related Documents

- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — full pipeline walkthrough
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — HBM/RAM/GCS allocation
- [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) — Test 4 runbook
- Reference: `maxtext/docs/guides/mimo_v2_flash_opt5_batch_size_scaling.md` — batch size sweep methodology
