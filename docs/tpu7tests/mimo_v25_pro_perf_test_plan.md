# MiMo-V2.5-Pro Performance Test Plan — 4-Node TPU v7x

## Status: Draft

---

## Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Baseline measurement — single request, record TP/EP config, prefill and decode tok/s | ⬜ Planned |
| **2** | Concurrent request sweep — vary `--max-running-requests`, measure decode throughput scaling | ⬜ Planned |
| **3** | Prefill length sweep — vary input length, measure prefill tok/s and TTFT | ⬜ Planned |
| **4** | Output length sweep — vary output length at optimal concurrency, measure decode tok/s | ⬜ Planned |
| **5** | Identify throughput-optimal concurrency; record final TP/EP/prefill/decode numbers | ⬜ Planned |

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

## Baseline (Phase 1)

Already measured from smoke test runs (2026-05-28, 2026-06-01):

| Metric | Measured value | Notes |
|--------|---------------|-------|
| `--max-running-requests` | 2 | Server config |
| Concurrent requests | 1 (single demo request) | Load level during measurement |
| Prefill tokens | 272 | Demo prompt length |
| Prefill throughput | — | Not directly measured yet |
| Decode throughput | **10.81 tok/s** | Steady-state single-request |
| Time to first token | ~5 s | Prefill 272 tokens |
| Max output tokens | 256 | Demo cap |
| Server startup | ~2h15m (warm XLA) | Weight loading dominates |

The 10.81 tok/s baseline is a **single-sequence** result. sglang-jax batches multiple
concurrent requests, so throughput is expected to scale with concurrency up to the
hardware saturation point (weight-bandwidth-bound → compute-bound transition).

---

## Phase 2 — Concurrent Request Sweep

### Hypothesis

sglang-jax decode is weight-bandwidth-bound at low concurrency: each decode step reads
~30 GB of FP8 expert weights per TensorCore regardless of how many tokens are being
decoded. Throughput should scale near-linearly with concurrency until:
1. **Compute bound**: FLOPS/TC saturated (unlikely before batch > 32 for these dimensions)
2. **HBM capacity**: KV cache exhaustion (~43 GB available per TC)
3. **ICI bandwidth**: all-reduce data volume grows with batch

### Sweep plan

Vary `--max-running-requests` and drive the server to saturation with a parallel
request script. For each concurrency level, measure steady-state decode throughput
and TTFT.

| Step | `--max-running-requests` | Effective batch | Notes |
|------|--------------------------|-----------------|-------|
| A | 1 | 1 | Single-request baseline |
| B | 2 | 2 | Current smoke test config |
| C | 4 | 4 | First scaling point |
| D | 8 | 8 | Monitor step latency growth |
| E | 16 | 16 | May enter mixed regime |
| F | 32 | 32 | Run only if E is still near-linear |

### KV cache capacity estimate

With tp-size=32, 70 attention layers, 8 KV heads, head_dim=128, BF16:

$$\text{KV per TC per sequence} \approx 2 \times 70 \times \frac{8}{32} \times L \times 128 \times 2 \approx 89.6 \text{ KB} \times L$$

where L is the sequence length. At L=512 tokens: ~46 MB per sequence per TC.
With ~43 GB KV cache per TC, this supports ~930 concurrent sequences at 512 tokens.
KV cache is not the limiting factor in this sweep — HBM temporaries during XLA
compilation and compute are more likely to bind first.

### Load generation

For each step, saturate the server with N parallel clients, each sending a request
with fixed input and output length:

```bash
# Set CONCURRENCY and SERVER for each sweep step
CONCURRENCY=4   # ← match --max-running-requests
SERVER="http://<rank0-pod-ip>:8080"
INPUT_TOKENS=512
OUTPUT_TOKENS=256

python3 scripts/perf_sweep.py \
  --server ${SERVER} \
  --concurrency ${CONCURRENCY} \
  --input-tokens ${INPUT_TOKENS} \
  --output-tokens ${OUTPUT_TOKENS} \
  --num-requests 50 \
  --output bench_concurrent_${CONCURRENCY}.json
```

### Metrics to record per step

1. **Decode throughput** (tok/s): total output tokens / total wall time across all requests
2. **TTFT** (ms): time from request submission to first output token, median and p90
3. **Decode step latency** (ms): per-step latency in the decode loop, derived from server logs
4. **Scaling efficiency**: `throughput_at_N / (baseline_throughput × N)`

---

## Phase 3 — Prefill Length Sweep

Fix `--max-running-requests 2`, vary input prompt length. Measures how prefill
throughput and TTFT change with prompt size.

| Step | Input tokens | Expected TTFT | Notes |
|------|-------------|---------------|-------|
| A | 128 | ~1–2 s | Short prompt |
| B | 256 | ~2–4 s | |
| C | 512 | ~4–8 s | |
| D | 1024 | ~8–15 s | Chunked prefill active (`--chunked-prefill-size 512`) |
| E | 2048 | ~15–30 s | 4 prefill chunks |
| F | 4096 | ~30–60 s | 8 prefill chunks |

**Prefill tok/s** = `input_tokens / TTFT`. Record at each step.

**Chunked prefill note**: prompts longer than `--chunked-prefill-size` (512) are
split into chunks. Prefill tok/s may decrease slightly for very long prompts due to
chunk coordination overhead.

```bash
INPUT_LEN=512   # ← change per step
OUTPUT_LEN=1    # fix output to 1 token to isolate prefill

python3 scripts/perf_sweep.py \
  --server ${SERVER} \
  --concurrency 1 \
  --input-tokens ${INPUT_LEN} \
  --output-tokens ${OUTPUT_LEN} \
  --num-requests 20 \
  --output bench_prefill_${INPUT_LEN}.json
```

---

## Phase 4 — Output Length Sweep

Fix `--max-running-requests` at the throughput-optimal level found in Phase 2.
Vary output length to characterize decode throughput at different generation budgets.

| Step | Output tokens | Notes |
|------|--------------|-------|
| A | 64 | Short generation |
| B | 128 | |
| C | 256 | Current smoke test setting |
| D | 512 | |
| E | 1024 | Long generation |

For each step record: decode tok/s, TTFT, total request latency.

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

### Benchmark server YAML

The benchmark requires the server to stay alive after the demo inference completes.
Use a modified job that removes `kill $SERVER_PID` from the rank0 script and adds
a long sleep, then access via `kubectl port-forward`.

```bash
# Port-forward rank0 HTTP server to localhost
kubectl port-forward pod/mimo-v25-pro-demo-0-<suffix> 8080:8080

# Test health
curl http://localhost:8080/health

# Send a request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MiMo-V2.5-Pro", "messages": [{"role": "user", "content": "..."}],
       "max_tokens": 256}'
```

### Benchmark script (to be created)

`scripts/perf_sweep.py` — parallel request driver with configurable:
- `--concurrency`: number of parallel requests
- `--input-tokens`: prompt length (padded with synthetic tokens if needed)
- `--output-tokens`: max tokens to generate
- `--num-requests`: total requests to send
- `--output`: JSON file for results

Records per-request: TTFT, total latency, output tokens, decode tok/s.
Aggregates: throughput, latency percentiles (p50, p90, p99).

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

1. Create `scripts/perf_sweep.py` — parallel request driver
2. Modify `scripts/mimo_v25_pro_demo_job.yaml` to keep server alive for benchmarking (long-running variant)
3. Run Phase 2 sweep (concurrency 1→32)
4. Run Phase 3 sweep (prefill length 128→4096)
5. Run Phase 4 sweep (output length 64→1024) at optimal concurrency
6. Record results and identify throughput-optimal `--max-running-requests`

---

## Related Documents

- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — full pipeline walkthrough
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — HBM/RAM/GCS allocation
- [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) — Test 4 runbook
- Reference: `maxtext/docs/guides/mimo_v2_flash_opt5_batch_size_scaling.md` — batch size sweep methodology
