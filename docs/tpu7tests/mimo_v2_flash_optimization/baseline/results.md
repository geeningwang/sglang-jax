# MiMo-V2-Flash sglang-jax Baseline Performance Results

**Date**: 2026-06-08
**Cluster**: `jingnw-tpu7-cluster`, zone `us-central1-c`
**Pool**: `jingnw-dws-tpu7-4ch` (tpu7x-standard-4t, topology 2x2x1, cloud-platform scope)
**Config**: tp=8, bf16, mem-fraction-static=0.75, page-size=16, chunked-prefill-size=2048, max-running-requests=32
**Startup**: Orbax checkpoint restore from `e0e89a7d/tp8_bfloat16/` (~7 min)
**GCS results**: `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-baseline/flash_baseline_20260608T100546Z.json`

---

## Phase 2: Concurrency Sweep (input=512 tok, output=256 tok)

| conc | tok/s | TPOT (ms) | e2e p50 (s) | e2e p90 (s) | vs conc=1 | efficiency |
|------|------:|----------:|------------:|------------:|----------:|----------:|
| 1    | 111.0 |       9.0 |       2.349 |       2.397 |     1.00× |      100% |
| 2    | 183.4 |      10.9 |       2.793 |       2.804 |     1.65× |       83% |
| 4    | 261.6 |      15.3 |       3.904 |       4.262 |     2.36× |       59% |
| **8** | **371.2** | **21.6** | **5.503** | **5.677** | **3.34×** | **42%** |
| 16   | 372.2 |      39.3 |      11.113 |      11.212 |     3.35× |       21% |
| 32   | 370.2 |      75.5 |      21.958 |      22.409 |     3.34× |       10% |

**Throughput plateau at conc=8** (~371 tok/s). Adding more concurrent requests beyond 8 increases latency proportionally without adding throughput. This is the HBM bandwidth ceiling — confirmed weight-bandwidth-bound, consistent with Maxtext opt4 post-mortem finding.

---

## Phase 3: Prefill / TTFT Sweep (output=1 tok, concurrency=1)

| input (tok) | TTFT p50 (s) | TTFT p90 (s) | prefill tok/s |
|------------:|-------------:|-------------:|--------------:|
|         128 |        0.052 |        0.064 |        2,462  |
|         256 |        0.053 |        0.053 |        4,830  |
|         512 |        0.056 |        0.057 |        9,143  |
|       1,024 |        0.109 |        0.110 |        9,395  |
|       2,048 |        0.135 |        0.136 |       15,170  |
|       4,096 |        0.190 |        0.191 |       21,558  |

Prefill throughput scales strongly with input length — compute-bound at longer contexts. TTFT under 200 ms even for 4K-token inputs.

---

## Phase 4: Output Length Sweep (input=512 tok, concurrency=16)

| output (tok) | tok/s | TPOT (ms) | e2e p50 (s) | e2e p90 (s) |
|-------------:|------:|----------:|------------:|------------:|
|           64 | 432.9 |      34.0 |       2.404 |       2.476 |
|          128 | 397.7 |      36.9 |       5.163 |       5.279 |
|          256 | 377.8 |      38.8 |      10.832 |      11.012 |
|          512 | 359.3 |      40.9 |      22.846 |      22.938 |
|        1,024 | 352.2 |      41.7 |      46.639 |      46.702 |

Throughput is relatively stable across output lengths (352–433 tok/s). TPOT grows slowly with output length due to KV cache growth, but remains manageable.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Peak decode throughput | **~372–433 tok/s** (conc=8–16, output=64–256) |
| Optimal concurrency | **8** (throughput plateau; no gain at 16 or 32) |
| conc=1 decode | **111 tok/s**, TPOT=9.0 ms |
| Peak throughput gain (conc=8 vs 1) | **3.34×** |
| TTFT @ 512 tok | **56 ms** (9,143 tok/s prefill) |
| TTFT @ 4096 tok | **190 ms** (21,558 tok/s prefill) |

### Interpretation

The throughput plateau at conc=8 confirms the **weight-bandwidth-bound** hypothesis:

- At tp=8, each TC holds ~40 GB of expert weights in HBM
- Each decode step reads all expert weights once per layer: 47 layers × 3 projections × 32 local experts × H × I × 2 bytes ≈ 80 GB/step/TC
- At conc=8, HBM bandwidth is saturated — adding more requests just serializes steps, trading throughput for latency

This directly points to **Opt A (FP8 weight quantization)** as the highest-value next step: halving expert weight reads would target the exact bottleneck, with potential 1.5–2× throughput gain.

**Prefill is not the bottleneck**: 9–21K tok/s prefill throughput is strong. Opt D (sparse MoE prefill) is lower priority than Opt A.

---

## Reproduce

```bash
# Submit the baseline job
kubectl apply -f scripts/mimo_v2_flash_1node_baseline_job.yaml

# Or run sweep directly against a running server
python3 scripts/perf_sweep_flash.py \
  --server http://localhost:8080 \
  --model "MiMo-V2-Flash" \
  --n-requests 20 \
  --phase 0 \
  --result-path /tmp/flash_baseline.json
```
