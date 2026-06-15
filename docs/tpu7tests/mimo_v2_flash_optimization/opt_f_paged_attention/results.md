# Opt F — Paged Attention Tuning: Results

**Date**: 2026-06-15 — in progress
**Status**: Benchmark job `mimo-v2-flash-1node-opt-f` submitted, awaiting results

---

## Approach

Sweep `--page-size` to find the optimal KV cache page granularity for the
MiMo-V2-Flash workload (input=512, output=256, tp=8, conc=8).

**Baseline**: page-size=16, TPOT=21.6ms, 371 tok/s @ conc=8 (2026-06-08)

### Why page-size matters

| page-size | pages per seq (in=512, out=256) | max KV waste | page-table entries |
|----------:|--------------------------------:|-------------:|-------------------:|
| 8         | 96                              | 7 tokens (3%)| high               |
| **16**    | **48**                          | **15 tokens (6%)** | **medium (baseline)** |
| 32        | 24                              | 31 tokens (12%)| low              |
| 64        | 12                              | 63 tokens (25%)| very low         |

Smaller pages → less KV cache waste for short/variable-length sequences.
Larger pages → less page-table metadata overhead per sequence.

The optimal page-size balances fragmentation waste against metadata overhead
at the target concurrency and output length.

---

## Test Configs

| Config | page-size | chunked-prefill-size | vs baseline |
|--------|----------:|---------------------:|-------------|
| Baseline (ref) | 16 | 2048 | — |
| A | 8 | 2048 | smaller pages |
| B | 32 | 2048 | larger pages |

---

## Results

### Config A: page-size=8

*(pending)*

| conc | tok/s | TPOT (ms) | vs baseline |
|-----:|------:|----------:|------------:|
| 1 | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |
| 16 | — | — | — |
| 32 | — | — | — |

Prefill (output=1, conc=1):

| input_tok | TTFT p50 | prefill tok/s |
|----------:|---------:|--------------:|
| 128 | — | — |
| 512 | — | — |
| 1024 | — | — |
| 4096 | — | — |

### Config B: page-size=32

*(pending)*

| conc | tok/s | TPOT (ms) | vs baseline |
|-----:|------:|----------:|------------:|
| 1 | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |
| 16 | — | — | — |
| 32 | — | — | — |

Prefill (output=1, conc=1):

| input_tok | TTFT p50 | prefill tok/s |
|----------:|---------:|--------------:|
| 128 | — | — |
| 512 | — | — |
| 1024 | — | — |
| 4096 | — | — |

---

## GCS Results

```
gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-opt-f/
  flash_opt_f_pagesz8_{timestamp}.json   — Config A full sweep
  flash_opt_f_pagesz32_{timestamp}.json  — Config B full sweep
```

Baseline reference:
```
gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-baseline/
  flash_baseline_20260608T100546Z.json
```

---

## Conclusion

*(pending — to be filled after benchmark completes)*
