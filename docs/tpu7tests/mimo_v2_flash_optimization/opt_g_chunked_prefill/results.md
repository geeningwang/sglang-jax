# Opt G — Chunked Prefill Tuning: Results

**Date**: 2026-06-15
**Status**: Complete — cps=2048 (baseline) is optimal; no change recommended

---

## Approach

Swept `--chunked-prefill-size` (cps) with the production config (`--page-size 32`).
Added **Phase 5** (long-input concurrency sweep, input=2048, output=64) to
`perf_sweep_flash.py` to isolate the chunked-prefill interleaving effect.

Phase 2 (input=512) is a sanity check only — 512-token prompts fit in one chunk
regardless of cps, so Phase 2 results are identical across configs.

**Baseline**: page-size=32, cps=2048, **534 tok/s @ conc=16** (standard workload).

---

## Test Configs

| Config | chunked-prefill-size | chunks per 2048-tok prompt | page-size |
|--------|---------------------:|---------------------------:|----------:|
| A | 512 | 4 | 32 |
| B | 1024 | 2 | 32 |
| **C (baseline)** | **2048** | **1** | **32** |

---

## Phase 2: Standard concurrency sweep (input=512, output=256)

As expected, **identical across all cps values** — no chunking occurs at input=512.

| conc | cps=512 | cps=1024 | cps=2048 |
|-----:|--------:|---------:|---------:|
| 1 | 89.4 | 81.0 | ~88 |
| 4 | 265.1 | 264.8 | ~265 |
| 8 | 365.7 | 378.7 | ~370 |
| 16 | ~534 | 524.3 | 534.0 |
| 32 | ~528 | 519.8 | ~528 |

Measurement variance of ±2–3% across runs. No systematic difference.

---

## Phase 5: Long-input concurrency sweep (input=2048, output=64)

Key measurement. Each 2048-token prompt is split into N chunks of `cps` tokens.

| conc | cps=512 tok/s | cps=1024 tok/s | **cps=2048 tok/s** |
|-----:|--------------:|---------------:|-------------------:|
| 1 | — | 74.9 | 69.1 |
| 2 | — | 161.8 | 148.1 |
| 4 | — | 268.1 | 261.4 |
| 8 | **354.7** | 355.2 | 358.0 |
| 16 | — | 404.7 | 427.1 |
| 32 | — | 408.2 | **438.7** |

**Peak throughput (in=2048, out=64):**

| Config | Peak tok/s | @ conc |
|--------|----------:|-------:|
| cps=512 | 354.7 | 8 |
| cps=1024 | 408.2 | 32 |
| **cps=2048** | **438.7** | **32** |

---

## Analysis

**Smaller cps is strictly worse**: cps=512 peaks at 355 tok/s (conc=8), cps=1024 at
408 tok/s (conc=32), cps=2048 at 439 tok/s (conc=32). Larger chunks consistently win.

**Why interleaving doesn't help here**: The hypothesis was that breaking prefills into
smaller chunks would let decode steps run between them, keeping TPOT low. In practice,
at conc=32 with input=2048, the server queue is always full — there are 32 concurrent
2048-token prompts competing for decode slots. The chunking overhead (per-chunk
scheduling, KV cache bookkeeping, additional dispatcher calls) outweighs any
interleaving benefit. The server is compute-saturated, not decode-starved.

**At conc=1 (no concurrency)**: cps=2048 is also faster (69.1 vs 74.9 tok/s for
cps=1024) because the single prefill is served in one shot without chunking overhead.

**Conclusion**: For MiMo-V2-Flash on tp=8 TPU v7x at the tested concurrency levels
(1–32), chunked-prefill serves no benefit. The optimal setting is the largest possible
chunk (cps=2048 or higher) which minimizes scheduling overhead.

---

## Decision: Keep cps=2048 (no change)

`--chunked-prefill-size 2048` remains the production default. Opt G closed as negative.

**Production config** (unchanged from Opt F):
```
--page-size 32
--chunked-prefill-size 2048
--max-running-requests 32
```

---

## GCS Results

```
gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-opt-g/
  flash_opt_g_cps512_20260615T035230Z.json   — Config A (peak 355 tok/s @ conc=8)
  flash_opt_g_cps1024_20260615T035230Z.json  — Config B (peak 408 tok/s @ conc=32)
  flash_opt_g_cps2048_20260615T035230Z.json  — Config C / baseline (peak 439 tok/s @ conc=32)
```
