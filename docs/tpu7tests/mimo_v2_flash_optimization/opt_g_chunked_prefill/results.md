# Opt G — Chunked Prefill Tuning: Results

**Date**: 2026-06-15 — in progress
**Status**: Benchmark job `mimo-v2-flash-1node-opt-g` submitted, awaiting results

---

## Approach

Sweep `--chunked-prefill-size` (cps) to find the optimal prefill chunk granularity
with the new production config (`--page-size 32`).

**Why cps matters at page-size=32**: With page-size=32 the engine sustains conc=16
(534 tok/s). At this concurrency, long-prompt prefills (2K+ tokens) can block decode
for hundreds of ms if served as a single chunk. Smaller cps interleaves decode steps
between prefill chunks, keeping TPOT low for concurrent requests.

**New measurement: Phase 5** (long-input concurrency sweep, input=2048, output=64).
Standard phases (2, 3, 4) use input≤512, so cps has no effect on them — all
512-token prefills fit in one chunk regardless of cps. Phase 5 is the discriminating
measurement.

**New baseline** (page-size=32, cps=2048): 534 tok/s @ conc=16 — measured 2026-06-15.

---

## Test Configs

| Config | chunked-prefill-size | chunks per 2048-tok prompt | page-size |
|--------|---------------------:|---------------------------:|----------:|
| A | 512 | 4 | 32 |
| B | 1024 | 2 | 32 |
| C (baseline) | 2048 | 1 | 32 |

---

## Results

### Phase 2: Standard concurrency sweep (input=512, output=256)

Expected: **identical across all cps values** — 512 tokens < all tested cps, so
no chunking occurs. Any difference is measurement noise.

| conc | cps=512 | cps=1024 | cps=2048 (baseline) |
|-----:|--------:|---------:|--------------------:|
| 8  | — | — | 371.8 |
| 16 | — | — | 534.0 |
| 32 | — | — | 527.6 |

*(pending — configs A and B)*

### Phase 5: Long-input concurrency sweep (input=2048, output=64)

Key measurement. Smaller cps → better TPOT for concurrent requests during long prefills.

| conc | cps=512 tok/s | cps=1024 tok/s | cps=2048 tok/s |
|-----:|--------------:|---------------:|---------------:|
| 1  | — | — | — |
| 2  | — | — | — |
| 4  | — | — | — |
| 8  | — | — | — |
| 16 | — | — | — |
| 32 | — | — | — |

*(all pending)*

---

## GCS Results

```
gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-opt-g/
  flash_opt_g_cps512_{timestamp}.json    — Config A
  flash_opt_g_cps1024_{timestamp}.json   — Config B
  flash_opt_g_cps2048_{timestamp}.json   — Config C (baseline reference)
```

---

## Conclusion

*(pending — to be filled after benchmark completes)*
