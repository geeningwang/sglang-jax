# Opt F — Paged Attention Tuning: Results

**Date**: 2026-06-15
**Status**: Complete — page-size=32 is the new recommended config

---

## Approach

Swept `--page-size` to find the optimal KV cache page granularity.
Baseline is page-size=16 (measured 2026-06-08). Two alternative configs tested
in a single provisioning slot (sequential server restarts).

**Baseline**: page-size=16, TPOT=21.6ms, **371 tok/s @ conc=8**, plateau at conc=8+.

### Why page-size matters

| page-size | pages per seq (in=512, out=256) | max KV waste | overhead |
|----------:|--------------------------------:|-------------:|---------|
| 8  | 96  | 7 tokens  (3%) | high page-table traversal |
| **16** | **48** | **15 tokens (6%)** | **baseline** |
| 32 | 24  | 31 tokens (12%) | low overhead |

Smaller pages → less KV cache fragmentation but more page-table entries to
traverse each step. Larger pages → less overhead but more wasted KV slots per
page boundary.

---

## Results

### Phase 2: Concurrency sweep (input=512, output=256)

| conc | p8 tok/s | **p16 (baseline)** | **p32 tok/s** | p32 vs baseline |
|-----:|---------:|-------------------:|--------------:|----------------:|
| 1    | 77.4     | **111**            | 90.3          | -19%            |
| 2    | 155.3    | **183**            | 169.8         | -7%             |
| 4    | 255.9    | **262**            | 260.7         | -1%             |
| 8    | 255.3    | **371**            | 365.6         | -1.5%           |
| 16   | 251.7    | **372**            | **526.8**     | **+42%**        |
| 32   | —        | **370**            | **528.3**     | **+43%**        |

### Phase 3: Prefill sweep (output=1, conc=1)

| in_tok | p8 TTFT | p16 TTFT (baseline) | p32 TTFT | p32 prefill tok/s |
|-------:|--------:|--------------------:|---------:|------------------:|
| 128    | 53ms    | 52ms                | 53ms     | 2,415             |
| 256    | —       | —                   | 55ms     | 4,655             |
| 512    | 44ms    | **56ms**            | **58ms** | 8,828             |
| 1024   | —       | 109ms               | 111ms    | 9,225             |
| 2048   | —       | —                   | 137ms    | 14,949            |
| 4096   | —       | 190ms               | 192ms    | 21,333            |

Prefill TTFT is essentially unchanged (within 2ms at all lengths).

### Phase 4: Output length sweep (input=512, conc=32)

| out_tok | p16 tok/s (est) | p32 tok/s | TPOT (ms) |
|--------:|----------------:|----------:|----------:|
| 64      | —               | 648.2     | 45.4      |
| 128     | —               | 590.8     | 49.6      |
| 256     | 370             | **533.3** | 55.0      |
| 512     | —               | 504.9     | 58.0      |
| 1024    | —               | 485.1     | 60.5      |

---

## Summary table

| Metric | page-size=8 | page-size=16 (baseline) | **page-size=32** | p32 vs p16 |
|--------|------------:|------------------------:|-----------------:|-----------:|
| Peak tok/s | 255.9 @ c=4 | 371 @ c=8 | **528 @ c=32** | **+42%** |
| TPOT @ conc=8 | 28.7ms | 21.6ms | **21.9ms** | ~flat |
| TTFT @ 512 tok | 44ms | 56ms | 58ms | +2ms |
| Peak prefill | ~21,900 tok/s | 21,558 tok/s | 21,333 tok/s | ~flat |

---

## Analysis

**page-size=8**: Worse at all concurrency levels. The 2× more page-table entries
per sequence cause extra overhead each decode step — throughput plateaus at
~256 tok/s vs 371 tok/s baseline at conc=8.

**page-size=32**: Unlocks a new throughput regime. The baseline plateau at conc=8
(371 tok/s) was caused by page-table overhead limiting effective concurrency.
With page-size=32 (half the page-table entries), the engine scales to conc=16-32
and achieves **528 tok/s — a 42% gain over baseline at optimal concurrency**.

At conc=8, page-size=32 matches the baseline (366 vs 371 tok/s, within 1.5%). So
switching to page-size=32 has no downside for existing conc=8 deployments and
provides substantial headroom for higher concurrency workloads.

The higher max KV waste (31 vs 15 tokens) does not hurt measured throughput —
at conc=32 with page-size=32, the server remains stable across all output lengths
up to 1024.

**TTFT is unaffected**: 56ms (p16) vs 58ms (p32) at 512-token input — within noise.

---

## Decision: Adopt page-size=32

Switch the production config from `--page-size 16` to `--page-size 32`.

**New peak**: 528 tok/s @ conc=32 with page-size=32 (+42% vs 371 baseline).

The recommendation is to re-run the baseline sweep with `--page-size 32
--max-running-requests 32` to establish the new official baseline with this config.

---

## GCS Results

```
gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-opt-f/
  flash_opt_f_pagesz8_20260615T020714Z.json   — Config A (page-size=8)
  flash_opt_f_pagesz32_20260615T020714Z.json  — Config B (page-size=32) ← winner
```
