# MiMo-V2-Flash TPU v7x Inference — Optimization Plan

**Cluster**: `jingnw-tpu7-cluster`, zone `us-central1-c`, GKE TPU v7x
**Model**: `XiaomiMiMo/MiMo-V2-Flash` (48 layers, 256 experts, hidden=4096, FP8 e4m3fn weights)
**Framework**: sglang-jax (`tpu7` branch)
**Last updated**: 2026-06-09

---

## Context: Maxtext Baseline (v6e-32)

Before sglang-jax work, the model was benchmarked on Maxtext (v6e-32, TP=4 EP=8, bf16).
The Maxtext optimization history is documented in:
`/home/jingnw_google_com/maxtext/docs/guides/mimo_v2_flash_tpu_perf_optimization.md`

### Key Maxtext findings (as of 2026-04-21, commit `711f591f`)

| Optimization | Decode | Throughput | Status |
|---|---|---|---|
| Baseline (no opt) | 71.7 ms | 447 tok/s | — |
| #1 Remove `jax.debug.print` from MoE gate | 56.5 ms | 566 tok/s | ✅ |
| #2 Sparse MoE (mblx.gmm, local only) | ~56.1 ms | ~570 tok/s | ⚠️ Meas. invalid |
| #3 Int8 KV cache | 60.1 ms | 533 tok/s | ❌ Rejected (+6% slower + quality regression) |
| #4 SWA KV cache truncation | 1797 ms | 18 tok/s | ❌ Rejected (0% gain; separate regression) |
| #5 shard_map sparse EP+TP dispatch | 160 ms | 200 tok/s | ❌ Reverted (all-gather overhead) |
| #6 Revert sparse; dense dispatch | 55.7 ms | 575 tok/s | ✅ |
| #7 scan_layers=true (bench-only; demo broken) | 68.3 ms | 468 tok/s | ⚠️ +23% overhead |
| #8 SWA fix + debug logging cleanup | 55.5 ms | 576 tok/s | ✅ |
| #9 ragged_all_to_all sparse EP routing | 101.5 ms | 316 tok/s | ❌ Reverted (83% regression) |
| #10 Revert opt4; prefill bench added | 55.4 ms | 578 tok/s | ✅ |
| **#11 Batch size scaling** (`per_device_batch_size=11`, total 352) | **129.2 ms** | **2,724 tok/s** | **✅ 4.72× gain** |
| #12 Flash attention, 16K context (BS=96) | 71.3 ms | 1,347 tok/s | ✅ New 16K config |

Prefill (scan=false, 512 tokens): **123.6 ms / 4,144 tok/s** (compute-bound, independent of batch).

### Critical lesson from opt4 post-mortem

AR decode at batch=32 / T=32 is **weight-bandwidth-bound**. Each step reads ~1.5 GB
of expert weights per device (47 MoE layers × 3 projections × 32 local experts ×
H × I × 2 bytes). MoE intermediates at T=32 are only 128 KB/layer — negligible.
All three sparse dispatch variants (gmm local, shard_map, ragged_all_to_all) failed
because they don't reduce weight reads; they only reduce intermediates.

**For decode:** target weight bandwidth (quantization, larger batch, fewer EP shards).
**For prefill (T=512):** sparse routing IS valid — intermediates are 2.68 GB/layer (32× larger).

---

## sglang-jax Baseline (TPU v7x)

**Measured**: 2026-06-08, `jingnw-dws-tpu7-4ch`, tp=8, bf16, max-running-requests=32
**Full results**: [`baseline/results.md`](baseline/results.md)
**GCS**: `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-baseline/flash_baseline_20260608T100546Z.json`

### Concurrency sweep (input=512, output=256)

| conc | tok/s | TPOT (ms) | e2e p50 | vs c=1 |
|------|------:|----------:|--------:|-------:|
| 1    |  111  |       9.0 |  2.35 s |  1.00× |
| 2    |  183  |      10.9 |  2.79 s |  1.65× |
| 4    |  262  |      15.3 |  3.90 s |  2.36× |
| **8** | **371** | **21.6** | **5.50 s** | **3.34×** |
| 16   |  372  |      39.3 | 11.11 s |  3.35× |
| 32   |  370  |      75.5 | 21.96 s |  3.34× |

**Plateau at conc=8** — throughput saturates at ~371 tok/s regardless of concurrency beyond 8.

### Prefill / TTFT (output=1, conc=1)

| input | TTFT p50 | prefill tok/s |
|------:|---------:|--------------:|
|   128 |   52 ms  |         2,462 |
|   512 |   56 ms  |         9,143 |
| 1,024 |  109 ms  |         9,395 |
| 4,096 |  190 ms  |        21,558 |

### Key summary

| Metric | Value |
|--------|-------|
| Peak decode throughput | **~372–433 tok/s** (conc=8+, short outputs) |
| Optimal concurrency | **8** (HBM bandwidth ceiling) |
| conc=1 decode | **111 tok/s**, TPOT=9.0 ms |
| TTFT @ 512 tok | **56 ms** |
| TTFT @ 4096 tok | **190 ms** |
| Startup (Orbax restore) | ~7 min |

The throughput plateau at conc=8 **confirms weight-bandwidth-bound decode**, consistent with Maxtext opt4 findings. Adding requests beyond 8 only increases latency, not throughput. This makes **Opt A (FP8 weight quantization)** the clear highest-priority next step.

The sglang-jax implementation uses a different serving stack than Maxtext
(JAX-based SGLang engine vs Maxtext's MaxEngine), so Maxtext opt numbers
do not transfer directly. The lessons about which approaches work/fail do transfer.

---

## Optimization Plan

Ordered by expected decode throughput gain and implementation difficulty.

---

### Opt A — FP8 / Int8 weight quantization

**Priority: Highest** → **Revised: Opt A2 is next active work (see below)**

> **Finding (2026-06-09)**: MoE expert weights are already FP8 in HBM — the loader
> does NOT dequantize them. The original plan assumption was wrong. See
> [`opt_a_weight_quant/results.md`](opt_a_weight_quant/results.md) for full details.
>
> The remaining lever is **Opt A2: keep attention weights in FP8** instead of
> dequantizing at load time. Attention weights (Q/K/V/O across 48 layers) are
> currently loaded as BF16 (~3.2 GB/step per TC). Keeping them FP8 saves ~1.6 GB/step
> (~4-5% decode bandwidth reduction). This is now the highest-priority next step.

**Opt A1 (W8A8 activation quant for MoE)**: Deprioritized. MoE decode is
bandwidth-bound, not compute-bound. Quantizing the tiny activation tensors
(8 tokens × 4096 × 2B ≈ 3 MB) against 36 GB of weight reads is negligible.

**Opt A2 (attention FP8 at load time)**: Active next step.
- Attention weights: 48 layers × 4 proj × 4096² × 2B (BF16) = 3.2 GB/step per TC
- With FP8: 1.6 GB/step per TC — saves 1.6 GB/step
- Fraction of total weight bandwidth (~38 GB/step): ~4%
- **Expected gain**: ~4-5% TPOT reduction at conc=8

**Steps for Opt A2**:
1. In `loader.py`, identify where attention weights are dequantized from FP8 → BF16.
   Check `dequant_fused_kv()` and any BF16 cast in `load_weights`.
2. Modify to keep Q/K/V/O in FP8 in HBM. Ensure the attention kernel handles FP8 input.
3. Check `flashattention_backend.py` — verify the Pallas flash-attention kernel
   accepts FP8 weight dtype or needs a cast inserted before the matmul.
4. Benchmark at conc=8. Quality gate: spot-check reasoning outputs.

**Risk**: Medium — flash attention kernel may not accept FP8 weights directly;
may need a per-layer cast or kernel modification.

---

### Opt B — Larger batch size / continuous batching

**Priority: High (proven in Maxtext)**

**Rationale**: Maxtext opt5 proved near-linear decode throughput scaling with batch
size up to the HBM OOM boundary (`per_device_batch_size=11`, 4.72× gain). The same
principle applies to sglang-jax: since decode is weight-bandwidth-bound, increasing
batch amortises weight reads across more tokens per step.

sglang-jax uses a paged KV cache and continuous batching by design, so this is
partially free — the engine naturally fills batch slots from the request queue.
The question is the optimal `max-running-requests` and `page-size` settings.

**Steps**:
1. Run throughput sweep: `--max-running-requests` = 4, 8, 16, 32 at fixed prompt
   length. Record tok/s and latency percentiles.
2. Identify OOM boundary (similar to Maxtext's `per_device_batch_size=12` OOM).
3. Set `--max-running-requests` to the throughput-optimal value.
4. Test with mixed prompt lengths (continuous batching scenario).

**Expected gain**: 3–5× decode throughput vs single-request baseline.
**Risk**: Low (config-only change).

---

### Opt C — Remove per-step host sync in inference loop

**Priority: Medium (quick win)** → **CLOSED: 0% measured gain**

> **Finding (2026-06-09)**: Full code investigation + benchmark showed zero gain.
> See [`opt_c_host_sync/results.md`](opt_c_host_sync/results.md).
>
> The overlap design (`future_token_ids_map` on-device, `jax.copy_to_host_async`,
> `launch_done` event) is already well-optimized. The F=3.9ms fixed overhead per
> step is **fixed TPU compute** — attention, layer norms, MoE router topk, and the
> future token ID map round-trip — none of which can be pipelined away.
>
> Opt C-A (move ForwardBatch/SamplingMetadata prep to background thread, commit
> `10a5699`) was implemented and benchmarked. TPOT at conc=8: 21.6ms → 21.6ms.
> Code is kept (clean refactoring, no downside) but provides no throughput gain.

---

### Opt D — Sparse MoE dispatch for prefill (ragged all-to-all)

**Priority: Medium (prefill-only)**

**Rationale**: The Maxtext opt4 post-mortem proved ragged_all_to_all IS valid for
prefill (T=512, intermediates 2.68 GB/layer → 32× reduction from sparse routing)
but NOT for decode (T=1 per sequence, weight-bound). Current sglang-jax prefill:
~3,500–4,600 tok/s in the 2-node demo. Sparse prefill dispatch could cut TTFT
significantly for long prompts.

**Prerequisite**: sglang-jax's prefill kernel must support ep-aware ragged routing.
Check whether `FusedEPMoE` already does sparse routing during prefill or uses the
same dense dispatch as decode.

**Steps**:
1. Audit `FusedEPMoE.__call__` — determine if it branches on prefill vs decode.
2. If dense for both: implement a prefill-only sparse path using `ragged_all_to_all`
   or equivalent TPU v7x collective. Reference: `RoutedMoE.sparse_matmul` in
   `maxtext/src/maxtext/layers/moe.py`.
3. Benchmark prefill at T=512, 1024, 4096 before/after.
4. Ensure decode path is untouched (no regression).

**Expected gain**: 30–50% TTFT reduction at T=512; larger at longer contexts.
**Risk**: Medium (new collective in the prefill path; debug with `jax.debug.print`
before removing).

---

### Opt E — Speculative decoding

**Priority: Medium-High for per-sequence latency**

**Rationale**: Orthogonal to batch throughput scaling. A draft model produces K
candidate tokens per step; the full model verifies them in one pass. With acceptance
rate ~70–80%, effective per-sequence throughput increases 2–3× without changing
the serving batch size. This is the primary lever for reducing time-to-completion
for individual users.

**Draft model candidates** (for MiMo-V2-Flash):
- A smaller MiMo variant (if publicly available)
- Shared-expert-only forward pass (skip the routed MoE, use only dense MLP layers)
  — fast draft, lower acceptance rate
- Self-speculative: early exit after N layers of the full model

**Steps**:
1. Decide on draft model architecture (shared-expert-only is lowest complexity).
2. Implement `speculative_generate` wrapper: draft K tokens, batch-verify, accept/reject.
3. Measure acceptance rate on a representative prompt distribution.
4. Benchmark effective tok/s and TTFT vs single-model decode.

**Expected gain**: 2–3× per-sequence latency reduction.
**Risk**: High (significant implementation; acceptance rate is workload-dependent).

---

### Opt F — Paged attention tuning (page-size, max-prefill-size)

**Priority: Low-Medium**

**Rationale**: sglang-jax already uses paged attention. The `--page-size` and
`--chunked-prefill-size` flags control memory granularity and prefill chunking.
Current settings (`--page-size 16`, `--chunked-prefill-size 2048`) may not be
optimal for all workloads.

**Steps**:
1. Sweep `--page-size` (8, 16, 32, 64) and measure HBM utilization and decode throughput.
2. Sweep `--chunked-prefill-size` (512, 1024, 2048, 4096) and measure TTFT.
3. Pick optimal values for the target workload (long prompt vs short prompt).

**Expected gain**: 5–15% improvement in HBM efficiency and TTFT.
**Risk**: Very low (config-only).

---

## Status Summary (2026-06-09)

| Opt | Description | Status | Gain |
|-----|-------------|--------|------|
| A (expert FP8) | MoE expert weights already FP8 — no action | ✅ Closed | 0% (already done) |
| **A2 (attention FP8)** | Keep attention weights FP8 at load time | **⏳ Next** | **~4-5% expected** |
| B (batch scaling) | Sweep already done: plateau at conc=8 | ✅ Closed | Diminishing returns |
| C (host sync) | Overlap design already optimal; 0% measured | ✅ Closed | 0% measured |
| D (sparse prefill) | Not yet investigated | 🔲 Backlog | ~30-50% TTFT |
| E (speculative) | Not yet investigated | 🔲 Backlog | ~2-3× per-seq latency |
| F (page tuning) | Not yet investigated | 🔲 Backlog | 5-15% HBM efficiency |

**Baseline**: 371 tok/s, TPOT=21.6ms @ conc=8 (2026-06-08)
**Current**: 371 tok/s (no improvement yet — Opt A and C closed with 0% gain)
**Next target**: ~388-390 tok/s (+4-5%) from Opt A2

## Tracking

Results for each optimization go in subdirectories:

```
docs/tpu7tests/mimo_v2_flash_optimization/
  plan.md                    ← this file
  baseline/results.md        ← baseline sweep (2026-06-08) ✅
  opt_a_weight_quant/        ← FP8/int8 weight quantization ✅ closed (0%)
  opt_b_batch_scaling/       ← batch size sweep (plateau at conc=8) ✅
  opt_c_host_sync/           ← host sync removal ✅ closed (0% measured)
  opt_d_sparse_prefill/      ← sparse MoE prefill 🔲 backlog
  opt_e_speculative/         ← speculative decoding 🔲 backlog
  opt_f_paged_attention/     ← paged attention tuning 🔲 backlog
```

Each subdirectory should contain at minimum:
- `results.md` — benchmark numbers (before/after), quality gate outcome, decision
- command templates used to reproduce the results

---

## What NOT to try (informed by Maxtext lessons)

| Approach | Reason to skip |
|----------|---------------|
| KV-int8 quantization | Tested in Maxtext opt3: +6% slower + quality regression on reasoning |
| Dense sparse MoE dispatch (local gmm only) | Tested in Maxtext opt2: negligible gain (weight-bound, not intermediate-bound) |
| shard_map all-gather sparse dispatch | Tested in Maxtext opt5: 160 ms (all-gather overhead dominates) |
| ragged_all_to_all for decode (T=1/seq) | Tested in Maxtext opt4: 83% regression — weight-bound, collectives dominate |
| scan_layers=true for decode | Tested in Maxtext: +23% decode overhead (no cross-iteration prefetch in while_loop) |
| SWA KV cache truncation | Tested in Maxtext opt3: 0% improvement at tested config |

---

## References

- Maxtext perf optimization doc: `/home/jingnw_google_com/maxtext/docs/guides/mimo_v2_flash_tpu_perf_optimization.md`
- Maxtext opt4 post-mortem (ragged A2A): `/home/jingnw_google_com/maxtext/docs/guides/mimo_v2_flash_opt4_ragged_a2a_sparse_moe_plan.md`
- Maxtext opt5 batch scaling: `/home/jingnw_google_com/maxtext/docs/guides/mimo_v2_flash_opt5_batch_size_scaling.md`
- sglang-jax 1-node demo results: [`../mimo_v2_flash_progress.md`](../mimo_v2_flash_progress.md)
- GKE infrastructure: [`../gke_tpu7x_env_setup.md`](../gke_tpu7x_env_setup.md)
- HBM resource allocation: [`../gke_tpu7x_resource_allocation.md`](../gke_tpu7x_resource_allocation.md)
