# Opt A — FP8 / Int8 Weight Quantization: Analysis Results

**Date**: 2026-06-09
**Status**: A1 (expert quant) closed — already FP8. A2 (attention FP8) closed — <0.3% gain, not worth implementing.

---

## Key Finding: MoE Expert Weights Are Already FP8

Codebase inspection (`python/sgl_jax/srt/layers/fused_moe.py`,
`python/sgl_jax/srt/configs/model_config.py`,
`python/sgl_jax/srt/model_loader/loader.py`) revealed:

| Layer | Dtype in HBM | Notes |
|-------|-------------|-------|
| MoE w1/w2/w3 (47 layers) | **float8_e4m3fn** | FP8 in HBM; scale multiply inside Pallas kernel |
| MoE activations (tokens) | bfloat16 | W8A16 only; `activation_quantized_dtype` stored but unused in `__call__` |
| Attention Q/K/V/O (48 layers) | bfloat16 | Dequantized from FP8 checkpoint at load time |
| Layer-0 dense MLP | bfloat16 | Dequantized at load time |

The HF Flash config specifies FP8 activations, but `model_config.py` explicitly skips
enabling `moe_activation_dtype` with comment "using weight-only FP8 (BF16 activations)".

The original Opt A assumption was wrong. Expert weights were never BF16 — they are
already FP8 in HBM. The Orbax checkpoint name `tp8_bfloat16/` refers to the BF16
attention weights (dequantized at load time), not the expert weights.

---

## Remaining Opt A Avenues

### A1 — W8A8 (activate `moe_activation_dtype` for MoE)

Currently unused: `FusedEPMoE.__call__()` receives BF16 activations and FP8 weights.
Enabling activation quantization would make the matmuls fully FP8.

Bandwidth math (conc=8 decode, TPOT=21.6ms):
- Expert weights per step: 47 layers × 3 proj × 32 local experts × 4096 × 1024 × 1B (FP8) ≈ 36 GB
- Activation tensors per step: 47 layers × conc=8 × 4096 × 2B (BF16) ≈ 3 MB — negligible

Activations are 1000× smaller than weights. Quantizing them saves essentially nothing
on bandwidth. W8A8 benefits are compute-side (faster MXU throughput for fully-FP8 matmuls),
not bandwidth-side.

Verdict: W8A8 is low priority for decode (bandwidth-bound, not compute-bound).
May help prefill at very long contexts (compute-bound at T=4096), but TTFT is already
fast (190ms at 4K tokens). Deprioritized.

### A2 — Attention FP8 (skip dequantization at load time) ✅ CLOSED — <0.3% gain

**Corrected analysis (2026-06-09)**:

The original estimate ("4-5% gain from 48 layers × 4 proj × 4096² savings") was wrong by ~20×.
Three compounding errors:

1. **o_proj already FP8**: Not dequantized in `load_weights()`. Only q/k/v are BF16 targets.
2. **TP=8 sharding**: Both input (4096→512 per TC) and output heads are sharded.
3. **GQA: 1 global SWA KV head**: Flash uses aggressive GQA — from the SWA eviction doc,
   each device has 1 KV head (head_dim=192+128=320) at TP=16. K/V projections are tiny.

**Corrected bandwidth math** (per TC per step):

| Weights | Size per TC |
|---------|-------------|
| MoE FP8 (47 layers × 3 proj × 32 experts) | ~18.9 GB |
| SWA attn q/k/v BF16 (39 layers) | ~43 MB |
| Full attn q/k/v BF16 (9 layers) | ~10 MB |
| **Attention total** | **~53 MB** |

Opt A2 savings (BF16 → FP8): ~27 MB = **0.14% of MoE bandwidth**.

Even doubling q_head assumptions gives <0.3%.

**Verdict**: Not worth implementing. Requires full checkpoint rebuild (24-min slow-path run)
or post-load re-quantization (loses original FP8 scales), for <0.3% TPOT improvement.

**Status**: Closed. No implementation needed.

---

## Profiling Attempts (for completeness)

Four runs of `scripts/mimo_v2_flash_1node_profile_job.yaml` were made.
All four crashed with SIGKILL when `POST /start_profile` was called:

| Run | host_tracer_level | profile_duration | Outcome |
|-----|------------------|------------------|---------|
| 1 | 2 | unbounded | Timeout (no requests in-flight when called) |
| 2 | 2 | unbounded | SIGKILL during burst |
| 3 | 2 | 30s | SIGKILL during burst |
| 4 | 1 | 30s | SIGKILL during burst |

Root cause hypothesis: `jax.profiler.start_trace()` (called by the sglang-jax
`/start_profile` handler) triggers JAX XLA recompilation with profiling
instrumentation, which exhausts the 900Gi container RAM limit on TPU v7x (8 chips).

Alternative not yet attempted: set `JAX_PROFILER_PORT=9999` in server env and use
`tf.profiler.experimental.client.trace('grpc://localhost:9999', logdir, 2000)`
to capture a bounded trace via gRPC, bypassing the sglang-jax HTTP handler.

---

## Decision: Close Opt A2; Pivot to Opt E

Both A1 and A2 are closed:
- A1: Expert weights already FP8.
- A2: Attention FP8 gives <0.3% gain — not worth checkpoint rebuild.

Opt C is also closed (0% measured gain; XLA dependency serializes steps regardless).

**Next active optimization: Opt E (speculative decoding)**. Potential: ~2-3× per-sequence
latency reduction. See `../plan.md` for full priority table.

---

## [OLD] Decision: Skip Profiling, Pivot to Opt C

The baseline TPOT scaling data gives a cleaner signal than profiling:

| conc | TPOT (ms) | linear model check |
|------|----------:|--------------------|
| 8    | 21.6      | 2.21×8 + 3.9 = 21.6 ✓ |
| 16   | 39.3      | 2.21×16 + 3.9 = 39.3 ✓ |
| 32   | 75.5      | 2.21×32 + 3.9 = 74.6 ≈ ✓ |

Fixed per-step overhead F ≈ 3.9ms (18% of TPOT at conc=8). Does not scale with
batch size — signature of per-step host work: EOS detection, CPU-side sampling,
or scheduler coordination.

Removing 3.9ms: TPOT 21.6 → 17.7ms, throughput 371 → ~452 tok/s (+22%).

Next step: Implement Opt C (host sync removal). See `../opt_c_host_sync/`.
