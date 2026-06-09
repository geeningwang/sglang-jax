# Opt E — Speculative Decoding (Flash MTP): Implementation & Results

**Date**: 2026-06-09
**Status**: Implementation complete — not yet benchmarked

---

## Approach

MiMo-V2-Flash ships pre-trained MTP (multi-token-prediction) weights in
`model_mtp.safetensors` alongside the main model. Each MTP layer is a single
SWA-attention + dense-MLP decoder block that predicts the next K tokens jointly
with the main model.

We implement the draft model using the EAGLE speculative-decoding framework
(`--speculative-algorithm EAGLE`), treating the MTP layer as the draft model.

---

## Flash MTP Weight Format

From `model_mtp.safetensors` header inspection:

| Weight | Shape | Dtype |
|--------|-------|-------|
| `model.mtp.layers.{i}.self_attn.q_proj.weight` | [12288, 4096] | fp8_e4m3fn |
| `model.mtp.layers.{i}.self_attn.q_proj.weight_scale_inv` | [96, 32] | float32 |
| `model.mtp.layers.{i}.self_attn.k_proj.weight` | [1536, 4096] | fp8_e4m3fn |
| `model.mtp.layers.{i}.self_attn.k_proj.weight_scale_inv` | [12, 32] | float32 |
| `model.mtp.layers.{i}.self_attn.v_proj.weight` | [1024, 4096] | fp8_e4m3fn |
| `model.mtp.layers.{i}.self_attn.v_proj.weight_scale_inv` | [8, 32] | float32 |
| `model.mtp.layers.{i}.self_attn.o_proj.weight` | [4096, 8192] | bfloat16 |
| `model.mtp.layers.{i}.self_attn.attention_sink_bias` | [64] | bfloat16 |
| `model.mtp.layers.{i}.mlp.gate_proj.weight` | [16384, 4096] | fp8_e4m3fn |
| `model.mtp.layers.{i}.mlp.gate_proj.weight_scale_inv` | [128, 32] | float32 |
| `model.mtp.layers.{i}.mlp.up_proj.weight` | [16384, 4096] | fp8_e4m3fn |
| `model.mtp.layers.{i}.mlp.up_proj.weight_scale_inv` | [128, 32] | float32 |
| `model.mtp.layers.{i}.mlp.down_proj.weight` | [4096, 16384] | fp8_e4m3fn |
| `model.mtp.layers.{i}.mlp.down_proj.weight_scale_inv` | [32, 128] | float32 |
| `model.mtp.layers.{i}.enorm.weight` | [4096] | bfloat16 |
| `model.mtp.layers.{i}.hnorm.weight` | [4096] | bfloat16 |
| `model.mtp.layers.{i}.eh_proj.weight` | [4096, 8192] | bfloat16 |
| `model.mtp.layers.{i}.final_layernorm.weight` | [4096] | bfloat16 |
| `model.mtp.layers.{i}.input_layernorm.weight` | [4096] | bfloat16 |
| `model.mtp.layers.{i}.pre_mlp_layernorm.weight` | [4096] | bfloat16 |

Key difference from V2.5-Pro MTP: **separate q/k/v FP8 weights** (not fused QKV).
o_proj and eh_proj are BF16 (no scale_inv).

---

## Implementation

Four files modified:

### 1. `python/sgl_jax/srt/models/mimo_v2_nextn.py`

Added `MiMoV2FlashMTPForCausalLM` class. Key points:
- Reuses `MiMoV2ModelNextN` (same SWA-attention + dense-MLP block structure)
- Adds `_kv_buffers: dict[int, dict] = {}` for raw FP8 weight storage
- `_create_weight_mappings()` routes q/k/v FP8 to `__KV_Q/K/V_WEIGHT__0` buffers
- `_create_weight_mappings()` routes o_proj/eh_proj BF16 to `__KV_O/EH_WEIGHT__0` buffers
- `_create_weight_mappings()` maps MLP to QuantizedLinear `weight_q`/`weight_scale`
- `load_weights()`: calls `_uniform_block_dequant` on q/k/v, sets o_proj/eh_proj
  as BF16 LinearBase directly, then `dequant_fp8_layers` for MLP

### 2. `python/sgl_jax/srt/configs/model_config.py`

- Added `self.is_draft_model = is_draft_model` (stored for loader use)
- Added `MiMoV2FlashForCausalLM → MiMoV2FlashMTPForCausalLM` architecture remapping

### 3. `python/sgl_jax/srt/model_loader/loader.py`

- Draft models skip Orbax checkpoint caching (avoids tree-mismatch with target)

### 4. `python/sgl_jax/srt/utils/weight_utils.py`

- Added `Q_WEIGHT`, `Q_SCALE`, `O_WEIGHT`, `EH_WEIGHT` buffer keys to `__KV_*` handler

---

## Launch Command

```bash
python3 -m sgl_jax.launch_server \
  --model-path /mnt/gcs/mimo-v2-flash-hf-weights \
  --trust-remote-code \
  --tp-size 8 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --page-size 16 \
  --chunked-prefill-size 2048 \
  --max-running-requests 32 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path /mnt/gcs/mimo-v2-flash-hf-weights \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --host 0.0.0.0 --port 8080 --nnodes 1 --node-rank 0
```

**Note**: `--speculative-draft-model-path` uses the same directory as the main
model (since `model_mtp.safetensors` is there). The Orbax checkpoint hash
collision is avoided by the `is_draft_model` flag in the loader, which forces
draft models to always use the safetensors slow-path (no Orbax). The MTP model
is tiny (~1.7 GB weights), so slow-path loading adds only ~10-20 s.

---

## Benchmark Plan

Sweep (input=512, output=256):

| Setting | conc | Expected |
|---------|------|---------|
| Baseline (no spec) | 8 | 21.6 ms TPOT |
| Spec EAGLE (K=3, topk=4) | 8 | ~10-12 ms if acceptance ~70% |
| Spec EAGLE (K=5, topk=4) | 8 | ~8-10 ms if acceptance ~70% |

Target: 2-3× per-sequence latency reduction.

---

## Results (TBD)

| Setting | TPOT (ms) | Acceptance rate | tok/s |
|---------|----------:|----------------:|------:|
| Baseline | 21.6 | — | 371 |
| Spec K=3 | TBD | TBD | TBD |
| Spec K=5 | TBD | TBD | TBD |

---

## TODO

- [ ] Create GKE benchmark YAML for Opt E
- [ ] Resolve `speculative-draft-model-path` same-path issue (test if Orbax skip works)
- [ ] Run benchmark, record TPOT and acceptance rate
- [ ] Tune K (num_steps) and topk for best throughput
- [ ] Update this doc with measured results
