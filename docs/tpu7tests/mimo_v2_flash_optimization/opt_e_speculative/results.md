# Opt E — Speculative Decoding (Flash MTP): Implementation & Results

**Date**: 2026-06-09 – 2026-06-11
**Status**: Benchmark job resubmitted (commit `9c41046`) — awaiting TPU provisioning

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

## Implementation (2026-06-09)

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

## Bugs Fixed During Integration (2026-06-09 – 2026-06-10)

Getting SPEC_EXTEND and SPEC_DECODE precompile to pass required fixing several
bugs across the paged attention kernel and EAGLE worker:

| Bug | Symptom | Fix | Commit |
|-----|---------|-----|--------|
| Vocab-logits all-gather inside JIT | `FAILED_PRECONDITION` on second precompile call | Moved all-gather outside JIT | — |
| TracerBoolConversionError | Crash on `if forward_mode.is_decode()` inside traced code | Added axis to `static_argnames` | — |
| OOB DMA in last BQ block | `Semaphore has nonzero value` at runtime | bq_sz divisibility loop in `get_default_block_sizes` | `893368d` |
| BQ double-buffering semaphore crash (bq_sz=1, EAGLE DECODE MIXED, bs=1) | `Semaphore has nonzero value` at runtime for draft decode | `bq_sz = max_num_tokens` for small token counts; limited to `page_size` threshold | `8c24cc4`, `bcbed33` |
| BQ double-buffering semaphore crash (sliding-window MIXED, bs≥4) | `Semaphore has nonzero value` at runtime for bs=4 precompile (`RPAm-p_16-bq_4_4-bkv_2048_1024-sw_128.1`) | `bq_sz = max_num_tokens` for any `sliding_window is not None`, uncapped; forces num_bq=1 | `9c41046` |
| `hidden_states` sharding mismatch in nextn | XLA INVALID_ARGUMENT on reshard | Reshard outside JIT with `jax.sharding.reshard` | `2ac21f0`, `6749b06` |

### Root-cause fix: Mosaic dynamic DMA size (2026-06-10)

**Error**:
```
MosaicError: INTERNAL: Mosaic failed to compile TPU kernel:
Failed to prove that a dynamic slice size along dimension 0 is divisible by the tiling (8).
MLIR: memref<272x256xi32, #tpu.tiled<(8,128),[2,1]>, #tpu.memory_space<hbm>> → memref<?x256xi32>
```

**Root cause**: SPEC_DECODE precompile compiles a `TARGET_VERIFY` kernel with
`custom_mask = tree_mask` (non-None). Inside `_fetch_mask`, the DMA source size
was:
```python
load_kvmask_sz = jnp.minimum(bkv_sz, mask_left)   # dynamic JAX value
```
Even though this equals `bkv_sz` or a multiple of 8 at runtime (kv_len is always
aligned to page_size=16), Mosaic cannot prove it statically for a tiled HBM memref
and rejects the compilation.

**Why it appeared after `8c24cc4`**: That commit set `bq_sz = max_num_tokens` for
`TARGET_VERIFY`, giving it a non-trivial `bq_sz`. Mosaic then compiled the DMA into
a loop body and applied stricter static-divisibility checks.

**Failed approaches** (each required ~20 min to iterate: commit → push → delete job
→ resubmit → wait for provisioning + Orbax restore + precompile):

| Attempt | Change | Why it failed |
|---------|--------|---------------|
| `bcbed33` | `lax.fori_loop(unroll=True)`, static bound | Mosaic still checks each unrolled copy |
| `f86a2eb` | Python `for i in range(bq_sz)` | Generates unconditional DMA calls; same check |
| `85a5135` | `lax.fori_loop(0, load_q_sz, unroll=False)` | `load_q_sz` dynamic bound doesn't help; DMA size still dynamic |
| `bcbed33` | Add `page_size` threshold for `bq_sz` selection | Reduced bq_sz correctly but DMA size remained dynamic |

**Correct fix** (commit `cd041a6`):

Use `bkv_sz` (a Python int captured in the kernel closure) as the DMA size:
```python
# _fetch_mask loop_body — bkv_sz is a Python int, statically divisible by 8
_async_copy(
    custom_mask_ref.at[pl.ds(start, bkv_sz)],   # static size
    kvmask_vmem_ref.at[i],
    sem, wait,
)
```

Pad `custom_mask` with 2048 zero rows before the kernel call so reading a full
`bkv_sz` block never goes out of bounds:
```python
custom_mask = jnp.pad(custom_mask, ((0, 2048), (0, 0)))
```

**Correctness**: rows beyond `kv_len` correspond to zero KV cache entries. The
attention output for those positions is zero regardless of the mask value
(V=0 → contribution=0). Result is numerically identical to the original code.

**File**: [ragged_paged_attention_v3.py](python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py)

---

## Launch Configuration

**Benchmark job**: `scripts/mimo_v2_flash_1node_opt_e_bench_job.yaml`

```
--speculative-algorithm EAGLE
--speculative-num-steps 4
--speculative-eagle-topk 5
--page-size 16
--max-running-requests 32
```

Draft model path points to the same directory as the main model
(`/mnt/gcs/mimo-v2-flash-hf-weights`); the loader's `is_draft_model` flag
forces the slow-path safetensors loader, avoiding Orbax hash collisions.

---

## Benchmark Plan

Sweep (input=512, output=256):

| Setting | conc | Expected |
|---------|------|---------|
| Baseline (no spec) | 8 | 21.6 ms TPOT |
| Spec EAGLE K=4, topk=5 | 8 | ~8-12 ms if acceptance ~60-70% |

Target: 2-3× per-sequence latency reduction.

---

## Results (pending)

| Setting | TPOT (ms) | Acceptance rate | tok/s |
|---------|----------:|----------------:|------:|
| Baseline | 21.6 | — | 371 |
| Spec K=4, topk=5 | TBD | TBD | TBD |

Job `mimo-v2-flash-1node-opt-e` resubmitted 2026-06-11, commit `9c41046`.
Awaiting TPU node provisioning → Orbax restore (~7 min) → precompile → results.
