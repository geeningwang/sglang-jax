# NanoGPT vs MiMo-V2.5-Pro: Architecture Comparison

## Overview

| | NanoGPT (GPT-2) | MiMo-V2.5-Pro |
|---|---|---|
| **Purpose** | Educational / research training demo | Production inference serving |
| **Scale** | ~124M params (GPT-2 small) | ~1.02T params total (MoE, ~42B active per token) |
| **Framework** | Flax NNX (training model) | Flax NNX + SGLang-JAX serving stack |
| **Files** | `examples/nanogpt/model.py` | `python/sgl_jax/srt/models/mimo_v2_pro.py` (subclass of `mimo_v2_flash.py`) |
| **Entry class** | `GPT` | `MiMoV2ForCausalLM` |

---

## Architecture Dimensions

### Normalization
- **NanoGPT**: Standard `LayerNorm` (mean + variance). Custom implementation because NNX only ships `RMSNorm` in its layer library.
- **MiMo-V2.5-Pro**: `RMSNorm` (variance only, no mean subtraction). Two norms per layer (pre-attention, pre-MLP), same as GPT-2's pre-norm layout but with a *floating residual* pattern — the residual is passed between layers explicitly rather than added inside each layer, reducing fused ops.

### Positional Encoding
- **NanoGPT**: Learned absolute position embedding (`wpe`, shape `[block_size, n_embd]`), added to token embeddings at the input. Simple and fixed-context.
- **MiMo-V2.5-Pro**: Rotary Position Embedding (RoPE, NeoX-style), applied inside attention to Q and K only. `partial_rotary_factor` allows RoPE to cover only a fraction of the head dimension. Supports `rope_scaling` for context extension (e.g. YaRN).

### Attention
- **NanoGPT**: Standard Multi-Head Attention (MHA). Fused QKV projection, manual causal mask via `jnp.tril`, full sequence attention. All heads for Q/K/V. No caching.
- **MiMo-V2.5-Pro**: Hybrid attention — each layer is either **Sliding Window Attention (SWA)** or **full attention**, determined by `hybrid_layer_pattern` config. Grouped-Query Attention (GQA) with separate `num_key_value_heads`. Separate `v_head_dim` (V can be smaller than Q/K head dim). Optional learnable attention-sink bias per head. Uses `RadixAttention` (paged KV cache) for efficient long-context serving.

  **Pro vs Flash**: The Pro variant stores Q, K, V as a single fused `qkv_proj` weight in the checkpoint (not three separate projections). The weight loader splits it into separate `q_proj`, `k_proj`, `v_proj` after loading, including a special per-shard FP8 dequantization path.

### MLP
- **NanoGPT**: Two-layer GELU MLP: `Linear(n_embd → 4·n_embd) → GELU → Linear(4·n_embd → n_embd)`.
- **MiMo-V2.5-Pro**: Three-layer SwiGLU MLP: `gate_proj` and `up_proj` run in parallel, element-wise `silu(gate) * up`, then `down_proj`. On **MoE layers** (determined by `moe_layer_freq`), the MLP is replaced by a Mixture-of-Experts block (`MiMoV2Moe`) with a learned router, top-K expert selection, and `EPMoE`/`FusedEPMoEV2` expert dispatch. Non-MoE layers use the standard dense SwiGLU.

### Linear Layers
- **NanoGPT**: Custom `Linear(nnx.Module)` with `(in, out)` weight layout, no sharding, no quantization.
- **MiMo-V2.5-Pro**: `LinearBase` throughout, with `kernel_axes` annotations for tensor-parallel sharding (`"tensor"` axis). Supports optional FP8 static quantization (`weight_q` + `weight_scale`), with per-head dequantization for K/V projections.

### Weight Loading
- **NanoGPT**: Direct assignment to `nnx.Param.value` in `sample.py`. HuggingFace safetensors → numpy → `jnp.array(...)` into each parameter attribute.
- **MiMo-V2.5-Pro**: Multi-stage pipeline via `WeightLoader` + `WeightMapping` (explicit HF-name → JAX-path mappings with flags for transpose, head-dim padding, KV-head replication). After bulk loading: FP8 dequant for Q (per-layer), fused K/V dequant (cross-boundary per-head blocks), dense MLP layer-0 dequant, KV head replication for TP alignment.

### Inference Serving
- **NanoGPT**: Standalone autoregressive Python loop. `generate_step` is `@jax.jit`'d with `nnx.split/merge`. One token per JIT dispatch. No KV cache management.
- **MiMo-V2.5-Pro**: Integrated into the SGLang-JAX serving runtime. `__call__` accepts a `ForwardBatch` (carries token positions, paged KV pool references, speculative info). Returns `(output, kv_fused_dict, callback_flags, None)`. KV cache is managed externally by `RadixAttention` across requests.

### Speculative Decoding
- **NanoGPT**: None.
- **MiMo-V2.5-Pro**: Supported via `MiMoV2MTPForCausalLM` (Multi-Token Prediction draft model in `mimo_v2_nextn.py`). Up to 3 SWA-attention draft layers; each takes the target model's hidden states + next-token embedding, projects down, runs one decoder layer, and proposes a draft token. LM head is shared with the target model at runtime.

### Deployment
- **NanoGPT**: Single device or 4-chip `jax.pmap` (data parallel). GKE job runs on `tpu7x-standard-4t` (2x2x1).
- **MiMo-V2.5-Pro**: Multi-host TPU pods (typically 4 hosts, 32 chips). Tensor Parallel (`tp=8`), Data Parallel (`dp=2`), Expert Parallel (`ep=8`). Served via GKE with GCSFuse or NFS-backed weight loading.

### Pallas Computation Kernels

- **NanoGPT**: Pure JAX throughout — `jnp.einsum`, `jax.nn.softmax`, standard matrix multiplications. All ops are expressed in high-level JAX primitives and compiled by XLA. No custom kernels.

- **MiMo-V2.5-Pro**: Three critical operations are implemented as hand-written **Pallas kernels** (`jax.experimental.pallas`, TPU backend). Pallas is JAX's low-level kernel language that gives explicit control over VMEM, DMA, semaphores, and the MXU — the same level of control as CUDA for GPU but for TPU. This is necessary because XLA's auto-scheduling cannot always overlap compute and memory transfers at the granularity required for peak TPU utilization.

  **1. Flash Attention — `ragged_paged_attention_v3`**
  The production attention kernel (`kernels/ragged_paged_attention/ragged_paged_attention_v3.py`). Selected whenever `FlashAttention` backend is active; `RadixAttention` itself is just a thin metadata holder — the Pallas kernel does all the real work. Key capabilities:
  - **Pipelined double-buffered DMA**: Q, K/V, and output tiles all use `pltpu.make_async_copy` with two VMEM double-buffers. The MXU computes on one tile while the DMA engine fetches the next, hiding HBM latency almost entirely.
  - **Fused KV cache update**: New K/V tokens are scattered into their page slots at the end of the kernel pass, eliminating a separate scatter operation.
  - **Interleaved KV layout**: K and V heads are interleaved in the cache pages so that loading one DMA chunk covers a K/V head pair — halving DMA descriptor overhead and enabling tighter GQA unrolling.
  - **Built-in SWA**: Sequence-specific window start offsets are computed at the kernel level; blocks outside the window are skipped without returning to Python.
  - **Attention sink**: A virtual sink token is included in the softmax without materializing an extra KV entry (replaces l=0/m=−∞ initialization with pre-computed sink logits).
  - **DP-aware indexing**: `cu_kv_lens`-based page addressing lets each DP rank's kernel shard operate on its own compact page range without coordination.

  **2. Fused EP MoE — `FusedEPMoEV2`**
  The Strix-style double-buffer MoE kernel (`kernels/fused_moe/v2/kernel.py`). Activated only for MiMo V2 architectures (the `moe_backend='fused_v2'` config field). Key capabilities:
  - **No JAX collectives**: The entire EP all-to-all scatter (tokens to expert devices) and gather (results back to token owners) is performed inside the Pallas kernel via `pltpu.make_async_remote_copy` with `DeviceIdType.MESH`. Tokens never leave Pallas to go through JAX's collective framework.
  - **Weight streaming with double-buffering**: W1/W3/W2 weight tiles are pre-fetched from HBM while the MXU computes on the previous tile. W2 DMA starts during W1/W3 accumulation.
  - **Token data stays in VMEM**: Each token sub-tile is loaded once per expert iteration, not re-read from HBM per weight tile.
  - **FP8 dequant in VMEM**: FP8 weights arrive from HBM and are dequantized to bf16 in VMEM scratch before the dot product — the dequant cost is hidden by the DMA latency.
  - **In-kernel shared experts**: MiMo's shared-expert portion is fused into the same `pl.pallas_call` alongside routed experts.
  - An older v1 kernel (`FusedEPMoE`) and a non-fused GMM path (`EPMoE`, backed by `megablox_gmm_kernel`) also exist as fallbacks for other architectures.

  **3. KV Cache Update — `update_kv_cache`**
  A smaller Pallas kernel that scatters new K/V tokens into their page slots. Used when the fused-in-attention KV update path is disabled.

  The bottom line: Pallas gives MiMo the ability to overlap compute and memory at the hardware level, perform all-to-all communication without surfacing it through the JAX collective graph, and maintain cache-resident intermediate state across the full FFN computation — none of which is achievable with plain JAX primitives.

### Data Parallel / Expert Parallel (DP/EP)

- **NanoGPT**: Data parallelism via `jax.pmap` — parameters are replicated across all chips, each chip processes a different micro-batch shard. No expert parallelism (no MoE). Gradient sync via `jax.lax.pmean`. Simple and sufficient for a 124M model.

- **MiMo-V2.5-Pro**: A two-axis device mesh drives three distinct parallelism strategies simultaneously.

  **Device mesh**
  ```
  mesh shape: [dp_size, tp_size // dp_size]
  axis names: ["data",  "tensor"]
  ```
  For a 4-host, 32-chip deployment: `dp=2, tp=8`, giving a `[2, 4]` mesh. EP is not a separate mesh axis — it reuses the full `data × tensor` product (`ep = dp × (tp/dp) = 8`).

  **Tensor Parallel (TP)** — pre-existing in SGLang-JAX.
  `LinearBase` weights are sharded along the `"tensor"` axis via `kernel_axes`. Column-parallel projections shard output features; row-parallel projections shard input features. An implicit all-reduce across `"tensor"` completes each row-parallel layer.

  **Data Parallel (DP)** — extended for MiMo.
  Each DP rank owns an independent sub-batch of requests. The scheduler assigns incoming requests round-robin to DP ranks and pads each rank's batch to a common `per_dp_bs` size. Attention metadata (`cu_q_lens`, `cu_kv_lens`, `page_indices`) is laid out DP-rank-contiguous and sharded with `P("data")`, so the attention Pallas kernel shard for rank `r` processes only rank `r`'s requests. **Attention is fully independent across DP ranks — no cross-DP communication.**

  The original SGLang-JAX already had `dp_size` as a server argument and a 2D mesh, but the attention metadata computation had `if dp > 1: 2D else: 1D` branches throughout. For MiMo, these were unified into a single code path using a new `_per_dp_cumsum` helper and `per_dp_bs_size` fields in `ModelWorkerBatch`, making multi-DP a first-class supported configuration rather than a bolt-on.

  **Expert Parallel (EP)** — added for MiMo.
  MoE expert weights are sharded with `P(("data", "tensor"), None, None)` — experts are split across the full mesh, so each device owns `n_routed_experts / ep_size` experts. All routing and communication happen **inside the Pallas kernel** via `pltpu.make_async_remote_copy` (token scatter to expert device, result gather back to token owner). Because there is no separate EP mesh or JAX collective, DP and EP coexist on the same `["data", "tensor"]` mesh without conflict. `FusedEPMoEV2` is the only MoE backend that supports this topology; it is gated to MiMo V2 architectures by `_FUSED_MOE_V2_SUPPORTED_ARCHITECTURES`.

---

## Tensor Inventory

### GPT-2 124M (NanoGPT) — Complete Tensor List

All weights are `float32`. Parenthesized counts mark tensors absent from the `bias=False` training checkpoint.

**Global (× 1)**

| Tensor | Shape | Params | Notes |
|---|---|---|---|
| `wte` | `(50304, 768)` | 38,633,472 | Token embedding; also serves as LM head — **weight-tied, no separate lm_head tensor** |
| `wpe` | `(1024, 768)` | 786,432 | Learned absolute position embedding |

**Per Transformer Block (× 12, all blocks identical)**

| Tensor | Shape | Params / block | × 12 | Notes |
|---|---|---|---|---|
| `h[i].ln_1.scale` | `(768,)` | 768 | 9,216 | Pre-attention LayerNorm γ |
| `h[i].ln_1.bias` | `(768,)` | *(768)* | *(9,216)* | β — absent if `bias=False` |
| `h[i].attn.c_attn.kernel` | `(768, 2304)` | 1,769,472 | 21,233,664 | Fused Q+K+V (out = 3 × 768) |
| `h[i].attn.c_attn.bias` | `(2304,)` | *(2,304)* | *(27,648)* | |
| `h[i].attn.c_proj.kernel` | `(768, 768)` | 589,824 | 7,077,888 | Attention output projection |
| `h[i].attn.c_proj.bias` | `(768,)` | *(768)* | *(9,216)* | |
| `h[i].ln_2.scale` | `(768,)` | 768 | 9,216 | Pre-MLP LayerNorm γ |
| `h[i].ln_2.bias` | `(768,)` | *(768)* | *(9,216)* | β — absent if `bias=False` |
| `h[i].mlp.c_fc.kernel` | `(768, 3072)` | 2,359,296 | 28,311,552 | MLP expand (4× hidden) |
| `h[i].mlp.c_fc.bias` | `(3072,)` | *(3,072)* | *(36,864)* | |
| `h[i].mlp.c_proj.kernel` | `(3072, 768)` | 2,359,296 | 28,311,552 | MLP contract |
| `h[i].mlp.c_proj.bias` | `(768,)` | *(768)* | *(9,216)* | |

**Final LayerNorm (× 1)**

| Tensor | Shape | Params | Notes |
|---|---|---|---|
| `ln_f.scale` | `(768,)` | 768 | Final LayerNorm γ |
| `ln_f.bias` | `(768,)` | *(768)* | β — absent if `bias=False` |

**Totals**

| | bias=True | bias=False (checkpoint) | Memory |
|---|---|---|---|
| `wte` + `wpe` | 39,419,904 | 39,419,904 | 150.4 MiB |
| 12 × blocks | 85,054,464 | 84,953,088 | 324.1 MiB |
| `ln_f` | 1,536 | 768 | ~3 KiB |
| **Grand total** | **124,475,904** | **124,373,760** | **≈ 474.4 MiB** |

> `wte` is weight-tied to the LM head: `logits = hidden @ wte.T`. No separate `lm_head` tensor exists. Full tensor list with checkpoint path names: [examples/nanogpt/PORTING_NOTES_EN.md](examples/nanogpt/PORTING_NOTES_EN.md) Appendix.

---

### MiMo-V2.5-Pro — Tensor Inventory

**70 layers: layer 0 dense + layers 1–69 MoE. 10 full-attention + 60 SWA layers (hybrid). Runtime dtype: `bfloat16`; checkpoint dtype: `e4m3fnuz` (FP8), every linear weight paired with a `weight_scale` tensor.**

#### Global Tensors (× 1)

| Tensor | Shape | Params | Notes |
|---|---|---|---|
| `model.embed_tokens.embedding` | `(152576, 6144)` | 937,689,088 | Token embedding |
| `model.norm.scale` | `(6144,)` | 6,144 | Final RMSNorm γ — no bias |
| `lm_head.embedding` | `(152576, 6144)` | 937,689,088 | Output projection — **not tied** to embed_tokens |

Global subtotal: **1,875,384,320** ≈ 1.75B params

#### Per-Layer Tensors — All 70 Layers

SWA and full-attention layers have identical tensor shapes; the difference is the attention mask, not the weights.

**Normalization:**

| Tensor | Shape | Params / layer | Notes |
|---|---|---|---|
| `input_layernorm.scale` | `(6144,)` | 6,144 | Pre-attention RMSNorm γ — no bias |
| `post_attention_layernorm.scale` | `(6144,)` | 6,144 | Pre-MLP RMSNorm γ — no bias |

**Attention:**

| Tensor | Shape | Params / layer | Notes |
|---|---|---|---|
| `self_attn.q_proj.weight` | `(6144, 24576)` | 150,994,944 | Q: 128 heads × 192 head_dim |
| `self_attn.k_proj.weight` | `(6144, 1536)` | 9,437,184 | K: 8 KV heads × 192 head_dim |
| `self_attn.v_proj.weight` | `(6144, 1024)` | 6,291,456 | V: 8 KV heads × 128 v_head_dim |
| `self_attn.o_proj.weight` | `(16384, 6144)` | 100,663,296 | O: 128 heads × 128 v_head_dim → 6144 |
| `self_attn.attention_sink_bias` | `(128,)` | 128 | Per-Q-head logit bias for attention sink |

> In the checkpoint (Pro variant), Q/K/V are stored as a single fused `qkv_proj` weight; the weight loader splits them on load.

Per-layer subtotal: **267,399,296** × 70 layers = **18,717,950,720** ≈ 18.72B params

#### Layer 0 — Dense MLP

| Tensor | Shape | Params | Notes |
|---|---|---|---|
| `mlp.gate_proj.weight` | `(6144, 16384)` | 100,663,296 | SwiGLU gate branch |
| `mlp.up_proj.weight` | `(6144, 16384)` | 100,663,296 | SwiGLU up branch |
| `mlp.down_proj.weight` | `(16384, 6144)` | 100,663,296 | Project back to hidden |

Dense MLP subtotal: **301,989,888** ≈ 302M params

#### Layers 1–69 — MoE Block (× 69 layers)

**Router (× 69 layers):**

| Tensor | Shape | Params / layer | Notes |
|---|---|---|---|
| `mlp.moe_gate.kernel` | `(6144, 384)` | 2,359,296 | Expert router: hidden → 384 logits |
| `mlp.correction_bias` | `(384,)` | 384 | Per-expert bias for `noaux_tc` routing |

**Expert Weights (× 384 experts per layer; EP-sharded across devices at runtime):**

| Tensor | Shape / expert | Params / expert | Params (× 384) | Notes |
|---|---|---|---|---|
| `experts[j].gate_proj.weight` | `(6144, 2048)` | 12,582,912 | 4,831,838,208 | SwiGLU gate |
| `experts[j].up_proj.weight` | `(6144, 2048)` | 12,582,912 | 4,831,838,208 | SwiGLU up |
| `experts[j].down_proj.weight` | `(2048, 6144)` | 12,582,912 | 4,831,838,208 | Project back to hidden |

Per expert: **37,748,736** params; all 384 experts per layer: **14,495,514,624** ≈ 14.50B  
Per MoE layer (router + experts): ≈ **14.498B**  
69 MoE layers total: ≈ **~1,000.3B** ≈ 1T

#### FP8 Scale Tensors (checkpoint-only)

Each linear weight is stored as `e4m3fnuz` (FP8) and paired with a `weight_scale` for dequantization:
- Attention Q: FP8 dequant applied per-layer by the weight loader after splitting `qkv_proj`
- KV projections: per-head-block cross-boundary dequant, with KV heads replicated for TP alignment
- MoE expert weights: dequantized in VMEM inside the `FusedEPMoEV2` Pallas kernel (cost hidden by DMA overlap)

Scale tensors are not counted in the parameter totals above.

#### Grand Total

| Category | Params | Share |
|---|---|---|
| Global (embed_tokens + lm_head + norm) | 1,875,384,320 | 0.18% |
| Attention × 70 layers | 18,717,950,720 | 1.83% |
| Dense MLP (layer 0 only) | 301,989,888 | 0.03% |
| MoE expert weights × 69 layers | ~1,000,353,327,000 | 97.95% |
| **Grand total** | **~1,021,248,651,904** | **~1.02T** |

**Active params per token** = attention (all 70 layers) + dense MLP + top-8 experts × 69 MoE layers  
≈ 18.72B + 0.30B + 1.75B + (8 × 37.75M × 69) ≈ **~42B**

---

### Structural Comparison

| Dimension | NanoGPT (GPT-2 124M) | MiMo-V2.5-Pro (1.02T / 42B active) |
|---|---|---|
| **Total params** | 124M | ~1.02T (8,233× larger) |
| **Active params / token** | 124M (fully dense) | ~42B (338×; only 8/384 = 2.1% of experts active) |
| **Layer count** | 12 (all identical) | 70 (1 dense + 69 MoE) |
| **Attention type** | Full MHA, all layers | Hybrid: 10 full-attn + 60 SWA (window = 128 tokens) |
| **Q / KV heads** | 12 / 12 (no GQA) | 128 / 8 (16× KV head sharing) |
| **Head dim (Q/K, V)** | 64, 64 | 192, 128 (asymmetric) |
| **Token embedding** | `(50304, 768)` = 38.6M | `(152576, 6144)` = 937.7M (24× larger) |
| **Position embedding** | `wpe (1024, 768)` = 786K | None — RoPE applied inside attention, no learned position tensor |
| **LM head** | Weight-tied to `wte` (0 extra params) | Separate `(152576, 6144)` = 937.7M |
| **Attention kernel** | Fused `c_attn (768, 2304)` + `c_proj (768, 768)` | Separate `q/k/v/o_proj`; fused `qkv_proj` in checkpoint, split on load |
| **Normalization** | LayerNorm — scale + bias (2 tensors/norm) | RMSNorm — scale only, no bias (1 tensor/norm) |
| **MLP type** | Dense 2-layer GELU (expand/contract) | Dense SwiGLU (layer 0); MoE SwiGLU (layers 1–69) |
| **MLP intermediate** | 768 → 3072 → 768 (4× expand) | 6144 → 16384 (dense); 6144 → 2048 per expert (MoE) |
| **Expert count** | None | 384 total; 8 active per token (2.1%) |
| **Expert params** | — | 37.7M / expert; 14.5B all 384 per MoE layer |
| **Router** | None | `(6144, 384)` gate + `(384,)` correction bias |
| **Attention sink bias** | None | `(128,)` per layer (one per Q head) |
| **dtype** | float32 | bfloat16 (FP8 in checkpoint) |
| **Vocab size** | 50304 (padded from 50257) | 152576 |
| **Max context** | 1024 tokens | 1M tokens (RoPE + SWA) |

---

## Summary

NanoGPT is a clean, self-contained GPT-2 implementation for learning and experimentation — one training model, one generation script, minimal dependencies. MiMo-V2.5-Pro is a production-grade system: hybrid MoE architecture, paged KV cache serving, FP8 quantization, speculative decoding, multi-host tensor/data/expert parallelism, and hand-written Pallas kernels that bypass XLA's scheduler to achieve peak TPU utilization. They share the same high-level decoder-only transformer skeleton (embedding → stacked blocks → LM head) and the same Flax NNX module system, but differ in virtually every implementation detail beneath that skeleton.

| Dimension | NanoGPT | MiMo-V2.5-Pro |
|---|---|---|
| Normalization | LayerNorm | RMSNorm + floating residual |
| Position | Learned absolute (`wpe`) | RoPE + partial factor + rope_scaling |
| Attention | Full MHA, causal mask | Hybrid SWA/full GQA, RadixAttention |
| MLP | 2-layer GELU | SwiGLU; MoE on selected layers |
| Linear | Custom `Linear`, no sharding | `LinearBase`, TP axes, FP8 |
| Weight loading | Direct `.value` assignment | `WeightLoader` + multi-stage FP8 dequant |
| Inference | Python loop, no KV cache | `ForwardBatch`, paged KV, serving runtime |
| Compute kernels | Plain JAX / XLA | Pallas: flash-attn, fused EP MoE, KV scatter |
| Data parallel | `jax.pmap` replica | Multi-DP, `["data","tensor"]` mesh, DP-sharded metadata |
| Expert parallel | None | In-Pallas all-to-all, no JAX collectives |
| Deployment | 4-chip pmap | 4-host TPU pod, tp=8 / dp=2 / ep=8 |
