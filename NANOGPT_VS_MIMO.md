# NanoGPT vs MiMo-V2.5-Pro: Architecture Comparison

## Overview

| | NanoGPT (GPT-2) | MiMo-V2.5-Pro |
|---|---|---|
| **Purpose** | Educational / research training demo | Production inference serving |
| **Scale** | ~124M params (GPT-2 small) | ~56B params (MoE, ~8B active) |
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
