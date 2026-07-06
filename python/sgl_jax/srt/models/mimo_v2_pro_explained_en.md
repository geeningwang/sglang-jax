# mimo_v2_pro.py / mimo_v2_flash.py — Comprehensive Explanation

This document walks through every class, method, and significant line of the
MiMo-V2.5-Pro model implementation, explaining **what** each piece does, **why**
it is written that way, and **how** the pieces fit together in the SGLang-JAX
serving stack.

The implementation spans two files:

| File | Role |
|---|---|
| `mimo_v2_flash.py` | Base classes — all network modules, weight-loading plumbing, inference `__call__` |
| `mimo_v2_pro.py` | Thin subclass — overrides weight loading to handle the fused `qkv_proj` FP8 checkpoint format |

Reading order: understand `mimo_v2_flash.py` first, then `mimo_v2_pro.py` as a delta.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Imports](#2-imports)
3. [MiMoV2MLP — Dense SwiGLU MLP](#3-mimov2mlp--dense-swiglu-mlp)
4. [MiMoV2Moe — Mixture-of-Experts Block](#4-mimov2moe--mixture-of-experts-block)
5. [MiMoV2Attention — Hybrid GQA Attention](#5-mimov2attention--hybrid-gqa-attention)
6. [MiMoV2DecoderLayer — Single Transformer Layer](#6-mimov2decoderlayer--single-transformer-layer)
7. [MiMoV2Model — Full Layer Stack](#7-mimov2model--full-layer-stack)
8. [MiMoV2FlashForCausalLM — Flash Variant (Causal LM Head)](#8-mimov2flashforcausallm--flash-variant-causal-lm-head)
9. [MiMoV2ForCausalLM — Pro Variant (Fused QKV Override)](#9-mimov2forcausallm--pro-variant-fused-qkv-override)
10. [Weight Loading Pipeline](#10-weight-loading-pipeline)
11. [Complete Tensor Inventory](#11-complete-tensor-inventory)
12. [Summary and Key Design Decisions](#12-summary-and-key-design-decisions)

---

## 1. Architecture Overview

MiMo-V2.5-Pro is a **decoder-only transformer** with 1.02 trillion total
parameters and approximately 42 billion active parameters per token. Its key
architectural properties:

| Property | Value |
|---|---|
| Total layers | 70 |
| Layer 0 | Dense SwiGLU MLP |
| Layers 1–69 | MoE SwiGLU (384 experts, top-8) |
| Attention type | Hybrid: 10 full-attention + 60 SWA (window=128) |
| Q heads | 128 |
| KV heads | 8 (GQA, 16× sharing) |
| Q/K head_dim | 192 |
| V head_dim | 128 (asymmetric!) |
| Hidden size | 6144 |
| Vocab size | 152576 |
| MoE intermediate | 2048 per expert |
| Dense intermediate | 16384 (layer 0 only) |
| Runtime dtype | bfloat16 |
| Checkpoint dtype | e4m3fnuz (FP8) |
| Weight tying | False (separate `lm_head`) |

The network is a standard residual stack:

```
input_ids
  → embed_tokens                   (vocab → hidden)
  → for each of 70 decoder layers:
      residual = hidden_states
      hidden_states = input_layernorm(hidden_states)
      hidden_states = self_attn(hidden_states) + residual   # floating residual
      residual = hidden_states
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = mlp(hidden_states) + residual          # floating residual
  → norm (final RMSNorm)
  → lm_head                        (hidden → vocab logits)
```

**Floating residual** means the residual is not accumulated inside each sub-layer
but is kept separate and added by the decoder layer after each sub-block. This
enables the two additions (after attention and after MLP) to fuse with the
subsequent layernorm in XLA, reducing HBM traffic.

---

## 2. Imports

### mimo_v2_flash.py

```python
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.fused_moe import FusedEPMoE, FusedEPMoEV2
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.parallel_utils import make_reduce_sharding
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
```

**Key imports explained:**

- **`LinearBase`** — the standard linear projection layer in SGLang-JAX. Accepts
  `kernel_axes` annotations that drive JAX's GSPMD sharding so XLA can shard the
  weight matrix across the device mesh automatically. For column-parallel
  projections `kernel_axes=(None, "tensor")` shards output features; for
  row-parallel projections `kernel_axes=("tensor", None)` shards input features.

- **`RMSNorm`** — Root Mean Square Normalization (no mean subtraction, no bias).
  Cheaper than LayerNorm and standard in modern LLMs.

- **`Embed` / `ParallelLMHead`** — Token embedding and the language model head.
  `Embed` stores a `(vocab, hidden)` weight sharded along the vocab ("tensor")
  axis. `ParallelLMHead` is the separate output projection when
  `tie_word_embeddings=False`.

- **`get_rope`** — Factory that returns a `RotaryEmbedding` module. Supports
  NeoX-style RoPE, `partial_rotary_factor`, and `rope_scaling` (e.g. YaRN for
  long-context extension).

- **`RadixAttention`** — A thin metadata holder for the paged KV cache. It does
  not compute attention itself; the actual computation is done by the Pallas
  `ragged_paged_attention_v3` kernel invoked at the serving runtime level.

- **`FusedEPMoEV2` / `FusedEPMoE` / `EPMoE`** — Three MoE expert dispatch backends.
  `FusedEPMoEV2` is the production Pallas kernel for MiMo (no JAX collectives,
  in-kernel EP all-to-all). `FusedEPMoE` is an older fused variant.
  `EPMoE` is a non-fused GMM fallback.

- **`GateLogit` / `TopK`** — The MoE router: `GateLogit` computes
  `hidden @ gate_kernel` to produce expert logits; `TopK` selects the top-K
  experts and renormalizes their weights.

- **`ForwardBatch`** — Carries all per-batch metadata (token positions, request
  IDs, paged KV pool references, DP rank assignments, etc.) for one serving step.

- **`WeightLoader` / `WeightMapping`** — The weight-loading system. `WeightMapping`
  is a dataclass describing how one HuggingFace tensor maps to one JAX parameter
  (target path, sharding, transpose, FP8 flags). `WeightLoader` reads safetensors
  files and applies each mapping.

- **`make_reduce_sharding`** — Helper that returns the `NamedSharding` to use after
  a row-parallel linear layer. When Sequence Parallel (SP) is enabled, the output
  is sharded along the sequence axis; otherwise it replicates along "data".

### mimo_v2_pro.py

```python
import logging
import jax
import jax.numpy as jnp
from transformers import PretrainedConfig
from sgl_jax.srt.layers.moe import create_moe_weights_mapping
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
```

The Pro file imports only what it needs to override: the base class, the MoE
weight mapping factory, and the weight-loading primitives. All network modules
are inherited unchanged from the Flash file.

---

## 3. MiMoV2MLP — Dense SwiGLU MLP

```python
class MiMoV2MLP(nnx.Module):
    def __init__(self, hidden_size, intermediate_size, mesh, layer_id=0, dtype=...):
        self.gate_proj = LinearBase(input_size=hidden_size,
                                    output_size=intermediate_size,
                                    kernel_axes=(None, "tensor"), ...)
        self.up_proj   = LinearBase(input_size=hidden_size,
                                    output_size=intermediate_size,
                                    kernel_axes=(None, "tensor"), ...)
        self.down_proj = LinearBase(input_size=intermediate_size,
                                    output_size=hidden_size,
                                    kernel_axes=("tensor", None), ...)
        self.act_fn = jax.nn.silu
```

**What it is.** A SwiGLU feed-forward network with three linear projections. Used
only for **layer 0** (the single dense MLP layer in the 70-layer stack).

**SwiGLU computation:**

```python
def __call__(self, hidden_states, *, out_sharding=None):
    a1, _ = self.gate_proj(hidden_states)   # gate path
    a2, _ = self.up_proj(hidden_states)     # up path
    intermediate_parallel = a2 * self.act_fn(a1)  # SiLU gate × up
    output, _ = self.down_proj(intermediate_parallel, out_sharding=out_sharding)
    return output
```

`gate_proj` and `up_proj` both expand `hidden_size` → `intermediate_size`
(6144 → 16384 for layer 0). They run in parallel (no dependency), though written
sequentially here. XLA's scheduler fuses them. `silu(gate) * up` is the gated
activation. `down_proj` contracts 16384 → 6144.

**Tensor parallelism.** `gate_proj` and `up_proj` use `kernel_axes=(None, "tensor")`:
column-parallel, each device holds a vertical slice of the output-feature
dimension. `down_proj` uses `kernel_axes=("tensor", None)`: row-parallel, each
device holds a horizontal slice of the input-feature dimension. The implicit
all-reduce after `down_proj` is handled by `make_reduce_sharding` in the caller.

---

## 4. MiMoV2Moe — Mixture-of-Experts Block

```python
class MiMoV2Moe(nnx.Module):
    def __init__(self, config, layer_id, mesh, dtype):
        num_experts          = config.n_routed_experts        # 384
        num_experts_per_tok  = config.num_experts_per_tok     # 8
        moe_intermediate_size = config.moe_intermediate_size  # 2048

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,  # 6144
            num_experts=num_experts,        # 384
            score_func=config.scoring_func, # "softmax"
        )
        self.topk_method = config.topk_method  # "noaux_tc"
        if self.topk_method == "noaux_tc":
            self.correction_bias = nnx.Param(jnp.zeros(num_experts, dtype=jnp.float32))
        else:
            self.correction_bias = None

        self.topk = TopK(topk=num_experts_per_tok, renormalize=config.norm_topk_prob)
        self.experts = FusedEPMoEV2(...)  # or FusedEPMoE / EPMoE
```

Used for **layers 1–69** (all 69 MoE layers).

### 4.1 Router (GateLogit + TopK + correction_bias)

`GateLogit` applies a single linear projection without bias:

```
router_logits = hidden_states @ moe_gate.kernel   # (tokens, 6144) @ (6144, 384) → (tokens, 384)
```

For `noaux_tc` routing, a learned **correction bias** is added before top-K
selection:

```
effective_logits = router_logits + correction_bias   # (tokens, 384)
topk_weights, topk_ids = TopK(effective_logits)     # (tokens, 8) each
```

**Why `noaux_tc`?** Traditional MoE routing trains with an auxiliary load-balancing
loss that penalizes routing imbalance. `noaux_tc` ("no auxiliary loss, target
count correction") replaces the aux loss with a per-expert bias term
(`correction_bias`) that is adjusted during training to steer each expert toward
its target token count. This achieves load balance without the instability that
auxiliary losses can introduce. The bias is float32 even if the model is bf16
because it accumulates small corrections over many gradient steps.

`TopK.renormalize=True` re-normalizes the softmax scores of the selected top-8
experts so they sum to 1, giving the correct weighted combination of expert
outputs.

### 4.2 Expert Dispatch

```python
def __call__(self, hidden_states, forward_batch, *, out_sharding):
    router_logits = self.moe_gate(hidden_states)
    correction_bias = self.correction_bias.value if self.correction_bias else None
    topk_weights, topk_ids = self.topk(router_logits, correction_bias=correction_bias)

    if self.use_fused:
        token_valid_mask = forward_batch.get_token_valid_mask(...)
        if token_valid_mask is not None:
            topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
        mlp_output = self.experts(hidden_states, topk_weights, topk_ids, ...)
    else:
        mlp_output = self.experts(hidden_states, topk_weights, topk_ids, ...)

    return mlp_output, topk_ids
```

**`token_valid_mask`** handles padded positions (padding tokens from the DP
batching layer). Padding positions get `topk_ids = -1`, which signals the
FusedEPMoEV2 kernel to skip those tokens. Without this, padding tokens would
route to experts and waste compute.

**`topk_ids` return value** — the caller (decoder layer) collects all layers'
routing decisions and passes them back through `__call__`, ultimately for
speculative decoding diagnostics or routing statistics.

### 4.3 Expert Backends

| Backend | Class | When used |
|---|---|---|
| `fused_v2` | `FusedEPMoEV2` | MiMo V2 (production; in-Pallas EP all-to-all) |
| `fused` | `FusedEPMoE` | Other MoE models (older fused variant) |
| `epmoe` | `EPMoE` | Fallback (non-fused GMM via megablox) |

`FusedEPMoEV2` implements Expert Parallelism entirely inside a Pallas kernel:
tokens are scattered to their assigned expert devices and results gathered back
without going through the JAX collective graph. Each device holds
`384 / ep_size` experts (e.g., 48 experts with ep=8).

---

## 5. MiMoV2Attention — Hybrid GQA Attention

```python
class MiMoV2Attention(nnx.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads,
                 head_dim, v_head_dim, sliding_window_size, ...):
        self.q_proj = LinearBase(hidden_size, num_heads * head_dim,
                                  kernel_axes=(None, "tensor"), ...)
        self.k_proj = LinearBase(hidden_size, num_kv_heads * head_dim,
                                  kernel_axes=(None, "tensor"), ...)
        self.v_proj = LinearBase(hidden_size, num_kv_heads * v_head_dim,
                                  kernel_axes=(None, "tensor"), ...)
        self.o_proj = LinearBase(num_heads * v_head_dim, hidden_size,
                                  kernel_axes=("tensor", None), ...)
        self.rotary_emb = get_rope(head_size=head_dim, ...)
        self.attn = RadixAttention(num_heads, head_dim, scaling,
                                    num_kv_heads, layer_id, v_head_dim,
                                    sliding_window_size)
        self.attention_sink_bias = nnx.Param(...) if attention_sink_bias else None
```

### 5.1 GQA with Asymmetric Head Dimensions

MiMo uses **Grouped Query Attention (GQA)** with unusual asymmetric dimensions:

| Projection | Heads | Head dim | Total output |
|---|---|---|---|
| Q | 128 | 192 | 24576 |
| K | 8 | 192 | 1536 |
| V | 8 | 128 | 1024 |
| O | input=128×128=16384 | — | 6144 |

Q has far more heads than K/V (16× sharing), which reduces KV cache memory by
16×. Q and K have head_dim=192 (used for RoPE and scaled dot-product). V has a
smaller head_dim=128, which reduces the output size of attention and thus the
size of `o_proj`. The output projection takes `128 × 128 = 16384` inputs (Q
heads × V head_dim) and maps back to hidden_size=6144.

### 5.2 V Padding for Fused KV Cache

```python
if self.v_head_dim != self.head_dim:
    pad_size = self.head_dim - self.v_head_dim  # 192 - 128 = 64
    v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_size)))
```

The paged KV cache (`RadixAttention`) stores K and V interleaved per page, so
both must have the same size. V is padded from 128 to 192 dims before entering
the cache. After attention, the output is sliced back to the true V size:

```python
if self.head_dim != self.v_head_dim:
    attn_output = attn_output.reshape(-1, self.q_head_num, padded_head_dim)
    attn_output = attn_output[..., :self.v_head_dim]   # slice off padding
    attn_output = attn_output.reshape(-1, self.q_head_num * self.v_head_dim)
```

### 5.3 Rotary Embeddings

```python
q, k = self.rotary_emb(positions, q, k)
```

RoPE is applied to Q and K only (not V), using NeoX-style rotation. `positions`
carries the absolute sequence position for each token (accounting for KV cache
prefixes in the paged cache). `partial_rotary_factor < 1.0` would apply RoPE
only to a fraction of the head dimension; MiMo uses 1.0 (full rotation).

### 5.4 Attention Value Scaling

```python
if self.attention_value_scale is not None:
    v = v * self.attention_value_scale   # 0.612 for MiMo-V2.5-Pro
```

A scalar multiplier applied to V before attention. This is an architectural
choice to stabilize output magnitudes at scale without changing the standard
softmax attention formula. Value is 0.612 for MiMo-V2.5-Pro.

### 5.5 Attention Sink Bias

```python
self.attention_sink_bias = nnx.Param(shape=(num_heads,), ...) if attention_sink_bias else None
```

An optional learned per-Q-head bias added to the attention logits for a virtual
"sink" token. The Pallas flash attention kernel implements the sink without
materializing an extra KV entry: it pre-computes the sink contribution as a
scalar bias and includes it in the softmax normalization (replacing
`l=0 / m=−∞` initialization). This enables very long contexts (up to 1M tokens)
to always attend to a consistent starting point, preventing attention entropy
collapse.

For MiMo-V2.5-Pro: SWA layers have sink bias if `add_swa_attention_sink_bias=True`
in config; full-attention layers if `add_full_attention_sink_bias=True`.

### 5.6 RadixAttention (Paged KV Cache)

```python
attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool,
                                   attention_sink=self.attention_sink_bias.value)
```

`RadixAttention` is a thin metadata holder. At call time it invokes the Pallas
`ragged_paged_attention_v3` kernel with:
- Q: `(tokens, q_heads, head_dim)` — per-layer per-device
- K/V: paged in `token_to_kv_pool` pages
- `sliding_window_size`: 128 for SWA layers, 0 for full-attention layers
- `kv_fused`: the new K/V page data to write back to the pool (fused update)

`kv_fused` is the new K/V data generated by this step; the layer returns it and
the causal LM class collects all layers' `kv_fused` into `layers_kv_fused` for
the pool update.

### 5.7 Hybrid SWA vs Full Attention

`MiMoV2DecoderLayer._is_swa_layer()` reads `hybrid_layer_pattern[layer_id]`:

```python
def _is_swa_layer(self, config) -> bool:
    hybrid = getattr(config, "hybrid_layer_pattern", None)
    if hybrid is not None and 0 <= self.layer_id < len(hybrid):
        return hybrid[self.layer_id] == 1   # 1=SWA, 0=full
    return False
```

Pattern value `1` → SWA (sliding_window_size=128); `0` → full attention
(sliding_window_size=0). Out of 70 layers: 10 full-attention, 60 SWA.

SWA and full-attention layers use the **same tensor shapes** (same Q/K/V/O head
counts and dimensions from `swa_num_attention_heads` == `num_attention_heads` ==
128 in the MiMo config). Only the attention mask differs, not the weights.

---

## 6. MiMoV2DecoderLayer — Single Transformer Layer

```python
class MiMoV2DecoderLayer(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype):
        self.is_layer_sparse = self._is_moe_layer(config)

        # Attention: SWA or full depending on hybrid_layer_pattern[layer_id]
        if self._is_swa_layer(config):
            self.self_attn = MiMoV2Attention(...swa config params...)
        else:
            self.self_attn = MiMoV2Attention(...full attn config params...)

        # MLP: dense (layer 0) or MoE (layers 1–69)
        if self.is_layer_sparse:
            self.mlp = MiMoV2Moe(config, layer_id, mesh, dtype)
        else:
            self.mlp = MiMoV2MLP(hidden_size, intermediate_size, mesh, layer_id, dtype)

        self.input_layernorm = RMSNorm(config.hidden_size, ...)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, ...)
```

**Two independent binary choices per layer:**
1. Attention: SWA (`hybrid_layer_pattern[i] == 1`) or full (`== 0`)
2. MLP: MoE (`moe_layer_freq[i]` truthy) or dense (falsy)

For MiMo-V2.5-Pro, these choices happen to be correlated (layer 0 is dense MLP
and is always full-attention), but the code treats them independently.

### 6.1 Forward Pass — Floating Residual Pattern

```python
def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool,
             residual=None):
    reduce_sharding = make_reduce_sharding(hidden_states, self.mesh, ...)

    # Attention sub-block
    if residual is not None:
        hidden_states += residual           # add previous layer's residual
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_fused = self.self_attn(...)
    hidden_states += jax.sharding.reshard(residual, reduce_sharding)

    # MLP sub-block
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if self.is_layer_sparse:
        hidden_states, topk_ids = self.mlp(hidden_states, forward_batch, ...)
    else:
        hidden_states = self.mlp(hidden_states, ...)
    residual = jax.sharding.reshard(residual, reduce_sharding)

    return hidden_states, residual, kv_fused, topk_ids
```

**Why return `residual` separately?** The caller (`MiMoV2Model`) accumulates
the residual across layers. On the first layer, `residual=None` is passed in, so
the first `hidden_states += residual` is skipped. This way the embedding output
does not need an extra copy — it becomes the first residual naturally. The final
layer's returned residual is added to `hidden_states` in `MiMoV2Model.__call__`
before the final norm.

**`jax.sharding.reshard(residual, reduce_sharding)`** — after the row-parallel
down-projection, the output is on `reduce_sharding`. The residual was on a
different sharding (column-parallel output). This call ensures the residual is on
the same sharding before addition, avoiding implicit JAX resharding overhead.

**`make_reduce_sharding`** returns the sharding for the all-reduce output of
row-parallel layers. Without sequence parallelism (SP off) this is a full
replication; with SP it shards along the sequence axis to avoid duplicating the
full hidden state on each chip.

---

## 7. MiMoV2Model — Full Layer Stack

```python
class MiMoV2Model(nnx.Module):
    def __init__(self, config, mesh, dtype):
        self.embed_tokens = Embed(num_embeddings=vocab_size,
                                   features=hidden_size,
                                   kernel_axes=("tensor", None), ...)
        self.layers = nnx.data([
            MiMoV2DecoderLayer(config=config, layer_id=i, ...)
            for i in range(config.num_hidden_layers)   # 70
        ])
        self.norm = RMSNorm(hidden_size, ...)
```

**`nnx.data([...])`** — wraps a Python list of NNX modules into an NNX-trackable
sequence. Without this, NNX's parameter traversal would not see the list
contents. `nnx.data` is the NNX equivalent of `nn.ModuleList` in PyTorch.

### 7.1 Forward Pass

```python
def __call__(self, forward_batch, token_to_kv_pool):
    residual = None
    hidden_states = self.embed_tokens(forward_batch.input_ids)
    layers_kv_fused = []
    layers_topk_ids = []

    for i, layer in enumerate(self.layers):
        hidden_states, residual, kv_fused, topk_ids = layer(
            forward_batch.positions, hidden_states,
            forward_batch, token_to_kv_pool, residual,
        )
        layers_kv_fused.append(kv_fused)
        layers_topk_ids.append(topk_ids)

    if residual is not None:
        hidden_states += residual   # final residual addition
    hidden_states = self.norm(hidden_states)
    return hidden_states, layers_kv_fused, layers_topk_ids
```

`layers_kv_fused` — one entry per layer, each is the new K/V data from that
layer's attention, used to update the paged KV pool after inference.

`layers_topk_ids` — one entry per layer (None for dense layers, `(tokens, 8)`
for MoE layers), returned to the causal LM class and ultimately passed back to
the serving runtime for routing statistics or speculative decoding.

---

## 8. MiMoV2FlashForCausalLM — Flash Variant (Causal LM Head)

```python
class MiMoV2FlashForCausalLM(nnx.Module):
    def __init__(self, config, mesh, dtype):
        self.model = MiMoV2Model(config, dtype=dtype, mesh=mesh)
        self._kv_buffers: dict[int, dict] = {}   # FP8 K/V staging buffer

        if not config.tie_word_embeddings:   # False for MiMo
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, ...)

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)
```

**`_kv_buffers`** — a Python dict keyed by `layer_idx`, each value holding the
raw FP8 K/V weights and scales before per-head dequantization. It exists only
during weight loading and is populated by the K/V weight mappings that redirect
FP8 K/V data to `__KV_K_WEIGHT__{idx}` / `__KV_V_WEIGHT__{idx}` keys instead of
the real parameter paths.

### 8.1 GCSFuse Cache Warm-Up

```python
@staticmethod
def _warmup_safetensors_cache(model_config):
    # Check if model_path is on a GCSFuse mount
    with open("/proc/mounts") as fp: ...
    if "fuse" not in mount_type:
        return   # skip on block-device mounts

    # Sequential bulk read of all .safetensors files
    def _read_file(path):
        buf = bytearray(4 * 1024 * 1024)
        with open(path, "rb") as f:
            while f.readinto(buf): pass

    with ThreadPoolExecutor(max_workers=min(8, len(st_files))) as executor:
        list(executor.map(_read_file, st_files))
```

**Why this exists.** Model weights are stored in Google Cloud Storage (GCS),
mounted via GCSFuse. GCSFuse caches file data in kernel page cache, but cold
random reads cost ~400 ms per tensor (one GCS API call per read). The MoE
checkpoint has thousands of expert weight tensors, so loading them cold would
take hours.

**The fix:** read every safetensors file sequentially once. Sequential reads are
served by GCSFuse's prefetcher and fill the page cache, turning subsequent random
reads into cache hits (~1 ms). This warm-up is skipped on block-device mounts
(GKE Persistent Disk, NFS) where it is not needed.

### 8.2 Weight Loading (Flash variant)

```python
def load_weights(self, model_config):
    self._warmup_safetensors_cache(model_config)
    self.loader = WeightLoader(...)
    weight_mappings = self._create_weight_mappings()
    self.loader.load_weights_from_safetensors(weight_mappings)

    if self.loader.is_static_quant:
        # 1. Dequant Q (per-layer)
        self.loader.dequant_fp8_layers(layers, specs=[("self_attn.q_proj", head_dim)])
        # 2. Fused KV dequant (cross K/V boundary blocks)
        self.loader.dequant_fused_kv(self._kv_buffers, layers, config)
        # 3. Layer-0 dense MLP dequant
        self.loader.dequant_fp8_layers(layers, specs=[...], layer_filter=lambda i, l: i==0)
        # 4. KV head replication for TP alignment
        self.loader.replicate_kv_heads(layers, specs=[...])
```

The four post-load steps are detailed in [§10 Weight Loading Pipeline](#10-weight-loading-pipeline).

### 8.3 `_create_weight_mappings`

Builds a flat `dict[str, WeightMapping]` covering all model parameters.
Structure:

```
global tensors:
  "model.embed_tokens.weight" → target "model.embed_tokens.embedding"
  "model.norm.weight"         → target "model.norm.scale"
  "lm_head.weight"            → target "lm_head.embedding"  (if not tie_word_embeddings)

per-layer (loop over 0..69):
  attention projections (q/k/v/o_proj)
  attention sink bias (conditional)
  layernorms (input_layernorm, post_attention_layernorm)
  MLP: dense (gate/up/down_proj) or MoE (gate + experts)
```

Each `WeightMapping` specifies:
- `target_path`: JAX parameter path (dot-separated attribute string)
- `sharding`: tuple of axis names (matched to `kernel_axes` on the `LinearBase`)
- `transpose`: whether to transpose before storing (HF stores weights as
  `(out, in)`; `LinearBase` expects `(in, out)`)
- `head_dim_padding` / `kv_head_padding`: for TP alignment of Q/K/V
- FP8 staging paths (`__KV_K_WEIGHT__{idx}`, etc.) for deferred dequantization

### 8.4 FP8 Attention Projection Handling (Flash variant)

For the Flash variant, Q, K, V are separate projections in the checkpoint:

- **Q** (`q_proj`): loaded directly with `weight_q` (FP8) + `weight_scale_inv`
  suffix, then dequantized per-layer in step 1 above.
- **K, V** (`k_proj`, `v_proj`): loaded into `_kv_buffers` staging dict (not the
  real parameter), then fused per-head dequantized in step 2. The "cross K/V
  boundary" refers to the FP8 quantization granularity: K and V tensors were
  quantized together per head block (one scale covering a block that may span
  the K/V boundary), so they must be dequantized jointly.

### 8.5 `__call__` — Inference Forward Pass

```python
def __call__(self, forward_batch, memory_pools, logits_metadata):
    kv_pool = memory_pools.token_to_kv_pool
    hidden_states, layers_kv_fused, layers_topk_ids = self.model(forward_batch, kv_pool)

    if not config.tie_word_embeddings:
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
    else:
        output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

    return output, {"token_to_kv_pool": layers_kv_fused}, True, layers_topk_ids
```

**Return value structure** — required by the SGLang-JAX model executor protocol:

| Position | Content | Used for |
|---|---|---|
| `output` | Logits output from `LogitsProcessor` | Token sampling / speculative verification |
| `{"token_to_kv_pool": layers_kv_fused}` | New K/V pages, one list per layer | KV pool update after inference |
| `True` | `callback_flags` (always True) | Triggers post-step callbacks |
| `layers_topk_ids` | Per-layer MoE routing decisions | Routing stats / speculative decoding |

`LogitsProcessor` selects the logits rows that correspond to the last token of
each request (for decode steps) or the last token of the prompt (for prefill
steps) via `logits_metadata`, applies temperature scaling and top-k/top-p
filtering, and returns the processed logit tensor.

### 8.6 `get_embed_and_head`

```python
def get_embed_and_head(self):
    embed = self.model.embed_tokens.embedding.value
    if not config.tie_word_embeddings:
        head = self.lm_head.embedding.value
    else:
        head = embed
    return embed, head
```

Called by the MTP (Multi-Token Prediction) draft model (`MiMoV2MTPForCausalLM`
in `mimo_v2_nextn.py`) to share the target model's embedding and LM head weights
at runtime without re-loading them.

---

## 9. MiMoV2ForCausalLM — Pro Variant (Fused QKV Override)

```python
class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):

    def __init__(self, config, mesh, dtype):
        super().__init__(config, mesh, dtype)
        self._fused_qkv_buffers: dict[int, dict] = {}
```

`MiMoV2ForCausalLM` inherits all network modules from Flash unchanged. The only
differences are in weight loading:

1. A second staging buffer `_fused_qkv_buffers` for the fused QKV FP8 data
2. Overridden `load_weights` that uses `dequant_fused_qkv` instead of
   `dequant_fp8_layers` for Q
3. Overridden `_create_layer_mappings` that handles `qkv_proj` (fused) instead of
   separate `q/k/v_proj` (split)

### 9.1 Why Pro Needs Special Handling

The Flash checkpoint stores `q_proj`, `k_proj`, and `v_proj` as three separate
tensors. The Pro checkpoint stores a **single fused `qkv_proj`** tensor that was
concatenated across tensor-parallel (TP) shards at quantization time:

```
qkv_proj layout in Pro checkpoint:
  [Q_shard_0 | K_shard_0 | V_shard_0 | Q_shard_1 | K_shard_1 | V_shard_1 | ...]
  ← shard 0 (tp_rank=0) →   ← shard 1 (tp_rank=1) →
```

Each shard block has its own FP8 scale. Splitting this back into separate Q/K/V
requires knowing the TP shard boundaries and applying the correct scale to each
shard block before splitting. This is implemented in
`WeightLoader.dequant_fused_qkv`.

### 9.2 Overridden `load_weights`

```python
def load_weights(self, model_config):
    self.loader = WeightLoader(...)
    weight_mappings = self._create_weight_mappings()   # calls overridden _create_layer_mappings
    self.loader.load_weights_from_safetensors(weight_mappings)

    if self.loader.is_static_quant:
        # Step 1: Dequant fused QKV (Pro-specific path)
        self.loader.dequant_fused_qkv(self._fused_qkv_buffers, self.model.layers, self.config)
        # Step 2: Layer-0 dense MLP dequant
        self.loader.dequant_fp8_layers(layers, specs=[gate/up/down], layer_filter=layer_0_dense)
        # Step 3: KV head replication for TP alignment
        self.loader.replicate_kv_heads(layers, specs=[k_proj, v_proj])
```

Note that Flash has 4 steps; Pro has only 3. The Flash `dequant_fp8_layers` for Q
and the separate `dequant_fused_kv` are replaced by a single `dequant_fused_qkv`
that handles Q/K/V together.

### 9.3 Overridden `_create_layer_mappings`

```python
def _create_layer_mappings(self, layer_idx):
    hf_qkv_key = f"model.layers.{layer_idx}.self_attn.qkv_proj"

    if is_fp8 and not qkv_ignored:
        # Stage raw FP8 fused QKV into buffer, not into the real parameter
        mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
            target_path=f"__FUSED_QKV_WEIGHT__{layer_idx}",
            sharding=(None, None), transpose=False,
        )
        mappings[f"{hf_qkv_key}.weight_scale_inv"] = WeightMapping(
            target_path=f"__FUSED_QKV_SCALE__{layer_idx}",
            sharding=(None, None), transpose=False,
        )
    else:
        # BF16 or quant-ignored: split Q/K/V normally from the fused tensor
        mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
            target_path=[q_proj.weight, k_proj.weight, v_proj.weight],
            sharding=(None, "tensor"),
            transpose=True,
            head_dim_padding=False,
            kv_head_padding=True,
        )
```

For FP8 checkpoints, the fused QKV weight and its scale are both redirected to
staging buffer paths (`__FUSED_QKV_WEIGHT__{idx}` / `__FUSED_QKV_SCALE__{idx}`).
`WeightLoader.load_weights_from_safetensors` recognizes these special paths and
stores the raw data in `_fused_qkv_buffers[layer_idx]` instead of the normal
parameter attributes.

For BF16 checkpoints (or quant-ignored layers), the fused tensor can be split
directly by `WeightMapping` using a list of three target paths. `WeightLoader`
splits it along the head dimension and assigns each slice to the corresponding
projection.

The rest of `_create_layer_mappings` (o_proj, sink bias, layernorms, MLP/MoE)
is identical to the Flash variant.

---

## 10. Weight Loading Pipeline

Weight loading is the most complex part of the implementation. Both variants
follow a two-phase approach:

### Phase 1: Bulk safetensors load

`WeightLoader.load_weights_from_safetensors(weight_mappings)` iterates over all
`WeightMapping` entries and for each:
1. Reads the tensor from the safetensors file on disk (or GCS via GCSFuse)
2. Transposes if `transpose=True`
3. Applies `head_dim_padding` / `kv_head_padding` for TP alignment
4. Shards the tensor according to `sharding` spec and places each shard on its
   device
5. Stores into the target parameter path (e.g., sets `layer.self_attn.q_proj.weight_q`)

For FP8 staging paths (`__KV_*` or `__FUSED_QKV_*`), step 5 instead stores into
`_kv_buffers` or `_fused_qkv_buffers`.

### Phase 2: FP8 dequantization (if `is_static_quant`)

The checkpoint is statically quantized to `e4m3fnuz` (FP8). After all raw data
is loaded, a multi-step dequantization converts to bfloat16 for actual inference.

#### Step A — Q dequantization (Flash only)

```python
self.loader.dequant_fp8_layers(layers, specs=[("self_attn.q_proj", head_dim)])
```

Each `q_proj` has `weight_q` (FP8) and `weight_scale` (bf16 scale). This reads
each layer's Q weight, applies `weight_q * weight_scale` to get bf16, and writes
the result back as `q_proj.weight`.

#### Step B — Fused KV dequantization (Flash) / Fused QKV (Pro)

**Flash:** `dequant_fused_kv(_kv_buffers, layers, config)`

K and V were quantized jointly (one FP8 scale per block spanning both K and V
channels). This function reads the staged raw FP8 data for K and V, dequantizes
each head block using its per-block scale (respecting the K/V boundary within the
block), then writes bf16 values into `k_proj.weight` and `v_proj.weight`.

**Pro:** `dequant_fused_qkv(_fused_qkv_buffers, layers, config)`

The staged fused QKV tensor has per-shard interleaved layout
`[Q_s0 | K_s0 | V_s0 | Q_s1 | K_s1 | V_s1 | ...]`. This function:
1. Iterates over TP shards
2. Dequantizes each shard block using its per-shard scale
3. Splits the dequantized shard into Q/K/V slices
4. Concatenates across shards and writes into `q_proj.weight`, `k_proj.weight`,
   `v_proj.weight`

#### Step C — Layer-0 dense MLP dequantization (both variants)

```python
self.loader.dequant_fp8_layers(
    layers,
    specs=[("mlp.gate_proj", None), ("mlp.up_proj", None), ("mlp.down_proj", None)],
    layer_filter=lambda idx, layer: idx == 0 and not layer.is_layer_sparse,
)
```

Only layer 0 has a dense MLP. `layer_filter` restricts dequantization to that
layer. MoE expert weights (layers 1–69) are dequantized **inside the Pallas
kernel** at runtime (in VMEM, overlapping with DMA), so they remain FP8 on disk
and in HBM.

#### Step D — KV head replication for TP alignment

```python
self.loader.replicate_kv_heads(
    layers,
    specs=[("self_attn.k_proj", head_dim), ("self_attn.v_proj", v_head_dim)],
    target_kv_heads_fn=lambda attn: attn.k_head_num,
)
```

In TP, each device shard holds a fraction of the Q heads. But K/V have only 8
heads total, which may be fewer than the number of TP shards. To give each shard
a complete copy of the K/V heads it needs, this step replicates K and V across
the TP dimension. For example, with TP=8 and 8 KV heads, each shard gets 1 KV
head — no replication needed. With TP=16, each shard gets 0.5 heads — the
heads would need replication.

### HF name → JAX parameter name mapping

| HuggingFace key | JAX parameter path |
|---|---|
| `model.embed_tokens.weight` | `model.embed_tokens.embedding` |
| `model.norm.weight` | `model.norm.scale` |
| `lm_head.weight` | `lm_head.embedding` |
| `model.layers.{i}.input_layernorm.weight` | `model.layers[i].input_layernorm.scale` |
| `model.layers.{i}.post_attention_layernorm.weight` | `model.layers[i].post_attention_layernorm.scale` |
| `model.layers.{i}.self_attn.q_proj.weight` (Flash) | `model.layers[i].self_attn.q_proj.weight_q` (FP8) |
| `model.layers.{i}.self_attn.qkv_proj.weight` (Pro) | staged → split into q/k/v (FP8) |
| `model.layers.{i}.self_attn.o_proj.weight` | `model.layers[i].self_attn.o_proj.weight_q` (FP8) |
| `model.layers.{i}.self_attn.attention_sink_bias` | same path |
| `model.layers.{i}.mlp.gate.weight` | `model.layers[i].mlp.moe_gate.kernel` |
| `model.layers.{i}.mlp.gate.e_score_correction_bias` | `model.layers[i].mlp.correction_bias` |
| `model.layers.{i}.mlp.experts.{j}.gate_proj.weight` | mapped by `create_moe_weights_mapping` |

---

## 11. Complete Tensor Inventory

**Runtime dtype: `bfloat16`. Checkpoint dtype: `e4m3fnuz` (FP8) for linear
weights; each FP8 weight is paired with a `weight_scale` for dequantization.
MoE expert weights remain FP8 in HBM and are dequantized in-kernel.**

### Global Tensors (× 1)

| Tensor | Shape | Params |
|---|---|---|
| `model.embed_tokens.embedding` | `(152576, 6144)` | 937,689,088 |
| `model.norm.scale` | `(6144,)` | 6,144 |
| `lm_head.embedding` | `(152576, 6144)` | 937,689,088 |

### Per-Layer Tensors — All 70 Layers

**Normalization:**

| Tensor | Shape | Params / layer |
|---|---|---|
| `input_layernorm.scale` | `(6144,)` | 6,144 |
| `post_attention_layernorm.scale` | `(6144,)` | 6,144 |

**Attention (SWA and full-attention layers have identical shapes):**

| Tensor | Shape | Params / layer | Notes |
|---|---|---|---|
| `self_attn.q_proj.weight` | `(6144, 24576)` | 150,994,944 | 128 Q heads × 192 |
| `self_attn.k_proj.weight` | `(6144, 1536)` | 9,437,184 | 8 KV heads × 192 |
| `self_attn.v_proj.weight` | `(6144, 1024)` | 6,291,456 | 8 KV heads × 128 |
| `self_attn.o_proj.weight` | `(16384, 6144)` | 100,663,296 | 128 × 128 v_head_dim → 6144 |
| `self_attn.attention_sink_bias` | `(128,)` | 128 | Optional per-Q-head |

Per-layer subtotal: **267,399,296** × 70 = **18,717,950,720** ≈ 18.72B

### Layer 0 — Dense MLP

| Tensor | Shape | Params |
|---|---|---|
| `mlp.gate_proj.weight` | `(6144, 16384)` | 100,663,296 |
| `mlp.up_proj.weight` | `(6144, 16384)` | 100,663,296 |
| `mlp.down_proj.weight` | `(16384, 6144)` | 100,663,296 |

Dense MLP subtotal: **301,989,888** ≈ 302M

### Layers 1–69 — MoE Block (× 69 layers)

| Tensor | Shape | Params / layer |
|---|---|---|
| `mlp.moe_gate.kernel` | `(6144, 384)` | 2,359,296 |
| `mlp.correction_bias` | `(384,)` | 384 |
| `mlp.experts[j].gate_proj.weight` × 384 | `(6144, 2048)` each | 4,831,838,208 |
| `mlp.experts[j].up_proj.weight` × 384 | `(6144, 2048)` each | 4,831,838,208 |
| `mlp.experts[j].down_proj.weight` × 384 | `(2048, 6144)` each | 4,831,838,208 |

Per MoE layer: ≈ **14.498B** × 69 = ≈ **1,000.3B** ≈ 1T

### Grand Total

| Category | Params |
|---|---|
| Global | 1,875,384,320 |
| Attention × 70 | 18,717,950,720 |
| Dense MLP (layer 0) | 301,989,888 |
| MoE expert weights × 69 | ~1,000,353,327,000 |
| **Total** | **~1.02T** |

**Active per token:** attention (all 70) + dense MLP + 8 experts × 69 layers ≈ **42B**

---

## 12. Summary and Key Design Decisions

### Flash vs Pro: one delta

The Flash variant (`MiMoV2FlashForCausalLM`) expects separate `q_proj`,
`k_proj`, `v_proj` weights in the checkpoint. The Pro variant
(`MiMoV2ForCausalLM`) expects a single fused `qkv_proj` weight. Everything
else — network architecture, forward pass, MoE routing, KV cache integration —
is identical. The subclass relationship (Pro inherits Flash) expresses this:
override only `load_weights` and `_create_layer_mappings`.

### MoE expert weights stay FP8 in HBM

Dense weights (embeddings, attention, layer-0 MLP) are dequantized to bf16 after
loading and stay bf16 in HBM. Expert weights (1.0T of the 1.02T total) remain
FP8 (1 byte/element) in HBM and are dequantized in VMEM inside the Pallas
`FusedEPMoEV2` kernel. This halves expert weight HBM bandwidth vs bf16 (from ~14
bytes/param-access to ~7), which is the dominant cost in MoE inference.

### Floating residual

Each decoder layer receives the previous layer's residual as a separate argument
and returns the new residual separately. This lets XLA schedule the residual
addition and layernorm as a single fused op, and avoids creating an intermediate
tensor for the sum inside the layer.

### TP axis names drive sharding

`LinearBase` `kernel_axes` annotations (`None`, `"tensor"`) bind to the named
axis of the 2D device mesh (`["data", "tensor"]`). No explicit `shard_map` or
`pjit` calls are needed; JAX's GSPMD propagates sharding automatically. Column-
parallel projections use `(None, "tensor")`; row-parallel use `("tensor", None)`.

### RadixAttention is stateless

`RadixAttention` holds no weight tensors — only layer metadata (head counts,
scaling, `sliding_window_size`). The actual KV cache is managed by the serving
runtime's `MemoryPools`; `RadixAttention` receives a reference via
`token_to_kv_pool`. This separation lets the same network weights serve multiple
concurrent requests with different KV cache states.

### EntryClass registration

```python
# mimo_v2_flash.py
EntryClass = [MiMoV2FlashForCausalLM]

# mimo_v2_pro.py
EntryClass = MiMoV2ForCausalLM
```

Both files set `EntryClass`, which the SGLang-JAX model registry reads to
discover the model class. Flash uses a list (allowing multiple entry points);
Pro uses a single class. The registry looks up this variable by name when a
config specifies `"architectures": ["MiMoV2ForCausalLM"]`.
