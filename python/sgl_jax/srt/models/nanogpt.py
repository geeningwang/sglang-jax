# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only NanoGPT (GPT-2) model compatible with HuggingFace weights.

Architecture: GPT-2 with absolute position embeddings, pre-norm blocks,
causal self-attention (MHA), and weight-tied token embedding / LM head.

Reference implementation:
  ~/transformer/nanogpt-tpu/model.py   (Flax Linen training model)
  https://github.com/karpathy/nanogpt  (original PyTorch)
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class NanoGPTConfig(PretrainedConfig):
    """HuggingFace-compatible config for the nanogpt / GPT-2 architecture.

    Exposes both nanogpt-style names (n_layer, n_head, n_embd) and the
    ModelConfig-expected aliases (num_hidden_layers, num_attention_heads,
    hidden_size, etc.) so that the serving stack can read architecture dims
    without special-casing.
    """

    model_type = "nanogpt"

    def __init__(
        self,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        block_size: int = 1024,
        vocab_size: int = 50304,
        bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.bias = bias
        self.dropout = dropout

        # Aliases expected by ModelConfig and WeightLoader
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head  # MHA — no GQA
        self.hidden_size = n_embd
        self.intermediate_size = 4 * n_embd
        self.max_position_embeddings = block_size
        self.head_dim = n_embd // n_head


class NanoGPTLayerNorm(nnx.Module):
    """Standard LayerNorm (mean + variance normalisation).

    GPT-2 uses standard LayerNorm, not RMSNorm.  We implement it directly
    with nnx.Param to follow the sgl_jax layer convention (see RMSNorm in
    python/sgl_jax/srt/layers/layernorm.py).
    """

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-5,
        use_bias: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.epsilon = epsilon
        self.scale = nnx.Param(jnp.ones((num_features,), dtype=param_dtype))
        self.bias: nnx.Param | None = (
            nnx.Param(jnp.zeros((num_features,), dtype=param_dtype)) if use_bias else None
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        out = self.scale.value * x_norm
        if self.bias is not None:
            out = out + self.bias.value
        return out.astype(orig_dtype)


class NanoGPTMLP(nnx.Module):
    """Two-layer MLP with GELU: n_embd → 4*n_embd → n_embd."""

    def __init__(
        self,
        n_embd: int,
        use_bias: bool,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        # Unsharded (kernel_axes=(None, None)) — GPT-2 124M is small enough
        # that we don't need tensor parallelism across the MLP.
        self.c_fc = LinearBase(
            input_size=n_embd,
            output_size=4 * n_embd,
            use_bias=use_bias,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.c_proj = LinearBase(
            input_size=4 * n_embd,
            output_size=n_embd,
            use_bias=use_bias,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.c_fc(x)
        x = jax.nn.gelu(x)
        x, _ = self.c_proj(x)
        return x


class NanoGPTAttention(nnx.Module):
    """Causal self-attention with fused QKV projection and paged KV cache.

    Uses a single c_attn LinearBase (n_embd → 3*n_embd) matching GPT-2's
    Conv1D layout, then splits into Q/K/V before dispatching to RadixAttention.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        use_bias: bool,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        # Fused QKV — mirrors GPT-2's single c_attn Conv1D weight
        self.c_attn = LinearBase(
            input_size=n_embd,
            output_size=3 * n_embd,
            use_bias=use_bias,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.c_proj = LinearBase(
            input_size=n_embd,
            output_size=n_embd,
            use_bias=use_bias,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.attn = RadixAttention(
            num_heads=n_head,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=n_head,  # MHA: kv_heads == q_heads
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, Any]:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape flat token dim: (-1, n_head, head_dim)
        num_tokens = hidden_states.shape[0]
        q = q.reshape(num_tokens, self.n_head, self.head_dim)
        k = k.reshape(num_tokens, self.n_head, self.head_dim)
        v = v.reshape(num_tokens, self.n_head, self.head_dim)

        # GPT-2 has no rotary embeddings — pass q/k straight to RadixAttention
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )

        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class NanoGPTBlock(nnx.Module):
    """Transformer block: pre-norm LayerNorm → attention → pre-norm LayerNorm → MLP."""

    def __init__(
        self,
        config: NanoGPTConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        self.ln_1 = NanoGPTLayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = NanoGPTAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            use_bias=config.bias,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.ln_2 = NanoGPTLayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = NanoGPTMLP(
            n_embd=config.n_embd,
            use_bias=config.bias,
            mesh=mesh,
            dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, Any]:
        # Pre-norm, then residual (no fused residual — GPT-2 is small)
        attn_out, kv_fused = self.attn(
            positions, self.ln_1(hidden_states), forward_batch, token_to_kv_pool
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, kv_fused


class NanoGPTModel(nnx.Module):
    """Core GPT-2 model: token + position embeddings → transformer blocks → LN."""

    def __init__(
        self,
        config: NanoGPTConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config

        # Token embedding — also used (tied) as the LM head
        self.embed_tokens = Embed(
            config.vocab_size,
            config.n_embd,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=(None, None),
            mesh=mesh,
        )

        # Learned absolute position embedding (GPT-2 style)
        self.wpe = nnx.Param(
            jnp.zeros((config.block_size, config.n_embd), dtype=dtype)
        )

        self.layers = [
            NanoGPTBlock(config=config, mesh=mesh, layer_id=i, dtype=dtype)
            for i in range(config.n_layer)
        ]

        self.ln_f = NanoGPTLayerNorm(config.n_embd, use_bias=config.bias)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list]:
        # Token + position embeddings
        tok_emb = self.embed_tokens(forward_batch.input_ids)
        pos_emb = self.wpe.value[forward_batch.positions]
        hidden_states = tok_emb + pos_emb

        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
            )
            layers_kv_fused.append(kv_fused)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, layers_kv_fused


class NanoGPTForCausalLM(nnx.Module):
    """NanoGPT (GPT-2) causal LM — SGLang-JAX serving model.

    Implements the serving interface expected by ModelRunner:
      - load_weights(model_config): loads GPT-2 safetensors weights
      - __call__(forward_batch, memory_pools, logits_metadata): serving forward pass

    Weight names in HuggingFace GPT-2 safetensors:
      transformer.wte.weight        → model.embed_tokens.embedding
      transformer.wpe.weight        → model.wpe
      transformer.h.{i}.ln_1.weight → model.layers.{i}.ln_1.scale
      transformer.h.{i}.ln_1.bias   → model.layers.{i}.ln_1.bias
      transformer.h.{i}.attn.c_attn.{weight,bias}  → model.layers.{i}.attn.c_attn.{weight,bias}
      transformer.h.{i}.attn.c_proj.{weight,bias}  → model.layers.{i}.attn.c_proj.{weight,bias}
      transformer.h.{i}.ln_2.*      → model.layers.{i}.ln_2.*
      transformer.h.{i}.mlp.c_fc.*  → model.layers.{i}.mlp.c_fc.*
      transformer.h.{i}.mlp.c_proj.*→ model.layers.{i}.mlp.c_proj.*
      transformer.ln_f.*            → model.ln_f.*
      lm_head.weight                → (tied to embed_tokens.embedding)

    GPT-2 uses PyTorch Conv1D which stores weights as (in, out), matching
    LinearBase's storage format — so transpose=False for all linear weights.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("NanoGPTForCausalLM config dtype: %s", self.dtype)

        self.model = NanoGPTModel(config=config, mesh=mesh, dtype=dtype)

        # ParallelLMHead weight is tied to the token embedding via tie_weights()
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.n_embd,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=(None, None),
            mesh=mesh,
        )
        self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("NanoGPT weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        # GPT-2 Conv1D stores (in, out) — same as LinearBase — so transpose=False.
        mappings = {
            "transformer.wte.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=(None, None),
                transpose=False,
            ),
            "transformer.wpe.weight": WeightMapping(
                target_path="model.wpe",
                sharding=(None, None),
                transpose=False,
            ),
            "transformer.ln_f.weight": WeightMapping(
                target_path="model.ln_f.scale",
                sharding=(None,),
                transpose=False,
            ),
            "transformer.ln_f.bias": WeightMapping(
                target_path="model.ln_f.bias",
                sharding=(None,),
                transpose=False,
            ),
        }

        for i in range(self.config.n_layer):
            prefix = f"transformer.h.{i}"
            tgt = f"model.layers.{i}"
            layer_mappings = {
                f"{prefix}.ln_1.weight": WeightMapping(
                    target_path=f"{tgt}.ln_1.scale", sharding=(None,), transpose=False
                ),
                f"{prefix}.ln_1.bias": WeightMapping(
                    target_path=f"{tgt}.ln_1.bias", sharding=(None,), transpose=False
                ),
                f"{prefix}.attn.c_attn.weight": WeightMapping(
                    target_path=f"{tgt}.attn.c_attn.weight",
                    sharding=(None, None),
                    transpose=False,
                ),
                f"{prefix}.attn.c_attn.bias": WeightMapping(
                    target_path=f"{tgt}.attn.c_attn.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.attn.c_proj.weight": WeightMapping(
                    target_path=f"{tgt}.attn.c_proj.weight",
                    sharding=(None, None),
                    transpose=False,
                ),
                f"{prefix}.attn.c_proj.bias": WeightMapping(
                    target_path=f"{tgt}.attn.c_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.ln_2.weight": WeightMapping(
                    target_path=f"{tgt}.ln_2.scale", sharding=(None,), transpose=False
                ),
                f"{prefix}.ln_2.bias": WeightMapping(
                    target_path=f"{tgt}.ln_2.bias", sharding=(None,), transpose=False
                ),
                f"{prefix}.mlp.c_fc.weight": WeightMapping(
                    target_path=f"{tgt}.mlp.c_fc.weight",
                    sharding=(None, None),
                    transpose=False,
                ),
                f"{prefix}.mlp.c_fc.bias": WeightMapping(
                    target_path=f"{tgt}.mlp.c_fc.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.mlp.c_proj.weight": WeightMapping(
                    target_path=f"{tgt}.mlp.c_proj.weight",
                    sharding=(None, None),
                    transpose=False,
                ),
                f"{prefix}.mlp.c_proj.bias": WeightMapping(
                    target_path=f"{tgt}.mlp.c_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            mappings.update(layer_mappings)

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused = self.model(
            forward_batch=forward_batch, token_to_kv_pool=kv_pool
        )

        # Weight-tied logits: lm_head.embedding == embed_tokens.embedding
        output = self.logits_processor(
            hidden_states, self.lm_head, logits_metadata
        )

        return output, {"token_to_kv_pool": layers_kv_fused}, [], None


EntryClass = NanoGPTForCausalLM
