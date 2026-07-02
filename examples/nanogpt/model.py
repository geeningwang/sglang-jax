"""
NanoGPT training model — Flax NNX port of ~/transformer/nanogpt-tpu/model.py.

Architecture is identical to nanogpt (GPT-2):
  - Causal self-attention with manual causal mask
  - MLP with GELU
  - Pre-norm transformer blocks
  - Weight tying: token embedding == lm_head projection

Key differences from the Flax Linen nanogpt-tpu version:
  - NNX modules (stateful) instead of Linen (functional)
  - Parameters stored as nnx.Param inside modules
  - nnx.split / nnx.merge used by train.py for pmap compatibility
  - nnx.Linear instead of LinearBase (no tensor parallelism needed for training)
  - dropout=0.0 default; if non-zero, pass rng key explicitly to __call__
"""

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size 50257 padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # bias in LayerNorm and Linear; False is slightly better


class LayerNorm(nnx.Module):
    """Standard LayerNorm with optional bias, implemented with nnx.Param.

    GPT-2 uses standard LayerNorm (mean + variance), not RMSNorm.
    We implement it manually to avoid dependency on nnx.LayerNorm's rngs requirement.
    """

    def __init__(
        self,
        num_features: int,
        use_bias: bool = True,
        epsilon: float = 1e-5,
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


class Linear(nnx.Module):
    """Dense linear layer, implemented with nnx.Param for pmap compatibility.

    Stores weight in (in, out) layout — same as LinearBase in the serving model.
    Avoids nnx.Linear's rngs requirement at construction time.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        std: float = 0.02,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.weight = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), (in_features, out_features), dtype=dtype)
            * std
        )
        self.bias: nnx.Param | None = (
            nnx.Param(jnp.zeros((out_features,), dtype=dtype)) if use_bias else None
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        out = x @ self.weight.value
        if self.bias is not None:
            out = out + self.bias.value
        return out


class CausalSelfAttention(nnx.Module):
    """Causal multi-head self-attention with fused QKV projection.

    Mirrors nanogpt-tpu/model.py CausalSelfAttention exactly:
      - Fused QKV via single Linear (n_embd → 3*n_embd)
      - Manual causal mask with jnp.tril
      - jnp.einsum for attention (compiled by XLA)
      - Output projection with scaled init (0.02 / sqrt(2 * n_layer))
    """

    def __init__(self, config: GPTConfig):
        cfg = config
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head
        self.dropout_rate = cfg.dropout

        # Fused QKV projection — std=0.02 per GPT-2 paper
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd, use_bias=cfg.bias, std=0.02)

        # Output projection — scaled init per GPT-2 paper
        proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.c_proj = Linear(cfg.n_embd, cfg.n_embd, use_bias=cfg.bias, std=proj_std)

    def __call__(
        self,
        x: jax.Array,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        B, T, C = x.shape
        head_size = self.head_size

        # Fused QKV
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape to (B, n_head, T, head_size)
        q = q.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(head_size)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn_weights = jnp.where(causal_mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Optional attention dropout
        if self.dropout_rate > 0.0 and rng is not None:
            rng, drop_rng = jax.random.split(rng)
            keep = jax.random.bernoulli(drop_rng, 1.0 - self.dropout_rate, attn_weights.shape)
            attn_weights = jnp.where(keep, attn_weights / (1.0 - self.dropout_rate), 0.0)

        y = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        y = self.c_proj(y)

        # Optional residual dropout
        if self.dropout_rate > 0.0 and rng is not None:
            keep = jax.random.bernoulli(rng, 1.0 - self.dropout_rate, y.shape)
            y = jnp.where(keep, y / (1.0 - self.dropout_rate), 0.0)

        return y


class MLP(nnx.Module):
    """Two-layer MLP: n_embd → 4*n_embd → n_embd with GELU activation."""

    def __init__(self, config: GPTConfig):
        cfg = config
        proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.c_fc = Linear(cfg.n_embd, 4 * cfg.n_embd, use_bias=cfg.bias, std=0.02)
        self.c_proj = Linear(4 * cfg.n_embd, cfg.n_embd, use_bias=cfg.bias, std=proj_std)
        self.dropout_rate = cfg.dropout

    def __call__(
        self,
        x: jax.Array,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        if self.dropout_rate > 0.0 and rng is not None:
            keep = jax.random.bernoulli(rng, 1.0 - self.dropout_rate, x.shape)
            x = jnp.where(keep, x / (1.0 - self.dropout_rate), 0.0)
        return x


class Block(nnx.Module):
    """Pre-norm transformer block: LN + Attention + LN + MLP."""

    def __init__(self, config: GPTConfig):
        self.ln_1 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = MLP(config)

    def __call__(
        self,
        x: jax.Array,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        attn_rng, mlp_rng = (jax.random.split(rng) if rng is not None else (None, None))
        x = x + self.attn(self.ln_1(x), rng=attn_rng)
        x = x + self.mlp(self.ln_2(x), rng=mlp_rng)
        return x


class GPT(nnx.Module):
    """GPT language model (nanogpt / GPT-2 architecture) in Flax NNX.

    Token embedding and position embedding stored as nnx.Param.
    Weight tying (wte == lm_head) is implemented by sharing the same
    nnx.Param object; the output logit projection uses wte.value.T.

    Usage in training via nnx.split / nnx.merge:
        graphdef, state = nnx.split(model)

        @functools.partial(jax.pmap, axis_name='devices')
        def train_step(state, ...):
            model = nnx.merge(graphdef, state)  # graphdef closed over
            ...
    """

    def __init__(self, config: GPTConfig):
        cfg = config
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None

        self.config = config

        # Token embedding — also used for output projection (weight tying)
        self.wte = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(1), (cfg.vocab_size, cfg.n_embd)) * 0.02
        )
        # Position embedding
        self.wpe = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(2), (cfg.block_size, cfg.n_embd)) * 0.02
        )

        self.blocks = nnx.List([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, use_bias=cfg.bias)
        self.dropout_rate = cfg.dropout

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        rng: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        """Forward pass.

        Args:
            idx:     Integer token indices, shape (B, T).
            targets: Target token indices for loss, shape (B, T). None for inference.
            rng:     Optional PRNGKey for dropout. Ignored when dropout=0.

        Returns:
            (logits, loss) where loss is None when targets is None.
        """
        cfg = self.config
        B, T = idx.shape
        assert T <= cfg.block_size, f"Sequence length {T} > block_size {cfg.block_size}"

        tok_emb = self.wte.value[idx]              # (B, T, n_embd)
        pos_emb = self.wpe.value[jnp.arange(T)]   # (T, n_embd)

        x = tok_emb + pos_emb

        # Embedding dropout
        if self.dropout_rate > 0.0 and rng is not None:
            rng, drop_rng = jax.random.split(rng)
            keep = jax.random.bernoulli(drop_rng, 1.0 - self.dropout_rate, x.shape)
            x = jnp.where(keep, x / (1.0 - self.dropout_rate), 0.0)

        for block in self.blocks:
            block_rng = None
            if self.dropout_rate > 0.0 and rng is not None:
                rng, block_rng = jax.random.split(rng)
            x = block(x, rng=block_rng)

        x = self.ln_f(x)

        if targets is not None:
            # Training: compute logits for all positions
            logits = x @ self.wte.value.T   # (B, T, vocab_size) — weight tying
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    logits.reshape(-1, cfg.vocab_size),
                    targets.reshape(-1),
                )
            )
        else:
            # Inference: only the last position
            logits = x[:, [-1], :] @ self.wte.value.T  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def num_params(self) -> int:
        """Count total trainable parameters."""
        state = nnx.state(self)
        return sum(v.size for v in jax.tree_util.tree_leaves(state))

    def estimate_mfu(
        self,
        fwdbwd_per_iter: float,
        dt: float,
        tpu_peak_tflops: float = 918.0,
    ) -> float:
        """Estimate model FLOPs utilization as a fraction of TPU peak FLOPS.

        tpu_peak_tflops: 918 TFLOPS for TPU v6e (Trillium), 2307 for v7x (Ironwood).
        Call with tpu_peak_tflops * num_devices to account for multi-chip setups.
        """
        cfg = self.config
        n = self.num_params() - cfg.block_size * cfg.n_embd  # exclude wpe
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * n + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_achieved = flops_per_fwdbwd * fwdbwd_per_iter / dt
        return flops_achieved / (tpu_peak_tflops * 1e12)
