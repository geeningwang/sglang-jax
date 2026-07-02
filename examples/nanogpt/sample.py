"""
sample.py — text generation demo for the SGLang-JAX NNX nanogpt port.

Mirrors nanogpt-tpu/sample.py in interface and behaviour, ported to
the Flax NNX model in model.py.

Supported weight sources (init_from):
  'gpt2'        — OpenAI GPT-2 small   (124M)  via HuggingFace
  'gpt2-medium' — OpenAI GPT-2 medium  (355M)
  'gpt2-large'  — OpenAI GPT-2 large   (774M)
  'gpt2-xl'     — OpenAI GPT-2 XL      (1.5B)
  'resume'      — load from out_dir/ckpt.msgpack  (NNX checkpoint)

Usage:
  python sample.py
  python sample.py --init_from=gpt2
  python sample.py --start="The meaning of life is" --num_samples=5
  python sample.py --start="FILE:prompt.txt"
  python sample.py --init_from=gpt2-medium --max_new_tokens=200 --temperature=0.9
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx, serialization

import tiktoken

from model import GPT, GPTConfig

# ── Config ────────────────────────────────────────────────────────────────────
init_from = "gpt2"      # 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl' | 'resume'
out_dir = "out"
start = "\n"            # prompt string, or "FILE:<path>"
num_samples = 3
max_new_tokens = 500
temperature = 0.8       # < 1.0 = sharper, > 1.0 = softer
top_k = 200             # keep only top-k logits; set to None to disable
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())  # noqa: S102
# -----------------------------------------------------------------------------

_GPT2_CONFIGS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
}


# ── Weight loading ────────────────────────────────────────────────────────────

def load_hf_weights(model: GPT, model_type: str) -> None:
    """Load OpenAI GPT-2 weights from HuggingFace safetensors into the NNX model.

    GPT-2 Conv1D stores weights as (in_features, out_features) — the same
    layout as our custom Linear layer — so no transposition is needed.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    print(f"Downloading {model_type} weights from HuggingFace ...")
    weights_path = hf_hub_download(model_type, "model.safetensors")
    pt = load_file(weights_path)  # pure numpy, no JAX tracing

    cfg = model.config

    # Token embedding — pad vocab from 50257 → 50304 for XLA alignment
    wte_np = pt["wte.weight"]  # (50257, n_embd)
    if wte_np.shape[0] < cfg.vocab_size:
        pad = np.zeros((cfg.vocab_size - wte_np.shape[0], cfg.n_embd), dtype=wte_np.dtype)
        wte_np = np.concatenate([wte_np, pad], axis=0)
    model.wte.value = jnp.array(wte_np)
    model.wpe.value = jnp.array(pt["wpe.weight"])

    # Final LayerNorm
    model.ln_f.scale.value = jnp.array(pt["ln_f.weight"])
    model.ln_f.bias.value  = jnp.array(pt["ln_f.bias"])

    # Transformer blocks
    for i, block in enumerate(model.blocks):
        block.ln_1.scale.value = jnp.array(pt[f"h.{i}.ln_1.weight"])
        block.ln_1.bias.value  = jnp.array(pt[f"h.{i}.ln_1.bias"])

        block.attn.c_attn.weight.value = jnp.array(pt[f"h.{i}.attn.c_attn.weight"])
        block.attn.c_attn.bias.value   = jnp.array(pt[f"h.{i}.attn.c_attn.bias"])
        block.attn.c_proj.weight.value = jnp.array(pt[f"h.{i}.attn.c_proj.weight"])
        block.attn.c_proj.bias.value   = jnp.array(pt[f"h.{i}.attn.c_proj.bias"])

        block.ln_2.scale.value = jnp.array(pt[f"h.{i}.ln_2.weight"])
        block.ln_2.bias.value  = jnp.array(pt[f"h.{i}.ln_2.bias"])

        block.mlp.c_fc.weight.value   = jnp.array(pt[f"h.{i}.mlp.c_fc.weight"])
        block.mlp.c_fc.bias.value     = jnp.array(pt[f"h.{i}.mlp.c_fc.bias"])
        block.mlp.c_proj.weight.value = jnp.array(pt[f"h.{i}.mlp.c_proj.weight"])
        block.mlp.c_proj.bias.value   = jnp.array(pt[f"h.{i}.mlp.c_proj.bias"])

    print(f"Loaded {model_type}: {model.num_params() / 1e6:.1f}M params")


def load_nnx_checkpoint(model: GPT, out_dir: str) -> None:
    """Resume from a checkpoint saved by train.py (NNX msgpack format)."""
    path = os.path.join(out_dir, "ckpt.msgpack")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No NNX checkpoint at {path}")
    _, state = nnx.split(model)
    with open(path, "rb") as f:
        outer = serialization.msgpack_restore(f.read())
    restored = serialization.from_state_dict(state, outer["state"])
    nnx.update(model, restored)
    print(f"Resumed from {path} (iter {int(outer['iter_num'])}, "
          f"val loss {float(outer['best_val_loss']):.4f})")


# ── Build model ───────────────────────────────────────────────────────────────
print(f"JAX devices: {jax.devices()}")

if init_from == "resume":
    path = os.path.join(out_dir, "ckpt.msgpack")
    with open(path, "rb") as f:
        outer = serialization.msgpack_restore(f.read())
    raw_args = outer["model_args"]
    # msgpack decodes bytes keys; normalise
    model_args = {(k.decode() if isinstance(k, bytes) else k): v
                  for k, v in raw_args.items()}
    cfg = GPTConfig(**model_args)
elif init_from in _GPT2_CONFIGS:
    cfg = GPTConfig(block_size=1024, bias=True, vocab_size=50304, dropout=0.0,
                    **_GPT2_CONFIGS[init_from])
else:
    raise ValueError(f"Unknown init_from: {init_from!r}")

model = GPT(cfg)

if init_from == "resume":
    load_nnx_checkpoint(model, out_dir)
else:
    load_hf_weights(model, init_from)

print(f"Model: {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}D  "
      f"block_size={cfg.block_size}  vocab_size={cfg.vocab_size}")

# ── JIT generation step ───────────────────────────────────────────────────────
# graphdef is static (closed over); state is the JAX pytree passed as argument.
graphdef, state = nnx.split(model)

_REAL_VOCAB = 50257   # GPT-2 BPE vocabulary; padded vocab_size is larger
_TOP_K = min(top_k, _REAL_VOCAB) if top_k is not None else _REAL_VOCAB


@jax.jit
def generate_step(state, window, rng_key):
    """One autoregressive step: sample the next token given a context window.

    Args:
        state:    nnx.State pytree (model params).
        window:   int32[1, block_size] — sliding context window.
        rng_key:  PRNGKey for categorical sampling.

    Returns:
        next_tok: scalar int32 — sampled next token id.
    """
    m = nnx.merge(graphdef, state)
    logits, _ = m(window)               # (1, 1, vocab_size) — last position only
    logits = logits[0, 0, :] / temperature

    # Mask tokens beyond the real GPT-2 vocab (padding indices)
    pad_mask = jnp.arange(cfg.vocab_size) >= _REAL_VOCAB
    logits = jnp.where(pad_mask, -jnp.inf, logits)

    # Top-k: zero out logits below the k-th largest
    kth_val = jnp.sort(logits)[-_TOP_K]
    logits = jnp.where(logits < kth_val, -jnp.inf, logits)

    return jax.random.categorical(rng_key, logits)


# ── Tokeniser ─────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa: E731
decode = lambda t: enc.decode(t)                                       # noqa: E731

if start.startswith("FILE:"):
    with open(start[5:]) as f:
        start = f.read()

start_ids = encode(start)
prompt = jnp.array(start_ids, dtype=jnp.int32)[None, :]  # (1, T_prompt)

# Left-pad prompt to block_size with the first prompt token so the compiled
# window shape is always (1, block_size) — one JIT compilation for all steps.
T = prompt.shape[1]
if T < cfg.block_size:
    pad_tok = int(prompt[0, 0])
    pad = jnp.full((1, cfg.block_size - T), pad_tok, dtype=jnp.int32)
    init_window = jnp.concatenate([pad, prompt], axis=1)
else:
    init_window = prompt[:, -cfg.block_size:]

# ── Generate ──────────────────────────────────────────────────────────────────
rng = random.PRNGKey(seed)

print(f"\nGenerating {num_samples} sample(s) — {max_new_tokens} new tokens each\n")
print("=" * 60)

for k in range(num_samples):
    window = init_window
    generated = list(start_ids)

    for step_i in range(max_new_tokens):
        rng, step_rng = random.split(rng)
        next_tok = int(generate_step(state, window, step_rng))
        generated.append(next_tok)
        # Slide context window by one position
        window = jnp.concatenate(
            [window[:, 1:], jnp.array([[next_tok]], dtype=jnp.int32)], axis=1
        )
        if k == 0 and step_i == 0:
            print("(first token compiled — subsequent steps run fast)")

    print(decode(generated))
    print("─" * 60)
