# nanogpt TPU → SGLang-JAX Port: Detailed Comparison

This document compares the original `nanogpt-tpu` implementation (Flax **Linen**) with
the SGLang-JAX port (Flax **NNX**), line by line.  Every meaningful difference is listed
with an explanation of *why* it changed.

Reference paths:
- Original: `~/transformer/nanogpt-tpu/{model,sample}.py`
- Port: `examples/nanogpt/{model,sample}.py`

---

## Part 1 — `model.py`

### 1.1 Imports

**Original (Linen)**
```python
import flax.linen as nn
```

**NNX port**
```python
from flax import nnx
```

**Why:** Linen (`flax.linen`) and NNX (`flax.nnx`) are two independent module
systems inside the Flax package.  Linen is the older, functional API; NNX is the
newer, stateful API introduced in Flax 0.8.  They cannot be mixed: a Linen
`nn.Module` cannot be a child of an NNX `nnx.Module` and vice versa.

---

### 1.2 `GPTConfig` — unchanged

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer:  int = 12
    n_head:   int = 12
    n_embd:   int = 768
    dropout:  float = 0.0
    bias:     bool = True
```

Identical in both versions.  `@dataclass` is a plain Python construct that both
frameworks accept as a config carrier.

---

### 1.3 Module class declaration

**Original**
```python
class CausalSelfAttention(nn.Module):
    config: GPTConfig          # class-level field annotation (Linen dataclass style)

    @nn.compact                # magic decorator: first call initialises params
    def __call__(self, x, training: bool = False):
        ...
```

**NNX port**
```python
class CausalSelfAttention(nnx.Module):
    # no class-level field annotation

    def __init__(self, config: GPTConfig):
        cfg = config
        ...                    # all sub-modules created here, stored as instance attrs
```

**Why:**
- **Linen** modules are *frozen dataclasses*.  Config fields are declared at the
  class level (`config: GPTConfig`).  Parameters and sub-modules are **not** stored
  as instance attributes; instead `@nn.compact` defers their creation to the first
  call and stores them inside an external pytree.  `self.param(...)` and `nn.Dense`
  calls inside `@nn.compact` register into that pytree.
- **NNX** modules are *normal Python objects*.  Config is a regular `__init__`
  argument.  Parameters and sub-modules are stored as instance attributes assigned
  in `__init__`.  There is no `@nn.compact`, no deferred initialisation.

---

### 1.4 LayerNorm

**Original**
```python
self.ln_1 = nn.LayerNorm(use_bias=cfg.bias)
```
`nn.LayerNorm` is a Linen built-in that creates its own `scale` and `bias` params
inside the external pytree on first call.

**NNX port**
```python
class LayerNorm(nnx.Module):
    def __init__(self, num_features, use_bias=True, epsilon=1e-5, ...):
        self.scale = nnx.Param(jnp.ones((num_features,), dtype=param_dtype))
        self.bias: nnx.Param | None = (
            nnx.Param(jnp.zeros((num_features,), dtype=param_dtype)) if use_bias else None
        )

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.var(x,  axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        out = self.scale.value * x_norm
        if self.bias is not None:
            out = out + self.bias.value
        return out
```

**Why a custom class instead of `nnx.LayerNorm`?**  `nnx.LayerNorm` exists but
requires an `rngs` argument at construction time (for future stochastic extensions),
which adds boilerplate.  The custom class avoids that and makes the `scale`/`bias`
attributes explicit, which simplifies HuggingFace weight assignment in `sample.py`.

**Why `jax.lax.rsqrt` instead of `jnp.sqrt`?**  `lax.rsqrt` is a single XLA op
(reciprocal square root) that fuses into one kernel; `1/jnp.sqrt(...)` would be
two ops.

---

### 1.5 Linear layer

**Original** — uses `nn.Dense` directly inline:
```python
qkv = nn.Dense(3 * C, use_bias=cfg.bias,
               kernel_init=nn.initializers.normal(0.02),
               name='c_attn')(x)
```
`nn.Dense` is a Linen built-in.  The `name=` argument is mandatory inside
`@nn.compact` to get a stable key in the param pytree.

**NNX port** — custom `Linear` class:
```python
class Linear(nnx.Module):
    def __init__(self, in_features, out_features, use_bias=True, std=0.02, ...):
        self.weight = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), (in_features, out_features)) * std
        )
        self.bias: nnx.Param | None = (
            nnx.Param(jnp.zeros((out_features,), dtype=dtype)) if use_bias else None
        )

    def __call__(self, x):
        out = x @ self.weight.value
        if self.bias is not None:
            out = out + self.bias.value
        return out
```

**Why a custom class?**
1. `nnx.Linear` requires `rngs` at construction time (same reason as LayerNorm).
2. The custom class stores weight in `(in_features, out_features)` layout, which
   matches both GPT-2's Conv1D convention and `LinearBase` in the SGLang-JAX
   serving model — no transposition needed at weight-loading time.
3. Explicit `self.weight` and `self.bias` attributes make HuggingFace weight
   assignment in `sample.py` straightforward: `block.attn.c_attn.weight.value = ...`

**`nn.Dense` kernel layout** is `(in, out)` too, so the math is identical; the
difference is only in how params are stored and accessed.

---

### 1.6 Dropout

**Original**
```python
attn_weights = nn.Dropout(cfg.dropout)(attn_weights, deterministic=not training)
```
`nn.Dropout` is a Linen module.  `deterministic=True` makes it a no-op (inference
mode).  It uses an `rngs={'dropout': key}` passed to `model.apply(...)` at the
call site — the caller is responsible for wiring the key through.

**NNX port**
```python
if self.dropout_rate > 0.0 and rng is not None:
    rng, drop_rng = jax.random.split(rng)
    keep = jax.random.bernoulli(drop_rng, 1.0 - self.dropout_rate, attn_weights.shape)
    attn_weights = jnp.where(keep, attn_weights / (1.0 - self.dropout_rate), 0.0)
```

**Why manual instead of `nnx.Dropout`?**
- `nnx.Dropout` holds a stateful `nnx.Rngs` object.  Under `jax.pmap` each device
  needs its own `Rngs`, which requires threading per-device keys through the
  replicated state — complicated.
- The manual approach is a transparent `jnp.where`; JAX/XLA compiles it to a
  masked multiply (same as the Dropout kernel under the hood).
- The guard `if self.dropout_rate > 0.0 and rng is not None` means inference
  (no `rng` passed) is a strict no-op with **zero** overhead, not a branch compiled
  into XLA.

---

### 1.7 Block structure

**Original**
```python
class Block(nn.Module):
    config: GPTConfig

    def setup(self):               # Linen lifecycle hook: called once before first use
        cfg = self.config
        self.ln_1 = nn.LayerNorm(use_bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(use_bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def __call__(self, x, training: bool = False):
        x = x + self.attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x
```

`setup()` is a Linen lifecycle hook that runs before the first call to `__call__`.
It is the alternative to `@nn.compact` for modules that need explicit sub-module
names (e.g. for weight loading by name).

**NNX port**
```python
class Block(nnx.Module):
    def __init__(self, config: GPTConfig):
        self.ln_1 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp  = MLP(config)

    def __call__(self, x, rng=None):
        attn_rng, mlp_rng = (jax.random.split(rng) if rng is not None else (None, None))
        x = x + self.attn(self.ln_1(x), rng=attn_rng)
        x = x + self.mlp(self.ln_2(x),  rng=mlp_rng)
        return x
```

**Why no `setup()`?**  NNX has no lifecycle hooks; `__init__` is the only
constructor.  All sub-modules are assigned as plain attributes — Python's own
attribute protocol.

**`rng` threading:**  Instead of a `training: bool` flag that the caller combines
with an external key, NNX passes the key explicitly when dropout is needed.  `None`
means inference.

---

### 1.8 Top-level GPT module — parameter declaration

**Original** (inside `@nn.compact __call__`):
```python
wte = self.param('wte', nn.initializers.normal(0.02), (cfg.vocab_size, cfg.n_embd))
wpe = self.param('wpe', nn.initializers.normal(0.02), (cfg.block_size, cfg.n_embd))
```
`self.param(name, init_fn, shape)` registers a leaf in the external param pytree
under the key `name`.  The initialiser is called once on first `model.apply(...)`.

**NNX port** (in `__init__`):
```python
self.wte = nnx.Param(
    jax.random.normal(jax.random.PRNGKey(1), (cfg.vocab_size, cfg.n_embd)) * 0.02
)
self.wpe = nnx.Param(
    jax.random.normal(jax.random.PRNGKey(2), (cfg.block_size, cfg.n_embd)) * 0.02
)
```
`nnx.Param` is a thin wrapper around a JAX array.  It is created eagerly at
construction time (no deferred initialisation).  The key difference: **the tensor
lives inside the module**, not in an external pytree.

**PRNGKey seeding:** The original draws keys from the `rngs` argument passed to
`model.apply(...)` at the call site, so the seed is controlled externally.  The NNX
version uses hard-coded seeds (`PRNGKey(1)`, `PRNGKey(2)`) because the initial
values are immediately overwritten by HuggingFace weights in `sample.py` or by
`nnx.split` + `nnx.merge` in training — the exact initialisation doesn't matter.

---

### 1.9 Layer list

**Original**
```python
for i in range(cfg.n_layer):
    x = Block(cfg, name=f'h_{i}')(x, training=training)
```
Blocks are created inline inside `@nn.compact` on every call; Linen caches them by
`name`.  The `name=f'h_{i}'` is mandatory so each block's params get a stable key
(`h_0`, `h_1`, …) in the pytree.

**NNX port**
```python
self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
```
A plain Python list of NNX modules.  NNX's `nnx.split` traverses object attributes
recursively; list elements are included automatically.  No explicit names are
needed — NNX uses the integer index as the pytree key.

**`nnx.data` note:**  The SGLang-JAX codebase sometimes uses `nnx.data([...])` for
module lists, but that API does not exist in Flax 0.8.5 (the dev environment's
version).  A plain Python list works identically because `nnx.split` handles lists
natively.

---

### 1.10 Weight tying

**Original** (inside `@nn.compact`):
```python
wte = self.param('wte', ...)    # local variable, same object used twice below
...
logits = x @ wte.T              # wte reused for lm_head projection
```
`wte` is a local JAX array returned by `self.param`.  Using it twice is trivially
cheap — it is the same tensor referenced in two `jnp.matmul` calls.

**NNX port**:
```python
self.wte = nnx.Param(...)       # attribute on self
...
logits = x @ self.wte.value.T   # .value unwraps the nnx.Param wrapper
```
Same idea; `.value` is needed because `nnx.Param` is a wrapper class, not a bare
array.  Accessing `.value` returns the underlying JAX array.

---

### 1.11 `estimate_mfu` signature

**Original**
```python
def estimate_mfu(self, params, fwdbwd_per_iter, dt, tpu_peak_tflops=2307.0):
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
```
Requires the external `params` pytree to be passed in because the Linen model
does not own its parameters.

**NNX port**
```python
def estimate_mfu(self, fwdbwd_per_iter, dt, tpu_peak_tflops=918.0):
    ...
def num_params(self):
    state = nnx.state(self)
    return sum(v.size for v in jax.tree_util.tree_leaves(state))
```
Parameters live inside the module, so `nnx.state(self)` extracts them without
any external argument.  `num_params()` is factored out as a separate helper method.

**Default `tpu_peak_tflops` discrepancy:**  The original defaults to 2307 (v7x
Ironwood); the NNX port defaults to 918 (v6e Trillium) because the training script
overrides it with `--tpu_peak_tflops=2307.0` when running on v7x.  Both are
correct for their respective use cases.

---

## Part 2 — `sample.py`

### 2.1 Imports

**Original**
```python
from flax.training import train_state
from flax import traverse_util
import optax
from model import GPTConfig, GPT     # Linen GPT
```

**NNX port**
```python
from flax import nnx, serialization
from model import GPT, GPTConfig     # NNX GPT
```

- `train_state` and `traverse_util` are Linen-era utilities for managing the
  `(params, opt_state)` bundle.  NNX has no equivalent because the model owns its
  own state.
- `optax` is not imported in the NNX sample.py because generation does not need
  an optimiser.  (The original imports it to reconstruct the optimizer structure
  for checkpoint loading, since the Linen checkpoint embeds `TrainState` including
  the optimizer state.)

---

### 2.2 Config section

Both files have an identical config block and `exec(open('configurator.py').read())`
pattern.  No difference.

---

### 2.3 HuggingFace weight loading — approach

**Original** — builds a Linen param pytree from scratch:
```python
def load_params_from_gpt2(model_type):
    ...
    params = {
        'wte': np.concatenate([wte_np, pad], axis=0),
        'wpe': pt['wpe.weight'],
        'ln_f': {'scale': pt['ln_f.weight'], 'bias': pt['ln_f.bias']},
    }
    for i in range(n_layer):
        params[f'h_{i}'] = {
            'ln_1': {'scale': ..., 'bias': ...},
            'attn': {
                'c_attn': {'kernel': pt[f'h.{i}.attn.c_attn.weight'],
                           'bias':   pt[f'h.{i}.attn.c_attn.bias']},
                'c_proj': {'kernel': ..., 'bias': ...},
            },
            ...
        }
    return params, model_args
```
The output is a nested dict matching the Linen pytree layout exactly:
`params['h_0']['attn']['c_attn']['kernel']`.  Note that Linen `nn.Dense` stores
weights under the key `'kernel'`.

**NNX port** — assigns directly to `nnx.Param.value`:
```python
def load_hf_weights(model: GPT, model_type: str) -> None:
    ...
    model.wte.value = jnp.array(wte_np)
    model.wpe.value = jnp.array(pt['wpe.weight'])
    model.ln_f.scale.value = jnp.array(pt['ln_f.weight'])
    model.ln_f.bias.value  = jnp.array(pt['ln_f.bias'])
    for i, block in enumerate(model.blocks):
        block.ln_1.scale.value = jnp.array(pt[f'h.{i}.ln_1.weight'])
        block.attn.c_attn.weight.value = jnp.array(pt[f'h.{i}.attn.c_attn.weight'])
        ...
```

**Key differences:**
| Aspect | Original (Linen) | NNX port |
|---|---|---|
| Return value | New dict `params` | `None` (mutates model in-place) |
| Key for weight | `'kernel'` (Linen `nn.Dense`) | `'weight'` (custom `Linear`) |
| Key for LayerNorm scale | `'scale'` | `'scale'` (same) |
| Intermediate dict | Built manually | Not needed |
| Device transfer | Separate `jax.tree_util.tree_map(jnp.array, params)` step | Happens at each `jnp.array(...)` call |

**Why `'kernel'` vs `'weight'`?**  Linen's `nn.Dense` stores weights under the
pytree key `'kernel'` (following the Flax/Haiku convention).  The custom `Linear`
in the NNX model stores under the attribute name `weight`, which becomes the
pytree key `'weight'` after `nnx.split`.

**Why in-place mutation?**  NNX modules are mutable Python objects.  Assigning to
`nnx.Param.value` updates the tensor held by the module's parameter directly.  This
avoids creating a separate params dict and then merging it back — the model is ready
to use immediately after `load_hf_weights` returns.

---

### 2.4 Checkpoint loading

**Original** — reconstructs a full `TrainState` and deserialises into it:
```python
def load_params_from_checkpoint(out_dir):
    # Must reconstruct the exact same pytree structure the checkpoint was saved with.
    temp_cfg = GPTConfig(...)
    model = GPT(temp_cfg)
    dummy_idx = jnp.zeros((1, temp_cfg.block_size), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, temp_cfg.block_size), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_idx, dummy_tgt)['params']

    tx = make_optimizer()
    state = train_state.TrainState.create(apply_fn=model.apply, params=init_params, tx=tx)
    target = {'state': state, 'iter_num': 0, 'best_val_loss': 1e9, ...}

    restored = serialization.from_bytes(target, raw)
    return restored['state'].params, restored['model_args']
```

This is complex because Flax's `from_bytes` requires a *target* pytree with the
exact same structure as the checkpoint.  To build that target, the code must run a
dummy forward pass to initialise params, then wrap them in `TrainState`.

**NNX port** — uses `from_state_dict` + `nnx.update`:
```python
def load_nnx_checkpoint(model: GPT, out_dir: str) -> None:
    _, state = nnx.split(model)
    with open(path, 'rb') as f:
        outer = serialization.msgpack_restore(f.read())
    restored = serialization.from_state_dict(state, outer['state'])
    nnx.update(model, restored)
```

`nnx.split(model)` extracts the current state as the target structure.  No dummy
forward pass is needed — the model was already initialised by `GPT(cfg)`.
`nnx.update` writes the restored state back into the model's `nnx.Param` objects
in-place.

**Incompatibility note:**  The two checkpoint formats are **not interchangeable**.
The Linen checkpoint embeds a `TrainState` (including optimizer moments and the
`apply_fn`); the NNX checkpoint embeds a raw `nnx.State` pytree + optimizer state.
Loading an old Linen checkpoint into the NNX model (or vice versa) will fail.  This
is why the GKE job uses a separate GCS path (`gpt2-124m-nnx/` vs `gpt2-124m/`).

---

### 2.5 Model creation and initialisation

**Original**
```python
cfg = GPTConfig(**model_args)
model = GPT(cfg)
# Params are NOT inside model — they live in the separate dict returned by load_params_*
params = jax.tree_util.tree_map(jnp.array, params)   # host → TPU device transfer
```
`model` here is a stateless callable; it holds only the `config`.  All tensors are
in `params`.  The explicit `tree_map(jnp.array, params)` copies every weight from
numpy (returned by `safetensors.load_file`) to the JAX default device.

**NNX port**
```python
model = GPT(cfg)
load_hf_weights(model, init_from)   # assigns jnp.array(...) to each nnx.Param.value
```
`model` holds all tensors.  The device transfer happens inside `load_hf_weights`
at each `jnp.array(...)` call — there is no separate step.

---

### 2.6 Generation — split/merge pattern

**Original** — `model.apply` with external params:
```python
@jax.jit
def _gen(params, window, rng_key):
    def step(carry, _):
        win, key = carry
        logits, _ = model.apply({'params': params}, win, training=False)
        logits = logits[:, 0, :] / temperature   # (1, vocab_size)
        ...
    _, tokens = jax.lax.scan(step, (window, rng_key), None, length=max_new_tokens)
    return tokens
```
`model.apply({'params': params}, ...)` is the Linen call convention: params are
passed as a dict, not stored in the model.  The `{'params': params}` dict is the
"variable collection" Linen expects.

Uses `jax.lax.scan` to compile all `max_new_tokens` steps into a **single XLA
program** — one dispatch, no Python loop overhead, maximum TPU utilisation.

**NNX port** — `nnx.split` + `nnx.merge`:
```python
graphdef, state = nnx.split(model)

@jax.jit
def generate_step(state, window, rng_key):
    m = nnx.merge(graphdef, state)
    logits, _ = m(window)                # (1, 1, vocab_size)
    logits = logits[0, 0, :] / temperature
    ...
    return jax.random.categorical(rng_key, logits)

# Python loop at the outer level
for step_i in range(max_new_tokens):
    rng, step_rng = random.split(rng)
    next_tok = int(generate_step(state, window, step_rng))
    ...
```

**`nnx.split(model)` → `(graphdef, state)`:**
- `graphdef` is a static, hashable description of the module tree (structure, types,
  metadata).  It is closed over by the jitted function — JAX traces through it once
  and caches the compiled program.
- `state` is the pure JAX pytree of all parameter arrays.  It is passed as a traced
  argument to `generate_step`, so JAX can dispatch different weights without
  recompilation.

**`nnx.merge(graphdef, state)`** reconstructs a live NNX model object inside the
jitted function.  This is the NNX equivalent of the Linen
`model.apply({'params': params}, ...)` call.

**Python loop vs `lax.scan`:**  The NNX port uses a Python `for` loop instead of
`lax.scan`.  The tradeoff:
- `lax.scan` compiles all steps into one XLA program → maximum throughput, but
  requires a static `length` and fixed carry/output shapes.
- Python loop calls `generate_step` once per token, each call dispatches to the
  cached XLA program.  First call is slow (JIT compile); subsequent calls are fast.
  The "first token compiled" print at `step_i == 0` marks this.
- For a demo that generates a few hundred tokens, the Python loop is simpler and
  fast enough.  For batch production throughput, `lax.scan` is preferred.

---

### 2.7 `logits[:, 0, :]` vs `logits[0, 0, :]`

**Original**
```python
logits = logits[:, 0, :] / temperature   # shape: (batch=1, 1, vocab) → (1, vocab)
```

**NNX port**
```python
logits = logits[0, 0, :] / temperature   # shape: (vocab,)
```

Both start from `(1, 1, vocab_size)` (batch=1, one position in inference mode).
The original slices with `[:, 0, :]` keeping the batch dimension; the NNX port
uses `[0, 0, :]` to get a 1-D vector directly.  Both are correct for batch=1;
the NNX version is slightly cleaner since `jax.random.categorical` accepts 1-D
logits without requiring a squeeze.

---

### 2.8 Top-k filtering

**Original** (inside `jax.lax.scan` body, compiled into XLA):
```python
top_vals = jnp.sort(logits, axis=-1)[..., -top_k_val]
logits   = jnp.where(logits < top_vals[..., None], -jnp.inf, logits)
```
`top_k_val = min(top_k, real_vocab)` is a Python int, computed once before JIT.
`jnp.sort` sorts ascending; `[..., -top_k_val]` picks the `top_k_val`-th largest.

**NNX port** (inside `@jax.jit generate_step`):
```python
kth_val = jnp.sort(logits)[-_TOP_K]     # _TOP_K is a module-level Python int
logits = jnp.where(logits < kth_val, -jnp.inf, logits)
```
Identical logic; `_TOP_K` is computed once at import time as a module-level
constant so it's a static Python int at JIT time (not a traced value), keeping the
compiled graph shape stable across calls.

---

### 2.9 Vocabulary masking

Both files mask the padded vocab tokens (indices ≥ 50257) with `-jnp.inf` before
sampling, so the model never generates tokens outside the real GPT-2 BPE vocab.

**Original**
```python
pad_mask = jnp.arange(vocab_size) >= real_vocab
logits   = jnp.where(pad_mask, -jnp.inf, logits)
```

**NNX port** — identical logic, same variable names.  No difference.

---

### 2.10 Generation output

**Original** — uses `lax.scan` which returns the full token sequence in one shot:
```python
_, tokens = jax.lax.scan(step, (window, rng_key), None, length=max_new_tokens)
# tokens: int32[max_new_tokens]
return jnp.concatenate([idx[0], gen_tokens])[None, :]  # (1, T + max_new_tokens)
```

**NNX port** — accumulates into a Python list during the loop:
```python
generated = list(start_ids)
for step_i in range(max_new_tokens):
    ...
    next_tok = int(generate_step(...))
    generated.append(next_tok)
print(decode(generated))
```
`int(...)` materialises the scalar from device to Python.  This is a host–device
sync per token (slower for large batches), but for a demo it is perfectly fine and
removes the scan boilerplate.

---

## Summary table

| Concept | Original (Linen) | NNX port |
|---|---|---|
| Module base class | `nn.Module` (frozen dataclass) | `nnx.Module` (plain Python class) |
| Parameter storage | External pytree (passed to `apply`) | Inside module as `nnx.Param` |
| Module initialisation | `@nn.compact` / `setup()` (deferred) | `__init__` (eager) |
| Layer name registration | `name=` arg required | Not needed |
| `model(x)` call | `model.apply({'params': p}, x)` | `model(x)` directly |
| Dropout | `nn.Dropout(r)(x, deterministic=...)` | Manual `bernoulli` + `jnp.where` |
| LayerNorm | `nn.LayerNorm` built-in | Custom `LayerNorm(nnx.Module)` |
| Linear layer | `nn.Dense` built-in | Custom `Linear(nnx.Module)` |
| Weight key for linear | `'kernel'` | `'weight'` |
| Weight tying | `wte` local var used twice | `self.wte.value` used twice |
| `estimate_mfu` signature | Requires external `params` | No external args (`nnx.state(self)`) |
| pmap / jit interface | params as positional arg | `nnx.split` → `(graphdef, state)` |
| HF weight loading | Builds param dict, returns it | Assigns to `nnx.Param.value` in-place |
| Checkpoint loading | `from_bytes` with `TrainState` target | `from_state_dict` + `nnx.update` |
| Generation loop | `jax.lax.scan` (one XLA dispatch) | Python `for` loop (one JIT per token) |
| Checkpoint format | Linen `TrainState` (incompatible) | NNX `State` pytree |
