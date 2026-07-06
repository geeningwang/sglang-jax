# NanoGPT NNX Porting Notes

**Source**: `~/transformer/nanogpt-tpu-nnx/` (Flax NNX, uses standard NNX layers)
**Destination**: `examples/nanogpt/` in this repo (Flax NNX, SGLang-JAX conventions)

Both are already Flax NNX implementations — this is not a Linen→NNX port. The differences are
about how NNX is used: the source relies on standard NNX layers (`nnx.Linear`, `nnx.LayerNorm`,
`nnx.Dropout`) and the `rngs` construction pattern; the destination replaces those with custom
parameter-only layers to match SGLang-JAX's serving-stack conventions and to support direct HF
weight assignment without needing a `rngs` object.

---

## Part 1 — model.py

### 1. Module construction: `rngs` removed

**Source** (`nanogpt-tpu-nnx/model.py`):
```python
class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        ...
        self.c_attn = nnx.Linear(C, 3 * C, ..., rngs=rngs)

class Block(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, ..., rngs=rngs)
        ...

class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        ...
        self.h = nnx.List([Block(config, rngs) for _ in range(config.n_layer)])
```

**Destination** (`examples/nanogpt/model.py`):
```python
class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig):   # no rngs
        ...
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd, ...)

class Block(nnx.Module):
    def __init__(self, config: GPTConfig):   # no rngs
        self.ln_1 = LayerNorm(config.n_embd, ...)
        ...

class GPT(nnx.Module):
    def __init__(self, config: GPTConfig):   # no rngs
        ...
        self.blocks = nnx.List([Block(cfg) for _ in range(cfg.n_layer)])
```

**Why**: Standard `nnx.Linear` and `nnx.LayerNorm` require an `rngs: nnx.Rngs` argument at
construction time (for parameter initialization). The destination uses custom `Linear` and
`LayerNorm` classes that initialize parameters from a fixed `jax.random.PRNGKey`, so no `rngs`
object is needed. This simplifies instantiation to `model = GPT(cfg)` and makes direct HF weight
assignment straightforward.

---

### 2. `nnx.Linear` → custom `Linear` class; weight attribute `kernel` → `weight`

**Source**:
```python
self.c_attn = nnx.Linear(
    C, 3 * C,
    use_bias=config.bias,
    kernel_init=nnx.initializers.normal(0.02),
    rngs=rngs,
)
# parameter attribute: self.c_attn.kernel  (standard NNX name)
```

**Destination**:
```python
class Linear(nnx.Module):
    def __init__(self, in_features, out_features, use_bias=True, std=0.02, dtype=jnp.float32):
        self.weight = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), (in_features, out_features)) * std
        )
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None

    def __call__(self, x):
        out = x @ self.weight.value
        if self.bias is not None:
            out = out + self.bias.value
        return out

self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd, use_bias=cfg.bias, std=0.02)
# parameter attribute: self.c_attn.weight  (renamed from kernel)
```

**Why**: The custom `Linear` avoids the `rngs` dependency and names the weight `weight` instead of
`kernel`. Both store weights in `(in_features, out_features)` layout — matching GPT-2's Conv1D
layout — so no transposition is needed when loading HF weights. The name change from `kernel` to
`weight` is reflected everywhere in the weight-loading code.

---

### 3. `nnx.LayerNorm` → custom `LayerNorm` class

**Source**:
```python
self.ln_1 = nnx.LayerNorm(config.n_embd, use_bias=config.bias, rngs=rngs)
# parameters: .scale, .bias  (standard NNX attribute names)
```

**Destination**:
```python
class LayerNorm(nnx.Module):
    def __init__(self, num_features, use_bias=True, epsilon=1e-5):
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias  = nnx.Param(jnp.zeros((num_features,))) if use_bias else None

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        mean   = jnp.mean(x, axis=-1, keepdims=True)
        var    = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        out    = self.scale.value * x_norm
        if self.bias is not None:
            out = out + self.bias.value
        return out.astype(orig_dtype)

self.ln_1 = LayerNorm(config.n_embd, use_bias=config.bias)
# parameters: .scale, .bias  (same attribute names as nnx.LayerNorm)
```

**Why**: `nnx.LayerNorm` requires `rngs`. The custom class has identical attribute names (`.scale`,
`.bias`) so HF weight assignment code is unchanged. The implementation manually computes mean and
variance, and casts to `float32` for numerical stability before casting back.

---

### 4. `nnx.Dropout` → manual Bernoulli dropout; `training: bool` → `rng: Optional[jax.Array]`

**Source**:
```python
# Construction
self.attn_drop  = nnx.Dropout(config.dropout, rngs=rngs)
self.resid_drop = nnx.Dropout(config.dropout, rngs=rngs)
self.drop       = nnx.Dropout(config.dropout, rngs=rngs)  # embedding dropout in GPT

# Call signature
def __call__(self, x, training: bool = False):
    ...
    attn_w = self.attn_drop(attn_w, deterministic=not training)
    y      = self.resid_drop(y, deterministic=not training)
```

**Destination**:
```python
# No Dropout module stored; dropout_rate stored as plain float
self.dropout_rate = cfg.dropout

# Call signature
def __call__(self, x, rng: Optional[jax.Array] = None):
    ...
    if self.dropout_rate > 0.0 and rng is not None:
        rng, drop_rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - self.dropout_rate, attn_weights.shape)
        attn_weights = jnp.where(keep, attn_weights / (1.0 - self.dropout_rate), 0.0)
```

**Why**: `nnx.Dropout` stores its own RNG state as an NNX variable, adding complexity for
`nnx.split/merge` workflows. The destination avoids this by using manual Bernoulli masking. The
`training: bool` flag is replaced by `rng: Optional[jax.Array]` — passing `rng=None` disables
dropout (inference mode) and passing a key enables it (training mode).

---

### 5. Block list renamed: `self.h` → `self.blocks`

**Source**:
```python
self.h = nnx.List([Block(config, rngs) for _ in range(config.n_layer)])

for block in self.h:
    x = block(x, training=training)
```

**Destination**:
```python
self.blocks = nnx.List([Block(cfg) for _ in range(cfg.n_layer)])

for block in self.blocks:
    x = block(x, rng=block_rng)
```

**Why**: `blocks` is more descriptive than `h`. Both use `nnx.List` — a plain Python `list` raises
a `ValueError` in newer Flax versions because Flax sees it as a static attribute containing data.

---

### 6. Causal mask fill value: `-jnp.inf` → `jnp.finfo(...).min`

**Source**:
```python
attn_w = jnp.where(causal_mask, attn_w, -jnp.inf)
```

**Destination**:
```python
attn_weights = jnp.where(causal_mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
```

**Why**: `-jnp.inf` is a Python `float64` constant. Using `jnp.finfo(dtype).min` produces the
minimum finite value for the actual computation dtype (e.g. `-3.4e38` for `float32`), which avoids
`nan` propagation through softmax when an entire attention row is masked.

---

### 7. `num_params()` method added; `estimate_mfu` param counting simplified

**Source** — no `num_params()` method; `estimate_mfu` filters to `nnx.Param`:
```python
n_params = sum(
    v.size for v in jax.tree_util.tree_leaves(nnx.state(self, nnx.Param))
)
```

**Destination** — `num_params()` added; counts all state leaves (no filter):
```python
def num_params(self) -> int:
    state = nnx.state(self)
    return sum(v.size for v in jax.tree_util.tree_leaves(state))
```

The filter is omitted because the custom modules only store `nnx.Param` leaves (no `nnx.BatchStat`
or other variable types), so the result is identical. `estimate_mfu` calls `self.num_params()`
instead of repeating the counting logic.

---

## Part 2 — sample.py

### 8. Default `init_from`: `'resume'` → `'gpt2'`

**Source**:
```python
init_from = 'resume'   # default: load from checkpoint
```

**Destination**:
```python
init_from = "gpt2"     # default: download GPT-2 pretrained weights
```

**Why**: The destination is a standalone inference demo. Defaulting to `'gpt2'` makes it runnable
out of the box without a pre-existing checkpoint.

---

### 9. Model construction: `GPT(cfg, rngs)` → `GPT(cfg)`

**Source**:
```python
rngs  = nnx.Rngs(params=0, dropout=42)
model = GPT(cfg, rngs)
```

**Destination**:
```python
model = GPT(cfg)   # no rngs
```

Direct consequence of the `rngs` removal in §1.

---

### 10. HF weight loading: `.kernel.value` → `.weight.value`

**Source** (uses `nnx.Linear`, attribute name is `kernel`; assigns via `[...]` slice syntax):
```python
block.attn.c_attn.kernel[...] = jnp.array(pt[f'h.{i}.attn.c_attn.weight'])
block.attn.c_proj.kernel[...] = jnp.array(pt[f'h.{i}.attn.c_proj.weight'])
block.mlp.c_fc.kernel[...]    = jnp.array(pt[f'h.{i}.mlp.c_fc.weight'])
block.mlp.c_proj.kernel[...]  = jnp.array(pt[f'h.{i}.mlp.c_proj.weight'])
```

**Destination** (uses custom `Linear`, attribute name is `weight`; assigns via `.value`):
```python
block.attn.c_attn.weight.value = jnp.array(pt[f"h.{i}.attn.c_attn.weight"])
block.attn.c_proj.weight.value = jnp.array(pt[f"h.{i}.attn.c_proj.weight"])
block.mlp.c_fc.weight.value    = jnp.array(pt[f"h.{i}.mlp.c_fc.weight"])
block.mlp.c_proj.weight.value  = jnp.array(pt[f"h.{i}.mlp.c_proj.weight"])
```

Two differences: the attribute name (`kernel` → `weight`) and the assignment style
(`param[...] = v` vs `param.value = v`). Both are valid NNX patterns that set the underlying
array. The source consistently uses `[...]` slice assignment; the destination uses `.value`.
The HF safetensors key names (e.g. `h.0.attn.c_attn.weight`) are unchanged in both.

---

### 11. Block iteration: `model.h` → `model.blocks`

**Source** — HF weight loading path (uses `[...]` slice assignment throughout):
```python
for i, block in enumerate(model.h):
    block.attn.c_attn.kernel[...] = jnp.array(pt[f'h.{i}.attn.c_attn.weight'])
    block.ln_1.scale[...]         = jnp.array(pt[f'h.{i}.ln_1.weight'])
    ...
```

**Destination** (uses `.value` assignment throughout):
```python
for i, block in enumerate(model.blocks):
    block.attn.c_attn.weight.value = jnp.array(pt[f"h.{i}.attn.c_attn.weight"])
    block.ln_1.scale.value         = jnp.array(pt[f"h.{i}.ln_1.weight"])
    ...
```

Two changes from source: block list name (`model.h` → `model.blocks`) and assignment style
(`[...]` → `.value`). The HF safetensors key names (`h.{i}.*`) and the LayerNorm attribute
names (`.scale`, `.bias`) are unchanged in both.

---

### 12. Checkpoint format: linen-compatible dict → native NNX `State` pytree

Both `nanogpt-tpu-nnx` and `sglang-jax` have their own `train.py`, but they use different
checkpoint formats, which forces different loaders in `sample.py`.

**Source `train.py`** (`nanogpt-tpu-nnx/train.py`) — saves in linen-compatible format:
```python
def _get_linen_params(model) -> dict:
    """Extract NNX model weights as a linen-compatible nested dict."""
    p = {
        'wte': np.array(model.wte[...]),
        'wpe': np.array(model.wpe[...]),
        'ln_f': {'scale': np.array(model.ln_f.scale[...])},
    }
    for i, block in enumerate(model.h):
        p[f'h_{i}'] = {
            'attn': {'c_attn': {'kernel': np.array(block.attn.c_attn.kernel[...])}, ...},
            ...
        }
    return p

def save_checkpoint(param_state_sharded, iter_num, best_val_loss):
    params = _get_linen_params(m)
    ckpt = {'state': {'params': params}, 'iter_num': iter_num, ...}
    serialization.to_bytes(ckpt)
```

Even though the model is NNX, `train.py` deliberately converts weights to a linen-style nested
dict before saving (`h_0`, `h_1`, ..., `kernel`, nested by layer). This preserves compatibility
with `nanogpt-tpu/sample.py` (the original Linen inference script).

**Source `sample.py`** (`nanogpt-tpu-nnx/sample.py`) — loads via `_assign_block` helper:
```python
def load_model_from_checkpoint(out_dir):
    outer = serialization.msgpack_restore(raw)
    p     = outer['state']['params']   # {'h_0': {'attn': {'c_attn': {'kernel': ...}}}, ...}

    model.wte[...] = jnp.array(p['wte'])
    for i, block in enumerate(model.h):
        _assign_block(block, p[f'h_{i}'], has_bias)
        # unpacks: p['attn']['c_attn']['kernel'] → block.attn.c_attn.kernel[...]
```

**Destination `train.py`** (`sglang-jax/examples/nanogpt/train.py`) — saves native NNX `State`:
```python
def save_checkpoint(state, opt_state, iter_num, best_val_loss):
    state0 = jax.tree_util.tree_map(lambda x: x[0], state)   # device-0 slice
    ckpt   = {"state": state0, "opt_state": opt0, "iter_num": iter_num, ...}
    serialization.to_bytes(ckpt)
```

`state` here is the raw `nnx.State` pytree from `nnx.split(model)`. Its structure mirrors the
model's Python attributes directly: `blocks[0].attn.c_attn.weight`, etc.

**Destination `sample.py`** (`sglang-jax/examples/nanogpt/sample.py`) — native NNX loader:
```python
def load_nnx_checkpoint(model, out_dir):
    _, state = nnx.split(model)
    outer    = serialization.msgpack_restore(f.read())
    restored = serialization.from_state_dict(state, outer["state"])
    nnx.update(model, restored)
```

`serialization.from_state_dict` can restore directly because the checkpoint `state` key already
holds the same pytree shape that `nnx.split` produces — no `_assign_block` helper needed.

**Why the formats differ**: The source uses a linen-compatible dict format (`h_0`, `kernel`) for
cross-project compatibility. The destination uses the raw NNX `State` pytree (`blocks[0]`, `weight`)
which is simpler but incompatible with the source's format.

---

### 13. Generation: `@nnx.jit + jax.lax.scan` → `@jax.jit + nnx.split/merge + Python loop`

**Source** — entire token loop compiled into one XLA program via `jax.lax.scan`:
```python
_gen_cache: dict = {}

def generate(model, idx, max_new_tokens, rng_key, temperature=1.0, top_k=None):
    cache_key = (id(model), top_k_val, float(temperature), max_new_tokens, vocab_size)
    if cache_key not in _gen_cache:
        @nnx.jit
        def _gen(model, window, rng_key):
            def step(carry, _):
                win, key  = carry
                logits, _ = model(win, training=False)   # (1, 1, vocab_size)
                logits    = logits[:, 0, :] / temperature
                # top-k filter + categorical sample
                win = jnp.concatenate([win[:, 1:], next_tok[:, None]], axis=1)
                return (win, key), next_tok[0]

            _, tokens = jax.lax.scan(step, (window, rng_key), None, length=max_new_tokens)
            return tokens

        _gen_cache[cache_key] = _gen

    return _gen_cache[cache_key](model, window, rng_key)
```

`@nnx.jit` automatically extracts model state before JIT and merges it back inside.
`jax.lax.scan` compiles all `max_new_tokens` steps into a single XLA program with zero Python
re-entry between steps.

**Destination** — one step per JIT dispatch, Python loop:
```python
graphdef, state = nnx.split(model)

@jax.jit
def generate_step(state, window, rng_key):
    m = nnx.merge(graphdef, state)
    logits, _ = m(window)           # (1, 1, vocab_size)
    logits = logits[0, 0, :] / temperature
    # top-k filter + categorical sample
    return jax.random.categorical(rng_key, logits)

for step_i in range(max_new_tokens):
    rng, step_rng = random.split(rng)
    next_tok = int(generate_step(state, window, step_rng))
    window = jnp.concatenate([window[:, 1:], jnp.array([[next_tok]])], axis=1)
```

Uses `@jax.jit` (not `@nnx.jit`), explicitly calling `nnx.split` once before the loop and
`nnx.merge` inside the JIT'd step. Each of the `max_new_tokens` steps is a separate JIT dispatch.

**Trade-offs**:

| | Source (`@nnx.jit + lax.scan`) | Destination (`@jax.jit + Python loop`) |
|---|---|---|
| Compilation | One XLA program for all N steps | One program per step (shape-cached) |
| Python overhead | Zero between steps | One Python call per step |
| Throughput | Higher for large `max_new_tokens` | Lower, but simpler |
| Inspectability | Cannot inspect mid-generation | Easy to inspect each token |
| `_gen_cache` | Needed (function defined inside `generate()`) | Not needed (module-level `@jax.jit`) |

---

### 14. `_gen_cache` → removed

**Source**:
```python
_gen_cache: dict = {}
cache_key = (id(model), top_k_val, float(temperature), max_new_tokens, vocab_size)
if cache_key not in _gen_cache:
    @nnx.jit
    def _gen(model, window, rng_key): ...
    _gen_cache[cache_key] = _gen
```

`_gen` is defined inside the `generate()` function, so a new Python object is created on every
call. Without the cache, `@nnx.jit` would recompile on each invocation because function identity
changes.

**Destination**: `generate_step` is a module-level function defined once with `@jax.jit`. JAX's
own compilation cache (keyed on function identity + argument abstract values) handles deduplication
automatically. No manual cache needed.

---

### 15. Prompt handling and output collection

**Source**:
```python
x = jnp.array(start_ids, dtype=jnp.int32)[None, :]   # (1, T)
# left-pad handled inside generate()
y    = generate(model, x, max_new_tokens, gen_rng, temperature=temperature, top_k=top_k)
text = decode(y[0].tolist())   # y includes prompt + generated tokens
print(text)
```

**Destination**:
```python
prompt = jnp.array(start_ids, dtype=jnp.int32)[None, :]
# left-pad to block_size before loop
T = prompt.shape[1]
pad_tok      = int(prompt[0, 0])
pad          = jnp.full((1, cfg.block_size - T), pad_tok, dtype=jnp.int32)
init_window  = jnp.concatenate([pad, prompt], axis=1)

generated = list(start_ids)   # Python list: prompt + generated tokens

for step_i in range(max_new_tokens):
    next_tok = int(generate_step(state, window, step_rng))
    generated.append(next_tok)
    window = jnp.concatenate([window[:, 1:], jnp.array([[next_tok]])], axis=1)

print(decode(generated))
```

The source left-pads inside `generate()` and returns the full sequence as a JAX array. The
destination left-pads before the loop, accumulates tokens in a plain Python list, and slides the
window array separately. The Python list avoids repeated JAX array concatenations on the output
side.

---

## Summary Table

| Aspect | `nanogpt-tpu-nnx` (source) | `sglang-jax/examples/nanogpt` (destination) |
|---|---|---|
| Linear layer | `nnx.Linear`, attribute `kernel` | Custom `Linear`, attribute `weight` |
| LayerNorm | `nnx.LayerNorm` | Custom `LayerNorm` (manual mean+var) |
| Dropout | `nnx.Dropout`, `training: bool` flag | Manual Bernoulli, `rng: Optional[Array]` |
| Module construction | `__init__(config, rngs: nnx.Rngs)` | `__init__(config)` — no rngs |
| Block list name | `self.h` | `self.blocks` |
| Causal mask fill | `-jnp.inf` | `jnp.finfo(dtype).min` |
| Default `init_from` | `'resume'` | `'gpt2'` |
| HF weight attr name | `.c_attn.kernel` (via `kernel[...] = v`) | `.c_attn.weight` (via `weight.value = v`) |
| Checkpoint resume | Own `train.py` saves linen-compatible format via `_get_linen_params()` (`h_0`, `kernel` keys); loaded via `_assign_block()` | Own `train.py` saves native NNX `State` pytree (`blocks[0]`, `weight` keys); loaded via `serialization.from_state_dict` |
| Generation | `@nnx.jit + jax.lax.scan` (one XLA program) | `@jax.jit + nnx.split/merge + Python loop` |
| Compile cache | Manual `_gen_cache` dict | JAX built-in (module-level `@jax.jit`) |
