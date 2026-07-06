# NanoGPT NNX 移植说明

**源代码**：`~/transformer/nanogpt-tpu-nnx/`（Flax NNX，使用标准 NNX 层）
**目标代码**：本仓库 `examples/nanogpt/`（Flax NNX，遵循 SGLang-JAX 约定）

两者均已是 Flax NNX 实现——这并非 Linen→NNX 的移植。差异在于 NNX 的使用方式：源代码依赖标准
NNX 层（`nnx.Linear`、`nnx.LayerNorm`、`nnx.Dropout`）以及 `rngs` 构造模式；目标代码则以自定义
的纯参数层替换之，以匹配 SGLang-JAX 服务栈的约定，并支持在不需要 `rngs` 对象的情况下直接赋值
HuggingFace 权重。

---

## 第一部分 — model.py

### 1. 模块构造：移除 `rngs`

**源代码** (`nanogpt-tpu-nnx/model.py`)：
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

**目标代码** (`examples/nanogpt/model.py`)：
```python
class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig):   # 无 rngs
        ...
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd, ...)

class Block(nnx.Module):
    def __init__(self, config: GPTConfig):   # 无 rngs
        self.ln_1 = LayerNorm(config.n_embd, ...)
        ...

class GPT(nnx.Module):
    def __init__(self, config: GPTConfig):   # 无 rngs
        ...
        self.blocks = nnx.List([Block(cfg) for _ in range(cfg.n_layer)])
```

**原因**：标准 `nnx.Linear` 和 `nnx.LayerNorm` 在构造时需要 `rngs: nnx.Rngs` 参数（用于参数初始化）。
目标代码使用自定义的 `Linear` 和 `LayerNorm` 类，通过固定的 `jax.random.PRNGKey` 初始化参数，
无需 `rngs` 对象。这将实例化简化为 `model = GPT(cfg)`，也使直接赋值 HF 权重更加方便。

---

### 2. `nnx.Linear` → 自定义 `Linear` 类；权重属性 `kernel` → `weight`

**源代码**：
```python
self.c_attn = nnx.Linear(
    C, 3 * C,
    use_bias=config.bias,
    kernel_init=nnx.initializers.normal(0.02),
    rngs=rngs,
)
# 参数属性：self.c_attn.kernel（标准 NNX 命名）
```

**目标代码**：
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
# 参数属性：self.c_attn.weight（从 kernel 改名）
```

**原因**：自定义 `Linear` 避免了 `rngs` 依赖，并将权重命名为 `weight` 而非 `kernel`。两者均以
`(in_features, out_features)` 布局存储权重——与 GPT-2 Conv1D 布局一致——因此加载 HF 权重时无需转置。
属性名从 `kernel` 改为 `weight` 的变化在所有权重加载代码中同步体现。

---

### 3. `nnx.LayerNorm` → 自定义 `LayerNorm` 类

**源代码**：
```python
self.ln_1 = nnx.LayerNorm(config.n_embd, use_bias=config.bias, rngs=rngs)
# 参数：.scale, .bias（标准 NNX 属性名）
```

**目标代码**：
```python
class LayerNorm(nnx.Module):
    def __init__(self, num_features, use_bias=True, epsilon=1e-5):
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias  = nnx.Param(jnp.zeros((num_features,))) if use_bias else None

    def __call__(self, x):
        orig_dtype = x.dtype
        x      = x.astype(jnp.float32)
        mean   = jnp.mean(x, axis=-1, keepdims=True)
        var    = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        out    = self.scale.value * x_norm
        if self.bias is not None:
            out = out + self.bias.value
        return out.astype(orig_dtype)

self.ln_1 = LayerNorm(config.n_embd, use_bias=config.bias)
# 参数：.scale, .bias（与 nnx.LayerNorm 相同的属性名）
```

**原因**：`nnx.LayerNorm` 需要 `rngs`。自定义类保留了相同的属性名（`.scale`、`.bias`），因此 HF
权重赋值代码无需改动。实现手动计算均值和方差，并在计算时转为 `float32` 以保证数值稳定性，最后
转回原始 dtype。

---

### 4. `nnx.Dropout` → 手动 Bernoulli Dropout；`training: bool` → `rng: Optional[jax.Array]`

**源代码**：
```python
# 构造时创建 Dropout 模块
self.attn_drop  = nnx.Dropout(config.dropout, rngs=rngs)
self.resid_drop = nnx.Dropout(config.dropout, rngs=rngs)
self.drop       = nnx.Dropout(config.dropout, rngs=rngs)  # GPT 中的嵌入 Dropout

# 调用签名
def __call__(self, x, training: bool = False):
    ...
    attn_w = self.attn_drop(attn_w, deterministic=not training)
    y      = self.resid_drop(y, deterministic=not training)
```

**目标代码**：
```python
# 不存储 Dropout 模块；只存 dropout_rate 浮点数
self.dropout_rate = cfg.dropout

# 调用签名
def __call__(self, x, rng: Optional[jax.Array] = None):
    ...
    if self.dropout_rate > 0.0 and rng is not None:
        rng, drop_rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - self.dropout_rate, attn_weights.shape)
        attn_weights = jnp.where(keep, attn_weights / (1.0 - self.dropout_rate), 0.0)
```

**原因**：`nnx.Dropout` 将自身的 RNG 状态存为 NNX 变量，增加了 `nnx.split/merge` 工作流的复杂度。
目标代码改用手动 Bernoulli 掩码。`training: bool` 标志替换为 `rng: Optional[jax.Array]`——传入
`rng=None` 即关闭 Dropout（推理模式），传入 key 则启用（训练模式）。

---

### 5. Block 列表重命名：`self.h` → `self.blocks`

**源代码**：
```python
self.h = nnx.List([Block(config, rngs) for _ in range(config.n_layer)])

for block in self.h:
    x = block(x, training=training)
```

**目标代码**：
```python
self.blocks = nnx.List([Block(cfg) for _ in range(cfg.n_layer)])

for block in self.blocks:
    x = block(x, rng=block_rng)
```

**原因**：`blocks` 比 `h` 更具描述性。两者均使用 `nnx.List`——在较新版本的 Flax 中，普通 Python
`list` 会因被视为包含数据的静态属性而抛出 `ValueError`。

---

### 6. 因果掩码填充值：`-jnp.inf` → `jnp.finfo(...).min`

**源代码**：
```python
attn_w = jnp.where(causal_mask, attn_w, -jnp.inf)
```

**目标代码**：
```python
attn_weights = jnp.where(causal_mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
```

**原因**：`-jnp.inf` 是 Python 的 `float64` 常量。使用 `jnp.finfo(dtype).min` 可生成实际计算
dtype 的最小有限值（如 `float32` 对应 `-3.4e38`），避免在注意力行全被掩码时，`softmax` 计算出
`nan`。

---

### 7. 新增 `num_params()` 方法；`estimate_mfu` 参数计数简化

**源代码**——无 `num_params()` 方法；`estimate_mfu` 中直接过滤计数：
```python
n_params = sum(
    v.size for v in jax.tree_util.tree_leaves(nnx.state(self, nnx.Param))
)
```

**目标代码**——新增 `num_params()`，计数所有 state 叶节点（不过滤）：
```python
def num_params(self) -> int:
    state = nnx.state(self)
    return sum(v.size for v in jax.tree_util.tree_leaves(state))
```

不过滤是因为自定义模块只存储 `nnx.Param` 叶节点（没有 `nnx.BatchStat` 等其他变量类型），结果
完全相同。`estimate_mfu` 调用 `self.num_params()` 而非重复计数逻辑。

---

## 第二部分 — sample.py

### 8. 默认 `init_from`：`'resume'` → `'gpt2'`

**源代码**：
```python
init_from = 'resume'   # 默认：从检查点加载
```

**目标代码**：
```python
init_from = "gpt2"     # 默认：下载 GPT-2 预训练权重
```

**原因**：目标代码是独立的推理演示脚本，开箱即可运行，无需预先存在的检查点。

---

### 9. 模型构造：`GPT(cfg, rngs)` → `GPT(cfg)`

**源代码**：
```python
rngs  = nnx.Rngs(params=0, dropout=42)
model = GPT(cfg, rngs)
```

**目标代码**：
```python
model = GPT(cfg)   # 无 rngs
```

直接源于第 1 条中移除 `rngs` 的变更。

---

### 10. HF 权重加载：`kernel[...] =` → `weight.value =`

**源代码**（使用 `nnx.Linear`，属性名为 `kernel`；用 `[...]` 切片赋值）：
```python
block.attn.c_attn.kernel[...] = jnp.array(pt[f'h.{i}.attn.c_attn.weight'])
block.attn.c_proj.kernel[...] = jnp.array(pt[f'h.{i}.attn.c_proj.weight'])
block.mlp.c_fc.kernel[...]    = jnp.array(pt[f'h.{i}.mlp.c_fc.weight'])
block.mlp.c_proj.kernel[...]  = jnp.array(pt[f'h.{i}.mlp.c_proj.weight'])
```

**目标代码**（使用自定义 `Linear`，属性名为 `weight`；用 `.value` 赋值）：
```python
block.attn.c_attn.weight.value = jnp.array(pt[f"h.{i}.attn.c_attn.weight"])
block.attn.c_proj.weight.value = jnp.array(pt[f"h.{i}.attn.c_proj.weight"])
block.mlp.c_fc.weight.value    = jnp.array(pt[f"h.{i}.mlp.c_fc.weight"])
block.mlp.c_proj.weight.value  = jnp.array(pt[f"h.{i}.mlp.c_proj.weight"])
```

两处变化：属性名（`kernel` → `weight`）和赋值方式（`param[...] = v` vs `param.value = v`）。
两者在 Flax NNX 中均可设置底层数组；源代码统一使用 `[...]` 切片赋值，目标代码统一使用 `.value`。
HF safetensors 的键名（如 `h.0.attn.c_attn.weight`）在两者中均不变。

---

### 11. Block 迭代：`model.h` → `model.blocks`

**源代码**——HF 权重加载路径（全程使用 `[...]` 切片赋值）：
```python
for i, block in enumerate(model.h):
    block.attn.c_attn.kernel[...] = jnp.array(pt[f'h.{i}.attn.c_attn.weight'])
    block.ln_1.scale[...]         = jnp.array(pt[f'h.{i}.ln_1.weight'])
    ...
```

**目标代码**（全程使用 `.value` 赋值）：
```python
for i, block in enumerate(model.blocks):
    block.attn.c_attn.weight.value = jnp.array(pt[f"h.{i}.attn.c_attn.weight"])
    block.ln_1.scale.value         = jnp.array(pt[f"h.{i}.ln_1.weight"])
    ...
```

两处变化：block 列表名（`model.h` → `model.blocks`）和赋值方式（`[...]` → `.value`）。
HF safetensors 键名（`h.{i}.*`）和 LayerNorm 属性名（`.scale`、`.bias`）在两者中均不变。

---

### 12. 检查点格式：Linen 兼容字典 → 原生 NNX `State` pytree

`nanogpt-tpu-nnx` 和 `sglang-jax` 各自都有 `train.py`，但使用不同的检查点格式，
因此 `sample.py` 中需要不同的加载器。

**源代码 `train.py`**（`nanogpt-tpu-nnx/train.py`）——以 Linen 兼容格式保存：
```python
def _get_linen_params(model) -> dict:
    """将 NNX 模型权重提取为 Linen 兼容的嵌套字典。"""
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

虽然模型是 NNX 格式，`train.py` 在保存时刻意将权重转换为 Linen 风格的嵌套字典
（`h_0`、`h_1`...、`kernel`，按层嵌套），以便与 `nanogpt-tpu/sample.py`（原始 Linen
推理脚本）保持兼容。

**源代码 `sample.py`**（`nanogpt-tpu-nnx/sample.py`）——通过 `_assign_block` 辅助函数加载：
```python
def load_model_from_checkpoint(out_dir):
    outer = serialization.msgpack_restore(raw)
    p     = outer['state']['params']   # {'h_0': {'attn': {'c_attn': {'kernel': ...}}}, ...}

    model.wte[...] = jnp.array(p['wte'])
    for i, block in enumerate(model.h):
        _assign_block(block, p[f'h_{i}'], has_bias)
        # 解包：p['attn']['c_attn']['kernel'] → block.attn.c_attn.kernel[...]
```

**目标代码 `train.py`**（`sglang-jax/examples/nanogpt/train.py`）——保存原生 NNX `State`：
```python
def save_checkpoint(state, opt_state, iter_num, best_val_loss):
    state0 = jax.tree_util.tree_map(lambda x: x[0], state)   # 取 device-0 切片
    ckpt   = {"state": state0, "opt_state": opt0, "iter_num": iter_num, ...}
    serialization.to_bytes(ckpt)
```

此处 `state` 是 `nnx.split(model)` 产生的原始 `nnx.State` pytree，其结构直接对应
模型的 Python 属性路径：`blocks[0].attn.c_attn.weight` 等。

**目标代码 `sample.py`**（`sglang-jax/examples/nanogpt/sample.py`）——原生 NNX 加载器：
```python
def load_nnx_checkpoint(model, out_dir):
    _, state = nnx.split(model)
    outer    = serialization.msgpack_restore(f.read())
    restored = serialization.from_state_dict(state, outer["state"])
    nnx.update(model, restored)
```

`serialization.from_state_dict` 可直接还原，因为检查点的 `state` 键已持有与
`nnx.split` 产生的相同 pytree 结构——无需 `_assign_block` 辅助函数。

**格式不兼容的原因**：源代码使用 Linen 兼容字典格式（`h_0`、`kernel`）以实现跨项目
兼容性；目标代码使用原始 NNX `State` pytree（`blocks[0]`、`weight`），更简洁但与源
格式不兼容。

---

### 13. 生成方式：`@nnx.jit + jax.lax.scan` → `@jax.jit + nnx.split/merge + Python 循环`

**源代码**——通过 `jax.lax.scan` 将整个 token 循环编译为单个 XLA 程序：
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
                # top-k 过滤 + 类别采样
                win = jnp.concatenate([win[:, 1:], next_tok[:, None]], axis=1)
                return (win, key), next_tok[0]

            _, tokens = jax.lax.scan(step, (window, rng_key), None, length=max_new_tokens)
            return tokens

        _gen_cache[cache_key] = _gen

    return _gen_cache[cache_key](model, window, rng_key)
```

`@nnx.jit` 在 JIT 前自动提取模型状态，内部自动合并。`jax.lax.scan` 将全部 `max_new_tokens` 步
编译为单个 XLA 程序，步间零 Python 开销。

**目标代码**——每步单独 JIT dispatch，外层 Python 循环：
```python
graphdef, state = nnx.split(model)

@jax.jit
def generate_step(state, window, rng_key):
    m = nnx.merge(graphdef, state)
    logits, _ = m(window)           # (1, 1, vocab_size)
    logits = logits[0, 0, :] / temperature
    # top-k 过滤 + 类别采样
    return jax.random.categorical(rng_key, logits)

for step_i in range(max_new_tokens):
    rng, step_rng = random.split(rng)
    next_tok = int(generate_step(state, window, step_rng))
    window = jnp.concatenate([window[:, 1:], jnp.array([[next_tok]])], axis=1)
```

使用 `@jax.jit`（非 `@nnx.jit`），在循环前显式调用 `nnx.split` 一次，在 JIT'd 函数内调用
`nnx.merge`。每步是独立的 JIT dispatch。

**权衡对比**：

| | 源代码（`@nnx.jit + lax.scan`） | 目标代码（`@jax.jit + Python 循环`） |
|---|---|---|
| 编译 | 全部 N 步一个 XLA 程序 | 每步一个程序（形状不变则命中缓存） |
| Python 开销 | 步间零开销 | 每步一次 Python 调用 |
| 吞吐量 | `max_new_tokens` 大时更高 | 较低，但代码更简单 |
| 可检视性 | 无法中途检查 token | 每步可直接查看生成 token |
| `_gen_cache` | 必须（函数定义在 `generate()` 内部） | 不需要（模块级 `@jax.jit` 自动缓存） |

---

### 14. `_gen_cache` → 移除

**源代码**：
```python
_gen_cache: dict = {}
cache_key = (id(model), top_k_val, float(temperature), max_new_tokens, vocab_size)
if cache_key not in _gen_cache:
    @nnx.jit
    def _gen(model, window, rng_key): ...
    _gen_cache[cache_key] = _gen
```

`_gen` 定义在 `generate()` 函数内部，每次调用都会创建新的 Python 函数对象。若无缓存，`@nnx.jit`
在函数对象变化时会触发重新编译。模块级字典以 `(model_id, 采样参数)` 为键缓存编译结果。

**目标代码**：`generate_step` 是模块级函数，以 `@jax.jit` 定义一次。JAX 内置编译缓存（以函数对象
标识 + 参数抽象值为键）自动处理去重，无需手动缓存。

---

### 15. Prompt 处理与输出收集

**源代码**：
```python
x = jnp.array(start_ids, dtype=jnp.int32)[None, :]   # (1, T)
# 左填充在 generate() 内部处理
y    = generate(model, x, max_new_tokens, gen_rng, temperature=temperature, top_k=top_k)
text = decode(y[0].tolist())   # y 包含 prompt + 生成 token
print(text)
```

**目标代码**：
```python
prompt = jnp.array(start_ids, dtype=jnp.int32)[None, :]
# 循环前先左填充到 block_size
T        = prompt.shape[1]
pad_tok  = int(prompt[0, 0])
pad      = jnp.full((1, cfg.block_size - T), pad_tok, dtype=jnp.int32)
init_window = jnp.concatenate([pad, prompt], axis=1)

generated = list(start_ids)   # Python 列表：prompt + 生成 token

for step_i in range(max_new_tokens):
    next_tok = int(generate_step(state, window, step_rng))
    generated.append(next_tok)
    window = jnp.concatenate([window[:, 1:], jnp.array([[next_tok]])], axis=1)

print(decode(generated))
```

源代码在 `generate()` 内部做左填充，返回包含 prompt 的完整序列 JAX 数组。目标代码在循环前填充，
用普通 Python 列表累积 token，JAX `window` 数组仅作为滑动上下文窗口单独维护。Python 列表避免了
在输出侧重复进行 JAX 数组拼接的开销。

---

## 附录：完整模型张量列表（GPT-2 124M）

所有参数张量均为 `float32`。源代码与目标代码的张量形状完全相同，区别仅在属性路径：
线性层权重在目标代码中为 `weight`，在源代码中为 `kernel`；block 列表在目标代码中为
`blocks[i]`，在源代码中为 `h[i]`。

默认训练使用 `bias=False`。"入检查点？"列以 `bias=False` 检查点为准；若 `bias=True`
则偏置张量存在。

**词嵌入与位置嵌入**

| 张量 — 目标路径 | 源路径 | 形状 | 参数量 |
|---|---|---|---|
| `wte`（词元嵌入） | `wte` | `(50304, 768)` | 38,633,472 |
| `wpe`（位置嵌入） | `wpe` | `(1024, 768)` | 786,432 |

**逐 Block × 12** — 目标 `blocks[i].*` / 源 `h[i].*`

| 张量 — 目标 | 源 | 形状 | 参数量 | 入检查点？（`bias=False`） |
|---|---|---|---|---|
| `ln_1.scale` | `ln_1.scale` | `(768,)` | 768 | 是 |
| `ln_1.bias` | `ln_1.bias` | `(768,)` | 768 | 否 |
| `attn.c_attn.weight` | `attn.c_attn.kernel` | `(768, 2304)` | 1,769,472 | 是 |
| `attn.c_attn.bias` | `attn.c_attn.bias` | `(2304,)` | 2,304 | 否 |
| `attn.c_proj.weight` | `attn.c_proj.kernel` | `(768, 768)` | 589,824 | 是 |
| `attn.c_proj.bias` | `attn.c_proj.bias` | `(768,)` | 768 | 否 |
| `ln_2.scale` | `ln_2.scale` | `(768,)` | 768 | 是 |
| `ln_2.bias` | `ln_2.bias` | `(768,)` | 768 | 否 |
| `mlp.c_fc.weight` | `mlp.c_fc.kernel` | `(768, 3072)` | 2,359,296 | 是 |
| `mlp.c_fc.bias` | `mlp.c_fc.bias` | `(3072,)` | 3,072 | 否 |
| `mlp.c_proj.weight` | `mlp.c_proj.kernel` | `(3072, 768)` | 2,359,296 | 是 |
| `mlp.c_proj.bias` | `mlp.c_proj.bias` | `(768,)` | 768 | 否 |
| **单 Block 小计（bias=True）** | | | **7,087,872** | |
| **单 Block 小计（bias=False）** | | | **7,079,424** | |
| **× 12 blocks（bias=True）** | | | **85,054,464** | |
| **× 12 blocks（bias=False）** | | | **84,953,088** | |

**最终 LayerNorm**

| 张量 — 目标 | 源 | 形状 | 参数量 | 入检查点？（`bias=False`） |
|---|---|---|---|---|
| `ln_f.scale` | `ln_f.scale` | `(768,)` | 768 | 是 |
| `ln_f.bias` | `ln_f.bias` | `(768,)` | 768 | 否 |

**总计**

| 模块 | 参数量（bias=True） | 参数量（bias=False） | 内存（float32，bias=False） |
|---|---|---|---|
| 嵌入层（`wte` + `wpe`） | 39,419,904 | 39,419,904 | 150.37 MiB |
| 12 × blocks | 85,054,464 | 84,953,088 | 324.08 MiB |
| `ln_f` | 1,536 | 768 | 3 KiB |
| **合计** | **124,475,904** | **124,373,760** | **474.45 MiB** |

> `wte` 通过权重共享同时作为输出投影（`logits = x @ wte.value.T`），两个版本均无独立的
> `lm_head` 张量。
>
> 验证：`124,373,760 × 4 字节 = 497,495,040 字节 ≈ 474.45 MiB`。`train.py` 保存的检查点
> 文件约大 3.7 KiB：额外字节为 msgpack 对元数据（`iter_num`、`best_val_loss`、
> `model_args`、`config`）的封装开销。

---

## 总结对比表

| 方面 | `nanogpt-tpu-nnx`（源代码） | `sglang-jax/examples/nanogpt`（目标代码） |
|---|---|---|
| 线性层 | `nnx.Linear`，属性名 `kernel` | 自定义 `Linear`，属性名 `weight` |
| 归一化层 | `nnx.LayerNorm` | 自定义 `LayerNorm`（手动均值+方差） |
| Dropout | `nnx.Dropout`，`training: bool` 标志 | 手动 Bernoulli，`rng: Optional[Array]` |
| 模块构造 | `__init__(config, rngs: nnx.Rngs)` | `__init__(config)` — 无 rngs |
| Block 列表名 | `self.h` | `self.blocks` |
| 因果掩码填充值 | `-jnp.inf` | `jnp.finfo(dtype).min` |
| 默认 `init_from` | `'resume'` | `'gpt2'` |
| HF 权重属性名 | `.c_attn.kernel`（用 `kernel[...] = v`） | `.c_attn.weight`（用 `weight.value = v`） |
| 检查点恢复 | 自有 `train.py` 通过 `_get_linen_params()` 保存 Linen 兼容格式（键：`h_0`、`kernel`）；由 `_assign_block()` 加载 | 自有 `train.py` 保存原生 NNX `State` pytree（路径：`blocks[0]`、`weight`）；由 `serialization.from_state_dict` 加载 |
| 生成方式 | `@nnx.jit + jax.lax.scan`（单 XLA 程序） | `@jax.jit + nnx.split/merge + Python 循环` |
| 编译缓存 | 手动 `_gen_cache` 字典 | JAX 内置（模块级 `@jax.jit`） |
