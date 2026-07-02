# nanogpt TPU → SGLang-JAX 移植详解

本文档逐行对比原始 `nanogpt-tpu` 实现（Flax **Linen**）与 SGLang-JAX 移植版（Flax **NNX**），并解释每处差异的原因。

参考路径：
- 原始版：`~/transformer/nanogpt-tpu/{model,sample}.py`
- 移植版：`examples/nanogpt/{model,sample}.py`

---

## 第一部分 — `model.py`

### 1.1 导入

**原始版（Linen）**
```python
import flax.linen as nn
```

**NNX 移植版**
```python
from flax import nnx
```

**原因：** Linen（`flax.linen`）和 NNX（`flax.nnx`）是 Flax 包内两套独立的模块系统。Linen 是较旧的函数式 API；NNX 是 Flax 0.8 引入的有状态 API。两者不能混用：Linen 的 `nn.Module` 不能作为 NNX `nnx.Module` 的子模块，反之亦然。

---

### 1.2 `GPTConfig` — 无变化

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

两个版本完全相同。`@dataclass` 是纯 Python 结构，两个框架都可以使用。

---

### 1.3 模块类声明

**原始版**
```python
class CausalSelfAttention(nn.Module):
    config: GPTConfig          # 类级别字段注解（Linen dataclass 风格）

    @nn.compact                # 魔法装饰器：首次调用时初始化参数
    def __call__(self, x, training: bool = False):
        ...
```

**NNX 移植版**
```python
class CausalSelfAttention(nnx.Module):
    # 无类级别字段注解

    def __init__(self, config: GPTConfig):
        cfg = config
        ...                    # 所有子模块在此创建，存储为实例属性
```

**原因：**
- **Linen** 模块是*冻结的 dataclass*。配置字段在类级别声明（`config: GPTConfig`）。参数和子模块**不**存储为实例属性；`@nn.compact` 将其创建推迟到首次调用时，并存入外部 pytree。`self.param(...)` 和 `nn.Dense` 在 `@nn.compact` 内部的调用会注册到该 pytree 中。
- **NNX** 模块是*普通 Python 对象*。配置是常规的 `__init__` 参数。参数和子模块作为实例属性存储在 `__init__` 中赋值。没有 `@nn.compact`，没有延迟初始化。

---

### 1.4 LayerNorm

**原始版**
```python
self.ln_1 = nn.LayerNorm(use_bias=cfg.bias)
```
`nn.LayerNorm` 是 Linen 内置层，首次调用时在外部 pytree 中创建自己的 `scale` 和 `bias` 参数。

**NNX 移植版**
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

**为什么不用 `nnx.LayerNorm`？** `nnx.LayerNorm` 存在，但构造时需要 `rngs` 参数（为未来随机扩展预留），会带来额外样板代码。自定义类避免了这一点，且使 `scale`/`bias` 属性显式可见，在 `sample.py` 中赋值 HuggingFace 权重时更加直接：`block.ln_1.scale.value = ...`。

**为什么用 `jax.lax.rsqrt` 而不是 `jnp.sqrt`？** `lax.rsqrt` 是单个 XLA 算子（倒数平方根），可融合成一个 kernel；`1/jnp.sqrt(...)` 需要两个算子。

---

### 1.5 线性层

**原始版** — 直接内联使用 `nn.Dense`：
```python
qkv = nn.Dense(3 * C, use_bias=cfg.bias,
               kernel_init=nn.initializers.normal(0.02),
               name='c_attn')(x)
```
`nn.Dense` 是 Linen 内置层。在 `@nn.compact` 内部必须指定 `name=` 参数，才能在参数 pytree 中获得稳定的键名。

**NNX 移植版** — 自定义 `Linear` 类：
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

**为什么自定义类？**
1. `nnx.Linear` 构造时需要 `rngs` 参数（与 LayerNorm 原因相同）。
2. 自定义类以 `(in_features, out_features)` 布局存储权重，与 GPT-2 Conv1D 约定和 SGLang-JAX 服务模型中的 `LinearBase` 一致——权重加载时无需转置。
3. 显式的 `self.weight` 和 `self.bias` 属性使 `sample.py` 中的 HuggingFace 权重赋值直接明了：`block.attn.c_attn.weight.value = ...`。

**`nn.Dense` 的 kernel 布局**也是 `(in, out)`，因此数学上完全相同；区别仅在于参数的存储和访问方式。

---

### 1.6 Dropout

**原始版**
```python
attn_weights = nn.Dropout(cfg.dropout)(attn_weights, deterministic=not training)
```
`nn.Dropout` 是 Linen 模块。`deterministic=True` 使其成为空操作（推理模式）。它通过调用点传入的 `rngs={'dropout': key}` 使用随机密钥——调用方负责传入密钥。

**NNX 移植版**
```python
if self.dropout_rate > 0.0 and rng is not None:
    rng, drop_rng = jax.random.split(rng)
    keep = jax.random.bernoulli(drop_rng, 1.0 - self.dropout_rate, attn_weights.shape)
    attn_weights = jnp.where(keep, attn_weights / (1.0 - self.dropout_rate), 0.0)
```

**为什么手动实现而不用 `nnx.Dropout`？**
- `nnx.Dropout` 持有有状态的 `nnx.Rngs` 对象。在 `jax.pmap` 下，每个设备需要自己的 `Rngs`，这要求在复制状态中传递每设备密钥——非常复杂。
- 手动实现是透明的 `jnp.where`；JAX/XLA 将其编译为带掩码的乘法（与 Dropout kernel 底层实现相同）。
- `if self.dropout_rate > 0.0 and rng is not None` 守卫意味着推理（不传 `rng`）是严格的空操作，**零开销**，不会将分支编译进 XLA。

---

### 1.7 Block 结构

**原始版**
```python
class Block(nn.Module):
    config: GPTConfig

    def setup(self):               # Linen 生命周期钩子：首次使用前调用一次
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

`setup()` 是 Linen 生命周期钩子，在首次调用 `__call__` 前运行。它是 `@nn.compact` 的替代方案，适用于需要显式子模块名称的场景（如按名称加载权重）。

**NNX 移植版**
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

**为什么没有 `setup()`？** NNX 没有生命周期钩子；`__init__` 是唯一的构造函数。所有子模块作为普通属性赋值——使用 Python 自身的属性协议。

**`rng` 传递：** 不再使用调用方与外部密钥组合的 `training: bool` 标志，而是在需要 dropout 时显式传递密钥。`None` 表示推理模式。

---

### 1.8 顶层 GPT 模块 — 参数声明

**原始版**（在 `@nn.compact __call__` 内部）：
```python
wte = self.param('wte', nn.initializers.normal(0.02), (cfg.vocab_size, cfg.n_embd))
wpe = self.param('wpe', nn.initializers.normal(0.02), (cfg.block_size, cfg.n_embd))
```
`self.param(name, init_fn, shape)` 在外部参数 pytree 中以键 `name` 注册一个叶节点。初始化函数在首次 `model.apply(...)` 时调用一次。

**NNX 移植版**（在 `__init__` 中）：
```python
self.wte = nnx.Param(
    jax.random.normal(jax.random.PRNGKey(1), (cfg.vocab_size, cfg.n_embd)) * 0.02
)
self.wpe = nnx.Param(
    jax.random.normal(jax.random.PRNGKey(2), (cfg.block_size, cfg.n_embd)) * 0.02
)
```
`nnx.Param` 是 JAX 数组的薄包装器。在构造时立即创建（无延迟初始化）。关键区别：**张量存在于模块内部**，而非外部 pytree。

**PRNGKey 种子：** 原始版从调用点传入 `model.apply(...)` 的 `rngs` 参数中获取密钥，种子由外部控制。NNX 版使用硬编码种子（`PRNGKey(1)`、`PRNGKey(2)`），因为初始值在 `sample.py` 中会立即被 HuggingFace 权重覆盖，或在训练中被 `nnx.split` + `nnx.merge` 替换——精确的初始化值无关紧要。

---

### 1.9 层列表

**原始版**
```python
for i in range(cfg.n_layer):
    x = Block(cfg, name=f'h_{i}')(x, training=training)
```
Block 在每次 `@nn.compact` 调用时内联创建；Linen 按 `name` 缓存它们。`name=f'h_{i}'` 是必须的，这样每个 block 的参数在 pytree 中才有稳定的键（`h_0`、`h_1`……）。

**NNX 移植版**
```python
self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
```
普通 Python 列表，包含 NNX 模块。`nnx.split` 递归遍历对象属性；列表元素自动包含在内。无需显式名称——NNX 使用整数索引作为 pytree 键。

**`nnx.data` 注意：** SGLang-JAX 代码库中有时使用 `nnx.data([...])` 存放模块列表，但该 API 在 Flax 0.8.5（开发环境版本）中不存在。普通 Python 列表效果完全相同，因为 `nnx.split` 原生处理列表。

---

### 1.10 权重绑定（Weight Tying）

**原始版**（在 `@nn.compact` 内部）：
```python
wte = self.param('wte', ...)    # 局部变量，下方两处使用同一对象
...
logits = x @ wte.T              # wte 复用于 lm_head 投影
```
`wte` 是 `self.param` 返回的局部 JAX 数组。使用两次成本极低——两个 `jnp.matmul` 调用引用同一张量。

**NNX 移植版**：
```python
self.wte = nnx.Param(...)       # self 的属性
...
logits = x @ self.wte.value.T   # .value 解包 nnx.Param 包装器
```
思路相同；需要 `.value` 是因为 `nnx.Param` 是包装类而非裸数组。访问 `.value` 返回底层 JAX 数组。

---

### 1.11 `estimate_mfu` 签名

**原始版**
```python
def estimate_mfu(self, params, fwdbwd_per_iter, dt, tpu_peak_tflops=2307.0):
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
```
需要传入外部 `params` pytree，因为 Linen 模型不拥有自己的参数。

**NNX 移植版**
```python
def estimate_mfu(self, fwdbwd_per_iter, dt, tpu_peak_tflops=918.0):
    ...
def num_params(self):
    state = nnx.state(self)
    return sum(v.size for v in jax.tree_util.tree_leaves(state))
```
参数存在于模块内部，`nnx.state(self)` 无需外部参数即可提取。`num_params()` 被抽取为独立辅助方法。

**默认 `tpu_peak_tflops` 差异：** 原始版默认 2307（v7x Ironwood）；NNX 移植版默认 918（v6e Trillium），因为训练脚本在 v7x 上运行时会通过 `--tpu_peak_tflops=2307.0` 覆盖。两者对各自用例均正确。

---

## 第二部分 — `sample.py`

### 2.1 导入

**原始版**
```python
from flax.training import train_state
from flax import traverse_util
import optax
from model import GPTConfig, GPT     # Linen GPT
```

**NNX 移植版**
```python
from flax import nnx, serialization
from model import GPT, GPTConfig     # NNX GPT
```

- `train_state` 和 `traverse_util` 是 Linen 时代用于管理 `(params, opt_state)` 捆绑包的工具。NNX 没有等价物，因为模型自身拥有状态。
- `sample.py` 中不导入 `optax`，因为推理不需要优化器。（原始版导入是为了重建检查点加载所需的优化器结构，因为 Linen 检查点嵌入了包含优化器状态的 `TrainState`。）

---

### 2.2 配置部分

两个文件的配置块和 `exec(open('configurator.py').read())` 模式完全相同，无差异。

---

### 2.3 HuggingFace 权重加载 — 方法对比

**原始版** — 从零构建 Linen 参数 pytree：
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
输出是嵌套字典，严格匹配 Linen pytree 布局：`params['h_0']['attn']['c_attn']['kernel']`。注意 Linen `nn.Dense` 将权重存于键 `'kernel'` 下。

**NNX 移植版** — 直接赋值给 `nnx.Param.value`：
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

**关键差异：**
| 方面 | 原始版（Linen） | NNX 移植版 |
|---|---|---|
| 返回值 | 新 dict `params` | `None`（就地修改模型） |
| 权重键名 | `'kernel'`（Linen `nn.Dense`） | `'weight'`（自定义 `Linear`） |
| LayerNorm scale 键名 | `'scale'` | `'scale'`（相同） |
| 中间字典 | 手动构建 | 不需要 |
| 设备传输 | 单独的 `jax.tree_util.tree_map(jnp.array, params)` 步骤 | 每个 `jnp.array(...)` 调用时完成 |

**为什么是 `'kernel'` vs `'weight'`？** Linen 的 `nn.Dense` 将权重存于 pytree 键 `'kernel'`（遵循 Flax/Haiku 约定）。NNX 模型中自定义 `Linear` 的属性名为 `weight`，经 `nnx.split` 后成为 pytree 键 `'weight'`。

**为什么就地修改？** NNX 模块是可变的 Python 对象。给 `nnx.Param.value` 赋值直接更新模块持有的张量。这避免了创建单独的参数字典再合并回来——`load_hf_weights` 返回后模型即可直接使用。

---

### 2.4 检查点加载

**原始版** — 重建完整 `TrainState` 并反序列化：
```python
def load_params_from_checkpoint(out_dir):
    # 必须重建与保存时完全相同的 pytree 结构
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

这很复杂，因为 Flax 的 `from_bytes` 需要一个与检查点结构**完全相同**的目标 pytree。为构建该目标，代码必须运行一次虚拟前向传播以初始化参数，然后将其包装进 `TrainState`。

**NNX 移植版** — 使用 `from_state_dict` + `nnx.update`：
```python
def load_nnx_checkpoint(model: GPT, out_dir: str) -> None:
    _, state = nnx.split(model)
    with open(path, 'rb') as f:
        outer = serialization.msgpack_restore(f.read())
    restored = serialization.from_state_dict(state, outer['state'])
    nnx.update(model, restored)
```

`nnx.split(model)` 提取当前状态作为目标结构。无需虚拟前向传播——模型已通过 `GPT(cfg)` 初始化。`nnx.update` 将恢复的状态就地写回模型的 `nnx.Param` 对象。

**不兼容说明：** 两种检查点格式**不可互换**。Linen 检查点嵌入了包含优化器状态和 `apply_fn` 的 `TrainState`；NNX 检查点嵌入了原始 `nnx.State` pytree + 优化器状态。将旧 Linen 检查点加载到 NNX 模型（或反之）会失败。这就是 GKE job 使用独立 GCS 路径（`gpt2-124m-nnx/` vs `gpt2-124m/`）的原因。

---

### 2.5 模型创建与初始化

**原始版**
```python
cfg = GPTConfig(**model_args)
model = GPT(cfg)
# params 不在 model 内——它们存在于 load_params_* 返回的独立字典中
params = jax.tree_util.tree_map(jnp.array, params)   # 主机 → TPU 设备传输
```
这里的 `model` 是无状态可调用对象；它只持有 `config`。所有张量在 `params` 中。显式的 `tree_map(jnp.array, params)` 将每个权重从 numpy（由 `safetensors.load_file` 返回）复制到 JAX 默认设备。

**NNX 移植版**
```python
model = GPT(cfg)
load_hf_weights(model, init_from)   # 将 jnp.array(...) 赋值给每个 nnx.Param.value
```
`model` 持有所有张量。设备传输在 `load_hf_weights` 内部每次 `jnp.array(...)` 调用时完成——无需单独步骤。

---

### 2.6 生成 — split/merge 模式

**原始版** — `model.apply` 传入外部参数：
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
`model.apply({'params': params}, ...)` 是 Linen 调用约定：参数以字典形式传入，不存储在模型中。`{'params': params}` 字典是 Linen 期望的"变量集合"。

使用 `jax.lax.scan` 将所有 `max_new_tokens` 步骤编译为**单个 XLA 程序**——一次调度，无 Python 循环开销，最大化 TPU 利用率。

**NNX 移植版** — `nnx.split` + `nnx.merge`：
```python
graphdef, state = nnx.split(model)

@jax.jit
def generate_step(state, window, rng_key):
    m = nnx.merge(graphdef, state)
    logits, _ = m(window)                # (1, 1, vocab_size)
    logits = logits[0, 0, :] / temperature
    ...
    return jax.random.categorical(rng_key, logits)

# 外层 Python 循环
for step_i in range(max_new_tokens):
    rng, step_rng = random.split(rng)
    next_tok = int(generate_step(state, window, step_rng))
    ...
```

**`nnx.split(model)` → `(graphdef, state)`：**
- `graphdef` 是模块树的静态、可哈希描述（结构、类型、元数据）。它被 jit 函数闭合捕获——JAX 追踪一次并缓存编译后的程序。
- `state` 是所有参数数组的纯 JAX pytree。作为追踪参数传给 `generate_step`，JAX 可以在不重新编译的情况下调度不同的权重。

**`nnx.merge(graphdef, state)`** 在 jit 函数内部重建一个活跃的 NNX 模型对象。这是 NNX 等价于 Linen 的 `model.apply({'params': params}, ...)` 调用。

**Python 循环 vs `lax.scan`：** NNX 移植版使用 Python `for` 循环而非 `lax.scan`。权衡如下：
- `lax.scan` 将所有步骤编译为一个 XLA 程序 → 最大吞吐量，但需要静态 `length` 和固定的 carry/输出形状。
- Python 循环每个 token 调用一次 `generate_step`，每次调度到缓存的 XLA 程序。首次调用慢（JIT 编译）；后续调用快。`step_i == 0` 时打印"first token compiled"标记此时刻。
- 对于生成数百个 token 的演示，Python 循环更简单且足够快。对于批量生产吞吐量，`lax.scan` 更优。

---

### 2.7 `logits[:, 0, :]` vs `logits[0, 0, :]`

**原始版**
```python
logits = logits[:, 0, :] / temperature   # shape: (batch=1, 1, vocab) → (1, vocab)
```

**NNX 移植版**
```python
logits = logits[0, 0, :] / temperature   # shape: (vocab,)
```

两者都从 `(1, 1, vocab_size)`（推理模式下 batch=1，一个位置）开始。原始版用 `[:, 0, :]` 保留 batch 维度；NNX 移植版用 `[0, 0, :]` 直接得到一维向量。对于 batch=1 两者均正确；NNX 版稍微简洁，因为 `jax.random.categorical` 直接接受一维 logits，无需额外压缩维度。

---

### 2.8 Top-k 过滤

**原始版**（在 `jax.lax.scan` 体内，编译进 XLA）：
```python
top_vals = jnp.sort(logits, axis=-1)[..., -top_k_val]
logits   = jnp.where(logits < top_vals[..., None], -jnp.inf, logits)
```
`top_k_val = min(top_k, real_vocab)` 是 Python int，在 JIT 前计算一次。`jnp.sort` 升序排序；`[..., -top_k_val]` 取第 `top_k_val` 大的值。

**NNX 移植版**（在 `@jax.jit generate_step` 内）：
```python
kth_val = jnp.sort(logits)[-_TOP_K]     # _TOP_K 是模块级 Python int
logits = jnp.where(logits < kth_val, -jnp.inf, logits)
```
逻辑完全相同；`_TOP_K` 在导入时作为模块级常量计算一次，JIT 时是静态 Python int（不是追踪值），保持编译图形状稳定。

---

### 2.9 词表掩码

两个文件都用 `-jnp.inf` 屏蔽填充词表 token（索引 ≥ 50257），确保模型不生成真实 GPT-2 BPE 词表之外的 token。两者逻辑相同，无差异。

---

### 2.10 生成输出

**原始版** — `lax.scan` 一次性返回完整 token 序列：
```python
_, tokens = jax.lax.scan(step, (window, rng_key), None, length=max_new_tokens)
# tokens: int32[max_new_tokens]
return jnp.concatenate([idx[0], gen_tokens])[None, :]  # (1, T + max_new_tokens)
```

**NNX 移植版** — 在循环中累积到 Python 列表：
```python
generated = list(start_ids)
for step_i in range(max_new_tokens):
    ...
    next_tok = int(generate_step(...))
    generated.append(next_tok)
print(decode(generated))
```
`int(...)` 将标量从设备具体化到 Python。这是每 token 一次主机-设备同步（大批量时较慢），但对演示完全足够，并去除了 scan 样板代码。

---

## 汇总对比表

| 概念 | 原始版（Linen） | NNX 移植版 |
|---|---|---|
| 模块基类 | `nn.Module`（冻结 dataclass） | `nnx.Module`（普通 Python 类） |
| 参数存储 | 外部 pytree（通过 `apply` 传入） | 模块内部，以 `nnx.Param` 存储 |
| 模块初始化 | `@nn.compact` / `setup()`（延迟） | `__init__`（立即） |
| 层名称注册 | 必须指定 `name=` 参数 | 不需要 |
| `model(x)` 调用 | `model.apply({'params': p}, x)` | 直接 `model(x)` |
| Dropout | `nn.Dropout(r)(x, deterministic=...)` | 手动 `bernoulli` + `jnp.where` |
| LayerNorm | `nn.LayerNorm` 内置 | 自定义 `LayerNorm(nnx.Module)` |
| 线性层 | `nn.Dense` 内置 | 自定义 `Linear(nnx.Module)` |
| 线性权重键名 | `'kernel'` | `'weight'` |
| 权重绑定 | 局部变量 `wte` 使用两次 | `self.wte.value` 使用两次 |
| `estimate_mfu` 签名 | 需要外部 `params` | 无外部参数（`nnx.state(self)`） |
| pmap / jit 接口 | params 作为位置参数 | `nnx.split` → `(graphdef, state)` |
| HF 权重加载 | 构建参数字典并返回 | 就地赋值给 `nnx.Param.value` |
| 检查点加载 | `from_bytes` 配合 `TrainState` 目标 | `from_state_dict` + `nnx.update` |
| 生成循环 | `jax.lax.scan`（单次 XLA 调度） | Python `for` 循环（每 token 一次 JIT） |
| 检查点格式 | Linen `TrainState`（不兼容） | NNX `State` pytree |
