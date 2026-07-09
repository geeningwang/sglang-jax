# moe.py — 逐行详解

本文档逐类、逐方法、逐关键行地讲解 `python/sgl_jax/srt/layers/moe.py`，阐明每段代码**做什么**、**为何这样写**，以及各部分在 SGLang-JAX 推理服务栈中**如何协作**。

`moe.py` 提供了**基于 GMM 的（非融合）专家并行 MoE 后端**以及**所有 MoE 后端共用的权重映射工具**。MiMo-V2-Flash 的生产后端是 `FusedEPMoEV2`（位于 `fused_moe.py`）；本文件的 `EPMoE` 是回退路径和参考实现。

---

## 目录

1. [在 MoE 后端体系中的定位](#1-在-moe-后端体系中的定位)
2. [导入说明](#2-导入说明)
3. [EPMoE.__init__ — 权重初始化与 Mesh 构造](#3-epmoe__init--权重初始化与-mesh-构造)
4. [_detect_device_capabilities — 平台探测](#4-_detect_device_capabilities--平台探测)
5. [_normalize_scale_for_gmm — 量化 Scale 布局归一化](#5-_normalize_scale_for_gmm--量化-scale-布局归一化)
6. [quantize_weights — 在线与静态量化](#6-quantize_weights--在线与静态量化)
7. [__call__ — 通过 shard_map 进行专家并行分发](#7-__call__--通过-shard_map-进行专家并行分发)
8. [_forward — 单专家分片计算](#8-_forward--单专家分片计算)
9. [_gmm_compute — 基于 megablox gmm 的三矩阵 SwiGLU](#9-_gmm_compute--基于-megablox-gmm-的三矩阵-swiglu)
10. [_dispatch — 分片内的专家偏移量](#10-_dispatch--分片内的专家偏移量)
11. [_permute — 按专家分配对 Token 排序](#11-_permute--按专家分配对-token-排序)
12. [_unpermute — 专家输出的加权聚合](#12-_unpermute--专家输出的加权聚合)
13. [_combine — Expert 轴 All-Reduce](#13-_combine--expert-轴-all-reduce)
14. [create_moe_weights_mapping — HF → JAX 权重映射](#14-create_moe_weights_mapping--hf--jax-权重映射)
15. [完整张量清单](#15-完整张量清单)
16. [总结与关键设计决策](#16-总结与关键设计决策)

---

## 1. 在 MoE 后端体系中的定位

`moe.py` 定义了三类对象：

| 导出项 | 作用 |
|---|---|
| `EPMoE` | 非融合的基于 GMM 的专家并行 MoE 层（本文件核心） |
| `FusedEPMoE`、`FusedEPMoEV2` | 从 `fused_moe.py` 重导出，保持向后兼容 |
| `GateLogit`、`TopK` | 从 `gate.py` 重导出，保持向后兼容 |
| `create_moe_weights_mapping` | 工具函数：为任意 MoE 后端生成 HF → JAX 权重映射 |

代码库中存在三种 MoE 后端：

| 后端 | 类 | 内核 | 使用场景 |
|---|---|---|---|
| `epmoe` | `EPMoE`（本文件） | 通过 `shard_map` 使用 megablox `gmm` | 回退路径；兼容 CPU/GPU |
| `fused` | `FusedEPMoE` | Pallas 融合内核 v1 | 较早的融合变体 |
| `fused_v2` | `FusedEPMoEV2` | Pallas 融合内核 v2（Strix 双缓冲） | MiMo 生产路径 |

`EPMoE` 使用 JAX 的 `shard_map` 原语让每台设备独立执行其专家分片，然后跨设备规约。这在概念上清晰且可移植，但比 Pallas 融合后端需要更显式的控制流。

---

## 2. 导入说明

```python
"""基于 GMM 的专家并行 MoE 层及权重映射工具。"""

import math
from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

# 向后兼容重导出：外部代码从本模块导入这些符号。
from sgl_jax.srt.layers.fused_moe import FusedEPMoE, FusedEPMoEV2  # noqa: F401
from sgl_jax.srt.layers.gate import GateLogit, TopK  # noqa: F401
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    quantize_tensor,
    quantize_tensor_simple,
)
from sgl_jax.srt.utils.weight_utils import WeightMapping
```

**关键导入说明：**

- **`shard_map`** — JAX 的 SPMD 原语，用于编写逐设备程序。在 `shard_map` 函数体内，每台设备运行相同的 Python 函数，但只看到分片输入的本地切片。集合操作（`psum`、`psum_scatter`）在设备间通信。`EPMoE` 使用 `shard_map` 覆盖 `expert` 轴，使每台设备只计算其本地专家切片。

- **`get_global_expert_location_metadata`** — EPLB（专家负载均衡）钩子。当启用冗余专家时（例如热门专家被复制到多台设备），此函数返回携带 `num_physical_experts` 的元数据对象（若某些专家被复制，则可能大于 `num_experts`）。`EPMoE` 用它来正确确定权重张量的大小。

- **`gmm`** — megablox 分组矩阵乘法内核。接收左侧 Token 矩阵、右侧权重张量和每组大小，计算稀疏分批 GEMM，其中每组（专家）有自己的权重块。比逐专家独立密集 GEMM 更高效。

- **`FusedEPMoE, FusedEPMoEV2`（重导出）** — 历史遗留：在后端拆分为独立文件之前，所有 MoE 代码都在 `moe.py` 中。外部代码仍使用 `from sgl_jax.srt.layers.moe import FusedEPMoEV2`。`# noqa: F401` 抑制"已导入但未使用"的 Lint 警告——这些是有意的重导出，不是死代码。

- **`GateLogit, TopK`（重导出）** — 同理。`GateLogit` 计算路由 logits（`hidden @ gate_kernel`）；`TopK` 选择 Top-K 专家。两者都定义在 `gate.py`，此处为向后兼容而暴露。

- **`named_scope`** — 性能分析装饰器。将 `__call__` 方法包裹在具名 XLA 操作作用域中，使其在分析器追踪（如 TensorBoard、Perfetto）中按名称可见。不影响正确性。

- **`quantize_tensor` / `quantize_tensor_simple`** — 将 float32/bf16 张量量化为更低精度 dtype（如 FP8、INT8），并返回量化值和缩放因子。`quantize_tensor` 支持块量化（按组缩放）；`quantize_tensor_simple` 是在 `_gmm_compute` 中用于激活快速量化的单次逐通道量化器。

- **`WeightMapping`** — 描述一个 HuggingFace 检查点张量 → 一个 JAX 参数映射的数据类：目标路径、分片规格、是否转置，以及可选的专家拼接轴。被 `create_moe_weights_mapping` 使用。

---

## 3. EPMoE.__init__ — 权重初始化与 Mesh 构造

```python
class EPMoE(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        quantization_config=None,
        physical_to_logical_map: "jax.Array | None" = None,
        pre_gather_quant_dtype=None,
    ):
```

**参数说明：**

| 参数 | 含义 |
|---|---|
| `hidden_size` | 隐藏维度（如 Flash 版为 4096，Pro 版为 6144） |
| `num_experts` | 逻辑专家数（如 Flash 版 256，Pro 版 384） |
| `num_experts_per_tok` | 每个 Token 选择的 Top-K 专家数（如 8） |
| `ep_size` | 专家并行度（如 8 表示 8 台设备各持有 `num_experts/8` 个专家） |
| `mesh` | 全局 JAX 设备网格（轴：`"data"`、`"tensor"`） |
| `intermediate_dim` | 专家 FFN 隐藏大小（如 2048） |
| `weight_dtype` | 存储权重的 dtype（通常为 bf16；被 `quantize_weights` 覆盖为 FP8） |
| `dtype` | 运行时计算 dtype（推理时始终为 bf16） |
| `activation` | 非线性函数名：`"silu"`（SwiGLU 门控）或 `"gelu"` |
| `layer_id` | 解码器层索引；用于查找 EPLB 元数据 |
| `quantization_config` | 可选配置，提供权重/激活量化 dtype |
| `physical_to_logical_map` | EPLB 映射：物理专家索引 → 逻辑专家索引 |
| `pre_gather_quant_dtype` | 若设置，在 `_gmm_compute` 的 gather 前对激活量化 |

### 第 44–62 行：实例属性初始化

```python
self.num_experts_per_tok = num_experts_per_tok
self.physical_to_logical_map = physical_to_logical_map
self.pre_gather_quant_dtype = pre_gather_quant_dtype

metadata = get_global_expert_location_metadata()
if metadata is not None and layer_id is not None:
    self.num_experts = metadata.num_physical_experts
else:
    self.num_experts = num_experts
```

**EPLB 物理专家数。** 当 EPLB 启用时，热门专家被复制到额外设备。`num_physical_experts` ≥ `num_experts`。权重张量大小为 `num_physical_experts`，使每个物理槽位有自己的权重副本。若 EPLB 关闭（metadata 为 None），直接使用 `num_experts`。

```python
self.intermediate_dim = intermediate_dim
self.weight_dtype = weight_dtype
self.dtype = dtype
self.layer_id = layer_id
self.ep_size = ep_size
self.original_mesh = mesh
self.mesh = mesh
self.activation = activation
self.hidden_size = hidden_size
```

`self.original_mesh` 和 `self.mesh` 初始时相同。分别保存两者可让未来代码区分推理时的 mesh（可能被序列并行等机制修改）和构造时传入的 mesh。

### 第 64–73 行：量化配置提取

```python
self.quantized_dtype = (
    quantization_config.get_moe_weight_dtype() if quantization_config else None
)
self.activation_quantized_dtype = (
    quantization_config.get_moe_activation_dtype() if quantization_config else None
)
self.weight_block_size = (
    getattr(quantization_config, "weight_block_size", None) if quantization_config else None
)
```

三个独立的量化开关：

- **`quantized_dtype`**：权重量化 dtype（如 FP8 权重为 `jnp.float8_e4m3fn`）。若为 None，权重保持 `weight_dtype`（bf16）。
- **`activation_quantized_dtype`**：激活量化 dtype。若设置，GEMM 前激活被量化为此 dtype，以实现低精度矩阵乘法。
- **`weight_block_size`**：块量化的 `[block_n, block_k]`。权重矩阵的每个 `block_k × block_n` 分块有独立的缩放因子。若为 None，使用逐通道（逐行）缩放。

### 第 75–88 行：EP 整除性检查与推导大小

```python
if self.num_experts % self.ep_size != 0:
    raise ValueError(...)
world_size = math.prod(self.mesh.shape.values())
self.tp_size = world_size // self.ep_size
self.experts_per_device = self.num_experts // self.ep_size
```

**约束：** 专家数必须能被 EP 设备数整除。若 `num_experts=256`、`ep_size=8`，每台设备持有 32 个专家。

`world_size` = mesh 中的总设备数（如 `tp=8` × `dp=2` = 16）。
`tp_size` = 每个专家组的设备数 = `world_size / ep_size`。这是每个 EP 分片内的张量并行度：路由到一个专家分片的 Token 进一步分配到 `tp_size` 台设备。

### 第 83–93 行：MoE Mesh 构造

```python
devices = self.mesh.devices.flatten()
self.moe_mesh = jax.sharding.Mesh(
    devices.reshape(self.ep_size, self.tp_size),
    axis_names=("expert", "tensor"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
)

abstract_mesh = self.mesh.abstract_mesh
self.updated_mesh = abstract_mesh.update(
    axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
)
```

**为何需要第二个 Mesh？** 传入的 `mesh` 有轴 `("data", "tensor")`——对应服务栈的数据并行和张量并行布局。专家并行需要名为 `"expert"` 的轴，以便 `shard_map` 的 `in_specs` 能显式寻址专家维度。

`self.moe_mesh` 将扁平设备列表重组为 `(ep_size, tp_size)` 并重命名轴为 `"expert"` 和 `"tensor"`。物理设备对象不变，只是以不同方式寻址。

`self.updated_mesh` 是同一想法的**抽象 Mesh** 版本，与 `jax.sharding.use_abstract_mesh(...)` 上下文管理器配合使用。抽象 Mesh 允许在不绑定到具体设备的情况下表达重分片操作，这是 JAX 新版分片 API 的要求。

**重要：** `"tensor"` 轴名在原始 mesh 和 moe_mesh 中共用。这确保 `EPMoE` 能正确地跨 `shard_map` 边界传播张量并行分片。

### 第 95–128 行：权重参数初始化

```python
with jax.sharding.use_abstract_mesh(self.updated_mesh):
    self.wi_0 = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (self.num_experts, hidden_size, intermediate_dim),
            dtype=weight_dtype,
            out_sharding=P("expert", None, "tensor"),
        )
    )
    self.wi_1 = nnx.Param(...)  # 与 wi_0 形状相同
    self.wo = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (self.num_experts, intermediate_dim, hidden_size),
            dtype=weight_dtype,
            out_sharding=P("expert", "tensor", None),
        )
    )
    self.wi_0_scale = None
    self.wi_1_scale = None
    self.wo_scale = None
```

**权重名称：**

| 属性 | 作用 | 形状 | HF 等价物 |
|---|---|---|---|
| `wi_0` | 门控投影（SwiGLU 门控分支） | `(E, hidden, intermediate)` | `gate_proj.weight`（已转置） |
| `wi_1` | 上投影（SwiGLU 上行分支） | `(E, hidden, intermediate)` | `up_proj.weight`（已转置） |
| `wo` | 下投影 | `(E, intermediate, hidden)` | `down_proj.weight`（已转置） |

命名 `wi_0/wi_1/wo` 来自 megablox 惯例：`wi` = "权重输入"（门控和上行），`wo` = "权重输出"（下行）。

**权重布局 `[E, k, n]`：** 每个权重形状为 `(num_experts, in_features, out_features)`。这是 HuggingFace 惯例 `(out_features, in_features)` 的**转置**。收缩轴（`k`）在输出轴（`n`）之前，符合 megablox `gmm` 内核期望的布局。

**`wi_0`/`wi_1` 的分片 `P("expert", None, "tensor")`：**
- `"expert"` 轴：跨 EP 设备切分——每台设备持有 `E/ep_size` 个专家。
- `None` 轴：`hidden_size` 不分片（在张量轴上复制）。
- `"tensor"` 轴：`intermediate_dim` 跨每个 EP 组内的 TP 设备切分。

**`wo` 的分片 `P("expert", "tensor", None)`：**
- 收缩轴（`intermediate_dim`）跨 `"tensor"` 切分。
- 输出轴（`hidden_size`）不分片。

这是 `wo` 的行并行惯例：每台 TP 设备计算 `intermediate_dim/tp_size` 输入通道的部分点积，需要 all-reduce 才能得到完整输出（后续由 `_forward` 中的 `psum` 处理）。

**缩放因子初始化为 None：** 缩放因子在初始化时为 `None`。它们通过 `quantize_weights(is_static=True)`（加载预量化检查点）或 `quantize_weights(is_static=False)`（对已加载的 bf16 权重进行在线量化）分配。`None` 默认值允许 `gmm` 内核在未配置量化时跳过缩放乘法。

**使用 `PRNGKey(0)` 随机初始化：** 这些权重立即被 `load_weights` 覆盖。`normal()` 调用以正确的形状和分片分配设备内存；值不重要。

---

## 4. _detect_device_capabilities — 平台探测

```python
def _detect_device_capabilities(self):
    try:
        devices = jax.devices()
        is_cpu_only = all(device.platform == "cpu" for device in devices)
        can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

        device_types = [device.platform for device in devices]
        primary_device = device_types[0] if device_types else "unknown"

        return can_use_ragged, primary_device
    except Exception as _:
        return False, "cpu"
```

**目的：** 检查运行时是否具备 TPU 特有原语。`jax.lax.ragged_all_to_all` 是用于可变大小全交换集合的仅 TPU JAX 操作。若它存在（TPU 环境）且设备非纯 CPU，则可使用更高效的 ragged 路径。

**注意：** 此方法在当前 `EPMoE` 实现中被定义但未调用。它是早期 EP 设计（使用 ragged all-to-all）的遗留死代码。生产 `EPMoE` 改用 `shard_map` + `gmm`。

---

## 5. _normalize_scale_for_gmm — 量化 Scale 布局归一化

```python
def _normalize_scale_for_gmm(
    self,
    scale: jax.Array | None,
    weight: jax.Array,
    *,
    scale_name: str,
) -> jax.Array | None:
    """将离线/运行时缩放因子张量归一化为 GMM 的 4D 布局。"""
```

`gmm` 内核期望缩放因子采用特定的 4D 布局：

```
[E, k_blocks, 1, out_dim]
 ^      ^      ^    ^
 |      |      |    输出通道（n）
 |      |      单例广播维度
 |      K 轴上的量化块数量
 专家维度
```

但检查点文件和不同量化方案产生的缩放因子形状各异。此方法将所有形式归一化为 GMM 约定。

### 支持的输入布局

**第 169–196 行：4D 输入（已符合或接近 GMM 要求）**

```python
if scale.ndim == 4:
    if scale.shape[0] != num_experts or scale.shape[2] != 1 or scale.shape[3] != out_dim:
        raise ValueError(...)
    ...
    return scale
```

若缩放因子已有 4 个维度且结构正确，直接返回。验证确保专家数、单例维度和输出维度与权重匹配。注释标注了一个与更严格 JAX 版本（jax 0.10.x）在 `ep_size == world_size` 时的分片注释问题。

**第 198–199 行：2D 逐通道缩放 `[E, out_dim]`**

```python
if scale.ndim == 2 and scale.shape == (num_experts, out_dim):
    return scale[:, None, None, :]
```

最简单的情况：每个专家每个输出通道一个缩放因子。在位置 1 和 2 插入两个单例维度，变为 `[E, 1, 1, out_dim]`，满足 `k_blocks=1`（逐通道而非逐块）的 GMM 4D 约定。

**第 201–245 行：3D 缩放——三种子情况**

```python
if scale.ndim == 3:
    if scale.shape == (num_experts, 1, out_dim):
        return scale[:, :, None, :]
```

`[E, 1, out_dim]` → 在轴 2 插入单例 → `[E, 1, 1, out_dim]`。

```python
    if scale.shape == (num_experts, out_dim, expected_k_blocks):
        scale_gmm = jnp.transpose(scale, (0, 2, 1))[:, :, None, :]
        return jax.sharding.reshard(scale_gmm, final_scale_sharding)
```

**离线块量化格式：** HuggingFace 以 `[E, out_blocks, in_blocks]`（输出优先）存储块量化缩放因子。GMM 内核期望 `[E, k_blocks, 1, out_dim]`（输入优先，按输出通道扩展）。`transpose(0,2,1)` 交换块轴，`[:, :, None, :]` 插入单例，`reshard` 将结果放置在正确的设备轴上。

```python
    if scale.shape == (num_experts, expected_out_blocks, expected_k_blocks):
        out_block_ids = jnp.arange(out_dim, dtype=jnp.int32) // block_size_out
        scale_per_out = scale.at[:, out_block_ids, :].get(...)
        scale_gmm = jnp.transpose(scale_per_out, (0, 2, 1))[:, :, None, :]
        return jax.sharding.reshard(scale_gmm, final_scale_sharding)
```

**更粗粒度的块量化：** 缩放因子每个 `block_out × block_k` 分块一个条目，而非每个输出通道。`scale.at[:, out_block_ids, :].get(...)` 通过 gather 将粗粒度块缩放扩展为逐通道布局，然后应用相同的转置+重塑路径。`out_block_ids[c]` = 通道 `c` 所属的输出块，因此同一块中的每个通道获得相同的缩放因子。

```python
    if scale.shape == (num_experts, expected_k_blocks, out_dim):
        return scale[:, :, None, :]
```

已是 `[E, k_blocks, out_dim]` 顺序；只需插入单例维度。

---

## 6. quantize_weights — 在线与静态量化

```python
def quantize_weights(self, is_static: bool = False):
    """就地量化 MoE 权重，或为静态加载初始化参数。"""
    if self.quantized_dtype is None:
        return
```

**提前返回：** 若未配置量化，此方法为空操作。大多数 bf16 推理不使用权重量化。

### 辅助函数：`_get_block_size_k`

```python
    def _get_block_size_k(*, hidden_size, intermediate_dim, weight_block_size) -> int | None:
```

从 `weight_block_size = [block_n, block_k]` 中提取 K 维块大小。EPMoE 沿轴 1（`[E, k, n]` 布局中的收缩/K 维）量化，因此只有 `block_k` 相关。验证 `hidden_size` 和 `intermediate_dim` 都能被 `block_k` 整除。

### 静态路径（`is_static=True`）

```python
    with jax.set_mesh(self.moe_mesh):
        if is_static:
            # 分配零填充的占位缩放因子张量
            self.wi_0_scale = nnx.Param(
                jnp.zeros((num_experts, k_blocks_wi, 1, intermediate_dim), ...),
                out_sharding=wi_scale_sharding,
            )
            ...
            return
```

**使用时机：** 预量化检查点已包含 FP8 权重和显式缩放因子。`quantize_weights(is_static=True)` 以正确的形状和分片创建缩放因子 `nnx.Param` 槽位，供权重加载器填充。此处不做量化计算——加载器直接写入真实缩放因子。

**缩放因子分片：**

| 缩放因子 | 分片 | 原因 |
|---|---|---|
| `wi_0_scale`、`wi_1_scale` | `P("expert", None, None, "tensor")` | 输出维度（`n`）跨 TP 切分 |
| `wo_scale` | `P("expert", None, None, None)` | 输出维度（`hidden_size`）完全复制 |

创建新参数前先 `del self.wi_0_scale`：NNX 将参数作为实例属性跟踪。当 `foo` 已存在时执行 `self.foo = new_value` 可能混淆 NNX 的图遍历，因为旧参数对象仍存在于 Python 内存中。显式删除在重绑定前将其从 NNX 的视图中移除。

### 动态路径（`is_static=False`）

```python
        # 沿 k 维（[g, k, n] 布局中的 axis=1）量化权重
        w0_value, w0_scale = quantize_tensor(self.quantized_dtype, self.wi_0.value, axis=1, block_size=block_size_k)
        w1_value, w1_scale = quantize_tensor(self.quantized_dtype, self.wi_1.value, axis=1, ...)
        wo_value, wo_scale = quantize_tensor(self.quantized_dtype, self.wo.value, axis=1, ...)

        self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", None, "tensor"))
        self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", None, "tensor"))
        self.wo  = nnx.Param(wo_value,  out_sharding=P("expert", "tensor", None))
```

**使用时机：** 权重从 bf16 检查点加载。在线量化将其就地转换为 `quantized_dtype`（如 FP8），释放 HBM。整个量化在 JAX 中运行（无需遍历专家的 Python 循环），XLA 可融合和流水线化量化操作。

`axis=1` 在 `[E, k, n]` 中沿 K（隐藏）维量化。缩放因子形状为 `[E, k_blocks, n]`（块量化）或 `[E, n]`（逐通道，`block_size_k=None` 时）。

**缩放因子重塑为 GMM 约定：**

```python
        if block_size_k is not None:
            w0_scale = w0_scale[:, :, None, :]    # [E, k_blocks, n] → [E, k_blocks, 1, n]
        else:
            w0_scale = w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1])
            # [E, n] → [E, 1, 1, n]
```

两种路径都产生 GMM 4D 布局 `[E, k_blocks, 1, n]`。对于逐通道量化，`k_blocks=1`。

---

## 7. __call__ — 通过 shard_map 进行专家并行分发

```python
@named_scope
def __call__(
    self,
    hidden_states,
    topk_weights,
    topk_ids,
    *,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
```

`@named_scope` 将方法体包裹在名为 `"EPMoE"` 的 XLA 性能分析作用域中。

**输入：**

| 张量 | 形状 | 含义 |
|---|---|---|
| `hidden_states` | `(T, H)` | Token 隐藏状态；`T` = 本 DP rank 的 Token 数，`H` = `hidden_size` |
| `topk_weights` | `(T, K)` | Top-K 专家的 softmax 归一化路由权重 |
| `topk_ids` | `(T, K)` | Top-K 选定专家的索引；值在 `[0, num_experts)` 范围内 |

**第 421–432 行：输出分片**

```python
    if out_sharding is None:
        out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))
    out_specs = P(
        *[
            "tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None
            for s in out_sharding.spec
        ]
    )
    scatter_on_tensor = "tensor" in out_specs
```

调用方（如 `MiMoV2Moe`）传入 `out_sharding`，指定输出如何在 `self.mesh`（轴：`data`、`tensor`）上分片。但 `shard_map` 运行在 `self.moe_mesh`（轴：`expert`、`tensor`）上。两个 Mesh 共用 `"tensor"` 轴名，因此转换只需提取哪些维度需要 `"tensor"` 分片，忽略 `"data"`（在 `shard_map` 内部无意义）。

`scatter_on_tensor = True` 表示输出应沿 `"tensor"` 轴做 scatter 规约（RS 模式）而非 all-reduce。这是序列并行路径，每台设备只保留输出的 `1/tp_size`。

**第 436–439 行：将输入重分片到 moe_mesh**

```python
    with jax.sharding.use_abstract_mesh(self.updated_mesh):
        hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
        topk_weights_reshard  = jax.sharding.reshard(topk_weights,  P(None))
        topk_ids_reshard      = jax.sharding.reshard(topk_ids,      P(None))
```

`use_abstract_mesh` 上下文告知 JAX 相对于 `self.updated_mesh`（轴：`expert`、`tensor`）解释分片规格。`P(None)` = 在所有轴上完全复制。这确保 `hidden_states`、`topk_weights` 和 `topk_ids` 在进入 `shard_map` 前跨 `expert` 轴复制，使每台专家设备能看到所有 Token 并在本地路由。

**第 441–456 行：缩放因子归一化**

```python
        w0_scale = self._normalize_scale_for_gmm(self.wi_0_scale.value if ..., ...)
        w1_scale = self._normalize_scale_for_gmm(self.wi_1_scale.value if ..., ...)
        wo_scale = self._normalize_scale_for_gmm(self.wo_scale.value if ..., ...)
```

磁盘上的缩放因子可能是各种布局。`_normalize_scale_for_gmm` 在传给 `shard_map` 之前将每个缩放因子转换为 `[E, k_blocks, 1, out_dim]`。在 `shard_map` 内，每台设备看到本地切片 `[experts_per_device, k_blocks, 1, out_dim]`。

**第 458–493 行：shard_map 调用**

```python
        result = shard_map(
            partial(self._forward, scatter_on_tensor=scatter_on_tensor),
            mesh=self.moe_mesh,
            in_specs=(
                P(None),              # hidden_states：复制
                P(None),              # topk_weights：复制
                P(None),              # topk_ids：复制
                P("expert", None, "tensor"),   # wi_0
                P("expert", None, "tensor"),   # wi_1
                P("expert", "tensor", None),   # wo
                P("expert", None, None, "tensor"),  # w0_scale
                P("expert", None, None, "tensor"),  # w1_scale
                P("expert", None, None, None),      # wo_scale
                P("expert", None, "tensor"),   # bias0（None）
                P("expert", None, "tensor"),   # bias1（None）
                P("expert", None, None),       # biasO（None）
            ),
            out_specs=out_specs,
            check_vma=False,
        )(hidden_states_reshard, topk_weights_reshard, topk_ids_reshard,
          self.wi_0.value, self.wi_1.value, self.wo.value,
          w0_scale, w1_scale, wo_scale,
          None, None, None)
```

**`shard_map` 的工作方式：**
1. 沿指定 `"expert"` 轴切分每个输入：每台设备接收 `wi_0[local_expert_start:local_expert_end, ...]`，即其 `experts_per_device` 行权重。
2. 标记为 `P(None)`（Token）的输入在每台设备上复制——每台设备接收所有 Token，但只有本地专家权重。
3. 在每台设备上并行调用 `self._forward(...)`。
4. 按 `out_specs` 收集并放置输出。

**`check_vma=False`：** 禁用虚拟 Mesh 对齐验证。moe_mesh 和 abstract_mesh 经构造是一致的，但显式布局使 vma 检查不必要且可能较慢。

**三个 `None` 偏置：** `gmm` 内核为每次矩阵乘法接受可选的逐专家偏置。MiMo 架构不使用它们，传入 `None`。`shard_map` 仍需要它们的 `in_specs`（即使值为 `None`），以了解输入规格。

**第 495–498 行：将输出重分片回原始 Mesh**

```python
    return jax.sharding.reshard(result, out_sharding)
```

`shard_map` 的结果在 `self.moe_mesh`（轴：`expert`、`tensor`）上。`reshard` 将其转换为调用方在 `self.mesh`（轴：`data`、`tensor`）上期望的布局。下游操作（残差加法、层归一化）在 `self.mesh` 上运行，因此此重分片是正确性所必需的。

---

## 8. _forward — 单专家分片计算

```python
def _forward(
    self,
    hidden_states, topk_weights, topk_ids,
    w0_weights, w1_weights, wo_weights,
    w0_kernel_scale=None, w1_kernel_scale=None, wo_kernel_scale=None,
    w0_kernel_bias=None, w1_kernel_bias=None, wo_kernel_bias=None,
    *, scatter_on_tensor: bool = False,
):
```

此函数在 **`shard_map` 内部**运行——在每台设备上独立执行。设备看到权重张量的本地切片，但能看到所有 Token。

**第 517–523 行：张量形状归一化**

```python
    expert_shard_id = jax.lax.axis_index("expert")
    if hidden_states.ndim == 2:
        total_tokens = hidden_states.shape[0]
        batch_size, seq_len = 1, total_tokens
    else:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        total_tokens = batch_size * seq_len
```

`axis_index("expert")` 返回此设备在 `"expert"` 轴上的位置（0、1、…、ep_size-1）。这是设备的专家分片索引——它持有哪段连续的专家块。

`EPMoE` 支持 2D `(tokens, hidden)` 和 3D `(batch, seq, hidden)` 输入，归一化为 `batch_size` 和 `seq_len` 用于形状追踪。

**第 525–531 行：排列 → 分发 → 计算 → 反排列**

```python
    inputs_2d, token_indices, sorted_selected_experts, weights, group_sizes = self._permute(
        hidden_states, topk_ids, topk_weights
    )
    group_sizes = group_sizes.astype(jnp.int32)
    group_offset = self._dispatch(group_sizes, expert_shard_id)
    intermediate_output = self._gmm_compute(inputs_2d, token_indices, group_sizes, ...)
    output = self._unpermute(intermediate_output, sorted_selected_experts, weights, ...)
```

整体计算遵循四阶段流水线：

```
tokens → [_permute] → 按专家排序的 token-专家对
       → [_gmm_compute] → 专家输出（排序顺序）
       → [_unpermute] → 按专家加权求和，恢复 token 顺序
       → [psum/psum_scatter] → 跨 TP 设备 all-reduce
       → [_combine] → 跨 EP 设备 all-reduce
```

**第 557–567 行：TP 规约与 EP 合并**

```python
    if self.tp_size > 1:
        if scatter_on_tensor:
            output = jax.lax.psum_scatter(output, "tensor", scatter_dimension=0, tiled=True)
        else:
            output = jax.lax.psum(output, "tensor")
    if self.ep_size > 1:
        output = self._combine(output)
    return output
```

计算加权专家输出后，需要两次规约：

1. **TP 规约（`"tensor"` 轴）：** `wo` 的输出特征跨 `tp_size` 台设备切分（行并行投影）。每台设备计算了必须求和的部分点积。`psum` 是 all-reduce（AR）；`psum_scatter` 是 reduce-scatter（RS），用于序列并行——每台设备沿 Token 维度只保留规约输出的 `1/tp_size`。

2. **EP 合并（`"expert"` 轴）：** 每台专家设备计算了路由到其专家的 Token 的输出。未路由到此设备专家的 Token 在此设备上贡献零输出。`_combine` 对 `"expert"` 轴做 `psum`，使每台设备获得所有 Token 的所有专家输出之和。

**为何 EP psum 是正确的：** `_permute` 和 `_gmm_compute` 后，设备只持有路由到其专家的 Token 的非零值。未分配到此设备专家的 Token，输出为零（在 `_gmm_compute` 中如此初始化）。跨 `"expert"` 的 `psum` 然后累加所有设备的贡献，为每个 Token 产生正确的总输出。

---

## 9. _gmm_compute — 基于 megablox gmm 的三矩阵 SwiGLU

```python
def _gmm_compute(
    self, inputs_2d, token_indices, group_sizes,
    w0_kernel, w1_kernel, wo_kernel, group_offset,
    w0_kernel_scale=None, w1_kernel_scale=None, wo_kernel_scale=None,
    w0_kernel_bias=None, w1_kernel_bias=None, wo_kernel_bias=None,
):
```

**第 586–588 行：空批次守卫**

```python
    if token_indices.shape[0] == 0:
        return jnp.zeros((0, wo_kernel.shape[-1]), dtype=inputs_2d.dtype)
```

若无 Token 路由到此设备的专家（路由不均衡时可能发生），立即返回零张量。没有此守卫，`gmm` 会收到空输入，可能行为异常。

**第 590–599 行：gather 前可选的激活量化**

```python
    pre_gather_q = getattr(self, "pre_gather_quant_dtype", None)
    if pre_gather_q is not None:
        x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
        x = x_q[token_indices]
        x_scale = x_scale[token_indices]
        x = (x.astype(jnp.float32) * x_scale).astype(self.dtype)
    else:
        x = inputs_2d[token_indices].astype(self.dtype)
```

`token_indices` 将每个排序位置映射回原始 Token。gather `inputs_2d[token_indices]` 将 Token 重排为与其分配专家匹配的顺序。

**索引 GMM 模式：** 与其在 `_permute` 中物化完整的 `[M*top_k, D]` 排序张量再送入 `gmm`，不如在此延迟 gather。XLA 可将 gather 与 GEMM 融合，避免消耗 `M × top_k × D × dtype_bytes` HBM 的临时张量（top_k=8、D=4096、M=1024 的 bf16 约 67 MB）。

**第 610–638 行：两 GEMM 门控+上行（SwiGLU）**

```python
    gmm_kwargs = dict(
        group_sizes=group_sizes,
        preferred_element_type=self.dtype,
        group_offset=group_offset,
        maybe_quantize_lhs=act_q_dtype is not None,
        acc_dtype=jnp.float32,
    )

    layer_w0 = gmm(lhs=x, rhs=w0_kernel, rhs_scale=w0_kernel_scale, ..., **gmm_kwargs)
    layer_w1 = gmm(lhs=x, rhs=w1_kernel, rhs_scale=w1_kernel_scale, ..., **gmm_kwargs)
```

`group_sizes` 告知 GMM 内核此分片中每个专家分配了多少 Token。`group_offset` 是此设备的全局专家索引偏移（如设备 0 持有专家 0–31，设备 1 持有 32–63 → `group_offset=32`）。

`gmm` 运行批量 GEMM：对每个专家 `e`，计算 `output_e = x_group_e @ w0_kernel[e]`。`group_sizes` 数组告知内核排序的 `x` 中一个专家的 Token 块在哪里结束、下一个从哪里开始。

`preferred_element_type=self.dtype`（bf16）设置输出元素类型。`acc_dtype=jnp.float32` 为数值稳定性使用 float32 累积，然后舍入为 bf16 输出。

`maybe_quantize_lhs=True` 在 `act_q_dtype` 设置时启用内核内激活量化——LHS（`x`）在矩阵乘法前在 `gmm` 内核内量化为 `act_q_dtype`。

**第 641–648 行：SwiGLU 激活**

```python
    if self.activation == "silu":
        layer_act = jax.nn.silu(layer_w0)
    elif self.activation == "gelu":
        layer_act = jax.nn.gelu(layer_w0)
    intermediate_layer = jnp.multiply(layer_act, layer_w1)
```

SwiGLU：`output = silu(gate) * up`。`layer_w0` = 门控分支输出，`layer_w1` = 上行分支输出。逐元素乘法产生门控激活。形状：`[M*top_k, intermediate_dim]`（排序 Token 顺序，仅本地专家）。

**第 650–658 行：下投影**

```python
    return gmm(
        lhs=intermediate_layer,
        rhs=wo_kernel,
        rhs_scale=wo_kernel_scale,
        zero_initialize=True,
        ...
    )
```

`intermediate_layer @ wo_kernel` 将 `intermediate_dim → hidden_size`。`zero_initialize=True` 将输出累积器初始化为零——这是必需的，因为路由到其他专家的 Token 在此处无贡献，其输出槽位应为零（而非随机内存）。

---

## 10. _dispatch — 分片内的专家偏移量

```python
def _dispatch(self, group_sizes, expert_shard_id):
    if self.ep_size <= 1:
        return jnp.array(0, dtype=jnp.int32)
    group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)
    return group_offset
```

**目的：** 告知 `gmm` 内核此设备的全局专家索引偏移量。若设备 2 持有专家 64–95（`experts_per_device=32`），则 `group_offset = 2 * 32 = 64`。`gmm` 内核用此正确查找 `group_sizes[group_offset:group_offset+experts_per_device]`，得到本地专家的 Token 数量。

当 `ep_size=1`（无 EP）时，所有专家在一台设备上，偏移量为 0。

---

## 11. _permute — 按专家分配对 Token 排序

```python
def _permute(self, inputs, top_k_indices, top_k_weights):
```

**目的：** 将 Token 按其分配的专家排成连续组，产生 `gmm` 期望的布局。

**第 692–697 行：展平为 2D**

```python
    if len(inputs_shape) == 2:
        inputs_2d = inputs
        bsz_times_seq_len = inputs_shape[0]
    else:
        bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
        inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))
```

将 3D `(batch, seq, hidden)` 输入归一化为 2D `(tokens, hidden)`。

**第 700–707 行：按专家对 Token 排序**

```python
    flatten_selected_experts = jnp.ravel(top_k_indices)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
    token_indices = sorted_selected_experts // self.num_experts_per_tok
    group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)
```

`top_k_indices` 形状为 `(T, K)`，其中 `T=Token 数，K=top_k`。展平得到 `(T*K,)` 的专家 ID，每个 token-专家槽位一个。`argsort` 产生按升序专家顺序排列此平坦数组的索引。

**`token_indices`：** 因为 `top_k_indices` 的布局为 `[tok_0_exp_0, tok_0_exp_1, ..., tok_T_exp_K]`，将排序位置除以 `K` 得到原始 Token 索引。这让 `_gmm_compute` 无需物化完整排序隐藏状态张量即可 gather 正确的输入行。

**`group_sizes`：** `bincount` 统计 `T*K` 个 token-专家分配中分配给每个专家的数量。对于 `num_experts=256`，产生长度为 256 的数组，`group_sizes[e]` = 分配给专家 `e` 的 token-专家槽位数。`gmm` 内核用此知道排序 Token 列表中每个专家计算块的开始和结束位置。

**返回值：**

| 值 | 形状 | 含义 |
|---|---|---|
| `inputs_2d` | `(T, H)` | 原始隐藏状态，2D |
| `token_indices` | `(T*K,)` | 每个排序槽位映射到的原始 Token |
| `sorted_selected_experts` | `(T*K,)` | argsort 索引（在 `_unpermute` 中用于反排序） |
| `top_k_weights` | `(T, K)` | 路由权重（直接传递，不排序） |
| `group_sizes` | `(E,)` | 每个专家的 Token 数量 |

---

## 12. _unpermute — 专家输出的加权聚合

```python
def _unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, seq_len):
```

**目的：** 逆转排列，计算每个 Token 的专家输出加权和。

**第 718–727 行：长度修正**

```python
    if actual_tokens != expected_tokens:
        if actual_tokens > expected_tokens:
            intermediate = intermediate[:expected_tokens]
        else:
            padding_size = expected_tokens - actual_tokens
            padding = jnp.zeros((padding_size, intermediate.shape[1]), ...)
            intermediate = jnp.concatenate([intermediate, padding], axis=0)
```

由于内部对齐填充，`gmm` 内核返回的 Token 数量可能与 `sorted_selected_experts` 略有不同（megablox 后端将 LHS 填充到所需对齐并可能返回额外行）。此守卫裁剪或填充以确保反排序步骤具有匹配的形状。

**第 729–730 行：反排序**

```python
    argsort_indices = jnp.argsort(sorted_selected_experts, stable=True)
    unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)
```

`argsort(sorted_selected_experts)` 给出逆置换：排序列表中位置 `i` 对应原始布局中的哪个位置。将此索引应用于 `intermediate` 将 Token 恢复为原始顺序。

**第 732–748 行：专家加权合并**

```python
    reshaped_weights = jnp.reshape(weights, (total_tokens, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (total_tokens, self.num_experts_per_tok, -1),
    )
    intermediate_fp32 = reshaped_intermediate.astype(jnp.float32)
    weights_fp32 = reshaped_weights.astype(jnp.float32)

    output = jnp.einsum("BKE,BK -> BE", intermediate_fp32, weights_fp32)
```

反排序后，`unsort_intermediate` 形状为 `(T*K, H)`。重塑为 `(T, K, H)` 将每个 Token 的 K 个专家输出分组。

`einsum("BKE,BK -> BE")`：对每个 Token `B`，将 K 个专家输出 `BKE` 按权重 `BK` 加权求和，产生最终 `H` 维输出 `BE`。这是 MoE 加权合并：`output[t] = Σ_k weight[t,k] * expert_output[t,k]`。

**fp32 累积：** `weights` 和 `intermediate` 都转换为 float32 用于 einsum。加权求和是数值敏感的规约；在 bf16 中累积会损失精度，尤其是 top-K=8 需要求和 8 个值时。结果最终转回 `self.dtype`（bf16）。

---

## 13. _combine — Expert 轴 All-Reduce

```python
def _combine(self, data):
    return jax.lax.psum(data, "expert")
```

`_unpermute` 后，每台设备只持有其自身专家分片的加权输出之和。路由到其他设备专家的 Token 贡献零值。此 `psum` 跨 `"expert"` 轴累积所有设备的贡献，使每台设备获得其所有 Token 的正确总输出。

**为何不用 `psum_scatter`？** 与 TP 规约不同，EP 规约是真正的 all-reduce（AR）：每台设备都需要所有 Token 的完整结果，因为不同 Token 可能由不同设备上的专家服务，而拥有 Token KV 缓存状态的主机设备需要完整输出才能继续处理。

---

## 14. create_moe_weights_mapping — HF → JAX 权重映射

```python
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,
    expert_type_names: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
    expert_concat_axis_map: dict[str, int] = None,
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    physical_to_logical_map=None,
) -> dict:
    """为 MoE 层专家权重生成统一映射字典。"""
```

**目的：** HuggingFace 检查点将专家权重存储为 `num_experts` 个独立张量，每个专家一个。JAX `EPMoE` 和 `FusedEPMoE` 将其存储为每种投影类型的单个堆叠张量。此函数生成 `WeightMapping` 条目，指导权重加载器收集并堆叠这些独立张量。

**参数：**

| 参数 | 示例 | 含义 |
|---|---|---|
| `prefix` | `"model.layers.5"` | 此层的 HF 键前缀 |
| `target_prefix` | `"model.layers[5]"` | JAX 参数路径前缀 |
| `num_experts` | `256` | 逻辑专家数 |
| `expert_type_names` | `("gate_proj", "up_proj", "down_proj")` | HF 源权重名 |
| `expert_concat_axis_map` | `{"gate_proj": 0}` | 若某专家权重需沿特定轴拼接 |
| `moe_backend` | `"fused_v2"` | 决定目标属性名和分片 |
| `moe_path` | `"mlp"` | 每层内的子路径 |
| `source_expert_pattern` | `"experts.{i}"` | 每个专家 HF 键的模式（`.format(i=i)`） |
| `physical_to_logical_map` | `np.array([0,1,2,...])` | EPLB 映射 |

### 第 776–788 行：后端特定的目标属性名

```python
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend in ("fused", "fused_v2"):
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
```

`EPMoE` 使用 `wi_0/wi_1/wo`（megablox 命名）。`FusedEPMoE`/`FusedEPMoEV2` 使用 `w1/w3/w2`。此映射将 HF 源名称（如 `gate_proj`）转换为每个后端正确的 JAX 参数属性。

**为何是 `w1/w3/w2` 而非 `w1/w2/w3`？** 在融合后端中，惯例是：`w1` = 门控（SwiGLU 门控分支），`w3` = 上行（SwiGLU 上行分支），`w2` = 下行。此命名源自 Pallas 内核设计，其中 w1 和 w3 在 GEMM1 中一起计算，w2 是 GEMM2。

### 第 795–831 行：构建 WeightMapping 条目

```python
    for source_name, target_name in expert_type_map.items():
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]
```

对每种投影类型（`gate_proj/wi_0`、`up_proj/wi_1`、`down_proj/wo`）：
- `target_path_base` = 堆叠的 JAX 参数（如 `"model.layers[5].mlp.wi_0"`）
- `expert_keys` = `num_experts` 个 HF 张量键的列表，每个专家一个（如 `"model.layers.5.mlp.experts.0.gate_proj.weight"`、…、`"experts.255.gate_proj.weight"`）

```python
        if moe_backend == "epmoe":
            sharding = ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            transpose = True
        elif moe_backend in ("fused", "fused_v2"):
            sharding = (("data", "tensor"), None, None)
            transpose = True
```

**`epmoe` 的分片：**
- `wi_0/wi_1`：`P("expert", None, "tensor")` — 专家在 `"expert"` 轴切分，输出特征在 `"tensor"` 切分。
- `wo`：`P("expert", "tensor", None)` — 专家在 `"expert"` 切分，输入特征在 `"tensor"` 切分（行并行）。

**`fused/fused_v2` 的分片：** `(("data", "tensor"), None, None)` — 专家维度跨完整 EP mesh（`"data"` 和 `"tensor"` 轴的乘积）切分，其他维度不分片。融合内核使用自己的内部全交换，不需要 EPMoE 使用的 `"expert"`/`"tensor"` 切分。

**`transpose=True`** 适用于两者：HF 将权重存储为 `(out, in)`，但 EPMoE 的 `[E, k, n]` 布局期望 `(in, out)`。权重加载器在堆叠为 `[E, in, out]` 之前转置每个 `(out, in)` 专家权重。

```python
        mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
            target_path=[target_path_base] + expert_keys,
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
            physical_to_logical_map=physical_to_logical_map,
        )
```

**`__MOE_EXPERTS__` 前缀：** 一个哨兵，告知权重加载器使用专家堆叠代码路径而非普通张量赋值路径。当加载器看到 `target_path` 为列表时，逐一读取每个 `expert_keys` 张量，转置，沿新的专家轴 0 拼接，并将结果堆叠张量写入 `target_path_base`。

**`physical_to_logical_map`：** 当 EPLB 启用时，某些物理专家槽位持有热门逻辑专家的副本。此映射定义哪个逻辑专家填充每个物理槽位。权重加载器用它将正确的源张量复制到额外的物理槽位。

---

## 15. 完整张量清单

对于有 `E` 个专家、`H` 隐藏大小、`I` 中间大小、`ep_size` EP 度的模型：

### 权重参数

| 属性 | 形状 | 分片（moe_mesh） | 描述 |
|---|---|---|---|
| `wi_0` | `(E, H, I)` | `P("expert", None, "tensor")` | 门控投影权重 |
| `wi_1` | `(E, H, I)` | `P("expert", None, "tensor")` | 上行投影权重 |
| `wo` | `(E, I, H)` | `P("expert", "tensor", None)` | 下投影权重 |
| `wi_0_scale` | `(E, k_b, 1, I)` 或 None | `P("expert", None, None, "tensor")` | 门控权重量化缩放因子 |
| `wi_1_scale` | `(E, k_b, 1, I)` 或 None | `P("expert", None, None, "tensor")` | 上行权重量化缩放因子 |
| `wo_scale` | `(E, k_b, 1, H)` 或 None | `P("expert", None, None, None)` | 下投影权重量化缩放因子 |

`k_b` = K 维上的量化块数 = `H // block_size_k`（逐通道量化时为 1）。

每台设备持有 `E / ep_size` 个专家（EPLB 冗余时可能更多）。

### 中间张量（`_forward` 内部，每次 `shard_map` 调用）

| 张量 | 形状 | 描述 |
|---|---|---|
| `flatten_selected_experts` | `(T*K,)` | 所有专家分配，展平 |
| `sorted_selected_experts` | `(T*K,)` | 专家顺序排序的 argsort 索引 |
| `token_indices` | `(T*K,)` | 每个排序槽位的原始 Token 索引 |
| `group_sizes` | `(E,)` | 所有 Token 中每个专家的 Token 数量 |
| `x`（gathered） | `(T*K, H)` | 排序专家顺序中的 Token 隐藏状态 |
| `layer_w0` | `(T*K, I)` | 门控分支输出 |
| `layer_w1` | `(T*K, I)` | 上行分支输出 |
| `intermediate_layer` | `(T*K, I)` | SwiGLU 门控激活 |
| `intermediate_output` | `(T*K, H)` | 下投影输出（专家顺序） |
| `unsort_intermediate` | `(T*K, H)` | 重排为原始 token-专家顺序 |
| `output` | `(T, H)` | 每个 Token 的加权专家和 |

---

## 16. 总结与关键设计决策

### EPMoE 与 FusedEPMoEV2 对比

`EPMoE` 在 JAX 集合层面使用 `shard_map` 和 megablox `gmm` 内核实现专家并行。每个步骤——排列、GMM、反排列、全规约——都是独立的 JAX 操作。XLA 将其调度为独立内核。

`FusedEPMoEV2`（Pallas）在单个内核调用中完成所有这些：EP 分发、专家 FFN 计算和结果聚合在 VMEM 中进行，DMA 流水线与 MXU 计算重叠。排序 Token 或专家输出不产生中间 HBM 往返。

`EPMoE` 更易于理解、调试和移植到新硬件。`FusedEPMoEV2` 在 TPU v7x 上更快，但特定于 TPU。

### shard_map 无需显式集合设计即可实现 EP

`shard_map` 让 EPMoE 代码可以像在单设备上操作本地专家切片一样编写。EP 全交换是隐式的：因为所有设备接收所有 Token（复制的 `P(None)` 输入），每台设备只计算其本地专家，不需要显式 Token 重分发（全交换 scatter/gather）。最后的 `psum` 合并结果。这避免了 ragged 全交换的复杂性，同时仍实现 EP 行为。

### 索引 GMM 延迟 gather

完整的 token-专家展开 `[T*K, H]` 从未在 HBM 中物化。相反，`_permute` 返回 `token_indices`（每槽位的 Token 映射），`_gmm_compute` 通过索引访问在 `gmm` 调用内执行 gather。对于 1024 个 Token、top_k=8、H=4096、bf16：与物化排序张量相比，节省约 `1024 * 8 * 4096 * 2 ≈ 67 MB` HBM。

### `create_moe_weights_mapping` 与后端无关

此函数通过单次调用为所有三个后端（`epmoe`、`fused`、`fused_v2`）生成权重映射，差异仅在于目标属性名和分片规格。这意味着模型代码（`mimo_v2_flash.py`）可以传入 `moe_backend=config.moe_backend` 并获得正确的映射，无需在模型本身中使用任何 if/else。

### 缩放因子布局归一化具有防御性

来自不同量化流水线的检查点产生不同布局的缩放因子。与其要求每个检查点产生 GMM 兼容的缩放因子，`_normalize_scale_for_gmm` 在加载时处理所有已知布局（2D、3D、4D、块优先、通道优先、逐通道）并将其转换为 GMM 约定 `[E, k_blocks, 1, out_dim]`。这使 `EPMoE` 与任何量化工具链兼容，无需修改内核。

### 重导出保持向后兼容性

`FusedEPMoE`、`FusedEPMoEV2`、`GateLogit` 和 `TopK` 从本模块重导出，尽管它们定义在别处。在融合后端拆分为独立文件之前，所有 MoE 代码都在此文件中。使用 `from sgl_jax.srt.layers.moe import FusedEPMoEV2` 的现有模型文件和测试无需修改即可继续工作。
