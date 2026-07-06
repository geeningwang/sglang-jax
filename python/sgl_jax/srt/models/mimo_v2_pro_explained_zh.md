# mimo_v2_pro.py / mimo_v2_flash.py — 全面解析

本文档逐类、逐方法地介绍 MiMo-V2.5-Pro 模型实现，解释每段代码**做什么**、**为什么这样写**，以及各模块如何在 SGLang-JAX 服务栈中协同工作。

实现跨越两个文件：

| 文件 | 职责 |
|---|---|
| `mimo_v2_flash.py` | 基类——所有网络模块、权重加载流程、推理 `__call__` |
| `mimo_v2_pro.py` | 薄包装子类——覆盖权重加载以处理融合 `qkv_proj` 的 FP8 检查点格式 |

阅读顺序：先理解 `mimo_v2_flash.py`，再将 `mimo_v2_pro.py` 视为其差量（delta）。

---

## 目录

1. [架构总览](#1-架构总览)
2. [导入](#2-导入)
3. [MiMoV2MLP — 稠密 SwiGLU MLP](#3-mimov2mlp--稠密-swiglu-mlp)
4. [MiMoV2Moe — 混合专家块](#4-mimov2moe--混合专家块)
5. [MiMoV2Attention — 混合 GQA 注意力](#5-mimov2attention--混合-gqa-注意力)
6. [MiMoV2DecoderLayer — 单个 Transformer 层](#6-mimov2decoderlayer--单个-transformer-层)
7. [MiMoV2Model — 完整层堆叠](#7-mimov2model--完整层堆叠)
8. [MiMoV2FlashForCausalLM — Flash 变体（因果 LM 头）](#8-mimov2flashforcausallm--flash-变体因果-lm-头)
9. [MiMoV2ForCausalLM — Pro 变体（融合 QKV 覆盖）](#9-mimov2forcausallm--pro-变体融合-qkv-覆盖)
10. [权重加载流程](#10-权重加载流程)
11. [完整张量清单](#11-完整张量清单)
12. [总结与关键设计决策](#12-总结与关键设计决策)

---

## 1. 架构总览

MiMo-V2.5-Pro 是一个**仅解码器（decoder-only）Transformer**，总参数量约 1.02 万亿，每 token 激活约 420 亿参数。主要架构参数：

| 属性 | 数值 |
|---|---|
| 总层数 | 70 |
| 第 0 层 | 稠密 SwiGLU MLP |
| 第 1–69 层 | MoE SwiGLU（384 专家，top-8） |
| 注意力类型 | 混合：10 层全注意力 + 60 层 SWA（窗口=128） |
| Q 头数 | 128 |
| KV 头数 | 8（GQA，16× 共享） |
| Q/K head_dim | 192 |
| V head_dim | 128（不对称！） |
| 隐藏层大小 | 6144 |
| 词表大小 | 152576 |
| MoE 中间维度 | 每专家 2048 |
| 稠密 MLP 中间维度 | 16384（仅第 0 层） |
| 运行时数据类型 | bfloat16 |
| 检查点数据类型 | e4m3fnuz（FP8） |
| 权重绑定 | False（lm_head 独立） |

网络是标准残差堆叠结构：

```
input_ids
  → embed_tokens                      （词表 → 隐藏）
  → 遍历 70 个解码层：
      residual = hidden_states
      hidden_states = input_layernorm(hidden_states)
      hidden_states = self_attn(hidden_states) + residual   # 浮动残差
      residual = hidden_states
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = mlp(hidden_states) + residual          # 浮动残差
  → norm（最终 RMSNorm）
  → lm_head                           （隐藏 → 词表 logit）
```

**浮动残差（floating residual）**：残差不在子层内部累加，而是由解码层在每个子块之后显式相加。这使 XLA 能够将残差加法与后续的 layernorm 融合为一个算子，减少 HBM 流量。

---

## 2. 导入

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

**关键导入说明：**

- **`LinearBase`** — SGLang-JAX 中的标准线性投影层。接受 `kernel_axes` 注解，驱动 JAX GSPMD 分片，使 XLA 能自动将权重矩阵分片到设备网格上。列并行投影用 `kernel_axes=(None, "tensor")` 按输出特征维度分片；行并行投影用 `kernel_axes=("tensor", None)` 按输入特征维度分片。

- **`RMSNorm`** — 均方根归一化（无均值减法，无偏置）。比 LayerNorm 更轻量，是现代大语言模型的标配。

- **`Embed` / `ParallelLMHead`** — Token 嵌入层和语言模型头。`Embed` 存储 `(vocab, hidden)` 权重，沿词表维度（"tensor" 轴）分片。当 `tie_word_embeddings=False` 时，`ParallelLMHead` 是独立的输出投影。

- **`get_rope`** — 返回 `RotaryEmbedding` 模块的工厂函数，支持 NeoX 风格 RoPE、`partial_rotary_factor` 和 `rope_scaling`（如 YaRN 长上下文扩展）。

- **`RadixAttention`** — 分页 KV 缓存的薄包装元数据持有者，不自行计算注意力。实际计算由服务运行时在 Pallas `ragged_paged_attention_v3` 内核中完成。

- **`FusedEPMoEV2` / `FusedEPMoE` / `EPMoE`** — 三种 MoE 专家调度后端。`FusedEPMoEV2` 是 MiMo 的生产级 Pallas 内核（无 JAX 集合通信，内核内 EP all-to-all）。`FusedEPMoE` 是较旧的融合变体。`EPMoE` 是基于 megablox GMM 的非融合回退路径。

- **`GateLogit` / `TopK`** — MoE 路由器：`GateLogit` 计算 `hidden @ gate_kernel` 得到专家 logit；`TopK` 选出 top-K 个专家并重归一化权重。

- **`ForwardBatch`** — 携带单次服务步骤的所有批次元数据（token 位置、请求 ID、分页 KV 池引用、DP rank 分配等）。

- **`WeightLoader` / `WeightMapping`** — 权重加载系统。`WeightMapping` 描述一个 HuggingFace 张量如何映射到一个 JAX 参数（目标路径、分片、转置、FP8 标志）。`WeightLoader` 读取 safetensors 文件并应用每个映射。

- **`make_reduce_sharding`** — 辅助函数，返回行并行线性层之后使用的 `NamedSharding`。启用序列并行（SP）时，输出沿序列轴分片；否则沿 "data" 轴复制。

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

Pro 文件只导入需要覆盖的内容：基类、MoE 权重映射工厂和权重加载原语。所有网络模块均从 Flash 文件原封不动继承。

---

## 3. MiMoV2MLP — 稠密 SwiGLU MLP

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

**功能说明。** 含三个线性投影的 SwiGLU 前馈网络。**仅用于第 0 层**（70 层堆叠中唯一的稠密 MLP 层）。

**SwiGLU 计算过程：**

```python
def __call__(self, hidden_states, *, out_sharding=None):
    a1, _ = self.gate_proj(hidden_states)   # gate 路径
    a2, _ = self.up_proj(hidden_states)     # up 路径
    intermediate_parallel = a2 * self.act_fn(a1)  # SiLU(gate) × up
    output, _ = self.down_proj(intermediate_parallel, out_sharding=out_sharding)
    return output
```

`gate_proj` 和 `up_proj` 均将 `hidden_size` 扩展到 `intermediate_size`（第 0 层为 6144 → 16384），两者并行执行（无数据依赖），由 XLA 调度器融合。`silu(gate) * up` 为门控激活。`down_proj` 将 16384 收缩回 6144。

**张量并行。** `gate_proj` 和 `up_proj` 使用 `kernel_axes=(None, "tensor")`：列并行，每个设备持有输出特征维度的一个垂直切片。`down_proj` 使用 `kernel_axes=("tensor", None)`：行并行，每个设备持有输入特征维度的一个水平切片。`down_proj` 之后的隐式 all-reduce 由调用方的 `make_reduce_sharding` 处理。

---

## 4. MiMoV2Moe — 混合专家块

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
        self.experts = FusedEPMoEV2(...)  # 或 FusedEPMoE / EPMoE
```

用于**第 1–69 层**（全部 69 个 MoE 层）。

### 4.1 路由器（GateLogit + TopK + correction_bias）

`GateLogit` 执行一次无偏置线性投影：

```
router_logits = hidden_states @ moe_gate.kernel   # (tokens, 6144) @ (6144, 384) → (tokens, 384)
```

对于 `noaux_tc` 路由，在 top-K 选择前加入可学习的**修正偏置**：

```
effective_logits = router_logits + correction_bias   # (tokens, 384)
topk_weights, topk_ids = TopK(effective_logits)     # (tokens, 8) each
```

**为什么用 `noaux_tc`？** 传统 MoE 路由通过辅助负载均衡损失来惩罚路由不均衡。`noaux_tc`（"无辅助损失，目标计数修正"）用每专家偏置项（`correction_bias`）代替辅助损失，在训练中调整该偏置使每个专家的路由 token 数趋向目标值，在不引入辅助损失不稳定性的情况下实现负载均衡。偏置使用 float32（即使模型是 bf16），因为它在训练中需要对微小梯度进行长期累积。

`TopK.renormalize=True` 对选出的 top-8 专家的 softmax 分数重归一化（使其和为 1），得到正确的专家输出加权组合。

### 4.2 专家调度

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
    ...
    return mlp_output, topk_ids
```

**`token_valid_mask`** 处理填充位置（来自 DP 批次对齐层的 padding token）。填充位置的 `topk_ids` 被设为 `-1`，`FusedEPMoEV2` 内核据此跳过这些 token，避免浪费算力。

**返回 `topk_ids`**：调用方（解码层）收集所有层的路由决策，并通过 `__call__` 逐层向上传递，最终用于投机解码诊断或路由统计。

### 4.3 专家后端

| 后端 | 类 | 使用场景 |
|---|---|---|
| `fused_v2` | `FusedEPMoEV2` | MiMo V2（生产级；Pallas 内核内 EP all-to-all） |
| `fused` | `FusedEPMoE` | 其他 MoE 模型（较旧的融合变体） |
| `epmoe` | `EPMoE` | 回退路径（基于 megablox 的非融合 GMM） |

`FusedEPMoEV2` 在 Pallas 内核内部完整实现专家并行：token 被 scatter 到对应专家所在设备，结果再 gather 回 token 所有者，全程不经过 JAX 集合通信框架。每个设备持有 `384 / ep_size` 个专家（如 ep=8 时每设备 48 个）。

---

## 5. MiMoV2Attention — 混合 GQA 注意力

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

### 5.1 GQA 与不对称 Head 维度

MiMo 使用**分组查询注意力（GQA）**，且 head 维度不对称：

| 投影 | 头数 | Head dim | 总输出 |
|---|---|---|---|
| Q | 128 | 192 | 24576 |
| K | 8 | 192 | 1536 |
| V | 8 | 128 | 1024 |
| O | 输入=128×128=16384 | — | 6144 |

Q 的头数远多于 K/V（16× 共享），KV 缓存内存减少 16×。Q 和 K 使用 head_dim=192（用于 RoPE 和缩放点积）。V 使用较小的 head_dim=128，减小注意力输出大小，进而缩小 `o_proj` 的规模。输出投影接收 `128 × 128 = 16384` 的输入（Q 头数 × V head_dim），映射回 hidden_size=6144。

### 5.2 V 填充以适配融合 KV 缓存

```python
if self.v_head_dim != self.head_dim:
    pad_size = self.head_dim - self.v_head_dim  # 192 - 128 = 64
    v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_size)))
```

分页 KV 缓存（`RadixAttention`）将 K 和 V 按页交错存储，因此两者必须大小相同。V 在进入缓存前从 128 维填充到 192 维。注意力计算后，输出被切回真实的 V 尺寸：

```python
if self.head_dim != self.v_head_dim:
    attn_output = attn_output.reshape(-1, self.q_head_num, padded_head_dim)
    attn_output = attn_output[..., :self.v_head_dim]   # 截掉填充
    attn_output = attn_output.reshape(-1, self.q_head_num * self.v_head_dim)
```

### 5.3 旋转位置编码

```python
q, k = self.rotary_emb(positions, q, k)
```

RoPE 仅作用于 Q 和 K（不作用于 V），使用 NeoX 风格旋转。`positions` 为每个 token 的绝对序列位置（考虑分页缓存中的 KV 前缀）。`partial_rotary_factor < 1.0` 可仅对 head 维度的一部分应用 RoPE；MiMo 使用 1.0（全量旋转）。

### 5.4 注意力值缩放

```python
if self.attention_value_scale is not None:
    v = v * self.attention_value_scale   # MiMo-V2.5-Pro 为 0.612
```

对 V 施加标量乘数，在不改变标准 softmax 注意力公式的情况下，稳定大规模训练中的输出量级。MiMo-V2.5-Pro 的值为 0.612。

### 5.5 注意力 Sink 偏置

```python
self.attention_sink_bias = nnx.Param(shape=(num_heads,), ...) if attention_sink_bias else None
```

可选的每 Q 头可学习偏置，加到虚拟"sink" token 的注意力 logit 上。Pallas Flash Attention 内核无需实体化额外的 KV 条目即可实现 sink：将 sink 贡献预计算为标量偏置，纳入 softmax 归一化（替代 `l=0 / m=−∞` 初始化）。这使超长上下文（最长 1M token）始终能关注到一个稳定的起始点，防止注意力熵坍缩。

对于 MiMo-V2.5-Pro：SWA 层在 `add_swa_attention_sink_bias=True` 时有 sink 偏置，全注意力层在 `add_full_attention_sink_bias=True` 时有 sink 偏置。

### 5.6 RadixAttention（分页 KV 缓存）

```python
attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool,
                                   attention_sink=self.attention_sink_bias.value)
```

`RadixAttention` 是无权重张量的元数据持有者，调用时触发 Pallas `ragged_paged_attention_v3` 内核：
- Q：`(tokens, q_heads, head_dim)` — 每层每设备
- K/V：按页存储在 `token_to_kv_pool` 中
- `sliding_window_size`：SWA 层为 128，全注意力层为 0
- `kv_fused`：本步骤产生的新 K/V 页数据（融合写回）

`kv_fused` 是本步新生成的 K/V 数据；解码层将其返回，因果 LM 类收集所有层的 `kv_fused` 成 `layers_kv_fused`，用于池更新。

### 5.7 SWA 与全注意力的选择

`MiMoV2DecoderLayer._is_swa_layer()` 读取 `hybrid_layer_pattern[layer_id]`：

```python
def _is_swa_layer(self, config) -> bool:
    hybrid = getattr(config, "hybrid_layer_pattern", None)
    if hybrid is not None and 0 <= self.layer_id < len(hybrid):
        return hybrid[self.layer_id] == 1   # 1=SWA，0=全注意力
    return False
```

值为 `1` → SWA（sliding_window_size=128）；`0` → 全注意力（sliding_window_size=0）。70 层中：10 层全注意力，60 层 SWA。

SWA 层与全注意力层的**张量形状完全相同**（MiMo 配置中 `swa_num_attention_heads` == `num_attention_heads` == 128）。两者的区别仅在于注意力掩码，而非权重。

---

## 6. MiMoV2DecoderLayer — 单个 Transformer 层

```python
class MiMoV2DecoderLayer(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype):
        self.is_layer_sparse = self._is_moe_layer(config)

        # 注意力：根据 hybrid_layer_pattern[layer_id] 选择 SWA 或全注意力
        if self._is_swa_layer(config):
            self.self_attn = MiMoV2Attention(...SWA 配置参数...)
        else:
            self.self_attn = MiMoV2Attention(...全注意力配置参数...)

        # MLP：稠密（第 0 层）或 MoE（第 1–69 层）
        if self.is_layer_sparse:
            self.mlp = MiMoV2Moe(config, layer_id, mesh, dtype)
        else:
            self.mlp = MiMoV2MLP(hidden_size, intermediate_size, mesh, layer_id, dtype)

        self.input_layernorm = RMSNorm(config.hidden_size, ...)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, ...)
```

**每层独立的两个二值选择：**
1. 注意力：SWA（`hybrid_layer_pattern[i] == 1`）或全注意力（`== 0`）
2. MLP：MoE（`moe_layer_freq[i]` 为真）或稠密（为假）

对于 MiMo-V2.5-Pro，这两个选择恰好相关（第 0 层为稠密 MLP 且为全注意力），但代码将二者独立处理。

### 6.1 前向传播 — 浮动残差模式

```python
def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool,
             residual=None):
    reduce_sharding = make_reduce_sharding(hidden_states, self.mesh, ...)

    # 注意力子块
    if residual is not None:
        hidden_states += residual           # 加入前一层的残差
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_fused = self.self_attn(...)
    hidden_states += jax.sharding.reshard(residual, reduce_sharding)

    # MLP 子块
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if self.is_layer_sparse:
        hidden_states, topk_ids = self.mlp(hidden_states, forward_batch, ...)
    else:
        hidden_states = self.mlp(hidden_states, ...)
    residual = jax.sharding.reshard(residual, reduce_sharding)

    return hidden_states, residual, kv_fused, topk_ids
```

**为什么单独返回 `residual`？** 调用方（`MiMoV2Model`）跨层累积残差。第一层传入 `residual=None`，跳过第一个 `hidden_states += residual`。这样嵌入输出无需额外拷贝，自然成为第一个残差。最终层返回的残差在 `MiMoV2Model.__call__` 中于最终 norm 之前被加入 `hidden_states`。

**`jax.sharding.reshard(residual, reduce_sharding)`** — 行并行 down-projection 后，输出处于 `reduce_sharding` 上，而残差处于不同的分片（列并行输出的分片）。此调用确保残差在相加前与输出处于相同分片，避免 JAX 隐式重分片开销。

**`make_reduce_sharding`** 返回行并行层 all-reduce 输出的分片。不启用序列并行（SP off）时为完全复制；启用 SP 时沿序列轴分片，避免在每个芯片上重复完整的隐藏状态。

---

## 7. MiMoV2Model — 完整层堆叠

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

**`nnx.data([...])`** — 将 NNX 模块的 Python 列表包装为 NNX 可追踪序列。若不这样做，NNX 的参数遍历将无法识别列表内容。`nnx.data` 是 PyTorch `nn.ModuleList` 在 Flax NNX 中的等价物。

### 7.1 前向传播

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
        hidden_states += residual   # 最终残差加法
    hidden_states = self.norm(hidden_states)
    return hidden_states, layers_kv_fused, layers_topk_ids
```

`layers_kv_fused` — 每层一个条目，为该层注意力生成的新 K/V 数据，推理后用于更新分页 KV 池。

`layers_topk_ids` — 每层一个条目（稠密层为 None，MoE 层为 `(tokens, 8)`），返回给因果 LM 类，最终传回服务运行时用于路由统计或投机解码。

---

## 8. MiMoV2FlashForCausalLM — Flash 变体（因果 LM 头）

```python
class MiMoV2FlashForCausalLM(nnx.Module):
    def __init__(self, config, mesh, dtype):
        self.model = MiMoV2Model(config, dtype=dtype, mesh=mesh)
        self._kv_buffers: dict[int, dict] = {}   # FP8 K/V 暂存缓冲区

        if not config.tie_word_embeddings:   # MiMo 为 False
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, ...)

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)
```

**`_kv_buffers`** — 以 `layer_idx` 为键的 Python dict，每个值持有逐头反量化前的原始 FP8 K/V 权重和尺度。仅在权重加载期间存在，由 K/V 权重映射填充（FP8 K/V 数据被重定向至 `__KV_K_WEIGHT__{idx}` / `__KV_V_WEIGHT__{idx}` 键而非真实参数路径）。

### 8.1 GCSFuse 缓存预热

```python
@staticmethod
def _warmup_safetensors_cache(model_config):
    # 检查 model_path 是否在 GCSFuse 挂载点上
    with open("/proc/mounts") as fp: ...
    if "fuse" not in mount_type:
        return   # 块设备挂载不需要预热

    # 顺序批量读取所有 .safetensors 文件
    def _read_file(path):
        buf = bytearray(4 * 1024 * 1024)
        with open(path, "rb") as f:
            while f.readinto(buf): pass

    with ThreadPoolExecutor(max_workers=min(8, len(st_files))) as executor:
        list(executor.map(_read_file, st_files))
```

**为什么需要这个。** 模型权重存储在 Google Cloud Storage（GCS）中，通过 GCSFuse 挂载。GCSFuse 将文件数据缓存在内核页缓存中，但冷随机读取每次约需 400 ms（每次读取对应一个 GCS API 调用）。MoE 检查点包含数千个专家权重张量，冷加载可能耗费数小时。

**解决方案：** 顺序读取每个 safetensors 文件一次。顺序读取由 GCSFuse 的预取器处理并填充页缓存，使后续随机读取变为缓存命中（约 1 ms）。块设备挂载（GKE Persistent Disk、NFS）无需此预热，故跳过。

### 8.2 权重加载（Flash 变体）

```python
def load_weights(self, model_config):
    self._warmup_safetensors_cache(model_config)
    self.loader = WeightLoader(...)
    weight_mappings = self._create_weight_mappings()
    self.loader.load_weights_from_safetensors(weight_mappings)

    if self.loader.is_static_quant:
        # 1. Q 反量化（逐层）
        self.loader.dequant_fp8_layers(layers, specs=[("self_attn.q_proj", head_dim)])
        # 2. 融合 KV 反量化（跨 K/V 边界逐头块）
        self.loader.dequant_fused_kv(self._kv_buffers, layers, config)
        # 3. 第 0 层稠密 MLP 反量化
        self.loader.dequant_fp8_layers(layers, specs=[...], layer_filter=lambda i, l: i==0)
        # 4. 张量并行对齐 KV 头复制
        self.loader.replicate_kv_heads(layers, specs=[...])
```

四个后处理步骤详见 [§10 权重加载流程](#10-权重加载流程)。

### 8.3 `_create_weight_mappings`

构建覆盖所有模型参数的平铺 `dict[str, WeightMapping]`，结构如下：

```
全局张量：
  "model.embed_tokens.weight" → 目标 "model.embed_tokens.embedding"
  "model.norm.weight"         → 目标 "model.norm.scale"
  "lm_head.weight"            → 目标 "lm_head.embedding"（若 tie_word_embeddings=False）

逐层（循环 0..69）：
  注意力投影（q/k/v/o_proj）
  注意力 sink 偏置（条件性）
  LayerNorm（input_layernorm、post_attention_layernorm）
  MLP：稠密（gate/up/down_proj）或 MoE（gate + experts）
```

每个 `WeightMapping` 指定：
- `target_path`：JAX 参数路径（点分隔属性字符串）
- `sharding`：轴名元组（与 `LinearBase` 上的 `kernel_axes` 对应）
- `transpose`：是否在存储前转置（HF 以 `(out, in)` 存储；`LinearBase` 期望 `(in, out)`）
- `head_dim_padding` / `kv_head_padding`：张量并行对齐 Q/K/V 时使用
- FP8 暂存路径（`__KV_K_WEIGHT__{idx}` 等）用于延迟反量化

### 8.4 FP8 注意力投影处理（Flash 变体）

Flash 变体中，Q、K、V 在检查点中是三个独立投影：

- **Q**（`q_proj`）：以 `weight_q`（FP8）+ `weight_scale_inv` 后缀直接加载，在步骤 1 中逐层反量化。
- **K、V**（`k_proj`、`v_proj`）：加载到 `_kv_buffers` 暂存 dict（非真实参数），在步骤 2 中融合逐头反量化。"跨 K/V 边界"指 FP8 量化粒度：K 和 V 张量以每头块联合量化（一个 scale 覆盖可能跨越 K/V 边界的块），因此必须联合反量化。

### 8.5 `__call__` — 推理前向传播

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

**返回值结构** — 由 SGLang-JAX 模型执行器协议要求：

| 位置 | 内容 | 用途 |
|---|---|---|
| `output` | `LogitsProcessor` 输出的 logit | token 采样 / 投机验证 |
| `{"token_to_kv_pool": layers_kv_fused}` | 新 K/V 页，每层一个列表 | 推理后更新 KV 池 |
| `True` | `callback_flags`（始终为 True） | 触发步骤后回调 |
| `layers_topk_ids` | 每层 MoE 路由决策 | 路由统计 / 投机解码 |

`LogitsProcessor` 通过 `logits_metadata` 选出每个请求最后一个 token（解码步骤）或 prompt 最后一个 token（预填步骤）对应的 logit 行，应用温度缩放和 top-k/top-p 过滤，并返回处理后的 logit 张量。

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

由 MTP（Multi-Token Prediction）草稿模型（`mimo_v2_nextn.py` 中的 `MiMoV2MTPForCausalLM`）调用，在运行时共享目标模型的嵌入和 LM Head 权重，无需重新加载。

---

## 9. MiMoV2ForCausalLM — Pro 变体（融合 QKV 覆盖）

```python
class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):

    def __init__(self, config, mesh, dtype):
        super().__init__(config, mesh, dtype)
        self._fused_qkv_buffers: dict[int, dict] = {}
```

`MiMoV2ForCausalLM` 原封不动继承 Flash 的所有网络模块。唯一区别在于权重加载：

1. 额外的暂存缓冲区 `_fused_qkv_buffers` 用于融合 QKV FP8 数据
2. 覆盖 `load_weights`：用 `dequant_fused_qkv` 代替 Flash 的 Q 单独反量化
3. 覆盖 `_create_layer_mappings`：处理 `qkv_proj`（融合）而非独立 `q/k/v_proj`

### 9.1 为什么 Pro 需要特殊处理

Flash 检查点将 `q_proj`、`k_proj`、`v_proj` 存为三个独立张量。Pro 检查点将其存为**单个融合 `qkv_proj`** 张量，该张量在量化时按张量并行（TP）分片拼接：

```
Pro 检查点中的 qkv_proj 布局：
  [Q_shard_0 | K_shard_0 | V_shard_0 | Q_shard_1 | K_shard_1 | V_shard_1 | ...]
  ←── shard 0（tp_rank=0） ───→   ←── shard 1（tp_rank=1） ───→
```

每个分片块有独立的 FP8 scale。将其还原为独立 Q/K/V 需要知道 TP 分片边界，并在拆分前对每个分片块应用正确的 scale。这由 `WeightLoader.dequant_fused_qkv` 实现。

### 9.2 覆盖的 `load_weights`

```python
def load_weights(self, model_config):
    self.loader = WeightLoader(...)
    weight_mappings = self._create_weight_mappings()   # 调用覆盖的 _create_layer_mappings
    self.loader.load_weights_from_safetensors(weight_mappings)

    if self.loader.is_static_quant:
        # 步骤 1：融合 QKV 反量化（Pro 专有路径）
        self.loader.dequant_fused_qkv(self._fused_qkv_buffers, self.model.layers, self.config)
        # 步骤 2：第 0 层稠密 MLP 反量化
        self.loader.dequant_fp8_layers(layers, specs=[gate/up/down], layer_filter=第0层稠密)
        # 步骤 3：张量并行对齐 KV 头复制
        self.loader.replicate_kv_heads(layers, specs=[k_proj, v_proj])
```

Flash 有 4 个步骤，Pro 只有 3 个。Flash 中对 Q 的 `dequant_fp8_layers` 和独立的 `dequant_fused_kv` 被单个 `dequant_fused_qkv` 取代，后者联合处理 Q/K/V。

### 9.3 覆盖的 `_create_layer_mappings`

```python
def _create_layer_mappings(self, layer_idx):
    hf_qkv_key = f"model.layers.{layer_idx}.self_attn.qkv_proj"

    if is_fp8 and not qkv_ignored:
        # 将原始 FP8 融合 QKV 暂存入缓冲区，而非真实参数
        mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
            target_path=f"__FUSED_QKV_WEIGHT__{layer_idx}",
            sharding=(None, None), transpose=False,
        )
        mappings[f"{hf_qkv_key}.weight_scale_inv"] = WeightMapping(
            target_path=f"__FUSED_QKV_SCALE__{layer_idx}",
            sharding=(None, None), transpose=False,
        )
    else:
        # BF16 或已忽略量化：从融合张量直接拆分 Q/K/V
        mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
            target_path=[q_proj.weight, k_proj.weight, v_proj.weight],
            sharding=(None, "tensor"),
            transpose=True,
            head_dim_padding=False,
            kv_head_padding=True,
        )
```

对 FP8 检查点，融合 QKV 权重及其 scale 均被重定向至暂存缓冲区路径（`__FUSED_QKV_WEIGHT__{idx}` / `__FUSED_QKV_SCALE__{idx}`）。`WeightLoader.load_weights_from_safetensors` 识别这些特殊路径，将原始数据存入 `_fused_qkv_buffers[layer_idx]` 而非正常参数属性。

对 BF16 检查点（或已忽略量化的层），融合张量可直接由 `WeightMapping` 通过三个目标路径列表拆分。`WeightLoader` 沿头维度拆分并分配到各投影。

其余内容（o_proj、sink 偏置、LayerNorm、MLP/MoE）与 Flash 变体相同。

---

## 10. 权重加载流程

权重加载是实现中最复杂的部分。两个变体均采用两阶段方式：

### 阶段 1：批量 safetensors 加载

`WeightLoader.load_weights_from_safetensors(weight_mappings)` 遍历所有 `WeightMapping` 条目，对每条：
1. 从磁盘（或 GCS/GCSFuse）的 safetensors 文件读取张量
2. 若 `transpose=True` 则转置
3. 应用 `head_dim_padding` / `kv_head_padding` 进行张量并行对齐
4. 按 `sharding` 规格分片，并将每个分片放到对应设备
5. 存入目标参数路径（如设置 `layer.self_attn.q_proj.weight_q`）

对 FP8 暂存路径（`__KV_*` 或 `__FUSED_QKV_*`），步骤 5 改为存入 `_kv_buffers` 或 `_fused_qkv_buffers`。

### 阶段 2：FP8 反量化（仅当 `is_static_quant` 时）

检查点以 `e4m3fnuz`（FP8）静态量化。全部原始数据加载完毕后，通过多步反量化转换为 bfloat16 以供推理。

#### 步骤 A — Q 反量化（仅 Flash）

```python
self.loader.dequant_fp8_layers(layers, specs=[("self_attn.q_proj", head_dim)])
```

每个 `q_proj` 有 `weight_q`（FP8）和 `weight_scale`（bf16 尺度）。读取每层的 Q 权重，应用 `weight_q * weight_scale` 得到 bf16，写回为 `q_proj.weight`。

#### 步骤 B — 融合 KV 反量化（Flash）/ 融合 QKV 反量化（Pro）

**Flash：** `dequant_fused_kv(_kv_buffers, layers, config)`

K 和 V 联合量化（每个跨 K/V 通道的块共享一个 FP8 scale）。该函数读取 K/V 的暂存原始 FP8 数据，使用每块 scale 反量化（考虑块内的 K/V 边界），再将 bf16 值写入 `k_proj.weight` 和 `v_proj.weight`。

**Pro：** `dequant_fused_qkv(_fused_qkv_buffers, layers, config)`

暂存的融合 QKV 张量布局为分片交错 `[Q_s0 | K_s0 | V_s0 | Q_s1 | K_s1 | V_s1 | ...]`。该函数：
1. 遍历 TP 分片
2. 用每分片 scale 反量化各分片块
3. 将反量化后的分片拆分为 Q/K/V 切片
4. 跨分片拼接，写入 `q_proj.weight`、`k_proj.weight`、`v_proj.weight`

#### 步骤 C — 第 0 层稠密 MLP 反量化（两个变体）

```python
self.loader.dequant_fp8_layers(
    layers,
    specs=[("mlp.gate_proj", None), ("mlp.up_proj", None), ("mlp.down_proj", None)],
    layer_filter=lambda idx, layer: idx == 0 and not layer.is_layer_sparse,
)
```

只有第 0 层有稠密 MLP。`layer_filter` 将反量化限制在该层。MoE 专家权重（第 1–69 层）在运行时于 Pallas 内核内部（VMEM 中，与 DMA 重叠）反量化，因此在磁盘和 HBM 中均保持 FP8。

#### 步骤 D — 张量并行对齐 KV 头复制（两个变体）

```python
self.loader.replicate_kv_heads(
    layers,
    specs=[("self_attn.k_proj", head_dim), ("self_attn.v_proj", v_head_dim)],
    target_kv_heads_fn=lambda attn: attn.k_head_num,
)
```

在 TP 中，每个设备分片持有 Q 头的一部分。但 K/V 只有 8 个头，可能少于 TP 分片数。为使每个分片都拥有所需的完整 K/V 头，该步骤在 TP 维度上复制 K 和 V。

### HF 名称 → JAX 参数名称映射

| HuggingFace 键 | JAX 参数路径 |
|---|---|
| `model.embed_tokens.weight` | `model.embed_tokens.embedding` |
| `model.norm.weight` | `model.norm.scale` |
| `lm_head.weight` | `lm_head.embedding` |
| `model.layers.{i}.input_layernorm.weight` | `model.layers[i].input_layernorm.scale` |
| `model.layers.{i}.post_attention_layernorm.weight` | `model.layers[i].post_attention_layernorm.scale` |
| `model.layers.{i}.self_attn.q_proj.weight`（Flash） | `model.layers[i].self_attn.q_proj.weight_q`（FP8） |
| `model.layers.{i}.self_attn.qkv_proj.weight`（Pro） | 暂存 → 拆分为 q/k/v（FP8） |
| `model.layers.{i}.self_attn.o_proj.weight` | `model.layers[i].self_attn.o_proj.weight_q`（FP8） |
| `model.layers.{i}.self_attn.attention_sink_bias` | 同名路径 |
| `model.layers.{i}.mlp.gate.weight` | `model.layers[i].mlp.moe_gate.kernel` |
| `model.layers.{i}.mlp.gate.e_score_correction_bias` | `model.layers[i].mlp.correction_bias` |
| `model.layers.{i}.mlp.experts.{j}.gate_proj.weight` | 由 `create_moe_weights_mapping` 映射 |

---

## 11. 完整张量清单

**运行时数据类型：`bfloat16`。检查点数据类型：线性权重为 `e4m3fnuz`（FP8），每个 FP8 权重均配有 `weight_scale` 用于反量化。MoE 专家权重保持 FP8 存于 HBM，在内核内反量化。**

### 全局张量（× 1）

| 张量 | 形状 | 参数量 |
|---|---|---|
| `model.embed_tokens.embedding` | `(152576, 6144)` | 937,689,088 |
| `model.norm.scale` | `(6144,)` | 6,144 |
| `lm_head.embedding` | `(152576, 6144)` | 937,689,088 |

### 逐层张量 — 全部 70 层

**归一化：**

| 张量 | 形状 | 每层参数量 |
|---|---|---|
| `input_layernorm.scale` | `(6144,)` | 6,144 |
| `post_attention_layernorm.scale` | `(6144,)` | 6,144 |

**注意力（SWA 与全注意力层形状相同）：**

| 张量 | 形状 | 每层参数量 | 说明 |
|---|---|---|---|
| `self_attn.q_proj.weight` | `(6144, 24576)` | 150,994,944 | 128 Q 头 × 192 |
| `self_attn.k_proj.weight` | `(6144, 1536)` | 9,437,184 | 8 KV 头 × 192 |
| `self_attn.v_proj.weight` | `(6144, 1024)` | 6,291,456 | 8 KV 头 × 128 |
| `self_attn.o_proj.weight` | `(16384, 6144)` | 100,663,296 | 128 × 128 v_head_dim → 6144 |
| `self_attn.attention_sink_bias` | `(128,)` | 128 | 可选，每 Q 头一个 |

每层小计：**267,399,296** × 70 = **18,717,950,720** ≈ 187.2 亿

### 第 0 层 — 稠密 MLP

| 张量 | 形状 | 参数量 |
|---|---|---|
| `mlp.gate_proj.weight` | `(6144, 16384)` | 100,663,296 |
| `mlp.up_proj.weight` | `(6144, 16384)` | 100,663,296 |
| `mlp.down_proj.weight` | `(16384, 6144)` | 100,663,296 |

稠密 MLP 小计：**301,989,888** ≈ 3.02 亿

### 第 1–69 层 — MoE 块（× 69 层）

| 张量 | 形状 | 每层参数量 |
|---|---|---|
| `mlp.moe_gate.kernel` | `(6144, 384)` | 2,359,296 |
| `mlp.correction_bias` | `(384,)` | 384 |
| `mlp.experts[j].gate_proj.weight` × 384 | 每专家 `(6144, 2048)` | 4,831,838,208 |
| `mlp.experts[j].up_proj.weight` × 384 | 每专家 `(6144, 2048)` | 4,831,838,208 |
| `mlp.experts[j].down_proj.weight` × 384 | 每专家 `(2048, 6144)` | 4,831,838,208 |

每 MoE 层：≈ **144.98 亿** × 69 = ≈ **~10,003 亿** ≈ 1T

### 参数量汇总

| 类别 | 参数量 |
|---|---|
| 全局 | 1,875,384,320 |
| 注意力 × 70 层 | 18,717,950,720 |
| 稠密 MLP（第 0 层） | 301,989,888 |
| MoE 专家权重 × 69 层 | ~1,000,353,327,000 |
| **总计** | **~1.02T** |

**每 token 激活参数量：** 所有 70 层注意力 + 稠密 MLP + 8 专家 × 69 层 ≈ **42B（约 420 亿）**

---

## 12. 总结与关键设计决策

### Flash 与 Pro：一个差量

Flash 变体（`MiMoV2FlashForCausalLM`）期望检查点中有独立的 `q_proj`、`k_proj`、`v_proj` 权重。Pro 变体（`MiMoV2ForCausalLM`）期望单个融合的 `qkv_proj` 权重。其他一切——网络架构、前向传播、MoE 路由、KV 缓存集成——完全相同。子类关系（Pro 继承 Flash）正是这种设计的体现：只覆盖 `load_weights` 和 `_create_layer_mappings`。

### MoE 专家权重保持 FP8 存于 HBM

稠密权重（嵌入、注意力、第 0 层 MLP）加载后反量化为 bf16 并以 bf16 保存在 HBM 中。专家权重（总量 1.02T 中的约 1T）保持 FP8（每元素 1 字节）存于 HBM，在 Pallas `FusedEPMoEV2` 内核内部的 VMEM 中反量化。这使专家权重 HBM 带宽相比 bf16 减半（从约每参数 14 字节访问降至约 7 字节），而这正是 MoE 推理的主要瓶颈。

### 浮动残差

每个解码层将前一层的残差作为独立参数接收，并将新残差单独返回。这使 XLA 能将残差加法与 LayerNorm 调度为单个融合算子，同时避免在层内部创建中间张量。

### TP 轴名称驱动分片

`LinearBase` 的 `kernel_axes` 注解（`None`、`"tensor"`）绑定到 2D 设备网格（`["data", "tensor"]`）的命名轴。无需显式 `shard_map` 或 `pjit` 调用；JAX GSPMD 自动传播分片。列并行投影使用 `(None, "tensor")`；行并行使用 `("tensor", None)`。

### RadixAttention 无状态

`RadixAttention` 不持有任何权重张量——仅保存层级元数据（头计数、缩放因子、`sliding_window_size`）。实际 KV 缓存由服务运行时的 `MemoryPools` 管理；`RadixAttention` 通过 `token_to_kv_pool` 接收引用。这种分离使相同的网络权重能够以不同的 KV 缓存状态同时服务多个并发请求。

### EntryClass 注册

```python
# mimo_v2_flash.py
EntryClass = [MiMoV2FlashForCausalLM]

# mimo_v2_pro.py
EntryClass = MiMoV2ForCausalLM
```

两个文件均设置 `EntryClass`，SGLang-JAX 模型注册表通过该变量名发现模型类。Flash 使用列表（允许多个入口点）；Pro 使用单个类。注册表在配置指定 `"architectures": ["MiMoV2ForCausalLM"]` 时查找此变量。
