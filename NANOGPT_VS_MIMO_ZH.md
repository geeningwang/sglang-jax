# NanoGPT 与 MiMo-V2.5-Pro 架构对比

## 总览

| | NanoGPT（GPT-2） | MiMo-V2.5-Pro |
|---|---|---|
| **用途** | 教学 / 研究训练演示 | 生产推理服务 |
| **规模** | ~1.24 亿参数（GPT-2 small） | ~1.02T 参数总量（MoE，每 token 约 420 亿激活参数） |
| **框架** | Flax NNX（训练模型） | Flax NNX + SGLang-JAX 服务栈 |
| **文件** | `examples/nanogpt/model.py` | `python/sgl_jax/srt/models/mimo_v2_pro.py`（继承自 `mimo_v2_flash.py`） |
| **入口类** | `GPT` | `MiMoV2ForCausalLM` |

---

## 各维度对比

### 归一化
- **NanoGPT**：标准 `LayerNorm`（均值 + 方差）。因为 NNX 层库只内置了 `RMSNorm`，所以采用自定义实现。
- **MiMo-V2.5-Pro**：`RMSNorm`（仅方差，不减均值）。每层两个 Norm（注意力前、MLP 前），与 GPT-2 的 Pre-norm 布局相同，但采用**浮动残差（floating residual）**模式——残差在层间显式传递，而非在层内相加，减少了融合算子的数量。

### 位置编码
- **NanoGPT**：可学习的绝对位置嵌入（`wpe`，形状 `[block_size, n_embd]`），在输入时与 token 嵌入相加。简单，上下文长度固定。
- **MiMo-V2.5-Pro**：旋转位置编码（RoPE，NeoX 风格），仅作用于注意力内部的 Q 和 K。`partial_rotary_factor` 控制 RoPE 覆盖 head 维度的比例。支持 `rope_scaling`（如 YaRN）以扩展上下文长度。

### 注意力机制
- **NanoGPT**：标准多头注意力（MHA）。融合 QKV 投影，通过 `jnp.tril` 手动构造因果掩码，全序列注意力，Q/K/V 头数相同，无 KV 缓存。
- **MiMo-V2.5-Pro**：**混合注意力**——每层由 `hybrid_layer_pattern` 配置决定使用**滑动窗口注意力（SWA）**或**全注意力**。分组查询注意力（GQA），K/V 头数独立设置。V 的 `head_dim` 可与 Q/K 不同（`v_head_dim`）。可选的每头可学习注意力 sink 偏置。使用 `RadixAttention`（分页 KV 缓存）支持高效长上下文服务。

  **Pro 与 Flash 的区别**：Pro 变体在检查点中将 Q、K、V 存储为单个融合的 `qkv_proj` 权重（而非三个独立投影）。权重加载器在加载后将其拆分为 `q_proj`、`k_proj`、`v_proj`，并包含特殊的逐分片 FP8 反量化路径。

### MLP
- **NanoGPT**：两层 GELU MLP：`Linear(n_embd → 4·n_embd) → GELU → Linear(4·n_embd → n_embd)`。
- **MiMo-V2.5-Pro**：三层 SwiGLU MLP：`gate_proj` 和 `up_proj` 并行运行，逐元素 `silu(gate) * up`，再经 `down_proj`。在 **MoE 层**（由 `moe_layer_freq` 决定），MLP 替换为混合专家（MoE）块（`MiMoV2Moe`），包含可学习路由器、Top-K 专家选择，以及 `EPMoE`/`FusedEPMoEV2` 专家调度。非 MoE 层使用标准稠密 SwiGLU。

### 线性层
- **NanoGPT**：自定义 `Linear(nnx.Module)`，权重形状 `(in, out)`，无分片，无量化。
- **MiMo-V2.5-Pro**：全程使用 `LinearBase`，带有 `kernel_axes` 注解支持张量并行分片（`"tensor"` 轴）。支持可选 FP8 静态量化（`weight_q` + `weight_scale`），K/V 投影支持逐头反量化。

### 权重加载
- **NanoGPT**：`sample.py` 中直接赋值给 `nnx.Param.value`。HuggingFace safetensors → numpy → `jnp.array(...)` 逐参数属性赋值。
- **MiMo-V2.5-Pro**：通过 `WeightLoader` + `WeightMapping` 的多阶段流水线（显式的 HF 名称 → JAX 路径映射，含转置、head 维度填充、KV 头复制标志）。批量加载后依次执行：Q 的 FP8 反量化（逐层）、融合 K/V 的反量化（跨边界逐头块）、第 0 层稠密 MLP 反量化、张量并行对齐的 KV 头复制。

### 推理服务
- **NanoGPT**：独立的自回归 Python 循环。`generate_step` 通过 `@jax.jit` + `nnx.split/merge` JIT 编译，每个 token 一次 JIT 调度，无 KV 缓存管理。
- **MiMo-V2.5-Pro**：集成到 SGLang-JAX 服务运行时。`__call__` 接收 `ForwardBatch`（包含 token 位置、分页 KV 池引用、投机解码信息）。返回 `(output, kv_fused_dict, callback_flags, None)`。KV 缓存由 `RadixAttention` 跨请求在外部管理。

### 投机解码
- **NanoGPT**：不支持。
- **MiMo-V2.5-Pro**：通过 `MiMoV2MTPForCausalLM`（`mimo_v2_nextn.py` 中的多 token 预测草稿模型）支持。最多 3 个 SWA 注意力草稿层；每层接收目标模型的隐藏状态 + 下一个 token 嵌入，投影降维后经过一个解码器层，输出草稿 token。LM head 在运行时与目标模型共享。

### 部署
- **NanoGPT**：单设备或 4 芯片 `jax.pmap`（数据并行）。GKE Job 运行在 `tpu7x-standard-4t`（2x2x1）节点上。
- **MiMo-V2.5-Pro**：多主机 TPU Pod（通常 4 个主机，32 个芯片）。张量并行 `tp=8`、数据并行 `dp=2`、专家并行 `ep=8`。通过 GKE 部署，权重从 GCSFuse 或 NFS 挂载加载。

### Pallas 计算内核

- **NanoGPT**：全程使用纯 JAX——`jnp.einsum`、`jax.nn.softmax`、标准矩阵乘法。所有算子均以高层 JAX 原语表达，由 XLA 编译。无自定义内核。

- **MiMo-V2.5-Pro**：三个关键操作以手写 **Pallas 内核**实现（`jax.experimental.pallas`，TPU 后端）。Pallas 是 JAX 的低层内核语言，提供对 VMEM、DMA、信号量和 MXU 的显式控制——类似 CUDA 之于 GPU，但针对 TPU。这是必要的，因为 XLA 的自动调度无法在 peak TPU 利用率所需的粒度上实现计算与内存传输的重叠。

  **1. Flash Attention — `ragged_paged_attention_v3`**
  生产级注意力内核（`kernels/ragged_paged_attention/ragged_paged_attention_v3.py`）。当 `FlashAttention` 后端激活时选用；`RadixAttention` 本身仅是元数据持有者——Pallas 内核完成所有实际计算。核心能力：
  - **流水线双缓冲 DMA**：Q、K/V 和输出 tile 均使用 `pltpu.make_async_copy` 和两个 VMEM 双缓冲。MXU 在计算一个 tile 的同时，DMA 引擎预取下一个 tile，几乎完全隐藏 HBM 延迟。
  - **融合 KV 缓存更新**：新 K/V token 在内核 pass 结尾散入其 page slot，省去独立的 scatter 操作。
  - **交错 KV 布局**：K 和 V 头在缓存 page 中交错存储，一次 DMA 加载即覆盖一个 K/V 头对——将 DMA 描述符开销减半，并支持更紧密的 GQA 展开。
  - **内置 SWA**：每个序列的窗口起始偏移在内核层计算；窗口外的 block 直接跳过，无需返回 Python。
  - **注意力 sink**：在 softmax 中包含一个虚拟 sink token，无需额外 KV 条目（以预计算的 sink logit 替代 l=0/m=−∞ 初始化）。
  - **DP 感知索引**：基于 `cu_kv_lens` 的 page 寻址使每个 DP rank 的内核分片在其紧凑 page 范围内独立运行，无需协调。

  **2. 融合 EP MoE — `FusedEPMoEV2`**
  Strix 风格的双缓冲 MoE 内核（`kernels/fused_moe/v2/kernel.py`）。仅对 MiMo V2 架构激活（配置字段 `moe_backend='fused_v2'`）。核心能力：
  - **无 JAX 集合通信**：EP all-to-all scatter（token 到专家设备）和 gather（结果回 token 所有者）全部在 Pallas 内核内通过 `pltpu.make_async_remote_copy`（`DeviceIdType.MESH`）完成，token 无需离开 Pallas 进入 JAX 集合框架。
  - **权重流式加载 + 双缓冲**：W1/W3/W2 权重 tile 在 MXU 计算上一个 tile 期间从 HBM 预取；W2 的 DMA 在 W1/W3 累加期间启动。
  - **token 数据驻留 VMEM**：每个 token 子 tile 在每次专家迭代中只加载一次，不会每个权重 tile 都从 HBM 重读。
  - **VMEM 内 FP8 反量化**：FP8 权重从 HBM 到达后在 VMEM scratch 中反量化为 bf16，再进行点积——反量化开销被 DMA 延迟隐藏。
  - **内核内共享专家**：MiMo 的共享专家部分与路由专家融合在同一个 `pl.pallas_call` 中。
  - 旧版 v1 内核（`FusedEPMoE`）和非融合 GMM 路径（`EPMoE`，基于 `megablox_gmm_kernel`）作为其他架构的回退选项仍然存在。

  **3. KV 缓存更新 — `update_kv_cache`**
  一个较小的 Pallas 内核，将新 K/V token 散入其 page slot。用于融合在注意力内部的 KV 更新路径被禁用时。

  总结：Pallas 让 MiMo 能够在硬件层面重叠计算与内存，在不暴露给 JAX 集合图的情况下执行 all-to-all 通信，并在整个 FFN 计算过程中保持缓存驻留的中间状态——这些均无法用纯 JAX 原语实现。

### 数据并行 / 专家并行（DP/EP）

- **NanoGPT**：通过 `jax.pmap` 实现数据并行——参数在所有芯片上复制，每个芯片处理不同的 micro-batch 分片。无专家并行（无 MoE）。梯度同步通过 `jax.lax.pmean`。对 124M 模型而言简单且足够。

- **MiMo-V2.5-Pro**：一个二维设备网格同时驱动三种不同的并行策略。

  **设备网格**
  ```
  网格形状：[dp_size, tp_size // dp_size]
  轴名称：  ["data",  "tensor"]
  ```
  对于 4 主机、32 芯片的部署：`dp=2, tp=8`，形成 `[2, 4]` 网格。EP 不是独立的网格轴——它复用 `data × tensor` 的全乘积（`ep = dp × (tp/dp) = 8`）。

  **张量并行（TP）** — SGLang-JAX 原有支持。
  `LinearBase` 权重通过 `kernel_axes` 沿 `"tensor"` 轴分片。列并行投影分片输出特征，行并行投影分片输入特征。每个行并行层的隐式 all-reduce 沿 `"tensor"` 轴完成。

  **数据并行（DP）** — 为 MiMo 扩展。
  每个 DP rank 拥有独立的请求子批次。调度器将传入请求轮询分配给各 DP rank，并将每个 rank 的批次填充到公共的 `per_dp_bs` 大小。注意力元数据（`cu_q_lens`、`cu_kv_lens`、`page_indices`）以 DP rank 连续方式排列，并以 `P("data")` 分片，使 rank `r` 的注意力 Pallas 内核分片仅处理 rank `r` 的请求。**注意力计算在各 DP rank 间完全独立——无跨 DP 通信。**

  原有 SGLang-JAX 已将 `dp_size` 作为服务参数并使用二维网格，但注意力元数据计算中充斥着 `if dp > 1: 2D else: 1D` 分支。为 MiMo 统一为单一代码路径，新增 `_per_dp_cumsum` 辅助函数和 `ModelWorkerBatch` 中的 `per_dp_bs_size` 字段，使多 DP 成为一流支持的配置，而非补丁式添加。

  **专家并行（EP）** — 为 MiMo 新增。
  MoE 专家权重以 `P(("data", "tensor"), None, None)` 分片——专家在整个网格上分布，每个设备拥有 `n_routed_experts / ep_size` 个专家。所有路由和通信均在 **Pallas 内核内**通过 `pltpu.make_async_remote_copy` 完成（token scatter 到专家设备，结果 gather 回 token 所有者）。由于没有独立的 EP 网格或 JAX 集合通信，DP 和 EP 在同一 `["data", "tensor"]` 网格上共存，互不冲突。`FusedEPMoEV2` 是唯一支持此拓扑的 MoE 后端；它通过 `_FUSED_MOE_V2_SUPPORTED_ARCHITECTURES` 限定只用于 MiMo V2 架构。

---

## 张量清单

### GPT-2 124M（NanoGPT）— 完整张量列表

所有权重均为 `float32`。括号内的参数量表示在 `bias=False` 训练检查点中不存在的张量。

**全局张量（× 1）**

| 张量 | 形状 | 参数量 | 说明 |
|---|---|---|---|
| `wte` | `(50304, 768)` | 38,633,472 | Token 嵌入；同时用作 LM Head——**权重共享，无单独的 lm_head 张量** |
| `wpe` | `(1024, 768)` | 786,432 | 可学习绝对位置嵌入 |

**每个 Transformer Block（× 12，所有块完全相同）**

| 张量 | 形状 | 每块参数量 | × 12 合计 | 说明 |
|---|---|---|---|---|
| `h[i].ln_1.scale` | `(768,)` | 768 | 9,216 | 注意力前 LayerNorm γ |
| `h[i].ln_1.bias` | `(768,)` | *(768)* | *(9,216)* | β — `bias=False` 时不存在 |
| `h[i].attn.c_attn.kernel` | `(768, 2304)` | 1,769,472 | 21,233,664 | 融合 Q+K+V 投影（输出 = 3 × 768） |
| `h[i].attn.c_attn.bias` | `(2304,)` | *(2,304)* | *(27,648)* | |
| `h[i].attn.c_proj.kernel` | `(768, 768)` | 589,824 | 7,077,888 | 注意力输出投影 |
| `h[i].attn.c_proj.bias` | `(768,)` | *(768)* | *(9,216)* | |
| `h[i].ln_2.scale` | `(768,)` | 768 | 9,216 | MLP 前 LayerNorm γ |
| `h[i].ln_2.bias` | `(768,)` | *(768)* | *(9,216)* | β — `bias=False` 时不存在 |
| `h[i].mlp.c_fc.kernel` | `(768, 3072)` | 2,359,296 | 28,311,552 | MLP 扩展（4× 隐藏维度） |
| `h[i].mlp.c_fc.bias` | `(3072,)` | *(3,072)* | *(36,864)* | |
| `h[i].mlp.c_proj.kernel` | `(3072, 768)` | 2,359,296 | 28,311,552 | MLP 收缩 |
| `h[i].mlp.c_proj.bias` | `(768,)` | *(768)* | *(9,216)* | |

**最终 LayerNorm（× 1）**

| 张量 | 形状 | 参数量 | 说明 |
|---|---|---|---|
| `ln_f.scale` | `(768,)` | 768 | 最终 LayerNorm γ |
| `ln_f.bias` | `(768,)` | *(768)* | β — `bias=False` 时不存在 |

**参数量汇总**

| | bias=True | bias=False（检查点） | 内存占用 |
|---|---|---|---|
| `wte` + `wpe` | 39,419,904 | 39,419,904 | 150.4 MiB |
| 12 × 块 | 85,054,464 | 84,953,088 | 324.1 MiB |
| `ln_f` | 1,536 | 768 | ~3 KiB |
| **合计** | **124,475,904** | **124,373,760** | **≈ 474.4 MiB** |

> `wte` 与 LM Head 共享权重：`logits = hidden @ wte.T`，不存在单独的 `lm_head` 张量。含检查点路径名的完整列表参见 [examples/nanogpt/PORTING_NOTES_ZH.md](examples/nanogpt/PORTING_NOTES_ZH.md) 附录。

---

### MiMo-V2.5-Pro — 张量清单

**70 层：第 0 层为稠密 MLP + 第 1–69 层为 MoE。10 层全注意力 + 60 层滑动窗口注意力（混合模式）。运行时数据类型：`bfloat16`；检查点数据类型：`e4m3fnuz`（FP8），每个线性权重均配有一个 `weight_scale` 张量。**

#### 全局张量（× 1）

| 张量 | 形状 | 参数量 | 说明 |
|---|---|---|---|
| `model.embed_tokens.embedding` | `(152576, 6144)` | 937,689,088 | Token 嵌入 |
| `model.norm.scale` | `(6144,)` | 6,144 | 最终 RMSNorm γ — 无偏置 |
| `lm_head.embedding` | `(152576, 6144)` | 937,689,088 | 输出投影 — **不与 embed_tokens 共享权重** |

全局小计：**1,875,384,320** ≈ 17.5 亿参数

#### 逐层张量 — 全部 70 层

SWA 层与全注意力层的张量形状完全相同；两者的区别在于注意力掩码，而非权重。

**归一化：**

| 张量 | 形状 | 每层参数量 | 说明 |
|---|---|---|---|
| `input_layernorm.scale` | `(6144,)` | 6,144 | 注意力前 RMSNorm γ — 无偏置 |
| `post_attention_layernorm.scale` | `(6144,)` | 6,144 | MLP 前 RMSNorm γ — 无偏置 |

**注意力：**

| 张量 | 形状 | 每层参数量 | 说明 |
|---|---|---|---|
| `self_attn.q_proj.weight` | `(6144, 24576)` | 150,994,944 | Q：128 头 × 192 head_dim |
| `self_attn.k_proj.weight` | `(6144, 1536)` | 9,437,184 | K：8 KV 头 × 192 head_dim |
| `self_attn.v_proj.weight` | `(6144, 1024)` | 6,291,456 | V：8 KV 头 × 128 v_head_dim |
| `self_attn.o_proj.weight` | `(16384, 6144)` | 100,663,296 | O：128 头 × 128 v_head_dim → 6144 |
| `self_attn.attention_sink_bias` | `(128,)` | 128 | 每 Q 头的注意力 sink logit 偏置 |

> 检查点（Pro 变体）中 Q/K/V 存储为单个融合的 `qkv_proj` 权重；权重加载器在加载时将其拆分。

每层小计：**267,399,296** × 70 层 = **18,717,950,720** ≈ 187.2 亿参数

#### 第 0 层 — 稠密 MLP

| 张量 | 形状 | 参数量 | 说明 |
|---|---|---|---|
| `mlp.gate_proj.weight` | `(6144, 16384)` | 100,663,296 | SwiGLU gate 分支 |
| `mlp.up_proj.weight` | `(6144, 16384)` | 100,663,296 | SwiGLU up 分支 |
| `mlp.down_proj.weight` | `(16384, 6144)` | 100,663,296 | 投影回隐藏维度 |

稠密 MLP 小计：**301,989,888** ≈ 3.02 亿参数

#### 第 1–69 层 — MoE 块（× 69 层）

**路由器（× 69 层）：**

| 张量 | 形状 | 每层参数量 | 说明 |
|---|---|---|---|
| `mlp.moe_gate.kernel` | `(6144, 384)` | 2,359,296 | 专家路由器：隐藏状态 → 384 个 logit |
| `mlp.correction_bias` | `(384,)` | 384 | `noaux_tc` 路由的每专家修正偏置 |

**专家权重（每层 × 384 个专家；运行时按 EP 分片）：**

| 张量 | 每专家形状 | 每专家参数量 | × 384 合计 | 说明 |
|---|---|---|---|---|
| `experts[j].gate_proj.weight` | `(6144, 2048)` | 12,582,912 | 4,831,838,208 | SwiGLU gate |
| `experts[j].up_proj.weight` | `(6144, 2048)` | 12,582,912 | 4,831,838,208 | SwiGLU up |
| `experts[j].down_proj.weight` | `(2048, 6144)` | 12,582,912 | 4,831,838,208 | 投影回隐藏维度 |

每专家合计：**37,748,736**；每层全部 384 个专家：**14,495,514,624** ≈ 144.96 亿  
每 MoE 层合计（路由器 + 专家）：≈ **144.98 亿**  
69 个 MoE 层合计：≈ **~10,003 亿** ≈ 1T

#### FP8 量化尺度张量（仅在检查点中）

每个线性权重以 `e4m3fnuz`（FP8）格式存储，并配有用于反量化的 `weight_scale` 张量：
- 注意力 Q 投影：权重加载器拆分 `qkv_proj` 后逐层进行 FP8 反量化
- KV 投影：跨边界逐头块反量化，并为张量并行对齐复制 KV 头
- MoE 专家权重：在 `FusedEPMoEV2` Pallas 内核的 VMEM 内完成反量化（与 DMA 重叠，开销被隐藏）

量化尺度张量不计入上述参数量统计。

#### 参数量汇总

| 类别 | 参数量 | 占比 |
|---|---|---|
| 全局（embed_tokens + lm_head + norm） | 1,875,384,320 | 0.18% |
| 注意力 × 70 层 | 18,717,950,720 | 1.83% |
| 稠密 MLP（仅第 0 层） | 301,989,888 | 0.03% |
| MoE 专家权重 × 69 层 | ~1,000,353,327,000 | 97.95% |
| **总计** | **~1,021,248,651,904** | **~1.02T** |

**每 token 激活参数量** = 所有 70 层注意力 + 稠密 MLP + 全局嵌入 + top-8 专家 × 69 个 MoE 层  
≈ 187.2 亿 + 3.0 亿 + 17.5 亿 + (8 × 3,775 万 × 69) ≈ **约 420 亿（42B）**

---

### 结构对比

| 维度 | NanoGPT（GPT-2 124M） | MiMo-V2.5-Pro（1.02T / 42B 激活） |
|---|---|---|
| **总参数量** | 1.24 亿 | ~1.02 万亿（约 8,233 倍） |
| **每 token 激活参数** | 1.24 亿（完全稠密） | ~420 亿（338 倍；仅 8/384 = 2.1% 专家激活） |
| **层数** | 12（全部相同） | 70（1 稠密 + 69 MoE） |
| **注意力类型** | 全量 MHA，所有层 | 混合：10 层全注意力 + 60 层 SWA（窗口 = 128 token） |
| **Q / KV 头数** | 12 / 12（无 GQA） | 128 / 8（KV 头 16× 共享） |
| **Head dim（Q/K，V）** | 64，64 | 192，128（不对称） |
| **Token 嵌入** | `(50304, 768)` = 3,863 万 | `(152576, 6144)` = 9.377 亿（约 24 倍） |
| **位置嵌入** | `wpe (1024, 768)` = 79 万 | 无——RoPE 在注意力内部计算，无可学习位置张量 |
| **LM Head** | 与 `wte` 共享权重（无额外参数） | 独立 `(152576, 6144)` = 9.377 亿 |
| **注意力投影** | 融合 `c_attn (768, 2304)` + `c_proj (768, 768)` | 独立 `q/k/v/o_proj`；检查点为融合 `qkv_proj`，加载时拆分 |
| **归一化** | LayerNorm — scale + bias（每个 norm 2 个张量） | RMSNorm — 仅 scale，无 bias（每个 norm 1 个张量） |
| **MLP 类型** | 稠密两层 GELU（扩展/收缩） | 稠密 SwiGLU（第 0 层）；MoE SwiGLU（第 1–69 层） |
| **MLP 中间维度** | 768 → 3072 → 768（4× 扩展） | 6144 → 16384（稠密）；6144 → 2048（每专家，MoE） |
| **专家数量** | 无 | 384 个总计；每 token 激活 8 个（2.1%） |
| **每专家参数量** | — | 3,775 万；全部 384 个合计每 MoE 层 144.96 亿 |
| **路由器** | 无 | `(6144, 384)` 门控核 + `(384,)` 修正偏置 |
| **注意力 sink 偏置** | 无 | `(128,)` 每层（每 Q 头一个） |
| **数据类型** | float32 | bfloat16（检查点为 FP8） |
| **词表大小** | 50304（从 50257 填充） | 152576 |
| **最大上下文长度** | 1024 token | 1M token（RoPE + SWA） |

---

## 总结

NanoGPT 是一个简洁、自包含的 GPT-2 实现，面向学习和实验——一个训练模型、一个生成脚本、依赖极少。MiMo-V2.5-Pro 是生产级系统：混合 MoE 架构、分页 KV 缓存服务、FP8 量化、投机解码、多主机张量/数据/专家并行，以及手写 Pallas 内核——绕过 XLA 调度器以实现 TPU 峰值利用率。两者共享相同的顶层解码器-Only Transformer 骨架（嵌入 → 堆叠 Block → LM Head）和相同的 Flax NNX 模块系统，但在这一骨架之下的几乎每个实现细节上都截然不同。

| 维度 | NanoGPT | MiMo-V2.5-Pro |
|---|---|---|
| 归一化 | LayerNorm | RMSNorm + 浮动残差 |
| 位置编码 | 可学习绝对编码（`wpe`） | RoPE + 部分因子 + rope_scaling |
| 注意力 | 全量 MHA，因果掩码 | 混合 SWA/全量 GQA，RadixAttention |
| MLP | 两层 GELU | SwiGLU；部分层 MoE |
| 线性层 | 自定义 `Linear`，无分片 | `LinearBase`，TP 轴，FP8 |
| 权重加载 | 直接 `.value` 赋值 | `WeightLoader` + 多阶段 FP8 反量化 |
| 推理服务 | Python 循环，无 KV 缓存 | `ForwardBatch`，分页 KV，服务运行时 |
| 计算内核 | 纯 JAX / XLA | Pallas：flash-attn、融合 EP MoE、KV scatter |
| 数据并行 | `jax.pmap` 副本 | 多 DP，`["data","tensor"]` 网格，DP 分片元数据 |
| 专家并行 | 无 | Pallas 内核内 all-to-all，无 JAX 集合通信 |
| 部署 | 4 芯片 pmap | 4 主机 TPU Pod，tp=8 / dp=2 / ep=8 |
