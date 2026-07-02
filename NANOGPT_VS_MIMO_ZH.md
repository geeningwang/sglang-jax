# NanoGPT 与 MiMo-V2.5-Pro 架构对比

## 总览

| | NanoGPT（GPT-2） | MiMo-V2.5-Pro |
|---|---|---|
| **用途** | 教学 / 研究训练演示 | 生产推理服务 |
| **规模** | ~1.24 亿参数（GPT-2 small） | ~560 亿参数（MoE，约 80 亿激活参数） |
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
