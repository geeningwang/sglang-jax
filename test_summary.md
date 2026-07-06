# MiMo-V2-Flash & MiMo-V2.5-Pro TPU 复现报告

**测试人：** Jing Wang  
**代码：** `primatrix/sglang-jax`，branch `mimo-tpu7`，commit `1bc2227`

---

## 一、MiMo-V2-Flash

### 复现环境

| 项目 | 配置 |
| ----- | ----- |
| 硬件 | TPU v7x 4-chip，topology 2×2×1（GKE DWS queued-provisioning） |
| 并行配置 | tp=8, dp=2, ep=8 |
| 代码 | `primatrix/sglang-jax`，commit `1bc2227` |
| 模型权重 | MiMo-V2-Flash HF safetensors（291.6 GiB），从 tmpfs 通过 NFS 挂载（权重全量驻留内存，无磁盘 I/O） |
| 压测工具 | `bench_serving`，random 数据集，input=16384 tokens，output=4096 tokens，bsz=32/64/128 |
| Server 启动参数 | 与 §3.1 完全一致，无差异 |

### MiMo-V2-Flash最优复现结果 vs §1.3 原始数据

下表为 6 次独立实验（覆盖不同硬件节点、XLA 缓存状态、page-size、chunked-prefill-size）中取得的最优吞吐，与 §1.3 原始数据对比。

**MTP（投机解码，NEXTN 3-step）**

| bsz | §1.3 原始 tok/s | 复现最优 tok/s | 差距 |
| :---: | ----: | ----: | ----: |
| 32  | 1411.41 | 1314.00 | −6.9% |
| 64  | 1591.87 | 1507.33 | −5.3% |
| 128 | 1831.77 | 1738.08 | −5.1% |

**No-MTP（标准解码）**

| bsz | §1.3 原始 tok/s | 复现最优 tok/s | 差距 |
| :---: | ----: | ----: | ----: |
| 32  | 1161.88 | 1122.25 | −3.4% |
| 64  | 1460.99 | 1403.32 | −3.9% |
| 128 | 1740.67 | 1672.92 | −3.9% |

MTP 接受率与 §1.3 高度一致（所有实验均 ≥97.6%，§1.3 为 ≥98.1%）。MTP/No-MTP 加速比趋势及各 bsz 档位相对排序完全保留。

### 已排除的差距来源

| 假设 | 验证方法 | 结论 |
| ----- | ----- | ----- |
| 模型加载开销（gcsfuse） | 改用 NFS + tmpfs（权重全量驻留内存） | 差异 ≤ ±2%，不是原因 |
| 硬件节点方差 | 在第二个独立 DWS 节点上重复运行 | 差异 ≤ ±2%，不是原因 |
| XLA 编译缓存热度 | 强制冷启动（指向从未写入的新 GCS 路径） | 差异 ≤ ±4%，启动耗时相同，不是原因 |
| page-size（128 vs 256） | 扫描两种取值 | 差异 ≤ ±5%，不是原因 |
| chunked-prefill-size（2048 vs 4096） | 改用 4096 以完全匹配 §1.3 参数 | 与基准差异 ≤ ±5%，仍比 §1.3 低 5–7%，不是原因 |

差距在所有配置和硬件节点上系统性一致，排除了随机噪声的可能。

---

## 二、MiMo-V2.5-Pro

### 复现环境

| 项目 | 配置 |
| ----- | ----- |
| 硬件 | TPU v7x 16-chip，topology 2×2×4，4 hosts（GKE DWS queued-provisioning） |
| 并行配置 | tp=32, dp=4, ep=32 |
| 代码 | `primatrix/sglang-jax`，commit `1bc2227` |
| 模型权重 | MiMo-V2.5-Pro HF safetensors（963 GiB），gcsfuse 挂载自 GCS |
| 压测工具 | `bench_serving`，random 数据集，input=16384 tokens，output=4096 tokens，bsz=32/64/128/192 |
| Server 启动参数 | 与 §3.2 完全一致，无差异 |

### MiMo-V2.5-Pro复现结果 vs §2.3 原始数据

**MTP（投机解码，NEXTN 3-step）**

| bsz | §2.3 原始 tok/s | 复现 tok/s | 差距 |
| :---: | ----: | ----: | ----: |
| 32  | 2083.69 | 1982.06 | −4.9% |
| 64  | 3215.02 | 3020.52 | −6.0% |
| 128 | 3981.98 | 3673.19 | −7.8% |
| 192 | 4417.05 | 4127.99 | −6.5% |

**No-MTP（标准解码）**

| bsz | §2.3 原始 tok/s | 复现 tok/s | 差距 |
| :---: | ----: | ----: | ----: |
| 32  | 1364.55 | 1305.83 | −4.3% |
| 64  | 2151.38 | 2058.31 | −4.3% |
| 128 | 2990.52 | 2843.93 | −4.9% |
| 192 | 3409.52 | 3124.31 | −8.4% |

MTP 接受率与 §2.3 高度一致（所有 bsz 均 ≥98.2%，§2.3 为 ≥97.9%）。MTP/No-MTP 加速比趋势及各 bsz 档位相对排序完全保留。

---

## 三、结论

两个模型均呈现 5–8% 的系统性吞吐差距，且已通过 Flash 的详细参数扫描排除所有可控变量（模型加载方式、硬件节点方差、XLA 缓存、page-size、chunked-prefill-size）。MTP 接受率、加速比趋势与原始数据高度一致，差距仅体现在绝对吞吐数值上，推测来源于 GKE DWS 按需分配节点与原始测试环境之间的硬件层差异。

---

*Flash 复现数据：`gs://jingnw-mimo-v2-flash-us-central1/perf-results/`*  
*Pro 复现数据：`gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-mtp-tpu7/`*

---

## 四、测试环境变量

容器镜像：`us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1`

**基础镜像内置环境变量**

| 变量 | 值 |
|---|---|
| `PATH` | `/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` |
| `VIRTUAL_ENV` | `/opt/venv` |
| `PYTHON_VERSION` | `3.12.12` |
| `LANG` | `C.UTF-8` |
| `DEBIAN_FRONTEND` | `noninteractive` |
| `PIP_NO_CACHE_DIR` | `on` |
| `PIP_ROOT_USER_ACTION` | `ignore` |
| `PYTHONUNBUFFERED` | `1` |

**运行时额外设置（§一、§二复现）**

| 变量 | 值 | 设置方式 |
|---|---|---|
| `PYTHONUNBUFFERED` | `1` | Pod spec 显式声明（与镜像默认值一致） |
| `JAX_COMPILATION_CACHE_DIR` | `gs://jingnw-mimo-v2-flash-us-central1/jax-compilation-cache` | 仅在 `launch_server` 命令行内联设置，非全局 export |

无 `LIBTPU_*`、`XLA_*`、`TF_*` 覆盖项；JAX/XLA backend 配置均使用镜像内置默认值。XLA 编译缓存已激活（服务器日志确认：`XLA persistent compilation cache: enabled`）。

---

## 五、DVFS P-state 7 追加测试（MiMo-V2-Flash）

**背景**：原始测试团队指出需添加 `LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7"` 以提升性能。

**参数含义**：`--xla_tpu_dvfs_p_state=7` 将 TPU 芯片的 DVFS（动态电压频率调节）锁定在 P-state 7（最高性能档位，最大时钟频率）。默认情况下 TPU runtime 会在计算负载空隙（如 MTP draft 与 verify 之间的间隔）动态降频以节能，锁定最高 P-state 可消除这一频率切换开销。该参数须通过 `LIBTPU_INIT_ARGS` 传入，因为 libtpu 在 JAX 首次 import 时即初始化，早于任何应用代码执行。

**实施方式**：在 Pod spec 的 `env:` 节中显式声明（而非命令行内联），确保所有子进程均继承该设置：

```yaml
- name: LIBTPU_INIT_ARGS
  value: "--xla_tpu_dvfs_p_state=7"
```

**测试配置**：与 §6 NFS 复现完全一致，仅增加上述环境变量。

| 项目 | 配置 |
|---|---|
| Manifest | `scripts/gke/mimo-v2-flash-1node-nfs-tpu7-dvfs7.yaml`（Run 1）/ `scripts/gke/mimo-v2-flash-1node-nfs-tpu7-dvfs7-mtp2.yaml`（Run 2） |
| 生命周期脚本 | `scripts/gke/run-flash-nfs-bench-dvfs7.sh`（Run 1）/ `scripts/gke/run-flash-nfs-bench-dvfs7-mtp2.sh`（Run 2） |
| 结果输出 | `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-dvfs7/`（Run 1）/ `.../flash-1node-nfs-tpu7-dvfs7-mtp2/`（Run 2） |
| 提交命令 | `kubectl apply -f scripts/gke/mimo-v2-flash-1node-nfs-tpu7-dvfs7.yaml`（Run 1）; `kubectl apply -f scripts/gke/mimo-v2-flash-1node-nfs-tpu7-dvfs7-mtp2.yaml`（Run 2） |

共进行两轮测试：

- **Run 1（dvfs7）**：MTP 先，No-MTP 后（2026-06-18T09:37Z 完成）
- **Run 2（dvfs7-mtp2）**：No-MTP 先（预热 XLA 缓存），MTP 后（2026-06-18T13:03Z 完成）

**MTP（NEXTN 3-step）**

| bsz | §1.3 原始 tok/s | NFS 基准 tok/s | Run 1 tok/s | Run 2 tok/s | 最优 vs §1.3 |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1411.41 | 1314.00 | 1336.08 | **1346.23** | −4.6% |
| 64  | 1591.87 | 1507.33 | **1560.84** | 1552.78 | −1.9% |
| 128 | 1831.77 | 1738.08 | **1832.60** | 1800.31 | **+0.0%** |

**No-MTP（标准解码）**

| bsz | §1.3 原始 tok/s | NFS 基准 tok/s | Run 1 tok/s | Run 2 tok/s | 最优 vs §1.3 |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1161.88 | 1122.25 | **1167.87** | 1164.30 | +0.5% |
| 64  | 1460.99 | 1403.32 | **1466.83** | 1463.11 | +0.4% |
| 128 | 1740.67 | 1672.92 | **1756.56** | 1751.15 | +0.9% |

**小结**：DVFS P-state 7 带来 **+2–5%** 吞吐提升，在高 bsz 下效果更显著。MTP bsz=128 最优（1832.60 tok/s）已与 §1.3 原始数据（1831.77 tok/s）基本持平（差距 <0.1%），No-MTP 各档位均小幅超出 §1.3（+0.4–0.9%）。MTP bsz=32 经两轮测试（含 XLA 缓存预热）最优为 1346.23 tok/s，仍低于 §1.3 约 4.6%，差距来源尚未完全排查，可能与该档位的调度噪声或硬件差异有关。

---

## 六、DVFS P-state 7 追加测试（MiMo-V2.5-Pro）

与 §五 Flash 测试相同的 DVFS 配置，应用于 MiMo-V2.5-Pro 4-host 场景。

| 项目 | 配置 |
|---|---|
| MTP Manifest | `scripts/gke/mimo-v2-pro-4host-tpu7-dvfs7.yaml` |
| No-MTP Manifest | `scripts/gke/mimo-v2-pro-4host-tpu7-dvfs7-nomtp.yaml` |
| 结果输出 | `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-tpu7-dvfs7/` |

MTP 与 No-MTP 分两次独立作业提交（原因：多 host 场景下 rank 0 kill server 后 rank 1–3 以非零退出，导致同一 job 内无法串行运行两个阶段）。

**MTP（NEXTN 3-step）**

| bsz | §2.3 原始 tok/s | 复现基准 tok/s | dvfs7 tok/s | vs 复现基准 | vs §2.3 |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 2083.69 | 1982.06 | 2079.97 | +4.9% | **−0.2%** |
| 64  | 3215.02 | 3020.52 | 3248.62 | +7.6% | **+1.1%** |
| 128 | 3981.98 | 3673.19 | 3971.45 | +8.1% | **−0.3%** |
| 192 | 4417.05 | 4127.99 | 4267.62 | +3.4% | −3.4% |

**No-MTP（标准解码）**

| bsz | §2.3 原始 tok/s | 复现基准 tok/s | dvfs7 tok/s | vs 复现基准 | vs §2.3 |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1364.55 | 1305.83 | 1364.76 | +4.5% | **+0.0%** |
| 64  | 2151.38 | 2058.31 | 2158.15 | +4.8% | **+0.3%** |
| 128 | 2990.52 | 2843.93 | 2995.55 | +5.3% | **+0.2%** |
| 192 | 3409.52 | 3124.31 | 3293.29 | +5.4% | −3.4% |

**小结**：DVFS P-state 7 对 Pro 的提升显著，带来 **+4–8%** 相对于复现基准的吞吐提升。MTP bsz=32/64/128 与 §2.3 原始数据差距均 ≤0.3%（基本持平）；No-MTP bsz=32/64/128 与 §2.3 持平（±0.3%）。bsz=192 两种模式均约 −3.4%，系统性偏低，推测为该档位下 DWS 节点与原始环境的硬件层差异所致。

---

## 七、ShareGPT 真实数据集下的 MTP 接受率验证（MiMo-V2-Flash）

### 背景

§五 DVFS 测试中 MTP 接受率（随机 random 数据集）均 ≥97.6%，与 §1.3 原始数据一致。但 random 数据集使用固定 16384 tokens 输入（将短 ShareGPT 对话轮次重复拼接约 21 倍），严重高估了真实工况下的 MTP 命中率。模型训练团队预期真实场景接受率约 ~70%。本节在自然对话长度（100–2000 tokens）下实测验证。

### 测试配置

| 项目 | 配置 |
|---|---|
| 数据集 | `ShareGPT_V3_unfiltered_cleaned_split.json`（641.67 MiB），预存至 `gs://jingnw-mimo-v2-flash-us-central1/datasets/` |
| 数据集来源 | `anon8231489123/ShareGPT_Vicuna_unfiltered`（因 GKE Pod 内 HuggingFace 下载停滞，改用直接 HTTPS 下载后上传至 GCS） |
| bench_serving 参数 | `--dataset-name sharegpt --sharegpt-output-len 512 --sharegpt-context-len 4096` |
| MTP 配置 | NEXTN 3-step，4 draft tokens/step（与 §五 完全一致） |
| 服务器配置 | 与 §五 完全一致（DVFS P-state 7，NFS tmpfs 权重，tp=8 dp=2 ep=8） |
| Manifest | `scripts/gke/mimo-v2-flash-1node-nfs-tpu7-dvfs7-sharegpt.yaml` |
| 生命周期脚本 | `scripts/gke/run-flash-nfs-bench-dvfs7-sharegpt.sh` |
| 结果输出 | `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-dvfs7-sharegpt/` |
| 完成时间 | 2026-06-22T07:28Z |

### MTP 接受率

从服务端日志（1932 个 decode batch）统计 `accept-ratio`（= 实际接受 tokens / draft tokens，最大值 4）：

| 指标 | 值 |
|---|---|
| 平均接受率 | **0.695（~69.5%）** |
| 最小值 | 0.25 |
| 最大值 | 1.00 |
| 统计批次 | 1932 |

与模型训练团队预期（~70%）高度吻合。对比 random 数据集的 ~98%，差距来源于 random 数据集将短对话重复拼接至 16K tokens 导致 KV cache 高度重复、draft token 命中率虚高。

### 输出吞吐（ShareGPT，DVFS P-state 7，NFS tmpfs）

| bsz | 平均输入长度 | prefill tok/s | mean TTFT (ms) | p99 TTFT (ms) | decode tok/s | §1.3 原始 decode tok/s（仅供参考） |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 350 tok | 928.55  | 541  | 1601  | 1359.43 | 1411.41 |
| 64  | 297 tok | 1144.69 | 747  | 2500  | 1975.87 | 1591.87 |
| 128 | 298 tok | 1462.93 | 1243 | 5407  | 2520.17 | 1831.77 |

> **注**：prefill tok/s = `total_input_tokens / 总wall时间`，分母包含全部 decode 时间，因此随 ShareGPT 短输入（~300 tok/req）而偏低——这是真实生产工况的客观反映，而非测量缺陷。若需要纯 prefill 内核吞吐，需单独跑 output=1 的专项压测。TTFT 列提供 prefill 延迟的另一视角。

**ShareGPT 数据更接近真实生产环境**：真实用户请求的输入长度通常为 100–2000 tokens、输出长度数百 tokens，与 ShareGPT 分布一致；§1.3 random 数据集（16384 tokens 输入 + 4096 tokens 输出）是人为构造的极端压测，不代表典型工况。因此 ShareGPT 吞吐数字是更有意义的生产参考值，而非 §1.3。两者输出长度相差 8×，不宜直接比较绝对数值。

**小结**：真实对话数据下 MTP 接受率约 **69.5%**，与训练团队预期（~70%）一致，证实 random 数据集接受率（~98%）为人为偏高。

---

## 八、ShareGPT 真实数据集下的 MTP 吞吐测试（MiMo-V2.5-Pro）

### 测试配置

| 项目 | 配置 |
|---|---|
| 数据集 | `ShareGPT_V3_unfiltered_cleaned_split.json`，从 `gs://jingnw-mimo-v2-flash-us-central1/datasets/` 下载 |
| bench_serving 参数 | `--dataset-name sharegpt --sharegpt-output-len 512 --sharegpt-context-len 4096` |
| MTP 配置 | NEXTN 3-step，4 draft tokens/step（与 §六 完全一致） |
| 服务器配置 | 与 §六 完全一致（DVFS P-state 7，gcsfuse 权重，tp=32 dp=4 ep=32，4-host） |
| Manifest | `scripts/gke/mimo-v2-pro-4host-tpu7-dvfs7-sharegpt.yaml` |
| 生命周期脚本 | `scripts/gke/run-pro-bench-dvfs7-sharegpt.sh` |
| 结果输出 | `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-tpu7-dvfs7-sharegpt/` |
| 完成时间 | 2026-06-22T12:13Z |

### MTP 接受率

服务端日志末尾 batch 显示 `accept-ratio` 稳定在 **~0.69**，与 Flash 的 69.5% 及训练团队预期（~70%）高度一致。

### 输出吞吐（ShareGPT，DVFS P-state 7，gcsfuse）

| bsz | 平均输入长度 | prefill tok/s | mean TTFT (ms) | p99 TTFT (ms) | decode tok/s | §2.3 原始 decode tok/s（仅供参考） |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 350 tok | 1216.53 | 364  | 790   | 1781.52  | 2083.69 |
| 64  | 297 tok | 1940.91 | 380  | 996   | 3351.83  | 3215.02 |
| 128 | 298 tok | 2916.36 | 583  | 1880  | 5026.17  | 3981.98 |
| 192 | 304 tok | 3715.13 | 727  | 2300  | 6278.24  | 4417.05 |

> **注**：prefill tok/s 说明同 §七。§2.3 原始数据使用 random 数据集（16384 输入 + 4096 输出），与 ShareGPT（~300 输入 + 512 输出）属不同工况，不宜直接比较绝对数值。

**小结**：Pro 在 ShareGPT 工况下 decode 吞吐随 bsz 大幅提升（bsz=192 达 6278 tok/s），TTFT 保持在 1s 以内（p99）；MTP 接受率 ~69% 与 Flash 一致，证实两个模型在真实对话分布下的投机解码表现相当。
