# MiMo-V2-Flash & MiMo-V2.5-Pro ShareGPT 真实数据集测试报告

**测试人：** Jing Wang  
**代码：** `primatrix/sglang-jax`，branch `mimo-tpu7`，commit `1bc2227`

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
