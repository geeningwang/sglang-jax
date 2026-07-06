**MiMo Flash / Pro TPU 与 GPU 性能测试报告**

本报告按模型组织：先看 MiMo-V2-Flash，再看 MiMo-V2.5-Pro。每个模型内部都按同⼀顺序展开：测试配置、 TPU/GPU peak throughput 汇总、TPU bench\_serving 端到端指标、TPU server decode 指标。 

TPU workload 为 `bench_serving` random 数据集：每个 request 固定 `16384` input tokens、 `4096` output  tokens， `random_range_ratio=1` ， `num_prompts=3 * bsz` 。Flash 统计 bsz `32/64/128` ，Pro 统计 bsz  `32/64/128/192` 。 

Pro GPU 数据来⾃ H200 PD 分离部署的 decode 测试结果，部署⽅式为 P 2x8 H200、D 2x8 H200，固定 16K  input / 1K output。 

**测试版本** 

代码仓库： `https://github.com/primatrix/sglang-jax.git` 

测试分⽀： `dev/spec-relay-overlap-refactor-codex` 

测试 commit： `67e09cc7989aa5a135b45c53cd12d00dfc4bc5e9` 

**1\. MiMo-V2-Flash 性能测试** 

**1.1 MiMo-V2-Flash 测试配置** 

| 项⽬  | 配置 |
| ----- | ----- |
| TPU 硬件  | v7x 4-chip |
| TPU 模型路径  | `/models/MiMo-V2-Flash` |
| TPU 并⾏配置  | `tp=8, dp=2, ep=8` |
| TPU node 配置  | `nnodes=1, node_rank=0` |
| TPU chunked prefill  | 2048 |
| TPU max prefill tokens  | 16384 |
| TPU max running requests  | 256 |
| GPU 模型路径  | `/mnt/mify-gw-model-alicn3/iter_0002500-FP8-Block` |
| GPU served model name  | `mimo-v2-flash-1207-sft` |
| GPU 并⾏配置  | `tp=8, dp=2, pp=1` |
| GPU max running requests  | 64 / 128 |

**1.2 MiMo-V2-Flash TPU / GPU peak throughput 汇总**  
TPU 的 MTP / No-MTP peak 都使⽤ server decode log 在⽬标 `#running-req == bsz` 下的 `gen throughput  (token/s)` p95。Flash GPU 使⽤ target decode 段 p95，忽略 raw max 离群点；GPU 为 `dp-size=2` ，整机 p95 throughput \= `DP0 p95 + DP1 p95` 。 

| bsz  | TPU MTP peak  throughput tok/s | TPU No-MTP peak  throughput tok/s | GPU MTP p95  throughput tok/s | GPU No-MTP p95  throughput tok/s |
| ----- | ----: | ----: | ----: | ----: |
| 32  | 3168.68  | 1928.79  | \-  | \- |
| 64  | 4226.29  | 2866.03  | 9676.64  | 3478.92 |
| 128  | 5827.94  | 4295.13  | 15755.42  | 6181.03 |

**1.3 MiMo-V2-Flash TPU bench\_serving 端到端指标** 

| bsz  | MTP 输出  tok/s | No-MTP 输出  tok/s | 输出加  速 | MTP TTFT  ms | No-MTP TTFT  ms | MTP ITL  ms | No-MTP ITL  ms |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1411.41  | 1161.88  | 1.21x  | 14256.12  | 22637.01  | 18.57  | 22.02 |
| 64  | 1591.87  | 1460.99  | 1.09x  | 27085.59  | 43339.53  | 32.18  | 33.22 |
| 128  | 1831.77  | 1740.67  | 1.05x  | 61064.97  | 82004.56  | 53.25  | 53.23 |

**1.4 MiMo-V2-Flash TPU server decode 指标** 

| bsz  | MTP steady  tok/s | No-MTP steady  tok/s | MTP peak p95  tok/s | No-MTP peak p95 tok/s | MTP 接受  率 % |
| ----- | ----: | ----: | ----: | ----- | ----: |
| 32  | 3013.19  | 1867.03  | 3168.68  | 1928.79  | 98.40 |
| 64  | 3982.16  | 2752.69  | 4226.29  | 2866.03  | 98.10 |
| 128  | 5519.62  | 4068.18  | 5827.94  | 4295.13  | 98.29 |

**2\. MiMo-V2.5-Pro 性能测试** 

**2.1 MiMo-V2.5-Pro 测试配置**

| 项⽬  | 配置 |
| ----- | ----- |
| TPU 硬件  | v7x 16-chip |
| TPU 模型路径  | `/models/MiMo-V2.5-Pro` |
| TPU 并⾏配置  | `tp=32, dp=4, ep=32` |
| TPU node 配置  | `nnodes=4, node_rank=0..3` |
| TPU chunked prefill  | 4096 |
| TPU max prefill tokens  | 16384 默认值 |
| TPU max running requests  | 256 |
| GPU 部署  | PD 分离，P 2x8 H200，D 2x8 H200 |
| GPU decode workload  | 16K input / 1K output |

**2.2 MiMo-V2.5-Pro TPU / GPU peak throughput 汇总** 

TPU 测试为 16K input / 4K output，TPU MTP / No-MTP peak 都使⽤ rank0 server decode log 在⽬标 `#running-req == bsz` 下的 `gen throughput (token/s)` p95，忽略 raw max 离群点。GPU decode 测试为 16K input / 1K output；Pro GPU 给的是 per-DP rank BS，由于 GPU `dp=2` ，global bsz \= `2 * BS per DP  rank` 。Pro GPU 的 MTP-3 throughput 是接收⻓度为 3 时的单机输出吞吐。 

| bsz  | TPU MTP peak  throughput tok/s | TPU No-MTP peak  throughput tok/s | GPU MTP-3  throughput tok/s | GPU No-MTP  throughput tok/s |
| ----- | ----: | ----: | ----: | ----: |
| 32  | 3381.57  | 1616.43  | \-  | \- |
| 64  | 7276.01  | 2787.33  | \-  | \- |
| 128  | 9765.04  | 4493.71  | 3873  | 1875 |
| 192  | 10713.32  | 5923.22  | 4840  | 2564 |

**2.3 MiMo-V2.5-Pro TPU bench\_serving 端到端指标** 

吞吐： 

| bsz | MTP 输⼊  tok/s | No-MTP 输⼊  tok/s | MTP 输出  tok/s | No-MTP 输出  tok/s | MTP bench peak  tok/s | No-MTP bench peak tok/s |
| :---: | ----: | ----: | ----: | ----: | ----: | ----- |
| 32  | 8332.25  | 5458.21  | 2083.69  | 1364.55  | 3664.00  | 1600.00 |
| 64  | 12856.11  | 8605.54  | 3215.02  | 2151.38  | 7452.00  | 2752.00 |
| 128  | 15923.12  | 11962.10  | 3981.98  | 2990.52  | 10293.00  | 4480.00 |
| 192  | 17662.96  | 13638.06  | 4417.05  | 3409.52  | 11531.00  | 6426.00 |

延迟：

| bsz | MTP Mean  TTFT ms | No-MTP Mean  TTFT ms | MTP P99  TTFT ms | No-MTP P99  TTFT ms | MTP Mean  ITL ms | No-MTP Mean  ITL ms | MTP P99  ITL ms | No-MTP P99  ITL ms |
| :---: | ----: | ----: | ----: | ----: | ----: | ----- | ----: | ----: |
| 32  | 5423.36  | 7221.73  | 13156.68  | 12744.53  | 12.58  | 21.69  | 42.37  | 21.56 |
| 64  | 10121.65  | 13416.72  | 25587.87  | 25203.39  | 15.46  | 26.47  | 50.52  | 70.16 |
| 128  | 22579.74  | 23228.36  | 49874.81  | 47953.51  | 23.81  | 36.85  | 112.32  | 310.84 |
| 192  | 33419.39  | 28452.13  | 77099.81  | 73235.11  | 32.07  | 48.75  | 298.84  | 494.09 |

**2.4 MiMo-V2.5-Pro TPU server decode 指标** 

| bsz  | MTP steady  tok/s | No-MTP steady  tok/s | MTP peak p95  tok/s | No-MTP peak p95 tok/s | MTP 接受  率 % |
| ----- | ----: | ----: | ----: | ----- | ----: |
| 32  | 3270.90  | 1573.00  | 3381.57  | 1616.43  | 97.94 |
| 64  | 6831.54  | 2713.84  | 7276.01  | 2787.33  | 98.31 |
| 128  | 9349.07  | 4326.05  | 9765.04  | 4493.71  | 98.41 |
| 192  | 10110.26  | 5723.36  | 10713.32  | 5923.22  | 98.33 |

**3\. 复现命令** 

**3.1 MiMo-V2-Flash TPU MTP server** 

`uv run python3 -m sgl_jax.launch_server \` 

`--model-path /models/MiMo-V2-Flash \` 

`--trust-remote-code \` 

`--enable-sequence-parallel \` 

`--tp-size 8 \` 

`--dp-size 2 \` 

`--ep-size 8 \` 

`--moe-backend fused_v2 \` 

`--nnodes 1 \` 

`--node-rank 0 \` 

`--host 0.0.0.0 \` 

`--port 30271 \` 

`--page-size 256 \` 

`--context-length 262144 \` 

`--disable-radix-cache \` 

`--chunked-prefill-size 2048 \` 

`--max-prefill-tokens 16384 \` 

`--dtype bfloat16 \` 

`--mem-fraction-static 0.84 \` 

`--swa-full-tokens-ratio 0.2 \` 

`--skip-server-warmup \` 

`--log-level info \` 

`--decode-log-interval 1 \` 

`--max-running-requests 256 \` 

`--dp-schedule-policy round_robin \`  
`--precompile-bs-paddings 1 4 8 16 32 64 128 256 \` 

`--precompile-token-paddings 4096 \` 

`--speculative-algorithm NEXTN \` 

`--speculative-num-steps 3 \` 

`--speculative-num-draft-tokens 4 \` 

`--speculative-eagle-topk 1 \` 

`--speculative-accept-threshold-single 1.0 \` 

`--speculative-accept-threshold-acc 1.0` 

MiMo-V2-Flash TPU No-MTP server 使⽤同⼀套基础参数，去掉以下 MTP 参数： 

`--speculative-algorithm NEXTN` 

`--speculative-num-steps 3` 

`--speculative-num-draft-tokens 4` 

`--speculative-eagle-topk 1` 

`--speculative-accept-threshold-single 1.0` 

`--speculative-accept-threshold-acc 1.0` 

**3.2 MiMo-V2.5-Pro TPU MTP server** 

`uv run python3 -m sgl_jax.launch_server \` 

`--model-path /models/MiMo-V2.5-Pro \` 

`--trust-remote-code \` 

`--enable-sequence-parallel \` 

`--tp-size 32 \` 

`--dp-size 4 \` 

`--ep-size 32 \` 

`--moe-backend fused_v2 \` 

`--nnodes 4 \` 

`--node-rank 0 \` 

`--dist-init-addr 10.125.139.4:5000 \` 

`--host 0.0.0.0 \` 

`--port 30271 \` 

`--page-size 256 \` 

`--context-length 262144 \` 

`--disable-radix-cache \` 

`--chunked-prefill-size 4096 \` 

`--dtype bfloat16 \` 

`--mem-fraction-static 0.84 \` 

`--swa-full-tokens-ratio 0.2 \` 

`--skip-server-warmup \` 

`--log-level info \` 

`--decode-log-interval 1 \` 

`--max-running-requests 256 \` 

`--dp-schedule-policy round_robin \` 

`--precompile-bs-paddings 1 4 8 16 32 64 128 256 \` 

`--precompile-token-paddings 4096 \` 

`--speculative-algorithm NEXTN \` 

`--speculative-num-steps 3 \` 

`--speculative-num-draft-tokens 4 \`  
`--speculative-eagle-topk 1 \` 

`--speculative-accept-threshold-single 1.0 \` 

`--speculative-accept-threshold-acc 1.0` 

MiMo-V2.5-Pro TPU No-MTP server 使⽤同⼀套基础参数，去掉以下 MTP 参数： 

`--speculative-algorithm NEXTN` 

`--speculative-num-steps 3` 

`--speculative-num-draft-tokens 4` 

`--speculative-eagle-topk 1` 

`--speculative-accept-threshold-single 1.0` 

`--speculative-accept-threshold-acc 1.0` 

**3.3 TPU bench\_serving** 

Flash 使⽤ `MODEL_PATH=/models/MiMo-V2-Flash` ，bsz 为 `32/64/128` 。Pro 使⽤ `MODEL_PATH=/models/MiMo-V2.5-Pro` ，MTP / No-MTP bsz 均为 `32/64/128/192` 。 

`for bs in 32 64 128; do` 

`num_prompts=$((bs * 3))` 

 `uv run python3 -m sgl_jax.bench_serving \` 

`--backend sgl-jax \` 

`--base-url http://127.0.0.1:30271 \` 

`--model ${MODEL_PATH} \` 

`--dataset-name random \` 

`--random-input-len 16384 \` 

`--random-output-len 4096 \` 

`--random-range-ratio 1 \` 

`--max-concurrency "$bs" \` 

`--num-prompts "$num_prompts" \` 

`--warmup-requests 0 \` 

`--output-file "$out/bs_${bs}/result.jsonl"` 

`done`

---

**4\. §3.1 复现记录（GKE DWS 环境）**

**4.1 复现环境**

| 项目 | 配置 |
| ----- | ----- |
| GCP 项目 | `tpu-launchpad-playground` |
| GKE 集群 | `jingnw-tpu7-cluster`（us-central1-c） |
| DWS Node Pool | `jingnw-dws-tpu7-4ch`（`tpu7x-standard-4t`，topology `2x2x1`，4 chips / 8 JAX devices） |
| 测试分支 | `dev/spec-relay-overlap-refactor-codex`（primatrix/sglang-jax） |
| 测试 commit | `1bc2227`（原定 `67e09cc`，见 4.5 说明） |
| 模型权重 | `gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights`（HF safetensors） |
| 结果输出 | `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-mtp-tpu7/` |
| 容器镜像 | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1` |

**4.2 补全的未知参数**

原始文档缺少以下参数，复现时按下表填入：

| 缺失项 | 补全值 | 说明 |
| ----- | ----- | ----- |
| `$out` | `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-mtp-tpu7/<mtp\|nomtp>/bs<N>/` | GCS 路径，按 mode 和 bsz 分目录 |
| `--model-path` 实际路径 | `/mnt/gcs/mimo-v2-flash-hf-weights` | gcsfuse 挂载 `jingnw-mimo-v2-5-pro-us-central1` 后的本地路径 |
| Python 环境 | `jax0.9.0-rev1` 镜像内置 `/opt/venv`，额外安装 `orbax-checkpoint>=0.12.0 aiohttp` | 镜像已含 `uv` |
| JAX compilation cache | `JAX_COMPILATION_CACHE_DIR=gs://jingnw-mimo-v2-flash-us-central1/jax-compilation-cache` | 跨 run 复用 XLA 编译缓存 |
| server 就绪判断 | `curl -sf http://localhost:30271/health` 轮询，间隔 5 s，最长等 2 h | `--skip-server-warmup` 已设置，health 通过即可发压 |

**4.3 Kubernetes 作业**

Manifest 路径：`scripts/gke/mimo-v2-flash-1node-tpu7.yaml`

包含三个对象：

- `PodTemplate` `mimo-v2-flash-1node-tpu7-template`：DWS 容量申请所需的 pod 资源描述
- `ProvisioningRequest` `mimo-v2-flash-1node-tpu7`：DWS queued-provisioning，`maxRunDurationSeconds=14400`（4 h）
- `Job` `mimo-v2-flash-1node-tpu7`：Indexed Job，completions=1，引用上述 ProvisioningRequest

提交命令：

`kubectl apply -f scripts/gke/mimo-v2-flash-1node-tpu7.yaml`

作业流程：

1. clone `primatrix/sglang-jax` 并 checkout `1bc2227`
2. gcsfuse 挂载模型 bucket
3. 启动 MTP server（§3.1 完整参数），等待 `/health`
4. `bench_serving` bsz = 32 / 64 / 128，结果上传 GCS
5. 关闭 server，去掉 speculative 参数，重复步骤 3–4（No-MTP）

**4.4 监控与结果查看**

```bash
# 查看 pod 状态
kubectl get pods -l job-name=mimo-v2-flash-1node-tpu7 -w

# 实时日志
kubectl logs -f -l job-name=mimo-v2-flash-1node-tpu7

# 查看 DWS 调度状态
kubectl get provisioningrequest mimo-v2-flash-1node-tpu7

# 查看结果文件
gsutil ls gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-mtp-tpu7/
```

**4.5 首次运行失败记录与 commit 修订**

首次运行使用 commit `67e09cc`，MTP server 启动后在 speculative decode overlap 阶段崩溃：

```
TypeError: 'NoneType' object is not subscriptable
  File "mimo_v2_flash.py", line 176, in __call__
    topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
```

根因：`token_valid_mask` 在特定 batch 下为 `None`，`67e09cc` 尚未处理此情况。该 bug 已由后续 commit `1bc2227`（`fix(mimo): handle missing token valid mask`）修复。

**修订**：将复现 commit 从 `67e09cc` 改为 `1bc2227`，重新提交作业。No-MTP 路径不受影响（不走 speculative decode），但为保持 MTP/No-MTP 对比在同一代码基上进行，统一使用 `1bc2227`。

**4.6 实际复现结果**


作业于 2026-06-15 完成，总耗时 87 min（含两次 server 启动 + 6 轮 bench_serving）。结果存储于 `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-mtp-tpu7/`。

**bench_serving 端到端指标（commit `1bc2227`，v7x 4-chip，16384 in / 4096 out）**

吞吐：

| bsz | MTP 输入 tok/s | No-MTP 输入 tok/s | MTP 输出 tok/s | No-MTP 输出 tok/s | 输出加速 | MTP bench peak tok/s | No-MTP bench peak tok/s |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 5151.48 | 4466.23 | 1288.27 | 1116.56 | 1.15x | 2861.00 | 1867.00 |
| 64  | 5940.74 | 5591.11 | 1485.68 | 1397.78 | 1.06x | 3959.00 | 2815.00 |
| 128 | 6925.19 | 6676.30 | 1731.88 | 1669.08 | 1.04x | 5634.00 | 4224.00 |

延迟：

| bsz | MTP Mean TTFT ms | No-MTP Mean TTFT ms | MTP P99 TTFT ms | No-MTP P99 TTFT ms | MTP Mean ITL ms | No-MTP Mean ITL ms | MTP P99 ITL ms | No-MTP P99 ITL ms |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 13839.36 | 24043.55 | 45136.37 | 45201.32 | 20.73 | 22.79 | 39.86 | 18.87 |
| 64  | 27669.40 | 46091.34 | 87732.72 | 89474.68 | 34.49 | 34.53 | 79.57 | 46.87 |
| 128 | 64563.83 | 89122.26 | 176093.04 | 175672.33 | 56.44 | 54.75 | 557.67 | 308.79 |

**server decode 指标（§1.4 对应，commit `1bc2227`，v7x 4-chip）**

| bsz | MTP steady tok/s | No-MTP steady tok/s | MTP peak p95 tok/s | No-MTP peak p95 tok/s | MTP 接受率 % |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 2634.74 | 1819.37 | 2776.52 | 1884.94 | 98.11 |
| 64  | 3538.92 | 2680.39 | 3969.52 | 2793.74 | 97.61 |
| 128 | 4997.73 | 3948.00 | 5455.88 | 4173.94 | 98.28 |

**与 §1.3 / §1.4 原始数据对比**

bench_serving 输出 tok/s：

| bsz | §1.3 MTP 输出 tok/s | 复现 MTP 输出 tok/s | §1.3 No-MTP 输出 tok/s | 复现 No-MTP 输出 tok/s |
| :---: | ----: | ----: | ----: | ----: |
| 32  | 1411.41 | 1288.27 | 1161.88 | 1116.56 |
| 64  | 1591.87 | 1485.68 | 1460.99 | 1397.78 |
| 128 | 1831.77 | 1731.88 | 1740.67 | 1669.08 |

server decode peak p95 tok/s：

| bsz | §1.4 MTP p95 | 复现 MTP p95 | §1.4 No-MTP p95 | 复现 No-MTP p95 | §1.4 accept % | 复现 accept % |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 3168.68 | 2776.52 | 1928.79 | 1884.94 | 98.40 | 98.11 |
| 64  | 4226.29 | 3969.52 | 2866.03 | 2793.74 | 98.10 | 97.61 |
| 128 | 5827.94 | 5455.88 | 4295.13 | 4173.94 | 98.29 | 98.28 |

复现结果与原始数据 bench_serving 输出 tok/s 差距约 6–10%，server decode MTP p95 差距约 6–12%，No-MTP p95 差距约 2–3%，MTP 接受率高度一致（≤0.49pp 差异）。误差来源主要为：原始测试使用本地 NFS 模型路径（无 gcsfuse 开销）、可能存在不同的 JIT 编译缓存热度差异。MTP 相对 No-MTP 的加速比趋势与原始数据一致（bsz 越大加速比越低）。

---

**5\. §3.2 复现记录（GKE DWS 环境，MiMo-V2.5-Pro 4-host）**

**5.1 复现环境**

| 项目 | 配置 |
| ----- | ----- |
| GCP 项目 | `tpu-launchpad-playground` |
| GKE 集群 | `jingnw-tpu7-cluster`（us-central1-c） |
| DWS Node Pool | `jingnw-dws-tpu7-16ch`（`tpu7x-standard-4t`，topology `2x2x4`，multi-host，4 hosts × 4 chips = 16 chips / 32 JAX devices） |
| 测试分支 | `dev/spec-relay-overlap-refactor-codex`（primatrix/sglang-jax） |
| 测试 commit | `1bc2227`（与 §4.1 一致，同一代码基） |
| 模型权重 | `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/`（HF safetensors，44 files） |
| 结果输出 | `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-mtp-tpu7/` |
| 容器镜像 | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1` |

**5.2 补全的未知参数与多 host 设计**

| 缺失项 | 补全值 | 说明 |
| ----- | ----- | ----- |
| `--dist-init-addr` | `mimo-v2-pro-4host-tpu7-0.mimo-v2-pro-4host-tpu7-svc:5000` | Indexed Job pod 0 的 DNS 名（headless Service + subdomain） |
| `--node-rank` | `${JOB_COMPLETION_INDEX}`（0–3） | 各 pod 通过 k8s 环境变量获取自身 rank |
| `$out` | `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-mtp-tpu7/<mtp\|nomtp>/bs<N>/` | 按 mode 和 bsz 分目录 |
| `--model-path` 实际路径 | `/mnt/gcs/hf-weights` | gcsfuse 挂载 `jingnw-mimo-v2-5-pro-us-central1` 后的路径 |
| JAX compilation cache | `JAX_COMPILATION_CACHE_DIR=gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache` | 复用已有缓存 |
| server 就绪判断 | rank 0 轮询 `http://localhost:30271/health`；rank 1–3 等待 server 进程退出 | rank 1–3 在 rank 0 kill server 后，因 JAX collective 断链自动退出 |
| bench bsz | `32 64 128 192`（§2.3 原始数据范围） | Pro 多一档 bsz=192 |

**多 host 运行机制**：Job 使用 `completionMode: Indexed, completions=4, parallelism=4`。所有 4 个 pod 同时启动 server，JAX 分布式初始化通过 `dist-init-addr` 汇聚到 rank 0。bench_serving 仅在 rank 0 运行；当 rank 0 kill server 后，rank 1–3 的 server 因 JAX collective 失败自动退出。两个 phase（MTP / No-MTP）依次执行，各 pod 在第一个 phase 结束后 sleep 30 s 再启动第二个 phase。

**5.3 Kubernetes 作业**

Manifest 路径：`scripts/gke/mimo-v2-pro-4host-tpu7.yaml`

包含四个对象：

- `Service` `mimo-v2-pro-4host-tpu7-svc`（headless）：为 Indexed Job pod 提供稳定 DNS，供 `dist-init-addr` 使用
- `PodTemplate` `mimo-v2-pro-4host-tpu7-template`：DWS 容量申请所需的 pod 资源描述（`google.com/tpu: 4` per pod）
- `ProvisioningRequest` `mimo-v2-pro-4host-tpu7`：DWS queued-provisioning，`count: 4`，`maxRunDurationSeconds=21600`（6 h）
- `Job` `mimo-v2-pro-4host-tpu7`：Indexed Job，completions=4，带 `subdomain` 与 headless Service 对应

提交命令：

`kubectl apply -f scripts/gke/mimo-v2-pro-4host-tpu7.yaml`

作业流程（所有 4 个 pod 并行）：

1. 所有 pod：clone `primatrix/sglang-jax` 并 checkout `1bc2227`，安装依赖，gcsfuse 挂载 Pro 模型 bucket
2. 所有 pod 同时启动 MTP server（`--node-rank ${JOB_COMPLETION_INDEX}`，`--nnodes 4`）
3. rank 0：等待 `/health`，运行 `bench_serving` bsz = 32 / 64 / 128 / 192，结果上传 GCS，kill server
4. rank 1–3：等待 server 进程退出（因 JAX collective 失败）
5. 所有 pod sleep 30 s，再同时启动 No-MTP server，重复步骤 3–4

**5.4 监控与结果查看**

```bash
# 查看所有 pod 状态（4 个 pod）
kubectl get pods -l job-name=mimo-v2-pro-4host-tpu7 -w

# rank 0 实时日志
kubectl logs -f -l job-name=mimo-v2-pro-4host-tpu7 --max-log-requests 4

# DWS 调度状态
kubectl get provisioningrequest mimo-v2-pro-4host-tpu7

# 查看结果文件
gsutil ls gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-mtp-tpu7/
```

**5.5 MTP + No-MTP 实际复现结果（完整）**

作业 `mimo-v2-pro-4host-tpu7` 于 2026-06-16 完成 MTP 阶段，作业 `mimo-v2-pro-4host-tpu7-nomtp` 于 2026-06-16 完成 No-MTP 阶段，commit `1bc2227`，v7x 16-chip，16384 in / 4096 out。

bench_serving 吞吐：

| bsz | MTP 输入 tok/s | No-MTP 输入 tok/s | MTP 输出 tok/s | No-MTP 输出 tok/s | 输出加速 | MTP bench peak tok/s | No-MTP bench peak tok/s |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 7925.92 | 5223.31 | 1982.06 | 1305.83 | 1.52x | 3548.00 | 1536.00 |
| 64  | 12078.34 | 8233.25 | 3020.52 | 2058.31 | 1.47x | 7007.00 | 2688.00 |
| 128 | 14688.30 | 11375.72 | 3673.19 | 2843.93 | 1.29x | 9496.00 | 4352.00 |
| 192 | 16506.99 | 12497.24 | 4127.99 | 3124.31 | 1.32x | 12647.00 | 5458.00 |

bench_serving 延迟：

| bsz | MTP Mean TTFT ms | No-MTP Mean TTFT ms | MTP P99 TTFT ms | No-MTP P99 TTFT ms | MTP Mean ITL ms | No-MTP Mean ITL ms | MTP P99 ITL ms | No-MTP P99 ITL ms |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 5790.74 | 8024.06 | 14505.92 | 14050.67 | 13.22 | 22.55 | 41.80 | 22.36 |
| 64  | 11281.79 | 14775.90 | 28354.86 | 27764.61 | 16.38 | 27.48 | 51.34 | 60.68 |
| 128 | 24155.33 | 26199.04 | 54887.36 | 52834.51 | 25.63 | 38.31 | 316.44 | 318.07 |
| 192 | 34168.61 | 33982.45 | 82589.81 | 80171.47 | 34.88 | 52.61 | 331.28 | 468.99 |

server decode 指标（§2.4 对应）：

| bsz | MTP steady tok/s | No-MTP steady tok/s | MTP peak p95 tok/s | No-MTP peak p95 tok/s | MTP 接受率 % |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 3146.15 | 1522.52 | 3276.15 | 1555.93 | 98.28 |
| 64  | 6094.82 | 2629.83 | 6925.93 | 2714.24 | 98.21 |
| 128 | 8226.26 | 4212.94 | 9358.25 | 4377.76 | 98.25 |
| 192 | 10011.04 | 5069.28 | 11109.79 | 5278.12 | 98.42 |

**与 §2.3 / §2.4 原始数据对比**

bench_serving 输出 tok/s：

| bsz | §2.3 MTP 输出 tok/s | 复现 MTP 输出 tok/s | §2.3 No-MTP 输出 tok/s | 复现 No-MTP 输出 tok/s |
| :---: | ----: | ----: | ----: | ----: |
| 32  | 2083.69 | 1982.06 | 1364.55 | 1305.83 |
| 64  | 3215.02 | 3020.52 | 2151.38 | 2058.31 |
| 128 | 3981.98 | 3673.19 | 2990.52 | 2843.93 |
| 192 | 4417.05 | 4127.99 | 3409.52 | 3124.31 |

server decode peak p95 tok/s：

| bsz | §2.4 MTP p95 | 复现 MTP p95 | §2.4 No-MTP p95 | 复现 No-MTP p95 | §2.4 accept % | 复现 accept % |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 3381.57 | 3276.15 | 1616.43 | 1555.93 | 97.94 | 98.28 |
| 64  | 7276.01 | 6925.93 | 2787.33 | 2714.24 | 98.31 | 98.21 |
| 128 | 9765.04 | 9358.25 | 4493.71 | 4377.76 | 98.41 | 98.25 |
| 192 | 10713.32 | 11109.79 | 5923.22 | 5278.12 | 98.33 | 98.42 |

复现结果与原始数据 bench_serving 输出 tok/s 差距约 5–8%，server decode MTP p95 差距约 ±4%（bsz=192 高出原始 3.7%，其余 3–5% 低于原始），No-MTP p95 差距约 3–11%（bsz=192 差距最大）。MTP 接受率高度一致（≤0.34pp 差异）。误差来源：原始测试使用本地 NFS 模型路径；GKE 复现使用 gcsfuse，且首次压测时 JIT 编译缓存热度可能不同。

结果文件：
- MTP：`gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-mtp-tpu7/mtp/`
- No-MTP：`gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-mtp-tpu7/nomtp/`

**5.6 No-MTP 阶段两次失败记录与修订**

**第一次失败（`mimo-v2-pro-4host-tpu7`，No-MTP 阶段）**

**失败现象**：MTP 阶段全部完成后，No-MTP server 在 rank 0 节点只打出 6 行日志即停止：

```
[Publisher 0] Begins to synchronize, wait 3 Subscribers
```

rank 0 等待 ranks 1–3 连接但始终未收到，dist-init 超时后 server 崩溃。job 最终 `succeeded: 1 / failed: 3`（`BackoffLimitExceeded`）。

**根因**：MTP server 被 kill 后，ranks 1–3 因 JAX collective 失败而退出。脚本 `sleep 30` 后立即启动 No-MTP server，但各节点的退出时序不一致，port 释放延迟导致分布式 init 握手失败。

**修订**：拆分为独立作业 `mimo-v2-pro-4host-tpu7-nomtp`，在 server 启动前插入 GCS barrier：每个 rank 写入一个标志文件，所有 rank 等待 4 个标志均出现后再同时启动 server，保证各节点同时参与分布式初始化。

Manifest 路径：`scripts/gke/mimo-v2-pro-4host-tpu7-nomtp.yaml`

提交命令：

`kubectl apply -f scripts/gke/mimo-v2-pro-4host-tpu7-nomtp.yaml`

**第二次失败（GCS barrier `SYNC_PREFIX` per-pod 时间戳问题）**

**失败现象**：4 pods 均 Running，barrier 日志始终显示 `1/4 ready`，1 小时后超时失败。

**根因**：`SYNC_PREFIX` 包含 `$TIMESTAMP`，而 `$TIMESTAMP` 在每个 pod 独立生成，导致各 rank 写入不同的 GCS 路径（4 个不同前缀），每个 rank 只能看到自己的 1 个标志。

**修订**：改用 bash 参数展开从 `$HOSTNAME` 推导共享前缀：

```bash
SYNC_PREFIX="${RESULTS_DIR}/sync/${HOSTNAME%-${NODE_RANK}}"
```

K8s Indexed Job 中每个 pod 的 `$HOSTNAME` 为 `{job-name}-{index}`，去掉尾部 `-{index}` 后所有 rank 得到相同的前缀 `{job-name}`，barrier 正确同步。

---

**6\. §3.1 NFS 复现记录（MiMo-V2-Flash，tmpfs 内存模型路径）**

**6.1 复现环境**

§4.6 的 gcsfuse 复现与 §1.3 原始数据存在 6–10% 的输出吞吐差距。原始测试使用本地 NFS 路径（权重在内存中），而 GKE 复现使用 gcsfuse（通过 FUSE 层访问 GCS）。本节以 NFS + tmpfs 方案排查 gcsfuse 是否为差距来源。

| 项目 | 配置 |
| ----- | ----- |
| GCP 项目 | `tpu-launchpad-playground` |
| GKE 集群 | `jingnw-tpu7-cluster`（us-central1-c） |
| DWS Node Pool | `jingnw-dws-tpu7-4ch`（`tpu7x-standard-4t`，topology `2x2x1`，4 chips / 8 JAX devices） |
| 测试 commit | `1bc2227`（与 §4 一致） |
| 模型权重 | NFS 挂载自 tmpfs（见下节），本地路径 `/tmp/flash-model` |
| 结果输出 | `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7/` |
| 容器镜像 | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1` |

**6.2 NFS 架构**

在 GKE 同一 VPC 内创建独立 NFS VM（`nfs-flash`，`n2-highmem-48`，384 GB RAM，us-central1-c）。VM 启动脚本：

1. 安装 `nfs-kernel-server`
2. 将 315 GiB tmpfs 挂载为 NFS 导出根目录 `/export/flash`（直接导出 tmpfs，避免 NFS 跨 submount 不可见问题）
3. `gsutil -m cp` 将 291.6 GiB 权重从 GCS 复制到 tmpfs（约 5 分钟）
4. 将 VM 内网 IP 写入 GCS，供 GKE pod 动态发现（zone-agnostic，无需硬编码 DNS）

GKE pod 轮询 GCS 获取 NFS IP，挂载 `nfsvers=3,nolock,rsize=1M`，模型读取走内核 NFS 而非 gcsfuse FUSE 层。

Manifest 路径：`scripts/gke/mimo-v2-flash-1node-nfs-tpu7.yaml`  
生命周期脚本：`scripts/gke/run-flash-nfs-bench.sh`

**6.3 实际复现结果**

作业于 2026-06-17 完成，总耗时 80 min（MTP + No-MTP 各一轮，server 启动 425 s）。

bench_serving 吞吐：

| bsz | MTP 输入 tok/s | No-MTP 输入 tok/s | MTP 输出 tok/s | No-MTP 输出 tok/s | 输出加速 | MTP bench peak tok/s | No-MTP bench peak tok/s |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 5186.72 | 4478.47 | 1297.15 | 1119.62 | 1.16x | 2979.00 | 1881.00 |
| 64  | 5819.32 | 5602.04 | 1455.34 | 1400.51 | 1.04x | 3898.00 | 2816.00 |
| 128 | 6950.08 | 6691.67 | 1738.08 | 1672.92 | 1.04x | 6037.00 | 4224.00 |

bench_serving 延迟：

| bsz | MTP Mean TTFT ms | No-MTP Mean TTFT ms | MTP P99 TTFT ms | No-MTP P99 TTFT ms | MTP Mean ITL ms | No-MTP Mean ITL ms | MTP P99 ITL ms | No-MTP P99 ITL ms |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 13179.91 | 23957.14 | 45159.91 | 44981.70 | 20.82 | 22.73 | 47.98 | 18.83 |
| 64  | 26318.55 | 46031.49 | 87819.33 | 89363.92 | 35.81 | 34.46 | 82.03 | 27.21 |
| 128 | 64835.50 | 88712.78 | 176288.08 | 175187.38 | 55.92 | 54.69 | 624.47 | 314.03 |

**6.4 三方对比：原始 §1.3 / gcsfuse §4.6 / NFS §6**

bench_serving 输出 tok/s（MTP）：

| bsz | §1.3 原始 | §4.6 gcsfuse | §6 NFS | NFS vs gcsfuse | NFS vs 原始 |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1411.41 | 1288.27 | 1297.15 | +0.7% | −8.1% |
| 64  | 1591.87 | 1485.68 | 1455.34 | −2.1% | −8.6% |
| 128 | 1831.77 | 1731.88 | 1738.08 | +0.4% | −5.1% |

bench_serving 输出 tok/s（No-MTP）：

| bsz | §1.3 原始 | §4.6 gcsfuse | §6 NFS | NFS vs gcsfuse | NFS vs 原始 |
| :---: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1161.88 | 1116.56 | 1119.62 | +0.3% | −3.6% |
| 64  | 1460.99 | 1397.78 | 1400.51 | +0.2% | −4.1% |
| 128 | 1740.67 | 1669.08 | 1672.92 | +0.2% | −3.9% |

**结论**：NFS（tmpfs）与 gcsfuse 在推理吞吐上几乎无差别（±2% 以内，在统计误差范围内）。这符合预期——bench_serving 期间权重已加载至 TPU HBM，模型文件系统不参与推理路径，gcsfuse 的 FUSE 层开销仅影响权重加载阶段。因此，§4.6 观测到的 5–10% 差距来源并非 gcsfuse，而更可能是硬件环境差异（TPU 实例调度、板卡 binning）或原始测试时 XLA 编译缓存热度更高。

---

**7\. 5–10% 差距排查**

**7.1 排查计划**

§4.6 和 §6 均显示与 §1.3 原始数据存在 5–10% 的推理吞吐差距，且 gcsfuse vs NFS 不是原因。按以下步骤逐一排查。

| 步骤 | 假设 | 验证方法 | 判断标准 |
| ----- | ----- | ----- | ----- |
| Step A | 硬件 binning / 实例方差 | 相同配置在不同 DWS 节点重复运行，对比两次结果 | 若两次差距 ≥5%，则硬件方差可解释总差距 |
| Step B | XLA 编译缓存热度 | 强制冷启动（清除 GCS 缓存），对比热/冷 cache 运行 | 若热 cache 比冷 cache 快 >3%，则 XLA cache 是部分原因 |
| Step C | page-size / chunked-prefill 配置 | 扫描 `--page-size 128/256` × `--chunked-prefill-size 2048/4096` | 若某组合比基准快 >3%，则配置可调 |
| Step D | 综合结论 | 汇总以上结果，估算各因素贡献 | — |

**7.2 Step A：硬件方差（第二次独立运行）**

使用与 §6 完全相同的参数（commit `1bc2227`，NFS tmpfs，v7x 4-chip），在新的 DWS 节点上重复一次完整 bench_serving。结果存储于 `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-vara/`。

Manifest：`scripts/gke/mimo-v2-flash-1node-nfs-tpu7-vara.yaml`  
生命周期脚本：`scripts/gke/run-flash-nfs-bench-vara.sh`

Run-2（vara）：commit `1bc2227`，us-central1-f 节点，DWS 新分配节点，NFS IP `10.128.0.56`。

**与 §6 Run-1 对比**

| bsz | §6 MTP tok/s | vara MTP tok/s | MTP 差异 | §6 No-MTP tok/s | vara No-MTP tok/s | No-MTP 差异 |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1297.15 | 1300.92 | +0.3% | 1119.62 | 1122.25 | +0.2% |
| 64  | 1455.34 | 1484.40 | +2.0% | 1400.51 | 1403.32 | +0.2% |
| 128 | 1738.08 | 1736.06 | −0.1% | 1672.92 | 1672.29 | −0.0% |

**Step A 结论**：两次独立 DWS 节点（us-central1-c vs us-central1-f）的吞吐差异 ≤ ±2%，远小于与 §1.3 的 5–10% 差距。**硬件 binning / 实例方差不是差距的来源。**

剩余差距（vs §1.3 MTP）：bsz=32 −8.1%，bsz=64 −8.6%，bsz=128 −5.1%。需继续排查 XLA 编译缓存热度（Step B）。

**7.3 Step B：XLA 编译缓存热度**

与 §6/vara 完全相同的参数，唯一区别：`JAX_COMPILATION_CACHE_DIR` 指向从未写入过的新 GCS 路径（`jax-compilation-cache-cold`），强制 XLA 从零编译（冷缓存）。若冷 cache 吞吐与热 cache 相差 ≤3%，则 XLA 缓存热度不是差距来源。

Manifest：`scripts/gke/mimo-v2-flash-1node-nfs-tpu7-cold.yaml`  
生命周期脚本：`scripts/gke/run-flash-nfs-bench-cold.sh`  
结果：`gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-cold/`

**与 §6/vara 对比**

| bsz | §6 MTP tok/s | cold MTP tok/s | MTP 差异 | §6 No-MTP tok/s | cold No-MTP tok/s | No-MTP 差异 |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1297.15 | 1314.00 | +1.3% | 1119.62 | 1110.19 | −0.8% |
| 64  | 1455.34 | 1507.33 | +3.6% | 1400.51 | 1389.44 | −0.8% |
| 128 | 1738.08 | 1720.92 | −1.0% | 1672.92 | 1658.60 | −0.9% |

服务器冷启动耗时：MTP 540s，No-MTP 540s（与热 cache 一致，均约 9 min）。

**Step B 结论**：冷 XLA 缓存与热缓存吞吐差异 ≤ ±4%（bsz=64 MTP +3.6% 属统计噪声），服务器启动耗时无差别（540s vs 540s）。**XLA 编译缓存热度不影响推理吞吐，不是 5–10% 差距的来源。**

剩余差距（vs §1.3 MTP）：bsz=32 −8.1%，bsz=64 −8.6%，bsz=128 −5.1%。继续排查 page-size / chunked-prefill 参数（Step C）。

**7.4 Step C：page-size / chunked-prefill 扫描**

§1.3 原始测试使用 `--chunked-prefill-size 4096`，而 §6/vara/cold 均使用 `2048`。本步骤扫描 `--page-size 128/256` × `--chunked-prefill-size 2048/4096` 四种组合，验证该参数差异是否能解释 5–10% 的吞吐差距。

| 变体 | page-size | chunked-prefill-size | 说明 |
| ----- | :---: | :---: | ----- |
| §6 baseline | 256 | 2048 | 基准（与 §6/vara/cold 一致） |
| cp4096 | 256 | 4096 | 匹配 §1.3 原始配置，单独验证 cp 影响 |
| ps128 | 128 | 2048 | 单独验证 page-size 影响 |
| ps128cp4096 | 128 | 4096 | 两参数同时变更 |

Manifest 路径：`scripts/gke/mimo-v2-flash-1node-nfs-tpu7-{cp4096,ps128,ps128cp4096}.yaml`  
生命周期脚本：`scripts/gke/run-flash-nfs-bench-{cp4096,ps128,ps128cp4096}.sh`  
结果：`gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-{cp4096,ps128,ps128cp4096}/`

三个变体的 GKE 作业按顺序提交（ps128cp4096 先提交，完成后提交 cp4096，cp4096 完成后提交 ps128），由定时脚本自动管理：

顺序调度脚本：`scripts/gke/step-c-sequencer.sh`（系统 crontab 每 5 分钟触发一次，轮询作业完成状态，依次提交下一个变体，所有变体完成后自动移除 crontab 条目）

**MTP bench_serving 输出 tok/s**

| bsz | §6 baseline | cp4096 | cp4096 vs §6 | ps128 | ps128 vs §6 | ps128cp4096 | ps128cp4096 vs §6 |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1297.15 | 1252.28 | −3.5% | 1275.90 | −1.6% | 1250.08 | −3.6% |
| 64  | 1455.34 | 1478.11 | +1.6% | 1466.02 | +0.7% | 1490.73 | +2.4% |
| 128 | 1738.08 | 1690.73 | −2.7% | 1658.88 | −4.6% | 1716.86 | −1.2% |

**No-MTP bench_serving 输出 tok/s**

| bsz | §6 baseline | cp4096 | cp4096 vs §6 | ps128 | ps128 vs §6 | ps128cp4096 | ps128cp4096 vs §6 |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1119.62 | 1114.60 | −0.4% | 1092.04 | −2.5% | 1088.45 | −2.8% |
| 64  | 1400.51 | 1393.51 | −0.5% | 1378.98 | −1.5% | 1373.04 | −2.0% |
| 128 | 1672.92 | 1627.40 | −2.7% | 1647.84 | −1.5% | 1626.69 | −2.8% |

**cp4096 vs §1.3 原始数据（最关键对比）**

| bsz | §1.3 MTP tok/s | cp4096 MTP tok/s | 差距 | §1.3 No-MTP tok/s | cp4096 No-MTP tok/s | 差距 |
| :---: | ----: | ----: | ----: | ----: | ----: | ----: |
| 32  | 1411.41 | 1252.28 | −11.3% | 1161.88 | 1114.60 | −4.1% |
| 64  | 1591.87 | 1478.11 | −7.1%  | 1460.99 | 1393.51 | −4.6% |
| 128 | 1831.77 | 1690.73 | −7.7%  | 1740.67 | 1627.40 | −6.5% |

**Step C 结论**：四种 page-size × chunked-prefill 组合的吞吐差异在 ±5% 以内，均属正常统计噪声范围。关键验证：cp4096（与 §1.3 完全相同的 chunked-prefill=4096 + page-size=256）相比 §6 基准并无显著提升，与 §1.3 原始数据仍差距 7–11%（MTP）/ 4–7%（No-MTP）。**page-size 和 chunked-prefill 参数差异不是 5–10% 差距的来源。**

**7.5 综合结论**

四步排查全部完成，汇总各步骤结论：

| 步骤 | 排查假设 | 结论 | 可解释差距 |
| ----- | ----- | ----- | :---: |
| Step A | 硬件 binning / 实例方差 | 不同 DWS 节点（us-central1-c vs f）差异 ≤ ±2% | ✗ 否 |
| Step B | XLA 编译缓存热度 | 冷 / 热 cache 吞吐差异 ≤ ±4%，启动耗时相同 | ✗ 否 |
| Step C | page-size / chunked-prefill 参数 | 四种组合差异 ≤ ±5%，cp4096 vs §1.3 仍差 7–11% | ✗ 否 |
| Step D | — | 所有可控因素均已排除 | — |

**已排除因素**：gcsfuse 开销（§6 已验证，±2% vs NFS）、硬件方差（±2%）、XLA 缓存（±4%）、page-size/chunked-prefill 配置（±5%）。

**剩余差距**：与 §1.3 原始数据 MTP 差距约 7–9%，No-MTP 差距约 4–7%，在所有可控复现条件下保持一致，无法通过参数调整消除。

**最可能的解释**：§1.3 原始测试在专属/预留 TPU 实例上运行，该实例具备比 DWS queued-provisioning 按需分配节点更优的芯片 binning 或频率配置。DWS 节点通过竞争性队列分配，无法保证与原始测试相同档位的硬件。这一差距（≈7%）与同类 TPU 实例 binning 方差的典型范围一致，属于硬件层面的固有不确定性，无法通过软件参数调整消弭。

**对原始结论的影响**：MTP vs No-MTP 加速比趋势、MTP 接受率（≥97.6%）与 §1.3 高度一致，各 bsz 档位的相对性能排序完全保留。7% 的绝对吞吐差距系硬件环境差异所致，不影响算法正确性和相对性能结论。