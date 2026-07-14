# MiMo-V2-Flash NonPD Benchmark on TPU v6e-16: Guide & Results

**Model**: MiMo-V2-Flash FP8 (PTQ)  
**Mode**: Non-disaggregated (single server handles both prefill and decode)  
**Hardware**: TPU v6e-16 — 4 physical hosts × 4 chips/host × 1 TensorCore/chip = **16 JAX devices**  
**Date**: 2026-07-13  
**Script**: [bench_v6e16_nonpd.sh](bench_v6e16_nonpd.sh)

---

## 1. Environment Preparation

### 1.1 Create the TPU v6e-16 VM

```bash
gcloud compute tpus tpu-vm create jingnw-node \
  --zone=us-east5-b \
  --accelerator-type=v6e-16 \
  --version=v2-alpha-tpuv6e
```

Verify it is ready:

```bash
gcloud compute tpus tpu-vm describe jingnw-node --zone=us-east5-b \
  --format="value(state, acceleratorConfig.type, acceleratorConfig.topology)"
# Expected: READY  V6E  4x4
```

Get the internal IP of each worker (needed to set `--dist-init-addr`):

```bash
gcloud compute tpus tpu-vm describe jingnw-node --zone=us-east5-b \
  --format="json(networkEndpoints)" | python3 -c "
import json, sys
for ep in json.load(sys.stdin).get('networkEndpoints', []):
    print(ep['ipAddress'])
"
# 10.202.0.29   ← worker 0 (use this as WORKER0_IP in the script)
# 10.202.15.225  ← worker 1
# 10.202.15.224  ← worker 2
# 10.202.15.223  ← worker 3
```

### 1.2 Model weights

Model weights live on an NFS server at `10.128.0.34:/export/flash` (us-central1, reachable from
us-east5-b over VPC). The benchmark script mounts this automatically. If using a different model
source, update `NFS_SERVER` and the mount path in the script, or replace the NFS mount section
with a GCS copy:

```bash
# Alternative: copy weights from GCS
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command="gsutil -m cp -r gs://<bucket>/mimo-v2-flash-fp8/ /tmp/flash-model/"
```

### 1.3 GCS bucket for results and coordination

The script uses GCS for two purposes:

| GCS path | Purpose |
|---|---|
| `gs://<bucket>/v6e16-barrier-w{0..3}` | 4-worker readiness barrier (temporary) |
| `gs://<bucket>/v6e16-done` | Completion signal from worker 0 to workers 1-3 (temporary) |
| `gs://<bucket>/perf-results/flash-v6e16-nonpd/` | Benchmark output (persistent) |
| `gs://<bucket>/jax-compilation-cache/` | XLA compilation cache (persistent, speeds up reruns) |

Make sure the TPU VM service account has `roles/storage.objectAdmin` on the bucket.

### 1.4 Script configuration

Edit [bench_v6e16_nonpd.sh](bench_v6e16_nonpd.sh) and update these variables at the top:

```bash
RESULTS_BUCKET="gs://jingnw-mimo-v2-flash-us-central1"   # your GCS bucket
NFS_SERVER="10.128.0.34"                                  # NFS server IP
WORKER0_IP="10.202.0.29"                                 # internal IP of worker 0
```

**Important**: `WORKER0_IP` must match what `gcloud … describe` returns for worker index 0.
Worker 0 acts as the JAX distributed coordinator; all other workers connect to it at startup.

### 1.5 Python environment note

The v6e TPU VMs ship with Python 3.10 (Ubuntu 22.04). `sglang-jax` requires Python ≥ 3.12.
The script installs **Miniconda** into `/tmp/miniconda3` on first run (~30 s) and uses that
Python 3.12 for all subsequent operations. The `/tmp` mount is per-VM-instance, so Miniconda
persists across SSH sessions but is lost on VM restart.

---

## 2. Running the Benchmark

### Step 1 — Upload script to GCS

```bash
gsutil cp scripts/tpu-vm/bench_v6e16_nonpd.sh \
  gs://jingnw-mimo-v2-flash-us-central1/scripts/bench_v6e16_nonpd.sh
```

### Step 2 — Clear any stale GCS coordination flags

If re-running after a previous attempt, remove leftover barrier/done flags:

```bash
gsutil -m rm -f \
  gs://jingnw-mimo-v2-flash-us-central1/v6e16-barrier-w{0,1,2,3} \
  gs://jingnw-mimo-v2-flash-us-central1/v6e16-done
```

### Step 3 — Create the launcher script

The launcher is a thin wrapper that downloads the latest bench script from GCS and runs it
detached (so it survives SSH disconnect):

```bash
cat > /tmp/v6e16_launcher.sh << 'EOF'
#!/bin/bash
set -euo pipefail
WORKER_ID=$(grep 'WORKER_ID' /tmp/tpu-env | tr -d "' " | cut -d: -f2)
gsutil cp gs://jingnw-mimo-v2-flash-us-central1/scripts/bench_v6e16_nonpd.sh \
  /tmp/bench_v6e16_nonpd.sh
chmod +x /tmp/bench_v6e16_nonpd.sh
nohup bash /tmp/bench_v6e16_nonpd.sh > /tmp/bench_w${WORKER_ID}.log 2>&1 &
echo "[w${WORKER_ID}] bench started PID=$! at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
gsutil cp /tmp/v6e16_launcher.sh \
  gs://jingnw-mimo-v2-flash-us-central1/scripts/v6e16_launcher.sh
```

### Step 4 — Launch on all 4 workers simultaneously

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='gsutil cp gs://jingnw-mimo-v2-flash-us-central1/scripts/v6e16_launcher.sh \
    /tmp/v6e16_launcher.sh && bash /tmp/v6e16_launcher.sh'
```

Expected output (all 4 workers confirm start within ~2 s of each other):

```
[w0] bench started PID=22081 at 2026-07-13T03:23:32Z
[w1] bench started PID=19579 at 2026-07-13T03:23:32Z
[w2] bench started PID=18447 at 2026-07-13T03:23:31Z
[w3] bench started PID=25449 at 2026-07-13T03:23:31Z
```

### Step 5 — Monitor progress

Tail the bench log on worker 0 (the coordinator and bench driver):

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
  --command="tail -f /tmp/bench_w0.log"
```

Tail the server log on worker 0 for detailed JAX/model progress:

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
  --command="tail -f /tmp/server_w0.log"
```

Check GCS for completed result files:

```bash
gsutil ls gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-v6e16-nonpd/
```

---

## 3. What Happens During the Run

### Phase 0 — Worker rank detection (all workers, ~0 s)

Each worker reads its rank from `/tmp/tpu-env`, a metadata file the TPU runtime writes at VM
creation:

```
WORKER_ID: '0'
ACCELERATOR_TYPE: 'v6e-16'
TOPOLOGY: '4x4'
```

### Phase 1 — Code install (~5 s on subsequent runs, ~35 s on first run)

Each worker independently:

1. Clones `geeningwang/sglang-jax` branch `mimo-tpu7-stage3` into `/tmp/workspace`
2. Checks whether `/tmp/miniconda3/bin/python3.12` exists; installs Miniconda if not
3. Installs `sglang-jax` and dependencies via `uv pip install --system -e "python[all]"`

### Phase 2 — NFS mount (~5 s)

Each worker mounts the model weights via NFS:

```
NFS: 10.128.0.34:/export/flash → /tmp/flash-model
Options: nfsvers=3, nolock, rsize/wsize=1M, hard, intr, timeo=600
```

After mount, the script verifies that ≥ 100 `.safetensors` files are present (the model has 145).

### Phase 3 — GCS barrier (~5 s)

All 4 workers must be ready before JAX initializes — if one worker enters `jax.distributed.initialize()`
while others are still installing packages, the coordinator handshake times out.

Each worker writes a flag to GCS (`v6e16-barrier-w{N}`) then polls until all 4 flags exist:

```
w0: writes gs://.../v6e16-barrier-w0
w1: writes gs://.../v6e16-barrier-w1
w2: writes gs://.../v6e16-barrier-w2
w3: writes gs://.../v6e16-barrier-w3
All workers: poll until 4/4 present, then proceed simultaneously
```

### Phase 4 — Server launch and JAX distributed init (~1 min)

All 4 workers start `sgl_jax.launch_server` at the same time with:

```
--nnodes 4  --node-rank {0..3}  --dist-init-addr 10.202.0.29:8088
```

Internally, the server calls `jax.distributed.initialize(coordinator_address="10.202.0.29:8088")`.
Worker 0 is the coordinator (publisher); workers 1-3 connect as subscribers:

```
[NP0] [Publisher 0] Begins to synchronize, wait 3 Subscribers
[NP0] [Publisher 0] receives 1 READY signal
[NP0] [Publisher 0] receives 2 READY signal
[NP0] [Publisher 0] receives 3 READY signal
[NP0] [Publisher 0] Succeeds to synchronize!
```

After synchronization, JAX sees all 16 devices across 4 hosts as one flat device array.
The mesh is shaped `(1, 16)` for `(data, tensor)` axes → TP=16, DP=1.

### Phase 5 — Weight loading (~20 min from NFS)

Weights load in two stages:

**Regular weights** (574 tensors, ~5 min):  
Attention projections, embedding, layer norms, FP8 scale factors for attention.  
These are small and load at ~1.5 it/s.

**MoE weights** (282 groups, ~15 min):  
Expert matrices w1/w2/w3 and their block-quantization scale factors.  
Each group has shape `(256, 4096, 2048)` at FP8 (~2 GB/group before EP sharding).  
With EP=16, each worker holds 256/16 = 16 experts locally.  
Loads at ~3.5 s/group due to NFS bandwidth and on-chip transpose.

```
Loading Regular Weights: 100%|██████| 574/574 [05:00]
Loading MoE Weights:     100%|██████| 282/282 [15:37]
```

Memory profiling is logged after all weights are placed:

```
TPU Memory profiling: available_kv_cache=7.1GB, max_tokens=155222, cell_size=49152 bytes
```

This means each 4-chip group (one worker) contributes 7.1 GB to the KV cache pool, totaling
~28 GB across all 16 chips. The memory split is roughly:

| Component | Estimate (all 16 chips) |
|---|---|
| Model weights (FP8, EP=16 sharded) | ~290 GB |
| XLA workspace / activation buffers | ~112 GB |
| KV cache | ~28 GB |
| **Total static (84% of 512 GB)** | **~430 GB** |

### Phase 6 — XLA precompilation (~2 min)

The scheduler runs precompile passes for all declared batch-size/token-length combinations
before opening for traffic:

```
[Scheduler] Begins to run worker precompile.
[EXTEND] Begin to precompile bs_paddings=[128] token_paddings=[2048]
[DECODE] Begin to precompile bs_paddings=[32, 64, 128]
[Scheduler] Completes worker precompile.
INFO: Application startup complete.   ← server is now healthy
```

Compiled artifacts are cached to `gs://<bucket>/jax-compilation-cache/` via
`JAX_COMPILATION_CACHE_DIR`. Subsequent runs with the same shapes skip compilation and
load from cache, reducing startup from ~2 min to seconds.

**Total time from script start to server ready: ~28 minutes.**

### Phase 7 — Benchmark (worker 0 only, ~90 min for bsz 32+64+128)

Worker 0 polls `/health` every 5 s. Once healthy, it drives three `bench_serving` runs:

```
python3.12 -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --base-url http://127.0.0.1:30271 \
  --model /tmp/flash-model \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 4096 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --seed 12345 \
  --max-concurrency <bsz> \
  --num-prompts <bsz*3>
```

Workers 1-3 keep their servers running but do no work — they simply poll GCS for the done flag.

The first request in each run triggers an additional JIT compilation (first real-data trace)
adding ~1 min 48 s of latency to that single request. Subsequent requests use compiled kernels.

Results for each `bsz` are uploaded to:
```
gs://<bucket>/perf-results/flash-v6e16-nonpd/bs{32,64,128}/result.jsonl
gs://<bucket>/perf-results/flash-v6e16-nonpd/bs{32,64,128}/bench.log
```

After all three runs, worker 0 uploads its server log and writes the done flag, which causes
workers 1-3 to upload their server logs and exit cleanly.

---

## 4. Benchmark Results

### Configuration

| Parameter | Value |
|---|---|
| TP size | 16 |
| DP size | 1 |
| EP size | 16 |
| dtype | bfloat16 |
| mem-fraction-static | 0.84 |
| context-length | 262144 |
| chunked-prefill-size | 2048 |
| max-prefill-tokens | 16384 |
| page-size | 256 |
| moe-backend | fused_v2 |
| Workload | 16384 input → 4096 output tokens |

### Results

| Metric | bsz=32 | bsz=64 | bsz=128 |
|---|---|---|---|
| Request throughput (req/s) | 0.20 | 0.21 | 0.21 |
| **Output tok/s** | **818** | **848** | **864** |
| Peak output tok/s | 1,326 | 1,326 | 1,352 |
| Input tok/s | 3,275 | 3,393 | 3,457 |
| Total tok/s | 4,093 | 4,242 | 4,321 |
| Effective concurrency | 28.7 | 56.3 | 111.0 |
| Mean TTFT (ms) | 36,409 | 147,106 | 363,936 |
| Median TTFT (ms) | 23,338 | 139,551 | 426,330 |
| P99 TTFT (ms) | 115,890 | 340,240 | 587,124 |
| **Median ITL (ms)** | **19.13** | **19.19** | **19.15** |
| P99 ITL (ms) | 20.51 | 20.40 | 20.43 |

### Analysis

**Decode throughput** is stable and predictable: median ITL holds at ~19 ms across all three
batch sizes, corresponding to ~52 tok/s per concurrent request. The v6e-16 system is not
decode-bottlenecked even at bsz=128.

**Output throughput** saturates around 864 tok/s at bsz=128. Peak decode-only throughput
(all requests decoding simultaneously, no prefill) reaches 1,352 tok/s.

**TTFT degrades sharply** with batch size because prefill and decode compete for the same chips.
Each 16384-token request requires 8 chunked-prefill steps (16384 ÷ 2048 chunk size), and at
bsz=128 the queue is 128 × 8 = 1024 steps deep. Disaggregated prefill/decode (1P1D) would
eliminate this contention and reduce TTFT to approximately one prefill-server pass.

**MoE MFU** is logged by the server at 3–6% per chip during precompile, which is expected for
memory-bandwidth-bound sparse MoE inference at these batch sizes. Higher batch sizes increase
MFU toward the arithmetic-intensive regime.
