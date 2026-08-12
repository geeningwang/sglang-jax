# MiMo-V2-Flash 1P1D Benchmark on TPU v6e-16: Guide & Results

**Model**: MiMo-V2-Flash FP8 (PTQ)  
**Mode**: Disaggregated Prefill/Decode (1P1D) — two independent v6e-16 VMs  
**Hardware**: 2 × TPU v6e-16 — each 4 physical hosts × 4 chips/host = **16 JAX devices per VM**  
**Date**: 2026-07-14  
**Branch**: `mimo-tpu7-stage3` (based on `primatrix/epic/mimo-pd-disggragation`)  
**Scripts**: [bench_v6e16_1p1d_prefill.sh](bench_v6e16_1p1d_prefill.sh) · [bench_v6e16_1p1d_decode.sh](bench_v6e16_1p1d_decode.sh)  
**See also**: [NonPD baseline](GUIDE_v6e16_nonpd_benchmark.md) (single-VM, NonPD)

---

## 1. Environment Preparation

### 1.1 Create two TPU v6e-16 VMs

1P1D requires **two separate** v6e-16 VMs: one for the prefill server and one for the decode server.
Each is created identically but plays a different role.

```bash
# Prefill VM (jingnw-node)
gcloud compute tpus tpu-vm create jingnw-node \
  --zone=us-east5-b \
  --accelerator-type=v6e-16 \
  --version=v2-alpha-tpuv6e

# Decode VM (jingnw-node2)
gcloud compute tpus tpu-vm create jingnw-node2 \
  --zone=us-east5-b \
  --accelerator-type=v6e-16 \
  --version=v2-alpha-tpuv6e
```

Verify both are ready:

```bash
gcloud compute tpus tpu-vm list --zone=us-east5-b \
  --format="table(name, state, acceleratorConfig.type, acceleratorConfig.topology)"
# Expected:
# NAME          STATE  TYPE  TOPOLOGY
# jingnw-node   READY  V6E   4x4
# jingnw-node2  READY  V6E   4x4
```

### 1.2 Get worker 0 IPs (required for distributed init and bootstrap)

Each VM's worker 0 acts as the JAX coordinator. Get both IPs:

```bash
# Prefill VM — worker 0 IP
gcloud compute tpus tpu-vm describe jingnw-node --zone=us-east5-b \
  --format="json(networkEndpoints)" | python3 -c "
import json, sys
eps = json.load(sys.stdin).get('networkEndpoints', [])
print('Prefill worker IPs:')
for ep in eps: print(' ', ep['ipAddress'])
print('  → use index 0 as PREFILL_W0_IP')
"
# Example: 10.202.0.29 ← worker 0

# Decode VM — worker 0 IP
gcloud compute tpus tpu-vm describe jingnw-node2 --zone=us-east5-b \
  --format="json(networkEndpoints)" | python3 -c "
import json, sys
eps = json.load(sys.stdin).get('networkEndpoints', [])
print('Decode worker IPs:')
for ep in eps: print(' ', ep['ipAddress'])
print('  → use index 0 as DECODE_W0_IP')
"
# Example: 10.202.15.227 ← worker 0
```

Update the scripts with the correct IPs:

| Variable | Location | Description |
|---|---|---|
| `PREFILL_W0_IP` | `bench_v6e16_1p1d_prefill.sh` and `bench_v6e16_1p1d_decode.sh` | JAX coordinator for prefill VM; bootstrap server host |
| `DECODE_W0_IP` | `bench_v6e16_1p1d_decode.sh` | JAX coordinator for decode VM |

### 1.3 Model weights

Same NFS server as the NonPD benchmark. Weights live at `10.128.0.34:/export/flash`
(us-central1, reachable from us-east5-b over VPC). Both scripts mount this automatically.

### 1.4 GCS bucket for results and coordination

The scripts use GCS for inter-VM coordination and result storage:

| GCS path | Purpose |
|---|---|
| `gs://<bucket>/v6e16-1p1d-p-barrier-w{0..3}` | 4-worker readiness barrier for prefill VM |
| `gs://<bucket>/v6e16-1p1d-d-barrier-w{0..3}` | 4-worker readiness barrier for decode VM |
| `gs://<bucket>/v6e16-1p1d-bootstrap-ready` | Signal from prefill worker 0 → decode: bootstrap is up |
| `gs://<bucket>/v6e16-1p1d-done` | Signal from decode worker 0 → prefill: benchmark complete |
| `gs://<bucket>/perf-results/flash-v6e16-1p1d/` | Benchmark output (persistent) |
| `gs://<bucket>/jax-compilation-cache/` | XLA compilation cache (shared with NonPD run) |

### 1.5 Script configuration

Edit both scripts and update the IP addresses at the top:

**`bench_v6e16_1p1d_prefill.sh`**:
```bash
PREFILL_W0_IP="10.202.0.29"    # internal IP of prefill jingnw-node worker 0
```

**`bench_v6e16_1p1d_decode.sh`**:
```bash
DECODE_W0_IP="10.202.15.227"   # internal IP of decode jingnw-node2 worker 0
PREFILL_W0_IP="10.202.0.29"    # bootstrap server is on prefill worker 0
```

### 1.6 Python environment note

Same as NonPD: the v6e VMs ship with Python 3.10. Both scripts install Miniconda into
`/tmp/miniconda3` on first run, providing Python 3.12 for sglang-jax. See
[GUIDE_v6e16_nonpd_benchmark.md](GUIDE_v6e16_nonpd_benchmark.md#15-python-environment-note)
for details.

**Note**: On freshly-created or re-created v6e VMs the apt cache is stale. Both scripts
run `sudo apt-get update` before installing `nfs-common` to avoid
`"Package 'nfs-common' has no installation candidate"`.

---

## 2. Running the Benchmark

### Step 1 — Upload scripts to GCS

```bash
gsutil cp scripts/tpu-vm/bench_v6e16_1p1d_prefill.sh \
  gs://jingnw-mimo-v2-flash-us-central1/scripts/bench_v6e16_1p1d_prefill.sh
gsutil cp scripts/tpu-vm/bench_v6e16_1p1d_decode.sh \
  gs://jingnw-mimo-v2-flash-us-central1/scripts/bench_v6e16_1p1d_decode.sh
```

### Step 2 — Clear any stale GCS coordination flags

If re-running after a previous attempt, remove leftover barrier/signal flags:

```bash
gsutil -m rm -f \
  gs://jingnw-mimo-v2-flash-us-central1/v6e16-1p1d-p-barrier-w{0,1,2,3} \
  gs://jingnw-mimo-v2-flash-us-central1/v6e16-1p1d-d-barrier-w{0,1,2,3} \
  gs://jingnw-mimo-v2-flash-us-central1/v6e16-1p1d-bootstrap-ready \
  gs://jingnw-mimo-v2-flash-us-central1/v6e16-1p1d-done
```

### Step 3 — Create launcher scripts

```bash
cat > /tmp/p_launcher.sh << 'EOF'
#!/bin/bash
gsutil cp gs://jingnw-mimo-v2-flash-us-central1/scripts/bench_v6e16_1p1d_prefill.sh /tmp/bench_prefill.sh
WID=$(grep "^WORKER_ID:" /tmp/tpu-env | awk -F"'" '{print $2}')
nohup bash /tmp/bench_prefill.sh > /tmp/bench_p_w${WID}.log 2>&1 &
echo "prefill-w${WID} started PID=$! at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF

cat > /tmp/d_launcher.sh << 'EOF'
#!/bin/bash
gsutil cp gs://jingnw-mimo-v2-flash-us-central1/scripts/bench_v6e16_1p1d_decode.sh /tmp/bench_decode.sh
WID=$(grep "^WORKER_ID:" /tmp/tpu-env | awk -F"'" '{print $2}')
nohup bash /tmp/bench_decode.sh > /tmp/bench_d_w${WID}.log 2>&1 &
echo "decode-w${WID} started PID=$! at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF

gsutil cp /tmp/p_launcher.sh gs://jingnw-mimo-v2-flash-us-central1/scripts/p_launcher.sh
gsutil cp /tmp/d_launcher.sh gs://jingnw-mimo-v2-flash-us-central1/scripts/d_launcher.sh
```

### Step 4 — Launch on both VMs simultaneously

Run both commands in the **same terminal session** (background + foreground or two windows):

```bash
# Terminal A (or background):
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='gsutil cp gs://jingnw-mimo-v2-flash-us-central1/scripts/p_launcher.sh /tmp/p_launcher.sh && bash /tmp/p_launcher.sh' &

# Terminal B (or foreground):
gcloud compute tpus tpu-vm ssh jingnw-node2 --zone=us-east5-b --worker=all \
  --command='gsutil cp gs://jingnw-mimo-v2-flash-us-central1/scripts/d_launcher.sh /tmp/d_launcher.sh && bash /tmp/d_launcher.sh'

wait
```

Expected output (all 8 workers confirm start within ~2 s of each other):

```
prefill-w0 started PID=8806 at 2026-07-13T09:15:01Z
prefill-w1 started PID=11367 at 2026-07-13T09:15:01Z
prefill-w2 started PID=12421 at 2026-07-13T09:15:01Z
prefill-w3 started PID=8710 at 2026-07-13T09:15:01Z
decode-w0 started PID=8625 at 2026-07-13T09:15:01Z
decode-w1 started PID=8648 at 2026-07-13T09:15:01Z
decode-w2 started PID=11316 at 2026-07-13T09:15:01Z
decode-w3 started PID=8603 at 2026-07-13T09:15:01Z
```

### Step 5 — Monitor progress

Tail the prefill worker 0 log (coordinator for prefill VM):

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
  --command="tail -f /tmp/bench_p_w0.log"
```

Tail the decode worker 0 log (coordinator for decode VM, drives bench_serving):

```bash
gcloud compute tpus tpu-vm ssh jingnw-node2 --zone=us-east5-b --worker=0 \
  --command="tail -f /tmp/bench_d_w0.log"
```

Tail the server logs for detailed JAX/model progress:

```bash
# Prefill server log
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
  --command="tail -f /tmp/server_prefill_w0.log"

# Decode server log
gcloud compute tpus tpu-vm ssh jingnw-node2 --zone=us-east5-b --worker=0 \
  --command="tail -f /tmp/server_decode_w0.log"
```

Check GCS for completed result files:

```bash
gsutil ls gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-v6e16-1p1d/
```

---

## 3. What Happens During the Run

### Phase 0 — Worker rank detection (all 8 workers, ~0 s)

Each worker reads its rank from `/tmp/tpu-env`. Both VMs run workers 0-3 independently —
the rank is local to each VM, not global.

### Phase 1 — Code install (~35 s on new VMs, ~5 s if Miniconda cached)

Each of the 8 workers independently:

1. Clones `geeningwang/sglang-jax` branch `mimo-tpu7-stage3` into `/tmp/workspace`
2. Runs `sudo apt-get update && sudo apt-get install -y nfs-common`
3. Installs Miniconda (if not cached in `/tmp/miniconda3`)
4. Installs `sglang-jax` and dependencies via `uv pip install --system -e "python[all]"`

### Phase 2 — NFS mount (~5 s)

Both VMs mount the same NFS share independently:

```
NFS: 10.128.0.34:/export/flash → /tmp/flash-model
```

After mount, each worker verifies ≥ 100 `.safetensors` files (model has 145).

### Phase 3P — Prefill-internal barrier (~5 s)

The 4 prefill workers write `v6e16-1p1d-p-barrier-w{0..3}` to GCS and wait until all 4 are
present. This ensures all prefill workers enter JAX distributed init simultaneously.

### Phase 3D — Decode waits for bootstrap-ready

The 4 decode workers poll GCS for the `v6e16-1p1d-bootstrap-ready` flag. They do NOT
start their barrier or JAX init yet — they're waiting for the prefill side to be ready first.

### Phase 4P — Prefill worker 0: bootstrap server

Prefill worker 0 (only) starts the disaggregation bootstrap rendezvous server:

```bash
python3.12 -m sgl_jax.srt.disaggregation.run_bootstrap \
  --host 0.0.0.0 --port 8998
```

The bootstrap server coordinates KV-cache transfer metadata between prefill and decode pods —
it is the rendezvous point for the `--disaggregation-bootstrap-url` handshake. Once the port
is confirmed bound (checked via `ss -tlnp`), prefill worker 0 writes the `bootstrap-ready` flag to GCS.

### Phase 4D — Decode-internal barrier (~5 s after bootstrap-ready)

Once decode workers see the `bootstrap-ready` flag, they write `v6e16-1p1d-d-barrier-w{0..3}`
and wait until all 4 are present. Then all 4 decode workers proceed to JAX init simultaneously.

### Phase 5P+5D — Server launch and JAX distributed init (~1 min per VM)

**Prefill VM** — all 4 workers launch the prefill server:

```
--disaggregation-mode prefill
--disaggregation-bootstrap-url http://<prefill-w0-ip>:8998
--nnodes 4  --node-rank {0..3}  --dist-init-addr <prefill-w0-ip>:8088
```

**Decode VM** — all 4 workers launch the decode server:

```
--disaggregation-mode decode
--disaggregation-bootstrap-url http://<prefill-w0-ip>:8998
--nnodes 4  --node-rank {0..3}  --dist-init-addr <decode-w0-ip>:8088
```

Each VM's 4 workers call `jax.distributed.initialize()` with their own VM's coordinator
address. The two VMs initialize completely independently — they are two separate JAX processes,
each seeing 16 local devices. The disaggregation protocol (bootstrap URL) handles the
inter-VM KV-cache transfer separately from JAX.

JAX distributed init proceeds identically to the NonPD case within each VM:

```
[NP0] [Publisher 0] Begins to synchronize, wait 3 Subscribers
[NP0] [Publisher 0] Succeeds to synchronize!
```

After init, each VM has a `(1, 16)` mesh for `(data, tensor)` → TP=16, DP=1 on 16 devices.

### Phase 6 — Weight loading (~20 min from NFS, on both VMs)

Both VMs load the full model weights simultaneously from NFS:

```
Loading Regular Weights: 100%|██████| 574/574 [~5 min]
Loading MoE Weights:     100%|██████| 282/282 [~15 min]
```

Each VM loads the complete model (not a shard of the other's weights). The disaggregation
is at the request level: the prefill VM executes prefill steps and pushes KV cache to the
decode VM, which then runs decode steps.

Memory budget per VM is identical to NonPD — see
[NonPD memory table](GUIDE_v6e16_nonpd_benchmark.md#phase-5--weight-loading-20-min-from-nfs).

### Phase 7 — XLA precompilation (~2 min per VM)

Both VMs independently compile XLA kernels for their declared batch/token shapes.
The compilation cache (`gs://<bucket>/jax-compilation-cache/`) is shared — if a shape
was compiled during a prior NonPD run, this phase is faster.

**Total time from script start to both servers ready: ~28 minutes.**

### Phase 8 — PD Router + Benchmark (decode worker 0 only)

Decode worker 0 polls `/health` on the decode server (port 10001). Once healthy, it starts
the **PD router** — a lightweight proxy that fans out each `/generate` request to both the
prefill and decode servers:

```bash
python3.12 -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill "http://<prefill-w0-ip>:10000" 8998 \
  --decode "http://127.0.0.1:10001" \
  --prefill-bootstrap-host <prefill-w0-ip> \
  --host 0.0.0.0 --port 30000
```

The router listens on port 30000 and coordinates the PD flow: it sends the request to the
prefill server (which does the prefill and pushes KV cache via the bootstrap rendezvous) and
to the decode server (which receives the KV cache and generates tokens). The client gets back
the decode server's response.

Once the router is healthy, decode worker 0 drives three `bench_serving` runs **through the
router**:

```bash
python3.12 -m sgl_jax.bench_serving \
  --base-url http://127.0.0.1:30000 \   # PD router
  --random-input-len 16384 \
  --random-output-len 4096 \
  --max-concurrency <bsz> \
  --num-prompts <bsz*3>
```

After all three runs, decode worker 0 writes the `done` flag, which signals the prefill
workers to terminate.

---

## 4. Benchmark Results

### Configuration

| Parameter | Value |
|---|---|
| Mode | 1P1D disaggregated |
| Prefill VM | jingnw-node (v6e-16, TP=16) |
| Decode VM | jingnw-node2 (v6e-16, TP=16) |
| TP size per VM | 16 |
| DP size | 1 |
| EP size | 16 |
| dtype | bfloat16 |
| mem-fraction-static | 0.84 |
| page-size | 256 |
| context-length | 262144 |
| chunked-prefill-size | 2048 |
| max-prefill-tokens | 16384 |
| moe-backend | fused_v2 |
| Workload | 16384 input → 4096 output tokens |
| Bootstrap port | 8998 (on prefill worker 0) |

### Results

*Completed: 2026-07-14 03:26 UTC on branch `mimo-tpu7-stage3`.*
*Raw logs: `gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-v6e16-1p1d/`*

| Metric | bsz=32 | bsz=64 | bsz=128 |
|---|---|---|---|
| Request throughput (req/s) | 0.92 | 0.92 | 0.92 |
| **Output tok/s** | **3,769** | **3,761** | **3,775** |
| Input tok/s | 15,077 | 15,044 | 15,101 |
| Total tok/s | 18,846 | 18,805 | 18,876 |
| Mean E2E latency (ms) | 29,181 | 58,267 | 115,955 |
| Median E2E latency (ms) | 34,486 | 69,451 | 138,406 |
| Mean TTFT (ms) | 0 ⚠ | 0 ⚠ | 0 ⚠ |
| Median ITL (ms) | 0 ⚠ | 0 ⚠ | 0 ⚠ |
| Benchmark duration (s) | 104 | 209 | 417 |
| Successful requests | 96/96 | 192/192 | 384/384 |

### Analysis

**Output throughput** is 4.1–4.3× higher than NonPD: 3,775 tok/s (1P1D) vs 924 tok/s (NonPD)
at bsz=128. This improvement comes from separating prefill and decode onto dedicated hardware
— the decode VM's chips are never interrupted by chunked-prefill steps.

**Request throughput** is stable at 0.92 req/s across all batch sizes, compared to 0.21-0.23
req/s for NonPD — a 4× improvement. The 1P1D system maintains constant request throughput
regardless of concurrency level because the two VMs pipeline prefill and decode independently.

**E2E latency** scales linearly with batch size: 29s (bsz=32), 58s (bsz=64), 116s (bsz=128).
NonPD latency at the same batch sizes is 134s, 255s, 492s — 4.2–4.6× worse.

**TTFT and ITL report 0 ms** (⚠) because `bench_serving` uses `pd_separated=False` (default).
In this mode, the client sends all requests through the PD router as a single endpoint; it
cannot distinguish the prefill leg from subsequent decode tokens. A `--pd-separated` benchmark
mode would be needed to measure TTFT and ITL accurately in disaggregated setups.

**Benchmark duration** is dramatically shorter: all 384 requests at bsz=128 complete in 417s
(~7 min) vs 1,702s (~28 min) for NonPD — the pipeline keeps both VMs saturated without
the prefill/decode contention that throttles NonPD.

### Comparison with NonPD baseline

| Metric | NonPD (1 VM) | 1P1D (2 VMs) | Change |
|---|---|---|---|
| Output tok/s @ bsz=128 | 924 | 3,775 | +4.1× |
| Request throughput @ bsz=128 | 0.23 req/s | 0.92 req/s | +4.0× |
| Mean E2E @ bsz=32 | 134,250 ms | 29,181 ms | −4.6× |
| Median TTFT @ bsz=32 | 22,226 ms | 0 ms ⚠ | Not measurable |
| Median ITL @ bsz=128 | 17.74 ms | 0 ms ⚠ | Not measurable |
| Hardware cost | 1 × v6e-16 | 2 × v6e-16 | 2× chip-hours |
| Benchmark duration @ bsz=128 | 1,702 s | 417 s | −4.1× |

The throughput and E2E latency gains are real and reflect the benefit of separating prefill
and decode onto dedicated hardware. At 2× hardware cost, 1P1D delivers >4× throughput —
a 2× improvement in throughput-per-chip.

TTFT/ITL metrics require `--pd-separated` mode for meaningful comparison.

NonPD baseline results: [GUIDE_v6e16_nonpd_benchmark.md](GUIDE_v6e16_nonpd_benchmark.md#4-benchmark-results)

---

## 5. Adapting for v6e-32 with dp_size=2

This guide targets v6e-16 with dp_size=1 (mesh `(1, 16)`). For v6e-32 with dp_size=2, the following changes apply:

| Parameter | v6e-16 (this guide) | v6e-32 dp_size=2 |
|---|---|---|
| `--tp-size` | 16 | 32 |
| `--dp-size` | 1 | 2 |
| `--ep-size` | 16 | 32 |
| `--nnodes` | 4 | 8 |
| Mesh shape | `(1, 16)` | `(2, 16)` |
| `attention_tp_size` | 16 | 16 |
| Workers per VM | 4 | 8 |
| JAX devices per VM | 16 | 32 |

Server launch commands and the router command are the same; only the flags above change. See [DOC_pd_environment.md](DOC_pd_environment.md) for the full v6e-32 server launch commands.

### Known limitation: E0100 OOM with long inputs

With `--mem-fraction-static 0.84` and dp_size=2 on v6e-32, `bench_serving` with `--random-input-len 16384` triggers `E0100: RuntimeBufferAllocationFailure` during KV extraction (`jnp.stack(layer_kvs)` in `_extract_req_kv`). The allocation requires ~768MB per request but only ~389MB is free. Short requests (e.g. "What is 2+2?") work correctly. See [DOC_pd_environment.md](DOC_pd_environment.md#e0100-oom-during-kv-extraction-with-large-inputs-dp_size2) for details.

### Verified (2026-08-12)

Simple request test passed on v6e-32 dp_size=2 (branch `mimo-tpu7-stage3`, commit 71658835). Output: "2 + 2 = 4", correct stop finish, 26.2s E2E latency (first request with XLA compilation).
