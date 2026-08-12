# PD Disaggregated Inference — Environment Reference

**Last updated:** 2026-08-12
**Branch:** mimo-tpu7-stage3
**Model:** MiMo-V2-Flash (292GB, 156 files including 145 safetensors)

---

## Hardware

| Role    | TPU VM          | Zone              | Type   | Workers | Chips |
|---------|-----------------|-------------------|--------|---------|-------|
| Prefill | jingnw-node     | asia-northeast1-b | v6e-32 | 8       | 32    |
| Decode  | jingnw-node2    | asia-northeast1-b | v6e-32 | 8       | 32    |

Each v6e-32 has 8 hosts × 4 chips × 1 TensorCore = 32 JAX devices.

**Network:** These VMs have **no internet access**. Code and dependencies are staged via GCS bucket `gs://jingnw-mimo-v2-flash-us-central1/staging/`.

---

## Software Stack

| Component         | Version / Path                                                |
|-------------------|---------------------------------------------------------------|
| Python            | 3.12 (Miniconda)                                              |
| Miniconda         | 25.1.1 at `/tmp/miniconda3/`                                  |
| JAX               | 0.10.2                                                        |
| JAXlib            | 0.10.2                                                        |
| Flax              | 0.12.4                                                        |
| Transformers      | 4.57.6                                                        |
| Safetensors       | 0.6.2                                                         |
| pathwaysutils     | 0.1.11 (installed from GCS-staged source; required at module level by scheduler) |
| sglang-jax        | dev (editable install from `/tmp/sglang-jax/python`)           |
| orbax-checkpoint  | ≥0.12.0                                                       |
| aiohttp           | latest                                                        |
| gcsfs             | latest (REQUIRED for GCS compilation cache)                    |
| uv                | latest (used for fast pip installs)                            |

### Installation commands

```bash
# 1. Miniconda
curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh" -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /tmp/miniconda3
export PATH="/tmp/miniconda3/bin:$PATH"

# 2. Clone repo
git clone -b mimo-tpu7-stage3 https://github.com/geeningwang/sglang-jax.git /tmp/sglang-jax
cd /tmp/sglang-jax

# 3. Install sglang-jax + dependencies
pip install uv -q
uv pip install --system -e "python[all]"
uv pip install --system "orbax-checkpoint>=0.12.0" aiohttp gcsfs -q
```

### Deploying code updates (offline VMs)

Since the VMs have no internet, code changes are deployed via GCS:

```bash
# On a machine with internet access:
tar czf /tmp/patch.tar.gz -C /path/to/sglang-jax/python .
gsutil cp /tmp/patch.tar.gz gs://jingnw-mimo-v2-flash-us-central1/staging/patch.tar.gz

# On each TPU VM worker:
gsutil cp gs://jingnw-mimo-v2-flash-us-central1/staging/patch.tar.gz /tmp/patch.tar.gz
tar xzf /tmp/patch.tar.gz --overwrite -C /tmp/sglang-jax/python
```

### Offline provisioning (jingnw-node3 pattern)

When provisioning a new offline VM from scratch using GCS-staged artifacts:

1. Install Miniconda from GCS-staged installer
2. Install sglang-jax editable from GCS-staged repo tarball (`--no-deps`)
3. Install JAX/jaxlib/numpy from GCS-staged pip wheels (`--no-index --find-links`)
4. Install pathwaysutils from GCS-staged source (see below)
5. Copy missing Python packages from a working node via GCS tarballs:
   - `soundfile` (including `_soundfile.py`, `_soundfile_data/` with `libsndfile_x86_64.so`)
   - `pybase64` (no cp312 wheel in pip-wheels; copy installed package from working node)
   - Correct versions of `transformers`, `huggingface_hub`, `safetensors`, `flax`
   - Other packages: `orjson`, `markupsafe`, `uvicorn`
6. Mount NFS model weights
7. Install nfs-common from GCS-staged debs (`nfs-debs.tar.gz`)

GCS staging bucket: `gs://jingnw-mimo-v2-flash-us-central1/staging/`

### Installing pathwaysutils (offline VMs)

`pathwaysutils` is not pre-installed on standard TPU VMs but is required by `scheduler.py` (top-level import). Since these VMs have no internet, install from a GCS-staged copy of the source:

```bash
# One-time prep (on a machine with internet):
git clone --depth 1 https://github.com/AI-Hypercomputer/pathways-utils.git /tmp/pathways-utils
tar czf /tmp/pathwaysutils_pkg.tar.gz -C /tmp/pathways-utils pathwaysutils
gsutil cp /tmp/pathwaysutils_pkg.tar.gz gs://jingnw-mimo-v2-flash-us-central1/staging/pathwaysutils_pkg.tar.gz

# On each TPU VM worker:
SITE=$(python3.12 -c "import site; print(site.getsitepackages()[0])")
gsutil cp gs://jingnw-mimo-v2-flash-us-central1/staging/pathwaysutils_pkg.tar.gz /tmp/pathwaysutils_pkg.tar.gz
tar xzf /tmp/pathwaysutils_pkg.tar.gz -C "$SITE"
```

The package is pure Python — no build step needed. Verify with `python3.12 -c "import pathwaysutils; print(pathwaysutils.__version__)"`.

**Note:** This install does NOT survive VM reboot (`/tmp` is volatile). Must re-install after reboot.

---

## Model Weights (NFS)

The model is served from an NFS filestore:

| Property | Value |
|----------|-------|
| NFS Server | `10.128.0.34` |
| Export path | `/export/flash` |
| Mount point | `/tmp/flash-model` |
| Size | 292GB (145 safetensors files) |

### Mount command

```bash
sudo apt-get update -qq && sudo apt-get install -y nfs-common
mkdir -p /tmp/flash-model
sudo mount -t nfs \
  -o nfsvers=3,nolock,rsize=1048576,wsize=1048576,hard,intr,timeo=600 \
  10.128.0.34:/export/flash /tmp/flash-model
```

**Note:** NFS mount does NOT survive VM reboot. Must re-mount after reboot.

---

## Runtime Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `LIBTPU_INIT_ARGS` | `--xla_tpu_dvfs_p_state=7` | Stable TPU frequency for deterministic perf |
| `JAX_COMPILATION_CACHE_DIR` | `gs://jingnw-mimo-v2-flash-us-central1/jax-compilation-cache` | GCS-backed XLA compilation cache |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout for log visibility |

---

## Server Launch Commands

Each worker must set `WORKER_ID` from GCE metadata before launching:
```bash
WORKER_ID=$(curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)
```

When launching via `gcloud compute tpus tpu-vm ssh`, use `nohup ... </dev/null >/tmp/<log> 2>&1 &` to fully detach from SSH.

### Mesh configuration

With `--dp-size 1`: mesh is `(1, 32)`, `attention_tp_size = 32`. All 32 devices form a single DP rank.

With `--dp-size 2`: mesh is `(2, 16)` with axes `("data", "tensor")`. Each DP rank spans 16 devices (4 hosts). `attention_tp_size = tp_size // dp_size = 16`.

**Current configuration:** dp_size=1 (see dp_rank=1 garbled output issue under Known Issues for why dp_size=2 is not used).

### Bootstrap (jingnw-node w0 only)

```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.run_bootstrap \
  --host 0.0.0.0 --port 8998 \
  </dev/null >/tmp/bootstrap.log 2>&1 &
```

### Prefill (jingnw-node, all 8 workers)

Must start **after** bootstrap is healthy (`curl http://localhost:8998/list_prefills` returns 200).

```bash
nohup python3.12 -m sgl_jax.launch_server \
  --model-path /tmp/flash-model --trust-remote-code \
  --enable-sequence-parallel --tp-size 32 --dp-size 1 --ep-size 32 \
  --moe-backend fused_v2 --nnodes 8 --node-rank $WORKER_ID \
  --dist-init-addr <jingnw-node-w0-ip>:8088 --host 0.0.0.0 --port 10000 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url http://<jingnw-node-w0-ip>:8998 \
  </dev/null >/tmp/prefill_server.log 2>&1 &
```

For dp_size=2, add `--dp-size 2` (instead of 1) and `--dp-schedule-policy round_robin`.

### Decode (each decode cluster, all 8 workers)

Same command for each decode cluster — only `--dist-init-addr` changes (must point to that cluster's own w0 IP).

```bash
nohup python3.12 -m sgl_jax.launch_server \
  --model-path /tmp/flash-model --trust-remote-code \
  --enable-sequence-parallel --tp-size 32 --dp-size 1 --ep-size 32 \
  --moe-backend fused_v2 --nnodes 8 --node-rank $WORKER_ID \
  --dist-init-addr <this-cluster-w0-ip>:8088 --host 0.0.0.0 --port 10001 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url http://<jingnw-node-w0-ip>:8998 \
  </dev/null >/tmp/decode_server.log 2>&1 &
```

For dp_size=2, add `--dp-size 2` (instead of 1) and `--dp-schedule-policy round_robin`.

**Note:** `--disable-overlap-schedule` is no longer needed. The SPMD race in the overlap decode event loop was fixed in commit 3c301255.

### Router (jingnw-node w0 only)

For 1P1D (1 decode cluster):
```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://<jingnw-node-w0-ip>:10000 8998 \
  --decode http://<jingnw-node2-w0-ip>:10001 \
  --host 0.0.0.0 --port 30000 \
  </dev/null >/tmp/router.log 2>&1 &
```

For 1P2D (2 decode clusters):
```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://<jingnw-node-w0-ip>:10000 8998 \
  --decode http://<jingnw-node2-w0-ip>:10001 \
  --decode http://<jingnw-node3-w0-ip>:10001 \
  --host 0.0.0.0 --port 30000 \
  </dev/null >/tmp/router.log 2>&1 &
```

The `--decode` flag supports `action="append"` — add one `--decode` per decode cluster.

---

## Port Assignments

| Service         | Port  | Host       |
|-----------------|-------|------------|
| Bootstrap       | 8998  | jingnw-node w0 |
| Prefill server  | 10000 | jingnw-node (all workers) |
| Decode server   | 10001 | jingnw-node2 (all workers) |
| Router          | 30000 | jingnw-node w0 |
| JAX coordinator (prefill)  | 8088 | jingnw-node w0 |
| JAX coordinator (decode)   | 8088 | jingnw-node2 w0 |

For 1P2D, add additional decode clusters on port 10001 with their own JAX coordinator on 8088.

---

## Startup Order

1. **Bootstrap** on jingnw-node w0 — must be up before prefill/decode register
2. **Prefill** on all 8 jingnw-node workers simultaneously — registers with bootstrap
3. **Decode** on all 8 jingnw-node2 workers simultaneously — registers with bootstrap, queries prefill peers
4. **Router** on jingnw-node w0 — waits for prefill and decode servers to be healthy

For 1P2D, additional decode clusters can start in parallel with decode #1 (step 3).

---

## IP Addresses

IPs change across VM recreations. Use `gcloud compute tpus tpu-vm describe <name> --zone asia-northeast1-b` to get current IPs, or check server logs.

The `--dist-init-addr` must point to worker 0's internal IP for each cluster. JAX assigns `process_index` independently — GCE worker 0 is NOT necessarily `jax.process_index() == 0`.

---

## Known Issues

### `/tmp` is volatile
Everything installed under `/tmp` (miniconda, sglang-jax, model mount) is lost on VM reboot. Re-provision after reboot.

### No internet on asia-northeast1-b VMs
These VMs cannot reach the internet. All code updates and package installations must be staged via GCS bucket. `pip install` from PyPI will fail.

### E0200 after kill -9
Using `kill -9` on JAX processes can corrupt TPU state, causing `E0200: RuntimeUnexpectedCoreHalt` on next run. Fix: reboot the VM to reset TPU state, then re-provision.

### Never use pkill on TPU VMs
`pkill -f` pattern-matches the SSH command itself and kills the SSH session. Use `kill` with specific PIDs instead. When run inside `gcloud compute tpus tpu-vm ssh --command`, the killed SSH session causes gcloud to retry, re-running the entire command — which can spawn duplicate server processes.

### Use SIGINT (kill -2) to stop JAX servers
JAX server processes may ignore SIGTERM (`kill`). Use `kill -2` (SIGINT) instead — it triggers clean shutdown. Do NOT use `kill -9` (see E0200 issue above).

### Missing gcsfs causes E0200 SPMD desync
Without `gcsfs`, the GCS-backed XLA compilation cache silently fails to read. Each worker recompiles from scratch, and non-deterministic XLA compilation produces different programs on different workers → `E0200`. Fix: `uv pip install --system gcsfs`.

### ~~Overlap schedule causes E0200 in disagg decode (multi-host)~~ — FIXED
Fixed in commit 3c301255. See [DOC_spmd_race_analysis.md](DOC_spmd_race_analysis.md).

### process_allgather int64→int32 truncation
`jax_enable_x64` is off by default on TPU. `process_allgather` silently truncates int64 to int32. Fixed: room IDs now generated in `[0, 2^31-1]` with `np.int32` dtype.

### Worker ID source
On these VMs, `/tmp/tpu-env` does NOT exist. Use GCE metadata:
```bash
curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number
```

### Process index mapping
JAX assigns `process_index` independently per cluster. GCE worker 0 is NOT necessarily `jax.process_index() == 0`. The mapping depends on which worker reaches the coordinator first. Use server logs to determine the current mapping.

### E0100 OOM during KV extraction with large inputs (dp_size=2)
With `--mem-fraction-static 0.84` and `dp_size=2` on v6e-32, KV extraction in `_extract_req_kv` (prefill.py) can hit `E0100: RuntimeBufferAllocationFailure` when processing long-input requests (e.g. 16384 tokens). The `jnp.stack(layer_kvs)` call attempts to allocate ~768MB per request but only ~389MB remains free. Single short requests (e.g. "What is 2+2?") work fine. The OOM triggers at high concurrency or with long inputs. Once OOM errors occur, the prefill server may stop processing new requests — requires a full server restart to recover. Potential mitigations: reduce `--mem-fraction-static`, reduce input length, or modify KV extraction to stream layers instead of stacking.

### dp_rank=1 produces garbled output (dp_size=2)
With `--dp-size 2` on v6e-32, requests routed to dp_rank=0 produce correct output, but requests routed to dp_rank=1 consistently produce garbled/nonsensical text (e.g. "\\OptionsResolver- Data-Integrity: 1.0.0.0" for "What is 2+2?"). The issue is reproducible across both 1P1D and 1P2D configurations and across different decode clusters (code verified identical via md5sum). The root cause is not yet identified — could be in prefill dp_rank=1 computation, KV transfer for dp_rank=1, or decode dp_rank=1 processing. With `--dp-schedule-policy round_robin`, approximately half of requests hit dp_rank=1 and produce wrong output.

### (Latent) No native timeout on `link.pull()`
`link.pull()` has no timeout parameter. If the prefill crashes after bootstrap publish but before `await_pull`, the pull blocks forever. The reaper flips a state flag but cannot interrupt the native C call.

### (Latent) Link caching with no reconnection
`wrapper.py` caches one link per `remote_addr`. If a link becomes stale (prefill restarts, network partition), all subsequent pulls to that address hang.

---

## Quick Smoke Test

After all servers and the router are healthy, verify with a simple request:

```bash
curl -s http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "What is 2+2?", "sampling_params": {"max_new_tokens": 64, "temperature": 0}}'
```

Expected: JSON response with `"text": "2 + 2 = 4"` (or similar), `finish_reason.type: "stop"`, and non-zero `completion_tokens`. First request includes XLA compilation and may take ~25s.

---

## Verification History

### 1P1D dp_size=1 on v6e-32 (2026-08-12)

| Property | Value |
|----------|-------|
| Branch | `mimo-tpu7-stage3` (commit 71658835) |
| Prefill VM | jingnw-node (v6e-32, asia-northeast1-b) |
| Decode VM | jingnw-node2 (v6e-32, asia-northeast1-b) |
| Mesh | `(1, 32)` — dp_size=1, attention_tp_size=32 |
| Test | `curl` "What is 2+2?" + "What is the capital of France?" × 4 → router (port 30000) |
| Result | **PASS** — all requests correct: "2 + 2 = 4", "The capital of France is **Paris**." |
| E2E latency | 25.1s (first request, includes XLA compilation); sub-second warm |
| Note | No dp_rank issue — dp_size=1 has only dp_rank=0. All requests consistent. |

### 1P2D dp_size=2 on v6e-32 (2026-08-12)

| Property | Value |
|----------|-------|
| Branch | `mimo-tpu7-stage3` (commit 71658835) |
| Prefill VM | jingnw-node (v6e-32, asia-northeast1-b) |
| Decode #1 VM | jingnw-node2 (v6e-32, asia-northeast1-b) |
| Decode #2 VM | jingnw-node3 (v6e-32, asia-northeast1-b) |
| Mesh | `(2, 16)` — dp_size=2, attention_tp_size=16 |
| Test | `curl` "What is 2+2?" × 4 → router (port 30000) |
| Result | **PARTIAL** — dp_rank=0 requests: correct output "2 + 2 = 4". dp_rank=1 requests: garbled output (see Known Issues). |
| E2E latency | 26.7s (first request, includes XLA compilation); ~0.3s warm (dp_rank=0) |
| Note | dp_rank=1 garbled output is a pre-existing issue also present in 1P1D — not caused by 1P2D routing. Code is identical across all 3 clusters (md5sum verified). |

### 1P1D dp_size=2 symmetric on v6e-32 (2026-08-12)

| Property | Value |
|----------|-------|
| Branch | `mimo-tpu7-stage3` (commit 71658835) |
| Prefill VM | jingnw-node (v6e-32, asia-northeast1-b) |
| Decode VM | jingnw-node2 (v6e-32, asia-northeast1-b) |
| Mesh | `(2, 16)` — dp_size=2, attention_tp_size=16 |
| Test | `curl` "What is 2+2?" → router (port 30000) |
| Result | **PASS** — correct output "2 + 2 = 4", 8 completion tokens, stop finish |
| E2E latency | 26.2s (first request, includes XLA compilation) |
| Note | `bench_serving` with 16384-token inputs at concurrency ≥4 triggers E0100 OOM in KV extraction (see Known Issues). Short requests work correctly. Only dp_rank=0 tested; dp_rank=1 garbled output discovered later in 1P2D testing. |
