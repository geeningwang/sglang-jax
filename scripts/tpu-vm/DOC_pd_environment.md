# PD Disaggregated Inference — Environment Reference

**Last updated:** 2026-08-10
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

### Mesh configuration (dp_size=2)

With `--tp-size 32 --dp-size 2 --ep-size 32`, the mesh is `(2, 16)` with axes `("data", "tensor")`. Each DP rank spans 16 devices (4 hosts). `attention_tp_size = tp_size // dp_size = 16`.

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
  --enable-sequence-parallel --tp-size 32 --dp-size 2 --ep-size 32 \
  --moe-backend fused_v2 --nnodes 8 --node-rank $WORKER_ID \
  --dist-init-addr <jingnw-node-w0-ip>:8088 --host 0.0.0.0 --port 10000 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url http://<jingnw-node-w0-ip>:8998 \
  </dev/null >/tmp/prefill_server.log 2>&1 &
```

### Decode (jingnw-node2, all 8 workers)

```bash
nohup python3.12 -m sgl_jax.launch_server \
  --model-path /tmp/flash-model --trust-remote-code \
  --enable-sequence-parallel --tp-size 32 --dp-size 2 --ep-size 32 \
  --moe-backend fused_v2 --nnodes 8 --node-rank $WORKER_ID \
  --dist-init-addr <jingnw-node2-w0-ip>:8088 --host 0.0.0.0 --port 10001 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url http://<jingnw-node-w0-ip>:8998 \
  </dev/null >/tmp/decode_server.log 2>&1 &
```

**Note:** `--disable-overlap-schedule` is no longer needed. The SPMD race in the overlap decode event loop was fixed in commit 3c301255.

### Router (jingnw-node w0 only)

```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://<jingnw-node-w0-ip>:10000 8998 \
  --decode http://<jingnw-node2-w0-ip>:10001 \
  --host 0.0.0.0 --port 30000 \
  </dev/null >/tmp/router.log 2>&1 &
```

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

---

## Startup Order

1. **Bootstrap** on jingnw-node w0 — must be up before prefill/decode register
2. **Prefill** on all 8 jingnw-node workers simultaneously — registers with bootstrap
3. **Decode** on all 8 jingnw-node2 workers simultaneously — registers with bootstrap, queries prefill peers
4. **Router** on jingnw-node w0 — waits for prefill and decode servers to be healthy

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
`pkill -f` pattern-matches the SSH command itself and kills the SSH session. Use `kill` with specific PIDs instead.

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

### (Latent) No native timeout on `link.pull()`
`link.pull()` has no timeout parameter. If the prefill crashes after bootstrap publish but before `await_pull`, the pull blocks forever. The reaper flips a state flag but cannot interrupt the native C call.

### (Latent) Link caching with no reconnection
`wrapper.py` caches one link per `remote_addr`. If a link becomes stale (prefill restarts, network partition), all subsequent pulls to that address hang.
