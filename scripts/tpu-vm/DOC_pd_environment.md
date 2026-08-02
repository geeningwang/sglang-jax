# PD Disaggregated Inference — Environment Reference

**Last updated:** 2026-08-02
**Branch:** mimo-tpu7-stage3
**Model:** MiMo-V2-Flash (292GB, 156 files including 145 safetensors)

---

## Hardware

| Role    | TPU VM          | Zone        | Type   | Workers | IPs (internal)                                      |
|---------|-----------------|-------------|--------|---------|-----------------------------------------------------|
| Prefill | jingnw-node     | us-east5-a  | v6e-16 | 4       | w0: 10.202.0.135, w1: 10.202.0.41, w2: 10.202.0.13, w3: 10.202.0.123 |
| Decode1 | jingnw-node2    | us-east5-a  | v6e-16 | 4       | w0: 10.202.0.162, w1: 10.202.15.208, w2: 10.202.15.205, w3: 10.202.15.226 |
| Decode2 | jingnw-node3    | us-east5-b  | v6e-16 | 4       | w0: 10.202.15.228, w1: 10.202.15.231, w2: 10.202.0.184, w3: 10.202.15.229 |

Each v6e-16 has 4 hosts × 4 chips × 1 TensorCore = 16 JAX devices.

---

## Software Stack

| Component         | Version / Path                                              |
|-------------------|-------------------------------------------------------------|
| Python            | 3.12 (Miniconda)                                            |
| Miniconda         | 25.1.1 at `/tmp/miniconda3/`                                |
| JAX               | 0.10.2                                                      |
| JAXlib            | 0.10.2                                                      |
| Flax              | 0.12.4                                                      |
| Transformers      | 4.57.6                                                      |
| Safetensors       | 0.6.2                                                       |
| sglang-jax        | dev (editable install from `/tmp/workspace/python`)          |
| orbax-checkpoint  | ≥0.12.0                                                     |
| aiohttp           | latest                                                      |
| gcsfs             | latest (REQUIRED for GCS compilation cache)                  |
| uv                | latest (used for fast pip installs)                          |

### Installation commands

```bash
# 1. Miniconda
curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh" -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /tmp/miniconda3
export PATH="/tmp/miniconda3/bin:$PATH"

# 2. Clone repo
git clone -b mimo-tpu7-stage3 https://github.com/geeningwang/sglang-jax.git /tmp/workspace
cd /tmp/workspace

# 3. Install sglang-jax + dependencies
pip install uv -q
uv pip install --system -e "python[all]"
uv pip install --system "orbax-checkpoint>=0.12.0" aiohttp gcsfs -q
```

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

### Bootstrap (jingnw-node w0 only)

```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.run_bootstrap \
  --host 0.0.0.0 --port 8998 \
  </dev/null >/tmp/bootstrap.log 2>&1 &
```

### Prefill (jingnw-node, all 4 workers)

Must start **after** bootstrap is healthy (`curl http://localhost:8998/list_prefills` returns 200).

```bash
nohup python3.12 -m sgl_jax.launch_server \
  --model-path /tmp/flash-model --trust-remote-code \
  --enable-sequence-parallel --tp-size 16 --dp-size 1 --ep-size 16 \
  --moe-backend fused_v2 --nnodes 4 --node-rank $WORKER_ID \
  --dist-init-addr 10.202.0.135:8088 --host 0.0.0.0 --port 10000 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url http://10.202.0.135:8998 \
  </dev/null >/tmp/prefill_server.log 2>&1 &
```

### Decode1 (jingnw-node2, all 4 workers)

```bash
nohup python3.12 -m sgl_jax.launch_server \
  --model-path /tmp/flash-model --trust-remote-code \
  --enable-sequence-parallel --tp-size 16 --dp-size 1 --ep-size 16 \
  --moe-backend fused_v2 --nnodes 4 --node-rank $WORKER_ID \
  --dist-init-addr 10.202.0.162:8088 --host 0.0.0.0 --port 10001 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url http://10.202.0.135:8998 \
  </dev/null >/tmp/decode_server.log 2>&1 &
```

### Decode2 (jingnw-node3, all 4 workers)

```bash
nohup python3.12 -m sgl_jax.launch_server \
  --model-path /tmp/flash-model --trust-remote-code \
  --enable-sequence-parallel --tp-size 16 --dp-size 1 --ep-size 16 \
  --moe-backend fused_v2 --nnodes 4 --node-rank $WORKER_ID \
  --dist-init-addr 10.202.15.228:8088 --host 0.0.0.0 --port 10001 \
  --page-size 256 --context-length 262144 --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --decode-log-interval 1 \
  --max-running-requests 256 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url http://10.202.0.135:8998 \
  </dev/null >/tmp/decode_server.log 2>&1 &
```

**Note:** `--disable-overlap-schedule` is no longer needed. The SPMD race in the overlap decode event loop was fixed in commit 3c301255 by reordering the loop to drain the forward thread before `process_allgather`.

### Router (jingnw-node w0 only)

For 1P1D (single decode):
```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://10.202.0.135:10000 8998 \
  --decode http://10.202.0.162:10001 \
  --host 0.0.0.0 --port 30000 \
  </dev/null >/tmp/router.log 2>&1 &
```

For 1P2D (two decode servers):
```bash
nohup python3.12 -u -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://10.202.0.135:10000 8998 \
  --decode http://10.202.0.162:10001 \
  --decode http://10.202.15.228:10001 \
  --host 0.0.0.0 --port 30000 \
  </dev/null >/tmp/router.log 2>&1 &
```

The `--decode` flag accepts multiple values (`nargs=1, action="append"`). The router's `mini_lb` randomly selects one decode server per request.

---

## Port Assignments

| Service         | Port  | Host       |
|-----------------|-------|------------|
| Bootstrap       | 8998  | jingnw-node w0 |
| Prefill server  | 10000 | jingnw-node (all workers) |
| Decode1 server  | 10001 | jingnw-node2 (all workers) |
| Decode2 server  | 10001 | jingnw-node3 (all workers) |
| Router          | 30000 | jingnw-node w0 |
| JAX coordinator (prefill)  | 8088 | jingnw-node w0 |
| JAX coordinator (decode1)  | 8088 | jingnw-node2 w0 |
| JAX coordinator (decode2)  | 8088 | jingnw-node3 w0 |

---

## Startup Order

1. **Bootstrap** on jingnw-node w0 — must be up before prefill/decode register
2. **Prefill** on all 4 jingnw-node workers simultaneously — registers with bootstrap
3. **Decode1** on all 4 jingnw-node2 workers simultaneously — registers with bootstrap, queries prefill peers
4. **Decode2** on all 4 jingnw-node3 workers simultaneously (for 1P2D) — same as Decode1
5. **Router** on jingnw-node w0 — waits for prefill and all decode servers to be healthy

---

## Known Issues

### `/tmp` is volatile
Everything installed under `/tmp` (miniconda, workspace, model mount) is lost on VM reboot. Re-provision after reboot using `provision_decode.sh` or the bench scripts.

### E0200 after kill -9
Using `kill -9` on JAX processes can corrupt TPU state, causing `E0200: RuntimeUnexpectedCoreHalt` on next run. Fix: reboot the VM to reset TPU state, then re-provision.

### Missing gcsfs causes E0200 SPMD desync
Without `gcsfs`, the GCS-backed XLA compilation cache (`JAX_COMPILATION_CACHE_DIR`) silently fails to read. Each worker recompiles from scratch, and non-deterministic XLA compilation produces different programs on different workers → `E0200: RuntimeUnexpectedCoreHalt` with "unexpected peer in launch group with different launch id." Fix: `uv pip install --system gcsfs`.

### ~~Overlap schedule causes E0200 in disagg decode (multi-host)~~ — FIXED
The overlap disagg decode event loop had an SPMD race condition: `process_decode_queue()` called `process_allgather` while the forward thread was executing `jit_jitted_sampler`. Fixed in commit 3c301255 by reordering the loop to drain the forward thread before any SPMD collective. `--disable-overlap-schedule` is no longer needed.

### process_allgather int64→int32 truncation
`jax_enable_x64` is off by default on TPU. `multihost_utils.process_allgather()` silently truncates int64 arrays to int32 (see JAX issue #18385). Fix: `generate_bootstrap_room()` now returns `[0, 2^31-1]` and `multihost_sync.py` uses `np.int32` dtype, avoiding truncation entirely.

### Worker ID source
On these VMs, `/tmp/tpu-env` does NOT exist. Use GCE metadata instead:
```bash
curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number
```

### Process index mapping
JAX assigns `process_index` independently per cluster. GCE worker 0 is NOT necessarily `jax.process_index() == 0`. The mapping depends on which worker reaches the coordinator first.

### Bootstrap registered process indices
Observed mapping depends on worker startup order and changes across VM recreations. Use server logs to determine the current mapping.

### (Latent) KV sharding mismatch with dp_size > 1
In the single-host direct-HBM path (no D2H staging), prefill uses `P(None, *pool_pspec[1:])` while decode uses `kv_pool.kv_sharding` = `P("data", None, "tensor", None, None)`. With `dp_size=1` these are equivalent (size-1 axis), but with `dp_size>1` the per-device buffer sizes differ and the pull could hang or fail. Not triggered in our setup (dp_size=1).

### (Latent) No native timeout on `link.pull()`
`link.pull()` (`wrapper.py`) has no timeout parameter. If the prefill crashes after bootstrap publish but before `await_pull`, the pull blocks forever. The reaper (30s default) flips a state flag but cannot interrupt the native C call, permanently sticking the worker thread. With `pull_worker_count` workers (default 4), 4 stuck pulls exhaust the pool and all subsequent pulls queue indefinitely.

### (Latent) Link caching with no reconnection
`wrapper.py` caches one link per `remote_addr`. If a link becomes stale (prefill restarts, network partition), all subsequent pulls to that address hang. There is no reconnection logic.
