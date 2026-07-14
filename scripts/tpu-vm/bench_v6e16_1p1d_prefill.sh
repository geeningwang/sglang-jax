#!/bin/bash
# MiMo-V2-Flash 1P1D Prefill server on v6e-16 (jingnw-node, us-east5-b)
# Runs on all 4 workers simultaneously via gcloud tpus tpu-vm ssh --worker=all.
#
# Hardware: v6e-16 (4x4), 4 hosts × 4 chips × 1 TensorCore = 16 JAX devices
# Server:   TP=16, DP=1, EP=16, nnodes=4, disaggregation-mode=prefill
# Partner:  decode server on jingnw-node2 (bench_v6e16_1p1d_decode.sh)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

RESULTS_BUCKET="gs://jingnw-mimo-v2-flash-us-central1"
RESULTS_DIR="${RESULTS_BUCKET}/perf-results/flash-v6e16-1p1d"
P_BARRIER_PREFIX="${RESULTS_BUCKET}/v6e16-1p1d-p-barrier"
BOOTSTRAP_READY_FLAG="${RESULTS_BUCKET}/v6e16-1p1d-bootstrap-ready"
DONE_FLAG="${RESULTS_BUCKET}/v6e16-1p1d-done"
NFS_SERVER="10.128.0.34"
MODEL_PATH="/tmp/flash-model"

# Prefill VM fixed IPs (jingnw-node, us-east5-b)
# Obtain via: gcloud compute tpus tpu-vm describe jingnw-node --zone=us-east5-b --format="json(networkEndpoints)"
PREFILL_W0_IP="10.202.0.29"
DIST_INIT_ADDR="${PREFILL_W0_IP}:8088"
BOOTSTRAP_PORT=8998
SERVER_PORT=10000
NNODES=4

ts() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"; }

# ── 0. Worker rank ────────────────────────────────────────────────────
WORKER_ID=$(grep "^WORKER_ID:" /tmp/tpu-env | awk -F"'" '{print $2}')
echo "$(ts) === [p-w${WORKER_ID}] MiMo-V2-Flash v6e-16 1P1D PREFILL starting ==="

# ── 1. Clone + install ────────────────────────────────────────────────
WORKDIR="/tmp/workspace"
echo "$(ts) [p-w${WORKER_ID}] Cloning geeningwang/sglang-jax branch mimo-tpu7-stage3..."
rm -rf "${WORKDIR}"
git clone https://github.com/geeningwang/sglang-jax.git "${WORKDIR}" 2>&1 | tail -3
cd "${WORKDIR}"
git checkout mimo-tpu7-stage3
echo "$(ts) [p-w${WORKER_ID}] HEAD: $(git rev-parse --short HEAD) — $(git log -1 --pretty=%s)"

echo "$(ts) [p-w${WORKER_ID}] Installing Python 3.12 via Miniconda..."
CONDA_DIR="/tmp/miniconda3"
if [ ! -x "${CONDA_DIR}/bin/python3.12" ]; then
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh" \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
fi
export PATH="${CONDA_DIR}/bin:${PATH}"
python3.12 --version

echo "$(ts) [p-w${WORKER_ID}] Installing uv + sglang-jax..."
pip install uv -q
uv pip install --system -e "python[all]" 2>&1 | tail -5
uv pip install --system "orbax-checkpoint>=0.12.0" aiohttp -q

# ── 2. Mount NFS model weights ────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y -qq nfs-common
mkdir -p "${MODEL_PATH}"
echo "$(ts) [p-w${WORKER_ID}] Mounting NFS ${NFS_SERVER}:/export/flash..."
for i in $(seq 1 60); do
  if sudo mount -t nfs \
    -o nfsvers=3,nolock,rsize=1048576,wsize=1048576,hard,intr,timeo=600 \
    "${NFS_SERVER}:/export/flash" "${MODEL_PATH}" 2>/dev/null; then
    echo "$(ts) [p-w${WORKER_ID}] NFS mounted (attempt ${i})"; break
  fi
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [p-w${WORKER_ID}] NFS not ready... (${i}×5s)"
  sleep 5
done
NFILES=$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)
echo "$(ts) [p-w${WORKER_ID}] Flash weights: ${NFILES} safetensors files"
[ "${NFILES}" -lt 100 ] && { echo "ERROR: expected 145 safetensors, got ${NFILES}"; exit 1; }

# ── 3. Prefill-internal barrier: all 4 workers ready before JAX init ──
echo "$(ts) [p-w${WORKER_ID}] Writing prefill barrier flag..."
echo "${WORKER_ID}" | gsutil cp - "${P_BARRIER_PREFIX}-w${WORKER_ID}"
for i in $(seq 1 240); do
  count=0
  for wid in 0 1 2 3; do
    gsutil ls "${P_BARRIER_PREFIX}-w${wid}" >/dev/null 2>&1 && count=$((count+1))
  done
  if [ "${count}" -eq "${NNODES}" ]; then
    echo "$(ts) [p-w${WORKER_ID}] All ${NNODES} prefill workers ready (${i}×5s elapsed)"; break
  fi
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [p-w${WORKER_ID}] waiting... ${count}/${NNODES}"
  sleep 5
done

# ── 4. Worker 0: start bootstrap server ──────────────────────────────
if [ "${WORKER_ID}" = "0" ]; then
  BLOG="/tmp/bootstrap.log"
  echo "$(ts) [p-w0] Starting bootstrap server on port ${BOOTSTRAP_PORT}..."
  python3.12 -m sgl_jax.srt.disaggregation.run_bootstrap \
    --host 0.0.0.0 --port "${BOOTSTRAP_PORT}" \
    >> "${BLOG}" 2>&1 &
  BOOTSTRAP_PID=$!
  # Wait for bootstrap to bind
  for i in $(seq 1 30); do
    ss -tlnp 2>/dev/null | grep -q ":${BOOTSTRAP_PORT} " && {
      echo "$(ts) [p-w0] Bootstrap ready after $((i*2))s"; break
    }
    kill -0 "${BOOTSTRAP_PID}" 2>/dev/null || {
      echo "$(ts) ERROR: bootstrap exited"; tail -20 "${BLOG}"; exit 1
    }
    sleep 2
  done
  # Signal decode that bootstrap is ready (write prefill IP)
  echo "${PREFILL_W0_IP}" | gsutil cp - "${BOOTSTRAP_READY_FLAG}"
  echo "$(ts) [p-w0] Bootstrap-ready flag written to GCS."
fi

# ── 5. Launch prefill server (all workers, TP=16 multi-host) ─────────
SLOG="/tmp/server_prefill_w${WORKER_ID}.log"
echo "$(ts) [p-w${WORKER_ID}] Starting prefill server (port ${SERVER_PORT})..."
PYTHONUNBUFFERED=1 \
LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7" \
JAX_COMPILATION_CACHE_DIR="${RESULTS_BUCKET}/jax-compilation-cache" \
python3.12 -m sgl_jax.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --enable-sequence-parallel \
  --tp-size 16 \
  --dp-size 1 \
  --ep-size 16 \
  --moe-backend fused_v2 \
  --nnodes "${NNODES}" \
  --node-rank "${WORKER_ID}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --page-size 256 \
  --context-length 262144 \
  --disable-radix-cache \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384 \
  --dtype bfloat16 \
  --mem-fraction-static 0.84 \
  --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup \
  --log-level info \
  --decode-log-interval 1 \
  --max-running-requests 256 \
  --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url "http://${PREFILL_W0_IP}:${BOOTSTRAP_PORT}" \
  >> "${SLOG}" 2>&1 &
SRV_PID=$!
echo "$(ts) [p-w${WORKER_ID}] Prefill server launched (PID=${SRV_PID})"

# ── 6. All workers: wait for done signal from decode ─────────────────
echo "$(ts) [p-w${WORKER_ID}] Waiting for decode done signal..."
while true; do
  if gsutil ls "${DONE_FLAG}" >/dev/null 2>&1; then
    echo "$(ts) [p-w${WORKER_ID}] Done signal received."
    break
  fi
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "$(ts) [p-w${WORKER_ID}] ERROR: prefill server exited unexpectedly."
    gsutil cp "${SLOG}" "${RESULTS_DIR}/server-prefill-w${WORKER_ID}-crash.log" || true
    exit 1
  fi
  sleep 10
done

gsutil cp "${SLOG}" "${RESULTS_DIR}/server-prefill-w${WORKER_ID}.log" || true
if [ "${WORKER_ID}" = "0" ]; then
  gsutil cp /tmp/bootstrap.log "${RESULTS_DIR}/bootstrap.log" || true
fi
kill "${SRV_PID}" 2>/dev/null || true
[ "${WORKER_ID}" = "0" ] && kill "${BOOTSTRAP_PID}" 2>/dev/null || true

echo "$(ts) [p-w${WORKER_ID}] Complete."
