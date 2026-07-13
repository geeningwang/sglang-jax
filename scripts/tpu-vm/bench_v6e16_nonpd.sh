#!/bin/bash
# MiMo-V2-Flash NonPD benchmark on v6e-16 (jingnw-node, us-east5-b)
# Run on all 4 workers via: gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all --command="..."
# Each worker independently runs this script; worker rank is read from /tmp/tpu-env.
#
# Hardware: v6e-16 (4x4), 4 hosts × 4 chips × 1 TensorCore = 16 JAX devices
# Server:   TP=16, DP=1, EP=16, nnodes=4
# Model:    MiMo-V2-Flash FP8, NFS-mounted from 10.128.0.34:/export/flash
# Results:  gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-v6e16-nonpd/

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

RESULTS_BUCKET="gs://jingnw-mimo-v2-flash-us-central1"
RESULTS_DIR="${RESULTS_BUCKET}/perf-results/flash-v6e16-nonpd"
BARRIER_PREFIX="${RESULTS_BUCKET}/v6e16-barrier"
DONE_FLAG="${RESULTS_BUCKET}/v6e16-done"
NFS_SERVER="10.128.0.34"
MODEL_PATH="/tmp/flash-model"
SERVER_PORT=30271
DIST_INIT_PORT=8088
# Worker 0 is the JAX coordinator; other workers connect to it.
# IP obtained from: gcloud compute tpus tpu-vm describe jingnw-node --zone=us-east5-b
WORKER0_IP="10.202.0.118"
DIST_INIT_ADDR="${WORKER0_IP}:${DIST_INIT_PORT}"
NNODES=4

ts() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"; }

# ── 0. Get worker rank ────────────────────────────────────────────
WORKER_ID=$(grep "^WORKER_ID:" /tmp/tpu-env | awk -F"'" '{print $2}')
echo "$(ts) === [w${WORKER_ID}] MiMo-V2-Flash v6e-16 NonPD bench starting ==="

# ── 1. Clone + install ────────────────────────────────────────────
WORKDIR="/tmp/workspace"
echo "$(ts) [w${WORKER_ID}] Cloning geeningwang/sglang-jax branch mimo-tpu7-stage2..."
rm -rf "${WORKDIR}"
git clone https://github.com/geeningwang/sglang-jax.git "${WORKDIR}" 2>&1 | tail -3
cd "${WORKDIR}"
git checkout mimo-tpu7-stage2
echo "$(ts) [w${WORKER_ID}] HEAD: $(git rev-parse --short HEAD) — $(git log -1 --pretty=%s)"

# Patch mesh_utils.py: revert jax.local_devices() → jax.devices() for TP=16 multi-host.
# (The branch has local_devices() for the GKE 1P1D case; we need devices() here.)
sed -i 's/jax\.local_devices()/jax.devices()/g' python/sgl_jax/srt/utils/mesh_utils.py
echo "$(ts) [w${WORKER_ID}] mesh_utils.py patched: local_devices → devices"

echo "$(ts) [w${WORKER_ID}] Installing Python 3.12 via Miniconda..."
CONDA_DIR="/tmp/miniconda3"
if [ ! -x "${CONDA_DIR}/bin/python3.12" ]; then
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh" \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
fi
export PATH="${CONDA_DIR}/bin:${PATH}"
python3.12 --version

echo "$(ts) [w${WORKER_ID}] Installing uv..."
pip install uv -q
echo "$(ts) [w${WORKER_ID}] Installing sglang-jax + deps..."
uv pip install --system -e "python[all]" 2>&1 | tail -5
uv pip install --system "orbax-checkpoint>=0.12.0" aiohttp -q

# ── 2. Mount NFS model weights ────────────────────────────────────
echo "$(ts) [w${WORKER_ID}] Installing nfs-common..."
sudo apt-get update -qq
sudo apt-get install -y -qq nfs-common

mkdir -p "${MODEL_PATH}"
echo "$(ts) [w${WORKER_ID}] Mounting NFS ${NFS_SERVER}:/export/flash → ${MODEL_PATH}..."
for i in $(seq 1 60); do
  if sudo mount -t nfs \
    -o nfsvers=3,nolock,rsize=1048576,wsize=1048576,hard,intr,timeo=600 \
    "${NFS_SERVER}:/export/flash" "${MODEL_PATH}" 2>/dev/null; then
    echo "$(ts) [w${WORKER_ID}] NFS mounted (attempt ${i})"
    break
  fi
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [w${WORKER_ID}] NFS not ready, retrying... (${i}×5s)"
  sleep 5
done
NFILES=$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)
echo "$(ts) [w${WORKER_ID}] Flash weights: ${NFILES} safetensors files"
if [ "${NFILES}" -lt 100 ]; then
  echo "$(ts) [w${WORKER_ID}] ERROR: expected 145 safetensors, got ${NFILES}. NFS mount may have failed."
  exit 1
fi

# ── 3. Barrier: wait for all 4 workers ready before JAX init ─────
echo "$(ts) [w${WORKER_ID}] Writing ready barrier flag..."
echo "${WORKER_ID}" | gsutil cp - "${BARRIER_PREFIX}-w${WORKER_ID}"

echo "$(ts) [w${WORKER_ID}] Waiting for all ${NNODES} workers to signal ready..."
for i in $(seq 1 240); do
  count=0
  for wid in 0 1 2 3; do
    gsutil ls "${BARRIER_PREFIX}-w${wid}" >/dev/null 2>&1 && count=$((count+1))
  done
  if [ "${count}" -eq "${NNODES}" ]; then
    echo "$(ts) [w${WORKER_ID}] All ${NNODES} workers ready (${i}×5s elapsed)"
    break
  fi
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [w${WORKER_ID}] waiting for other workers... ${count}/${NNODES} ready"
  sleep 5
done

# ── 4. Launch sgl-jax server (all workers, TP=16 multi-host) ─────
SLOG="/tmp/server_w${WORKER_ID}.log"
echo "$(ts) [w${WORKER_ID}] Starting server: TP=16 DP=1 EP=16 nnodes=${NNODES} rank=${WORKER_ID}..."
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
  --log-mfu \
  >> "${SLOG}" 2>&1 &
SRV_PID=$!
echo "$(ts) [w${WORKER_ID}] Server launched (PID=${SRV_PID})"

# ── 5. Worker 0: health check + benchmark; Workers 1-3: wait ─────
if [ "${WORKER_ID}" = "0" ]; then
  echo "$(ts) [w0] Waiting for /health on localhost:${SERVER_PORT}..."
  T0=$(date +%s)
  for i in $(seq 1 1440); do
    if curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; then
      echo "$(ts) [w0] Server healthy after $(($(date +%s)-T0))s"
      break
    fi
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
      echo "$(ts) ERROR: server exited early. Last 80 lines:"
      tail -80 "${SLOG}"
      gsutil cp "${SLOG}" "${RESULTS_DIR}/server-w0-crash.log" || true
      echo "error" | gsutil cp - "${DONE_FLAG}"
      exit 1
    fi
    [ $((i % 60)) -eq 0 ] && echo "$(ts) [w0] still waiting... $((i*5))s"
    sleep 5
  done

  echo ""
  echo "$(ts) [w0] === bench_serving NonPD: bsz 32 64 128 ==="
  for bs in 32 64 128; do
    np=$((bs * 3))
    RF="/tmp/bench_v6e16_bs${bs}.jsonl"
    BL="/tmp/bench_v6e16_bs${bs}.log"
    echo "$(ts) [w0] bsz=${bs} num_prompts=${np}"
    python3.12 -m sgl_jax.bench_serving \
      --backend sgl-jax \
      --base-url "http://127.0.0.1:${SERVER_PORT}" \
      --model "${MODEL_PATH}" \
      --dataset-name random \
      --random-input-len 16384 \
      --random-output-len 4096 \
      --random-range-ratio 1 \
      --max-concurrency "${bs}" \
      --num-prompts "${np}" \
      --warmup-requests 0 \
      --seed 12345 \
      --output-file "${RF}" \
      2>&1 | tee "${BL}" || true
    gsutil cp "${RF}" "${RESULTS_DIR}/bs${bs}/result.jsonl" || true
    gsutil cp "${BL}" "${RESULTS_DIR}/bs${bs}/bench.log"   || true
  done

  echo "$(ts) [w0] Uploading server log..."
  gsutil cp "${SLOG}" "${RESULTS_DIR}/server-w0.log" || true
  echo "$(ts) [w0] Signalling all workers: done."
  echo "done" | gsutil cp - "${DONE_FLAG}"
  kill "${SRV_PID}" 2>/dev/null || true

else
  # Workers 1-3: poll for done or server crash
  echo "$(ts) [w${WORKER_ID}] Waiting for done signal..."
  while true; do
    if gsutil ls "${DONE_FLAG}" >/dev/null 2>&1; then
      echo "$(ts) [w${WORKER_ID}] Done signal received."
      break
    fi
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
      echo "$(ts) [w${WORKER_ID}] Server exited unexpectedly."
      gsutil cp "${SLOG}" "${RESULTS_DIR}/server-w${WORKER_ID}-crash.log" || true
      exit 1
    fi
    sleep 10
  done
  gsutil cp "${SLOG}" "${RESULTS_DIR}/server-w${WORKER_ID}.log" || true
  kill "${SRV_PID}" 2>/dev/null || true
fi

echo "$(ts) [w${WORKER_ID}] Complete. Results: ${RESULTS_DIR}/"
