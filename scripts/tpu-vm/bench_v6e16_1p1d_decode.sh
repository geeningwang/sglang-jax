#!/bin/bash
# MiMo-V2-Flash 1P1D Decode server + benchmark on v6e-16 (jingnw-node2, us-east5-b)
# Runs on all 4 workers simultaneously via gcloud tpus tpu-vm ssh --worker=all.
#
# Hardware: v6e-16 (4x4), 4 hosts × 4 chips × 1 TensorCore = 16 JAX devices
# Server:   TP=16, DP=1, EP=16, nnodes=4, disaggregation-mode=decode
# Partner:  prefill server on jingnw-node (bench_v6e16_1p1d_prefill.sh)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

RESULTS_BUCKET="gs://jingnw-mimo-v2-flash-us-central1"
RESULTS_DIR="${RESULTS_BUCKET}/perf-results/flash-v6e16-1p1d"
D_BARRIER_PREFIX="${RESULTS_BUCKET}/v6e16-1p1d-d-barrier"
BOOTSTRAP_READY_FLAG="${RESULTS_BUCKET}/v6e16-1p1d-bootstrap-ready"
DONE_FLAG="${RESULTS_BUCKET}/v6e16-1p1d-done"
NFS_SERVER="10.128.0.34"
MODEL_PATH="/tmp/flash-model"

# Decode VM fixed IPs (jingnw-node2, us-east5-b)
DECODE_W0_IP="10.202.15.227"
DIST_INIT_ADDR="${DECODE_W0_IP}:8088"
# Prefill VM bootstrap (jingnw-node worker 0)
PREFILL_W0_IP="10.202.0.29"
BOOTSTRAP_PORT=8998
SERVER_PORT=10001
NNODES=4

ts() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"; }

# ── 0. Worker rank ────────────────────────────────────────────────────
WORKER_ID=$(grep "^WORKER_ID:" /tmp/tpu-env | awk -F"'" '{print $2}')
echo "$(ts) === [d-w${WORKER_ID}] MiMo-V2-Flash v6e-16 1P1D DECODE starting ==="

# ── 1. Clone + install ────────────────────────────────────────────────
WORKDIR="/tmp/workspace"
echo "$(ts) [d-w${WORKER_ID}] Cloning geeningwang/sglang-jax branch mimo-tpu7-stage3..."
rm -rf "${WORKDIR}"
git clone https://github.com/geeningwang/sglang-jax.git "${WORKDIR}" 2>&1 | tail -3
cd "${WORKDIR}"
git checkout mimo-tpu7-stage3
echo "$(ts) [d-w${WORKER_ID}] HEAD: $(git rev-parse --short HEAD) — $(git log -1 --pretty=%s)"

echo "$(ts) [d-w${WORKER_ID}] Installing Python 3.12 via Miniconda..."
CONDA_DIR="/tmp/miniconda3"
if [ ! -x "${CONDA_DIR}/bin/python3.12" ]; then
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh" \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
fi
export PATH="${CONDA_DIR}/bin:${PATH}"
python3.12 --version

echo "$(ts) [d-w${WORKER_ID}] Installing uv + sglang-jax..."
pip install uv -q
uv pip install --system -e "python[all]" 2>&1 | tail -5
uv pip install --system "orbax-checkpoint>=0.12.0" aiohttp -q

# ── 2. Mount NFS model weights ────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y -qq nfs-common
mkdir -p "${MODEL_PATH}"
echo "$(ts) [d-w${WORKER_ID}] Mounting NFS ${NFS_SERVER}:/export/flash..."
for i in $(seq 1 60); do
  if sudo mount -t nfs \
    -o nfsvers=3,nolock,rsize=1048576,wsize=1048576,hard,intr,timeo=600 \
    "${NFS_SERVER}:/export/flash" "${MODEL_PATH}" 2>/dev/null; then
    echo "$(ts) [d-w${WORKER_ID}] NFS mounted (attempt ${i})"; break
  fi
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [d-w${WORKER_ID}] NFS not ready... (${i}×5s)"
  sleep 5
done
NFILES=$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)
echo "$(ts) [d-w${WORKER_ID}] Flash weights: ${NFILES} safetensors files"
[ "${NFILES}" -lt 100 ] && { echo "ERROR: expected 145 safetensors, got ${NFILES}"; exit 1; }

# ── 3. Wait for prefill bootstrap to be ready ────────────────────────
echo "$(ts) [d-w${WORKER_ID}] Waiting for prefill bootstrap-ready flag..."
for i in $(seq 1 360); do
  gsutil ls "${BOOTSTRAP_READY_FLAG}" >/dev/null 2>&1 && {
    echo "$(ts) [d-w${WORKER_ID}] Bootstrap ready (${i}×5s elapsed)"; break
  }
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [d-w${WORKER_ID}] still waiting for prefill bootstrap... (${i}×5s)"
  sleep 5
done
gsutil ls "${BOOTSTRAP_READY_FLAG}" >/dev/null 2>&1 || {
  echo "$(ts) ERROR: bootstrap-ready flag never appeared"; exit 1
}

# ── 4. Decode-internal barrier: all 4 workers ready before JAX init ──
echo "$(ts) [d-w${WORKER_ID}] Writing decode barrier flag..."
echo "${WORKER_ID}" | gsutil cp - "${D_BARRIER_PREFIX}-w${WORKER_ID}"
for i in $(seq 1 240); do
  count=0
  for wid in 0 1 2 3; do
    gsutil ls "${D_BARRIER_PREFIX}-w${wid}" >/dev/null 2>&1 && count=$((count+1))
  done
  if [ "${count}" -eq "${NNODES}" ]; then
    echo "$(ts) [d-w${WORKER_ID}] All ${NNODES} decode workers ready (${i}×5s elapsed)"; break
  fi
  [ $((i % 12)) -eq 0 ] && echo "$(ts) [d-w${WORKER_ID}] waiting... ${count}/${NNODES}"
  sleep 5
done

# ── 5. Launch decode server (all workers, TP=16 multi-host) ──────────
SLOG="/tmp/server_decode_w${WORKER_ID}.log"
echo "$(ts) [d-w${WORKER_ID}] Starting decode server (port ${SERVER_PORT})..."
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
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url "http://${PREFILL_W0_IP}:${BOOTSTRAP_PORT}" \
  >> "${SLOG}" 2>&1 &
SRV_PID=$!
echo "$(ts) [d-w${WORKER_ID}] Decode server launched (PID=${SRV_PID})"

# ── 6. Worker 0: health check + benchmark; Workers 1-3: wait ─────────
if [ "${WORKER_ID}" = "0" ]; then
  echo "$(ts) [d-w0] Waiting for /health on localhost:${SERVER_PORT}..."
  T0=$(date +%s)
  for i in $(seq 1 1440); do
    if curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; then
      echo "$(ts) [d-w0] Decode server healthy after $(($(date +%s)-T0))s"; break
    fi
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
      echo "$(ts) ERROR: decode server exited early. Last 80 lines:"
      tail -80 "${SLOG}"
      gsutil cp "${SLOG}" "${RESULTS_DIR}/server-decode-w0-crash.log" || true
      echo "error" | gsutil cp - "${DONE_FLAG}"
      exit 1
    fi
    [ $((i % 60)) -eq 0 ] && echo "$(ts) [d-w0] still waiting... $((i*5))s"
    sleep 5
  done

  # ── 6a. Start PD router (fans out to prefill + decode) ──────────────
  ROUTER_PORT=30000
  RLOG="/tmp/router.log"
  echo "$(ts) [d-w0] Starting PD router on port ${ROUTER_PORT}..."
  python3.12 -m sgl_jax.srt.disaggregation.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://${PREFILL_W0_IP}:10000" "${BOOTSTRAP_PORT}" \
    --decode "http://127.0.0.1:${SERVER_PORT}" \
    --prefill-bootstrap-host "${PREFILL_W0_IP}" \
    --max-concurrent-requests 256 \
    --pd-prefill-max-inflight-requests 4 \
    --pd-router-admission-poll-ms 50 \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    >> "${RLOG}" 2>&1 &
  ROUTER_PID=$!
  echo "$(ts) [d-w0] Router launched (PID=${ROUTER_PID})"

  # Wait for router health (it proxies /health to decode)
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:${ROUTER_PORT}/health" >/dev/null 2>&1; then
      echo "$(ts) [d-w0] Router healthy"; break
    fi
    sleep 2
  done

  echo "$(ts) [d-w0] === bench_serving 1P1D: bsz 32 64 128 ==="
  for bs in 32 64 128; do
    np=$((bs * 3))
    RF="/tmp/bench_v6e16_1p1d_bs${bs}.jsonl"
    BL="/tmp/bench_v6e16_1p1d_bs${bs}.log"
    echo "$(ts) [d-w0] bsz=${bs} num_prompts=${np}"
    python3.12 -m sgl_jax.bench_serving \
      --backend sgl-jax \
      --base-url "http://127.0.0.1:${ROUTER_PORT}" \
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

  echo "$(ts) [d-w0] Uploading decode server + router logs..."
  gsutil cp "${SLOG}" "${RESULTS_DIR}/server-decode-w0.log" || true
  gsutil cp "${RLOG}" "${RESULTS_DIR}/router.log" || true
  echo "$(ts) [d-w0] Signalling done."
  echo "done" | gsutil cp - "${DONE_FLAG}"
  kill "${ROUTER_PID}" 2>/dev/null || true
  kill "${SRV_PID}" 2>/dev/null || true

else
  echo "$(ts) [d-w${WORKER_ID}] Waiting for done signal..."
  while true; do
    if gsutil ls "${DONE_FLAG}" >/dev/null 2>&1; then
      echo "$(ts) [d-w${WORKER_ID}] Done signal received."; break
    fi
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
      echo "$(ts) [d-w${WORKER_ID}] ERROR: decode server exited unexpectedly."
      gsutil cp "${SLOG}" "${RESULTS_DIR}/server-decode-w${WORKER_ID}-crash.log" || true
      exit 1
    fi
    sleep 10
  done
  gsutil cp "${SLOG}" "${RESULTS_DIR}/server-decode-w${WORKER_ID}.log" || true
  kill "${SRV_PID}" 2>/dev/null || true
fi

echo "$(ts) [d-w${WORKER_ID}] Complete. Results: ${RESULTS_DIR}/"
