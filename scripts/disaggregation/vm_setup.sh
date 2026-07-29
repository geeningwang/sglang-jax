#!/bin/bash
set -ex

# Setup script for TPU VM workers — installs JAX, sglang-jax, and tpu-raiden.
# Run via: gcloud compute tpus tpu-vm ssh ... --command="bash /tmp/vm_setup.sh"

export DEBIAN_FRONTEND=noninteractive

# Install system packages
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential curl iproute2 2>/dev/null

# Install JAX + libtpu
pip install -U --pre jax jaxlib libtpu requests aiohttp tqdm numpy httpx uvicorn fastapi pydantic \
  -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify JAX can see TPU
python3 -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.device_count()}')"

# Clone sglang-jax (use the working branch)
REPO_DIR="$HOME/sglang-jax"
if [ -d "$REPO_DIR" ]; then
  echo "sglang-jax already cloned, pulling latest..."
  cd "$REPO_DIR" && git pull || true
else
  git clone https://github.com/geeningwang/sglang-jax.git --branch mimo-tpu7-stage3 "$REPO_DIR"
fi
cd "$REPO_DIR"
pip install -e python[all] 2>/dev/null || pip install -e "python[all]"
pip install datasets huggingface_hub[cli]

# Install tpu-raiden from fork
RAIDEN_DIR="$HOME/tpu-raiden"
if [ -d "$RAIDEN_DIR" ]; then
  echo "tpu-raiden already cloned, pulling latest..."
  cd "$RAIDEN_DIR" && git pull || true
else
  git clone --depth 1 https://github.com/DigitalWNZ/tpu-raiden.git "$RAIDEN_DIR"
fi

# Try wheel first, fall back to source build
RAIDEN_OK=0
pip install tpu-raiden-jax \
  --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/ 2>/dev/null && \
  python3 -c "import tpu_raiden.frameworks.jax._tpu_raiden_jax" 2>/dev/null && \
  RAIDEN_OK=1 && echo "tpu-raiden installed from GAR wheel"

if [ "$RAIDEN_OK" != "1" ]; then
  echo "wheel unavailable; building tpu-raiden from source..."
  cd "$RAIDEN_DIR"
  HERMETIC_PYTHON_VERSION=3.10 ./build.sh jax
  export PYTHONPATH="$RAIDEN_DIR:${PYTHONPATH:-}"
  python3 -c "import tpu_raiden.frameworks.jax._tpu_raiden_jax" && \
    RAIDEN_OK=1 && echo "tpu-raiden built from source"
fi

if [ "$RAIDEN_OK" != "1" ]; then
  # Last resort: add fork to PYTHONPATH for the Python-level code
  export PYTHONPATH="$RAIDEN_DIR:${PYTHONPATH:-}"
  echo "WARNING: tpu-raiden native module not available; Python stubs only"
fi

# Download model
MODEL_PATH="$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B"
if [ ! -d "$MODEL_PATH" ]; then
  echo "Downloading model..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    local_dir='$MODEL_PATH',
    local_dir_use_symlinks=False,
)
print('Model downloaded successfully')
"
fi

echo "=== Setup complete ==="
echo "JAX: $(python3 -c 'import jax; print(jax.__version__)')"
echo "Devices: $(python3 -c 'import jax; print(jax.device_count())')"
echo "Model: $MODEL_PATH"
