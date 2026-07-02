# Train GPT-2 (124M) on OpenWebText on a TPU v6e-8 (Trillium) or v4-8.
#
# Usage (from examples/nanogpt/):
#   python train.py config/train_gpt2.py
#
# Effective batch size per update:
#   batch_size(12) × num_devices(8) × gradient_accumulation_steps(5)
#   = 480 sequences × 1024 tokens = 491,520 tokens/step
#
# Expected ~2.85 val loss after 600K iterations (~57 hours of TPU compute,
# split across ~3 DWS sessions of 24 hours each on a single v6e-8 pod).
#
# TPU peak TFLOPS (per chip):
#   918   for v6e (Trillium)
#   2307  for v7x (Ironwood)
# Override on the command line: --tpu_peak_tflops=2307

wandb_log = True
wandb_project = "owt"
wandb_run_name = "gpt2-124M-tpu-nnx"

# Data parallelism: 12 sequences per chip, 8 chips via pmap, 5 micro-steps.
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

# 300B tokens total; cosine decay over the full run.
max_iters = 600000
lr_decay_iters = 600000

# Eval / logging
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Regularisation
weight_decay = 1e-1
