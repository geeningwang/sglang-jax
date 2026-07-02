"""
GPT-2 training in Flax NNX on TPU.

Ported from ~/transformer/nanogpt-tpu/train.py (Flax Linen + pmap) to the
Flax NNX model in model.py.  Core design choices are preserved:

  - jax.pmap for data parallelism across all TPU chips
  - jax.lax.scan for gradient accumulation (single XLA dispatch per update)
  - Deferred loss materialisation (float(loss[0]) only every log_interval iters)
  - optax AdamW with warmup-cosine LR schedule
  - GCS checkpoint sync when GCS_CHECKPOINT env var is set

NNX-specific changes vs the Linen version:
  - Model is stateful: nnx.split(model) → (graphdef, state) pytree
  - graphdef is static (closed over by pmap'd functions)
  - Pure optax state tracks optimiser moments (not nnx.Optimizer, for pmap compat)
  - Weight decay mask operates on the flat nnx.State pytree leaves

Usage:
  python train.py config/train_gpt2.py
  python train.py config/train_gpt2.py --batch_size=8 --max_iters=100
"""

import functools
import os
import pickle
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax import serialization
from jax import random

from model import GPT, GPTConfig

# ── Default config (mirrors nanogpt-tpu/train.py) ─────────────────────────────
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' or 'resume'
# wandb
wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2-tpu-nnx"
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5
batch_size = 12          # micro-batch per TPU core
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# TPU
seed = 1337
tpu_peak_tflops = 918.0  # per chip: 918 for v6e (Trillium), 2307 for v7x (Ironwood)
# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # noqa: S102
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# ── Device setup ──────────────────────────────────────────────────────────────
devices = jax.devices()
num_devices = len(devices)
print(f"JAX devices: {devices}")
print(f"Running on {num_devices} device(s)")

tokens_per_iter = gradient_accumulation_steps * num_devices * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
data_dir = os.path.join("data", dataset)


def get_batch(split: str, total_batch_size: int):
    """Return (x, y) numpy arrays of shape (total_batch_size, block_size)."""
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix = np.random.randint(len(data) - block_size, size=(total_batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int32) for i in ix])
    return x, y


# ── Model ─────────────────────────────────────────────────────────────────────
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    model_args["vocab_size"] = meta["vocab_size"]
    print(f"found vocab_size = {meta['vocab_size']} in {meta_path}")
else:
    model_args["vocab_size"] = 50304
    print("defaulting to vocab_size=50304 (GPT-2 padded)")

cfg = GPTConfig(**model_args)
model = GPT(cfg)
print(f"number of parameters: {model.num_params() / 1e6:.2f}M")

# Split into (static graphdef, JAX-pytree state) for pmap compatibility
graphdef, init_state = nnx.split(model)


# ── Optimiser ─────────────────────────────────────────────────────────────────

def make_lr_schedule():
    if not decay_lr:
        return learning_rate
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_iters,
        decay_steps=lr_decay_iters,
        end_value=min_lr,
    )


def weight_decay_mask(state):
    """Weight decay on 2-D params only (weights/embeddings, not biases/LN scales)."""
    return jax.tree_util.tree_map(lambda p: p.ndim >= 2, state)


tx = optax.chain(
    optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
    optax.adamw(
        learning_rate=make_lr_schedule(),
        b1=beta1,
        b2=beta2,
        weight_decay=weight_decay,
        mask=weight_decay_mask,
    ),
)

init_opt_state = tx.init(init_state)

# ── pmap'd train / eval steps ─────────────────────────────────────────────────
# graphdef is static and closed over by these functions — it is NOT a traced JAX
# value.  pmap slices the batch (first) axis of state, opt_state, xs, ys, rngs.

@functools.partial(jax.pmap, axis_name="devices")
def train_step(state, opt_state, xs, ys, rngs):
    """One gradient update with accumulation compiled into a single XLA program.

    Args (shapes after pmap slices the device axis):
        state:     nnx.State pytree (per-device params)
        opt_state: optax state (per-device)
        xs:        int32[G, batch_size, block_size]  — all micro-batches stacked
        ys:        int32[G, batch_size, block_size]
        rngs:      uint32[G, 2]  — one PRNGKey per micro-step

    Returns: (new_state, new_opt_state, mean_loss)
    """

    def micro_step(carry, inputs):
        grads_acc, loss_acc = carry
        x, y, rng = inputs

        def loss_fn(state):
            m = nnx.merge(graphdef, state)
            _, loss = m(x, y, rng=rng)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        return (jax.tree_util.tree_map(jnp.add, grads_acc, grads), loss_acc + loss), None

    init_grads = jax.tree_util.tree_map(jnp.zeros_like, state)
    (grads_acc, loss_sum), _ = jax.lax.scan(
        micro_step, (init_grads, jnp.zeros(())), (xs, ys, rngs)
    )
    grads_acc = jax.tree_util.tree_map(
        lambda g: g / gradient_accumulation_steps, grads_acc
    )
    updates, new_opt_state = tx.update(grads_acc, opt_state, state)
    new_state = optax.apply_updates(state, updates)
    return new_state, new_opt_state, loss_sum / gradient_accumulation_steps


@functools.partial(jax.pmap, axis_name="devices")
def eval_step(state, x, y):
    m = nnx.merge(graphdef, state)
    _, loss = m(x, y)
    return jax.lax.pmean(loss, axis_name="devices")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _gcs_upload(local_path: str, gcs_uri: str) -> None:
    try:
        from google.cloud import storage as _gcs  # noqa: PLC0415

        bucket_name, blob_path = gcs_uri[5:].split("/", 1)
        _gcs.Client().bucket(bucket_name).blob(blob_path).upload_from_filename(local_path)
        print(f"checkpoint synced → {gcs_uri}")
    except Exception as exc:
        print(f"WARNING: GCS sync skipped ({exc})")


def save_checkpoint(state, opt_state, iter_num, best_val_loss):
    # Un-replicate: all devices hold identical copies; take device 0
    state0 = jax.tree_util.tree_map(lambda x: x[0], state)
    opt0 = jax.tree_util.tree_map(lambda x: x[0], opt_state)
    ckpt = {
        "state": state0,
        "opt_state": opt0,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "model_args": model_args,
        "config": config,
    }
    path = os.path.join(out_dir, "ckpt.msgpack")
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(ckpt))
    print(f"checkpoint saved to {path}")
    gcs_uri = os.environ.get("GCS_CHECKPOINT", "")
    if gcs_uri:
        _gcs_upload(path, gcs_uri)


def load_checkpoint(state, opt_state):
    path = os.path.join(out_dir, "ckpt.msgpack")
    with open(path, "rb") as f:
        raw = f.read()
    # Un-replicate reference objects for shape matching
    state0 = jax.tree_util.tree_map(lambda x: x[0], state)
    opt0 = jax.tree_util.tree_map(lambda x: x[0], opt_state)
    # msgpack_restore avoids strict key matching on new config fields
    outer = serialization.msgpack_restore(raw)
    restored_state = serialization.from_state_dict(state0, outer["state"])
    restored_opt = serialization.from_state_dict(opt0, outer["opt_state"])
    rep_state = jax.device_put_sharded([restored_state] * num_devices, devices)
    rep_opt = jax.device_put_sharded([restored_opt] * num_devices, devices)
    return rep_state, rep_opt, int(outer["iter_num"]), float(outer["best_val_loss"])


# ── Estimate loss ─────────────────────────────────────────────────────────────

def estimate_loss(state):
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split, batch_size * num_devices)
            x = x.reshape(num_devices, batch_size, block_size)
            y = y.reshape(num_devices, batch_size, block_size)
            loss = eval_step(state, x, y)
            losses.append(loss[0])
        out[split] = float(np.mean(losses))
    return out


# ── Initialise or resume ──────────────────────────────────────────────────────
iter_num = 0
best_val_loss = 1e9

key = random.PRNGKey(seed)
key, init_key = random.split(key)

# Replicate state and opt_state across all devices
state = jax.device_put_sharded([init_state] * num_devices, devices)
opt_state = jax.device_put_sharded([init_opt_state] * num_devices, devices)

if init_from == "resume":
    print(f"Resuming training from {out_dir}")
    state, opt_state, iter_num, best_val_loss = load_checkpoint(state, opt_state)
    print(f"  resumed at iter {iter_num}, best_val_loss {best_val_loss:.4f}")

# ── wandb ─────────────────────────────────────────────────────────────────────
if wandb_log:
    import wandb  # noqa: PLC0415

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ── Training loop ─────────────────────────────────────────────────────────────
running_mfu = -1.0
t0 = time.time()

while True:

    # ── Eval + checkpoint ──────────────────────────────────────────────────────
    if iter_num % eval_interval == 0:
        losses = estimate_loss(state)
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
        if wandb_log:
            lr_now = (
                float(make_lr_schedule()(iter_num)) if decay_lr else learning_rate
            )
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr_now,
                    "mfu": running_mfu * 100,
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                save_checkpoint(state, opt_state, iter_num, best_val_loss)

    if iter_num == 0 and eval_only:
        break

    # ── Pre-load all micro-batches, then single XLA dispatch ──────────────────
    batches = [
        get_batch("train", batch_size * num_devices)
        for _ in range(gradient_accumulation_steps)
    ]
    # xs/ys shape after stacking: (num_devices, G, batch_size, block_size)
    xs = jnp.array(
        np.stack(
            [b[0].reshape(num_devices, batch_size, block_size) for b in batches], axis=1
        )
    )
    ys = jnp.array(
        np.stack(
            [b[1].reshape(num_devices, batch_size, block_size) for b in batches], axis=1
        )
    )

    # rngs shape: (num_devices, G, 2)
    key, train_key = random.split(key)
    rngs = random.split(train_key, num_devices * gradient_accumulation_steps)
    rngs = rngs.reshape(num_devices, gradient_accumulation_steps, 2)

    state, opt_state, loss = train_step(state, opt_state, xs, ys, rngs)

    # ── Logging ───────────────────────────────────────────────────────────────
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = float(loss[0])  # single host sync; prior iters ran asynchronously
        if iter_num >= 5:
            # MFU: use device-0 state (identical across all devices after sync)
            m_ref = nnx.merge(graphdef, jax.tree_util.tree_map(lambda x: x[0], state))
            mfu = m_ref.estimate_mfu(
                fwdbwd_per_iter=batch_size * gradient_accumulation_steps * num_devices,
                dt=dt,
                tpu_peak_tflops=tpu_peak_tflops * num_devices,
            )
            running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, "
            f"time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )

    iter_num += 1
    if iter_num > max_iters:
        break

print("Training complete.")
