# MiMo-V2.5-Pro Weight Checkpoint Conversion

## Status: Blocked — JAX 0.8.1 cannot create float8 arrays on TPU via Orbax

**Root cause (confirmed 2026-06-03)**: JAX 0.8.1 on TPU cannot create `float8_e4m3fn`
JAX arrays via `jax.make_array_from_single_device_arrays` (the internal path Orbax uses
for array deserialization). Orbax leaves FP8 leaves as `ShapeDtypeStruct` instead of
actual arrays. Since MiMo-V2.5-Pro is **fully FP8-quantized** (all weights are
float8_e4m3fn), 100% of checkpoint tensors fail to restore.

**Tested approaches (all failed)**:
- Orbax 0.11.28: FP8 arrays → ShapeDtypeStruct ✗
- Orbax 0.12.0: same behavior ✗
- uint8 workaround: saves/restores correctly, but in-device bitcast OOMs (no HBM headroom)
- CPU bitcast via numpy: `jax.device_get()` fails for non-addressable shards ✗
- Per-shard `addressable_shards`: `make_array_from_single_device_arrays` OOMs ✗

**Path forward**: Requires JAX upgrade to a version that supports FP8 array creation
on TPU via the Orbax restore path. The NFS RAM-backed loading (~42 min) remains the
current fastest option.

Convert HuggingFace FP8 safetensors → sharded Orbax/zarr checkpoint so each
TensorCore loads only its 30 GB TP shard instead of the full 240 GB node shard.
Expected startup reduction: ~40 min → ~5 min (MoE loading phase).

---

## Background — Current Loading Pipeline

### Timing (measured, 4-node 2x2x4, tp-size=32)

| Phase | Duration | Notes |
|-------|----------|-------|
| sglang-jax install + NFS mount | ~5 min | One-time per job |
| Regular weights (557 tensors) | ~3 min | Fast, small tensors |
| **MoE weights (414 groups)** | **~40 min (NFS) / ~2h (gcsfuse)** | Bottleneck |
| KV cache profiling | ~1 min | |
| XLA warmup | ~55s (warm cache) | |

### What happens during MoE weight loading

For each of the 414 MoE groups, across all 70 layers × 6 weight types:

```
1. Read raw bytes from safetensors file via NFS/gcsfuse
   → 384 expert tensors, FP8, packed in one large contiguous span

2. CPU numpy: _bulk_read_file()
   → gsutil/NFS read → np.frombuffer → reshape to per-expert arrays
   → I/O burst: ~1-3s at ~1.8 Gbps per node

3. CPU numpy: _maybe_convert_epmoe_scale_for_kernel()
   → scale tensors: (num_experts, k_blocks, out_blocks)
                  → (num_experts, k_blocks, 1, out_dim_padded)   [GMM kernel layout]

4. CPU numpy: fused QKV construction (regular weights only)
   → Q, K, V projections concatenated into single tensor

5. JAX: make_array_from_callback()
   → TP-shard the stacked expert tensor across mesh
   → Each TC receives its 1/32 slice
   → Device transfer (HBM): fast, ~ms

6. JAX: model_param.value = stacked_weight.astype(dtype)
   → Final assignment to model param

   CPU idle (~3-5s between reads)
```

### Why loading is slow

The fundamental problem: **each node reads 240 GB from the weight source**
(all 34 safetensors files), then extracts only its **30 GB TP shard**
(1/8 of what its 8 TCs need = 240/8 = 30 GB per TC). The other 210 GB
read per node is discarded.

```
Network I/O per node:  240 GB   (read)
Useful data per node:   30 GB   (kept in HBM)
Wasted I/O:            210 GB   (87.5% of reads discarded)
```

With 3 NFS servers at ~12.5 GB/s each, the 240 GB/node read takes:
`240 GB / (37.5 GB/s ÷ 4 nodes) ≈ 25 min just for I/O`
(plus ~15 min CPU conversion = ~40 min total)

### NFS network utilization

From measured data:
- Per-pod inbound: ~1.8 Gbps (10s average)
- 4 pods aggregate: ~7.2 Gbps
- 3 NFS server NICs (3 × 100 Gbps): utilization **~19%**

The NIC is not the bottleneck — the **sequential read pattern** of the weight
loader is: read one group → CPU convert → next group (NIC idle during CPU work).

---

## Analysis — Three Approaches

### Approach A: Pre-convert safetensors (CPU conversions only)

Pre-apply steps 3 and 4 (scale reshape, QKV fusion) to the safetensors files
and save back to GCS.

| | |
|---|---|
| **Saves** | CPU conversion time (~5–10 min of the ~40 min total) |
| **Does not save** | I/O time (still reads 240 GB per node) |
| **Compatible with** | Any tp-size |
| **Verdict** | Marginal gain (~10–20% speedup) |

### Approach B: Sharded Orbax/zarr checkpoint ✅ RECOMMENDED

After the first full load completes, save the model's JAX state via Orbax.
Each TC writes only its local 30 GB shard. On subsequent runs, each TC reads
only its own shard directly — no conversion, no wasted I/O.

```
First run:  read 240 GB → convert → load → save 30 GB checkpoint per TC
Later runs: read 30 GB checkpoint per TC → done (8× less I/O per node)
```

| | |
|---|---|
| **Saves** | I/O time + all CPU conversion (90%+ of loading time) |
| **Expected load time** | ~30 GB / (NFS bandwidth per TC) ≈ 2–5 min |
| **Checkpoint size** | ~962 GB total (same as source), split into 32 shards |
| **Cache key** | `tp_size + model_path + dtype` — changing tp-size invalidates |
| **Requires** | All 4 nodes present for both save AND load |
| **Verdict** | **8× I/O reduction, eliminates all CPU conversion on reload** |

### Approach C: Pre-sharded safetensors (per-TC files)

Write 32 safetensors files (one per TC), each containing only that TC's weight
shard. No JAX dependency; load is just `safe_open` → `device_put`.

| | |
|---|---|
| **Saves** | I/O time + CPU conversion |
| **Compatible with** | Only one tp-size (like Approach B) |
| **Complexity** | Requires custom shard-extraction script |
| **Verdict** | Similar benefit to B but more complex; B is better |

---

## Plan — Approach B: Sharded Orbax Checkpoint

### GCS checkpoint path

```
gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/tp{tp_size}/
```

The `tp_size` in the path ensures the correct checkpoint is used and different
tp-size runs don't conflict. Future extension: add model hash or dtype.

### Save flow (first run only)

```
model.load_weights(model_config)          # existing path, ~40 min
    ↓
save_checkpoint(model, checkpoint_path)   # NEW: ~5 min, all 4 nodes participate
    ↓
exit (or continue to serve)
```

### Load flow (subsequent runs)

```
checkpoint exists at gs://.../tp32/ ?
    YES → load_checkpoint(model, checkpoint_path)   # NEW: ~5 min
    NO  → model.load_weights(model_config)          # existing path, ~40 min
          save_checkpoint(model, checkpoint_path)   # NEW: save for next time
```

### Checkpoint format

Use `flax.nnx` state + `orbax.checkpoint`:

```python
import orbax.checkpoint as ocp
from flax import nnx

# Save
state = nnx.state(model)
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save(gcs_path, state)

# Load
abstract_state = nnx.eval_shape(lambda: model_class(...))
state = checkpointer.restore(gcs_path, item=abstract_state)
nnx.update(model, state)
```

Orbax handles GCS paths natively via `gs://` URIs. Each process (one per node)
saves/loads only its local device's shards — no all-gather needed.

---

## Implementation Notes

### Files to modify

| File | Change |
|------|--------|
| `python/sgl_jax/srt/model_loader/loader.py` | Add `_save_checkpoint()`, `_load_checkpoint()`, checkpoint detection in `_get_model()` |
| `python/sgl_jax/srt/server_args.py` | Add `--checkpoint-path` flag (default: auto-derived from model path) |

### Key implementation details

**1. Checkpoint path derivation (if not explicitly set):**
```python
# Auto-derive from model_path and tp_size
import hashlib
model_hash = hashlib.md5(model_config.model_path.encode()).hexdigest()[:8]
checkpoint_path = (
    f"gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/"
    f"tp{mesh.size}/{model_hash}/"
)
```

**2. Multi-host save coordination:**
Orbax with GCS handles multi-host saves correctly when each process writes
its own shards. Use `ocp.CheckpointManager` with `multiprocessing_options`
for safe concurrent writes.

**3. Checkpoint validity check:**
Before loading, verify:
- Checkpoint directory exists in GCS
- `tp_size` matches current launch config
- Metadata file contains matching model hash

**4. Fallback:**
If checkpoint load fails (corrupt, wrong dtype, etc.), fall back to
`load_weights()` and re-save.

**5. Save timing:**
Save happens AFTER `load_weights()` completes, before the server starts
accepting requests. The save itself runs in parallel across all nodes
(each saves its shards concurrently) — expected ~5 min.

### Pseudocode for `_get_model` modification

```python
def _get_model(self, model_class, model_config):
    checkpoint_path = self._get_checkpoint_path(model_config)

    with jax.set_mesh(self.mesh):
        model = nnx.eval_shape(
            lambda: model_class(config, dtype=model_config.dtype, mesh=self.mesh)
        )

    # Apply quantization structure if needed
    model = self._apply_quantization(model, model_config)

    if self._checkpoint_exists(checkpoint_path):
        logger.info("Loading from checkpoint: %s", checkpoint_path)
        self._load_checkpoint(model, checkpoint_path)
    else:
        logger.info("No checkpoint found, loading from weights source")
        model.load_weights(model_config)
        logger.info("Saving checkpoint to: %s", checkpoint_path)
        self._save_checkpoint(model, checkpoint_path)

    return model
```

### Dependencies

`orbax-checkpoint` is already in the dependency tree via `flax`. Verify:
```bash
python3 -c "import orbax.checkpoint; print(orbax.checkpoint.__version__)"
```

---

## Expected Results

| Metric | Before (NFS) | After (checkpoint) |
|--------|-------------|-------------------|
| First-run load time | ~40 min | ~45 min (40 + 5 save) |
| Subsequent load time | ~40 min | **~5 min** |
| I/O per node | 240 GB | 30 GB |
| CPU conversion | Yes (scale reshape, QKV fusion) | **No** |
| Checkpoint size (GCS) | — | ~962 GB (32 shards) |
| tp-size portability | Any | Fixed to saved tp-size |

First-run cost: +5 min (save). Subsequent savings: ~35 min per run.
Break-even: after 2 runs (1 first-run + 1 fast reload = ~50 min vs 2 × 40 min = 80 min).

---

## Related Documents

- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — weight loading pipeline detail
- [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md) — performance measurements
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — HBM/RAM allocation
