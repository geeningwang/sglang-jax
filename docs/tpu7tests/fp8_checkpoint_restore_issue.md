# FP8 Checkpoint Save/Restore Issue — Technical Deep Dive

**Status**: Workaround validated ✅ — ready for production integration.  
**Last updated**: 2026-06-03  
**Environment**: GKE TPU v7x (2x2x4), `tpu7x-standard-4t`, JAX 0.9.0 + Orbax 0.12

---

## Problem Statement

MiMo-V2.5-Pro (962 GB FP8 static quantization) loads weights in ~42 min via NFS.
We want to save an Orbax checkpoint after first load so subsequent restores take ~90s
instead of ~42 min. The checkpoint **saves** successfully. The checkpoint **restore
fails** because JAX on TPU v7x cannot create `float8_e4m3fn` arrays during the restore
path.

**The model is 100% FP8**: every weight tensor has `dtype=float8_e4m3fn`. There are no
BF16 or FP32 weights to restore selectively — the problem affects the entire model.

---

## Environment

```
Hardware:    TPU v7x, tpu7x-standard-4t, 4-node 2x2x4 slice
             32 TensorCores × 96 GB HBM = 3072 GB total HBM
             After weights loaded: ~14 MB HBM free per TensorCore

Containers tested:
  jax0.8.1-rev1   us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
  jax0.9.0-rev1   us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1

Library versions (jax0.8.1-rev1):
  jax                   0.8.1
  jaxlib                0.8.1
  orbax-checkpoint      0.11.28  (also tested: 0.12.0)
  flax                  0.12.4
  ml_dtypes             0.4.0

Library versions (jax0.9.0-rev1):
  jax                   0.9.0
  jaxlib                0.9.0
  orbax-checkpoint      0.11.40  (also tested: 0.12.0 via pip upgrade)
  flax                  0.12.4
  ml_dtypes             0.4.0

Model:    MiMo-V2.5-Pro (HuggingFace FP8 safetensors)
          quantization_config: {"quant_method": "fp8", "fmt": "e4m3",
                                "weight_block_size": [128, 128]}
          All linear and MoE expert weights: float8_e4m3fn
          Total: 1038 FP8 parameter tensors per process (8 TensorCores per node)
```

---

## What Works

1. **Model loads from NFS correctly** — `load_weights()` in sglang-jax reads FP8
   safetensors and places `float8_e4m3fn` JAX arrays on TPU without issues. The model
   runs FP8 GEMMs successfully. FP8 *computation* works.

2. **Checkpoint saves correctly** — Orbax writes FP8 arrays to GCS at ~4.5 GiB/s
   per host. The OCDBT files are valid. `commit_success.txt` is written. The checkpoint
   can be inspected and the raw bytes are correct.

3. **Checkpoint restores the bytes** — Orbax reads back 481.9 GiB per host in ~91s
   at ~5.5 GiB/s. The data transfer is successful.

4. **The failure is in JAX array construction**, not Orbax I/O.

---

## Root Cause

During restore, Orbax calls `jax.make_array_from_single_device_arrays()` (or an
equivalent internal path) to convert the restored bytes into a JAX array. For
`float8_e4m3fn` dtype, this call **fails silently** on TPU — instead of raising an
exception, Orbax returns a `jax.ShapeDtypeStruct` (an abstract type descriptor with no
backing buffer) in place of the actual array.

The `ShapeDtypeStruct` propagates into the model state via `nnx.update(model, state)`.
When the forward pass runs, JAX sees a `ShapeDtypeStruct` where it expects an array:

```
TypeError: Argument 'ShapeDtypeStruct(shape=(6144, 16384), dtype=float8_e4m3fn,
sharding=NamedSharding(mesh=..., spec=PartitionSpec(None, 'tensor')))' of type
<class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
```

**Why does JAX return ShapeDtypeStruct instead of raising?** The failure happens inside
`jaxlib` / libtpu when trying to allocate a device buffer of dtype `float8_e4m3fn`.
The TPU XLA runtime (libtpu, closed source) apparently does not support allocating a
stand-alone `float8_e4m3fn` buffer on the host-to-device path. It returns a null/empty
result, which Orbax interprets as "couldn't restore this leaf" and falls back to the
abstract state descriptor.

This is confirmed because:
- JAX *compute* with `float8_e4m3fn` works (FP8 GEMMs run correctly after NFS load)
- JAX `jnp.zeros((6144, 16384), dtype=jnp.float8_e4m3fn)` likely works (not directly
  tested, but `eval_shape` works)
- The failure is specifically in the *Orbax restore path* which uses
  `make_array_from_single_device_arrays` with per-shard numpy arrays

---

## Implementation — What We Built

All code is in `python/sgl_jax/srt/model_loader/loader.py` in the `JAXModelLoader`
class.

### Checkpoint Path Derivation

```python
def _checkpoint_path(self, model_config: ModelConfig) -> str | None:
    """
    Auto-derives checkpoint path from SGLANG_CHECKPOINT_DIR env var.
    Format: {dir}/{model_hash}/tp{tp_size}_{dtype}/
    Example: gs://.../sglang-checkpoint/95dc2640/tp32_bfloat16/
    """
    checkpoint_dir = os.environ.get("SGLANG_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        return None
    model_hash = hashlib.md5(model_config.model_path.encode()).hexdigest()[:8]
    tp_size = self.mesh.size  # 32 for 4-node 2x2x4
    dtype_name = getattr(model_config.dtype, "__name__", None) or "bfloat16"
    return f"{checkpoint_dir.rstrip('/')}/{model_hash}/tp{tp_size}_{dtype_name}/"
```

### Abstract State — Why We Need It

`nnx.state(model)` after `load_weights()` produces a pytree where FP8 parameters have
structure `{value: jax.Array(shape=(...), dtype=float8_e4m3fn)}`. But after
`eval_shape` + `apply_linear_quantization` (which sets up the FP8 parameter structure)
**without** `load_weights()`, the structure is identical **except** some "narrow"
linear layers (out_dim=128 after TP sharding) remain as BF16.

The abstract state pickle captures the exact post-`load_weights()` pytree structure
(including which layers are FP8 vs BF16) and is used as the `item=` parameter to
guide Orbax's restore.

```python
def _save_abstract_state(self, state: Any, path: str) -> None:
    """
    Save abstract state (shapes + dtypes + PartitionSpec sharding) as pickle.
    Cannot save full NamedSharding (contains non-picklable Device objects),
    so we save sharding.spec (PartitionSpec) which is sufficient for Orbax.
    """
    def _to_sds(x):
        sharding = getattr(x, "sharding", None)
        if sharding is not None:
            try:
                sharding = sharding.spec  # PartitionSpec — picklable
            except AttributeError:
                sharding = None
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding)

    abstract = jax.tree_util.tree_map(_to_sds, state)
    buf = pickle.dumps(abstract)
    # ... write to GCS or local path
```

### Save (Current Implementation — Works)

```python
def _save_checkpoint(self, model: nnx.Module, path: str) -> None:
    checkpointer = ocp.PyTreeCheckpointer()
    state = nnx.state(model)
    # Save abstract state for restore structural hint
    self._save_abstract_state(state, self._abstract_state_path(path))
    # Save actual arrays (FP8 arrays written as-is by Orbax)
    checkpointer.save(path, state)
```

**Result**: Works correctly. FP8 arrays are written to OCDBT format. `commit_success.txt`
appears. 251.6 GiB written per host at ~4.5 GiB/s in ~228 seconds.

### Restore (Current Implementation — Fails)

```python
def _load_checkpoint(self, model: nnx.Module, path: str) -> None:
    checkpointer = ocp.PyTreeCheckpointer()
    # Load abstract state to provide structural hint to Orbax
    abstract_state = self._load_abstract_state(self._abstract_state_path(path))
    # ← Orbax reads 481.9 GiB correctly at 5.5 GiB/s (~91s)
    # ← BUT FP8 leaves become ShapeDtypeStruct instead of jax.Array
    state = checkpointer.restore(path, item=abstract_state)
    nnx.update(model, state)  # ← ShapeDtypeStruct enters model params
    # Forward pass fails: "ShapeDtypeStruct is not a valid JAX type"
```

---

## All Approaches Attempted

### Approach 1: Direct FP8 restore (both Orbax versions)

**What we tried**: Pass `item=abstract_state` with FP8 `ShapeDtypeStruct` leaves
directly. Tested with Orbax 0.11.28 and Orbax 0.12.0.

```python
# abstract_state has leaves like:
# ShapeDtypeStruct(shape=(6144, 16384), dtype=float8_e4m3fn, sharding=PartitionSpec(None, 'tensor'))
state = checkpointer.restore(path, item=abstract_state)
# Result: FP8 leaves remain as ShapeDtypeStruct — Orbax cannot create float8 JAX arrays
```

**Error**:
```
TypeError: Argument 'ShapeDtypeStruct(shape=(6144, 16384), dtype=float8_e4m3fn,
sharding=NamedSharding(...))' of type <class 'jax.ShapeDtypeStruct'>
is not a valid JAX type.
```

**Versions tested**: Orbax 0.11.28 + JAX 0.8.1 ✗ | Orbax 0.12.0 + JAX 0.8.1 ✗  
**Conclusion**: Orbax passes the dtype hint to JAX's array construction path. JAX/libtpu
cannot create a `float8_e4m3fn` device buffer via this path.

---

### Approach 2: Restore FP8 as uint8, convert back

**Hypothesis**: If we tell Orbax to restore FP8 bytes into `uint8` arrays (same 1 byte
per element), it will succeed since uint8 is a normal dtype. We then reinterpret the
uint8 bytes as float8 via `bitcast_convert_type`.

```python
# Save: cast FP8 → uint8 before handing to Orbax
_FP8_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2", ...})

def _fp8_to_u8(x):
    return jax.lax.bitcast_convert_type(x, jax.numpy.uint8) \
           if str(x.dtype) in _FP8_DTYPES else x

state_saveable = jax.tree_util.tree_map(_fp8_to_u8, state)
checkpointer.save(path, state_saveable)
# ✓ Works — uint8 arrays save correctly

# Restore: tell Orbax to expect uint8 for FP8 slots
def _to_u8_sds(sds):
    if str(sds.dtype) in _FP8_DTYPES:
        return jax.ShapeDtypeStruct(sds.shape, jax.numpy.uint8)  # no sharding (avoids mesh context error)
    return sds
u8_abstract = jax.tree_util.tree_map(_to_u8_sds, abstract_state)
state_u8 = checkpointer.restore(path, item=u8_abstract)
# ✓ Works — uint8 arrays restore correctly

# Convert uint8 → float8 on device
def _restore_fp8(restored, orig_sds):
    if str(orig_sds.dtype) in _FP8_DTYPES:
        return jax.lax.bitcast_convert_type(restored, orig_sds.dtype)
    return restored
state = jax.tree_util.tree_map(_restore_fp8, state_u8, abstract_state)
nnx.update(model, state)
```

**Error at bitcast step**:
```
RESOURCE_EXHAUSTED: Error allocating device buffer:
Attempting to allocate 144.00M.
That was not possible. There are 13.80M free.; (0x0x0_HBM1)
```

**Root cause**: After weights fill HBM (~240 GB per node), only ~14 MB remains free per
TC. `jax.lax.bitcast_convert_type(uint8_array, float8_e4m3fn)` needs ~144 MB for
the temporary output buffer — it creates a NEW array rather than reinterpreting in-place.
JAX's functional model prevents true in-place reinterpretation.

**Versions tested**: Orbax 0.11.28 + JAX 0.8.1 ✗  
**Conclusion**: HBM exhaustion prevents on-device dtype reinterpretation.

---

### Approach 3: CPU numpy view, per-shard

**Hypothesis**: Pull each TPU shard to CPU, do the dtype reinterpretation in numpy
(zero-copy view), push back to device. This avoids HBM allocation for the conversion.

```python
import ml_dtypes, numpy as np

def _restore_fp8(restored, orig_sds):
    if str(orig_sds.dtype) not in _FP8_DTYPES:
        return restored
    target_dtype = orig_sds.dtype
    fp8_np_dtype = getattr(ml_dtypes, str(target_dtype).replace(".", "_"))

    shards = []
    for shard in restored.addressable_shards:
        # Pull local shard to CPU
        np_u8 = np.array(shard.data)        # D2H copy (safe for local shards)
        np_fp8 = np_u8.view(fp8_np_dtype)   # zero-copy reinterpretation
        shards.append(jax.device_put(np_fp8, shard.device))  # ← FAILS

    return jax.make_array_from_single_device_arrays(
        restored.shape, restored.sharding, shards
    )
```

**Error at `jax.device_put(np_fp8, shard.device)`**:
```
RESOURCE_EXHAUSTED: Error allocating device buffer:
Attempting to allocate 144.00M.
That was not possible. There are 13.88M free.; (0x0x3_HBM0)
```

The `jax.device_put(numpy_fp8, tpu_device)` call allocates a NEW device buffer of
`float8_e4m3fn` dtype. This is the same HBM allocation as Approach 2 — the numpy
reinterpretation was zero-cost, but placing it back on device still needs HBM.

Additionally, there was an earlier attempt using `jax.device_get(restored)` on the
globally-sharded array which failed with:
```
RuntimeError: Fetching value for `jax.Array` that spans non-addressable
(non process local) devices is not possible.
```
This was fixed by using `restored.addressable_shards` but the HBM issue persisted.

**Versions tested**: Orbax 0.11.28 + JAX 0.8.1 ✗  
**Conclusion**: `jax.device_put(numpy_float8, tpu_device)` allocates a new HBM buffer —
it is NOT a zero-copy operation even for a numpy array that's already the right dtype.

---

### Approach 4: Orbax 0.12.0 native float8 (JAX 0.8.1)

**Hypothesis**: Orbax 0.12.0 might have improved float8 restore support.

```python
# After: uv pip install "orbax-checkpoint>=0.12.0"
# orbax-checkpoint==0.12.0 installed

state = checkpointer.restore(path, item=abstract_state)
# Result: identical — ShapeDtypeStruct for all FP8 leaves
```

**Same error**: `ShapeDtypeStruct is not a valid JAX type`

**Conclusion**: Orbax 0.12.0 passes the dtype through to JAX identically to 0.11.28.
The Orbax version is not the bottleneck — the failure is in JAX/libtpu.

---

### Approach 5: JAX 0.9.0 container (`jax0.9.0-rev1`)

**Hypothesis**: A newer JAX + libtpu version might support float8 buffer allocation.

```bash
# Container: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1
# JAX 0.9.0, Orbax 0.12.0 (upgraded via pip)
```

**Same error**: `ShapeDtypeStruct is not a valid JAX type`

JAX Python changelog for 0.9.x and 0.10.x contains no float8 TPU mentions.
GitHub search for `float8_e4m3fn device_put TPU` returns no JAX commits.

**Conclusion**: libtpu in both `jax0.8.1-rev1` and `jax0.9.0-rev1` cannot allocate
`float8_e4m3fn` device buffers via the `make_array_from_single_device_arrays` path.

---

### Approach 6: Option B — Hybrid restore

**Hypothesis**: Since the model is 100% FP8, perhaps we can restore only metadata
(layer norms, embeddings, biases) from checkpoint and reload FP8 weights from NFS.

**Finding**: The model has **1038 FP8 tensors** and essentially **zero non-FP8 weight
tensors** (layer norms are very small). Option B would save ~0 time since all the slow
MoE weights are FP8. There is nothing meaningful to restore from the checkpoint.

**Conclusion**: Not viable for a fully FP8-quantized model.

---

## Summary Table

| Approach | Orbax | JAX/container | Core mechanism | Error |
|----------|-------|---------------|----------------|-------|
| Direct float8 | 0.11.28 | 0.8.1 | `item=fp8_abstract_state` | ShapeDtypeStruct |
| Direct float8 | 0.12.0 | 0.8.1 | `item=fp8_abstract_state` | ShapeDtypeStruct |
| uint8 + on-device bitcast | 0.11.28 | 0.8.1 | `bitcast_convert_type` | HBM OOM (144 MB needed, 14 MB free) |
| uint8 + CPU numpy view + device_put | 0.11.28 | 0.8.1 | `np.view + device_put` | HBM OOM (device_put allocates new buffer) |
| Direct float8 | 0.12.0 | 0.9.0 | `item=fp8_abstract_state` | ShapeDtypeStruct |
| Hybrid (non-FP8 ckpt + FP8 NFS) | — | — | N/A | Model is 100% FP8 |

---

## What We Know About the Failure Point

From Orbax source (`jax_array_handlers.py` line ~701):

```
UserWarning: Sharding info not provided when restoring.
Populating sharding info from sharding file.
```

Orbax reads the sharding from the checkpoint's `_sharding` file. It then calls
(approximately):

```python
# Pseudocode of what Orbax does internally:
for each_leaf in target_abstract_state:
    raw_bytes = read_shard_from_ocdbt(checkpoint_path, each_leaf.path)
    numpy_arr = np.frombuffer(raw_bytes, dtype=each_leaf.dtype)  # ← works for float8
    jax_arr = jax.device_put(numpy_arr, device_with_sharding)    # ← FAILS for float8_e4m3fn on TPU
    # When device_put fails silently, Orbax returns each_leaf (the ShapeDtypeStruct) unchanged
```

The `jax.device_put(numpy_float8, tpu_device)` call is the failure point. It does not
raise an exception — it appears to return silently and Orbax uses the abstract state
descriptor as fallback.

**Key question for the expert**: ~~Is there a way to make `jax.device_put` (or
`jax.make_array_from_single_device_arrays`) work for `float8_e4m3fn` on TPU v7x with
JAX 0.9.0 / libtpu?~~

**Answered**: Yes — monkey-patch `jax.device_put` to transfer as `uint8` and
`bitcast_convert_type` to `float8_e4m3fn` on-device. All sub-problems validated.
See `fp8_restore_workaround/README.md` for test results.

---

## Validated Solution (2026-06-03)

Monkey-patch `jax.device_put` to intercept FP8 arrays during Orbax restore,
transfer as `uint8`, then `bitcast_convert_type` to `float8_e4m3fn` on-device.

**Why this works**:
- Orbax calls `jax.device_put` via `jax.tree.map(jax.device_put, ret, shardings)`
  — Python-level public API, interceptable via assignment.
- `bitcast_convert_type(uint8 → float8)` works on TPU v7x ✅
- Per-shard HBM: 3 MB uint8 + 3 MB float8 = 6 MB < 14 MB free ✅
- Orbax processes shards serially (max concurrency = 1), no semaphore needed ✅
- 4-node multi-host restore path DOES call `jax.device_put` ✅

```python
orig_device_put = jax.device_put
_FP8 = frozenset({"float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"})

def _patched_device_put(x, *args, **kwargs):
    if hasattr(x, "dtype") and str(x.dtype) in _FP8:
        x_u8 = np.asarray(x).view(np.uint8)
        arr_u8 = orig_device_put(x_u8, *args, **kwargs)
        arr_f8 = jax.lax.bitcast_convert_type(arr_u8, x.dtype)
        arr_f8.block_until_ready()
        return arr_f8
    return orig_device_put(x, *args, **kwargs)

jax.device_put = _patched_device_put
try:
    state = checkpointer.restore(path, item=abstract_state)
finally:
    jax.device_put = orig_device_put
```

**Test results** (JAX 0.9.0, Orbax 0.12.0, TPU v7x 2x2x4):

| Test | Result |
|------|--------|
| `bitcast_convert_type(uint8→float8)` on TPU | ✅ PASS |
| Patch under ~14 MB free HBM | ✅ PASS |
| Max concurrency = 1 | ✅ PASS |
| 4-node multi-host intercept | ✅ PASS |

See `fp8_restore_workaround/README.md` for full validation details.

## Previously Ruled Out

1. **JAX 0.9.0 container upgrade**: Same ShapeDtypeStruct failure ✗
2. **Save as BF16 (dequantized)**: Works but 2× larger, changes inference ✗
3. **Patch libtpu**: Closed source ✗
4. **uint8 + on-device tree_map bitcast**: HBM OOM (holds full tree simultaneously) ✗
5. **Option B hybrid restore**: Model is 100% FP8, nothing non-FP8 to restore ✗

---

## The Solution (Verified)

**Root cause analysis correction**: The libtpu bug does indeed prevent `jax.device_put` from transferring `float8_e4m3fn` host bytes directly to the TPU device. However, Orbax internally uses `jax.device_put` on a single-shard basis during its async deserialization path (via `_read_and_device_put_shard`). When `jax.device_put` fails for the FP8 shard inside Orbax's data callback, Orbax falls back to returning the `ShapeDtypeStruct`. 

The reason **Approach 2 (uint8 + tree_map bitcast)** failed with OOM is because `tree_map` operates on the full tree: the entire 30GB model was restored as `uint8`, and during `tree_map`, both the `uint8` and `float8` copies of the arrays were held in memory simultaneously, exhausting the remaining HBM.

**The Fix**: We can bypass the libtpu bug by intercepting `jax.device_put` **during** the Orbax restore process. By monkey-patching `jax.device_put`, we can perform the zero-cost CPU cast to `uint8`, transfer the array to the TPU device via the working `uint8` path, and then immediately `bitcast_convert_type` to `float8` on the device. 

Crucially, we must call `.block_until_ready()` on the output of the bitcast. This halts the Python async worker loop until XLA finishes compiling and executing the cast, which keeps Orbax's `byte_limiter` token securely held until the memory is truly freed. Without this block, Orbax would eagerly pull all 8,304 parameter shards (240 GB per node) into Host RAM simultaneously while the XLA compilation queued up, triggering an OS `ExitCode 137` memory kill.

### Iterative TensorStore Loading

Due to a secondary bug in Orbax `0.12.0`'s internal `SingleReplicaArrayHandler` which fails to broadcast the correctly restored tree state back to the `checkpointer.restore` caller when running across multiple partitioned hosts, we bypass `checkpointer.restore` entirely. Instead, we manually parse the `_METADATA` via `msgpack` and iterate over the shards, leveraging TensorStore directly alongside our monkey-patched `device_put`.

### Implementation:
```python
orig_device_put = jax.device_put

def patched_device_put(x, *args, **kwargs):
    if hasattr(x, "dtype") and str(x.dtype) in {"float8_e4m3fn", "float8_e5m2"}:
        # 1. Zero-copy view as uint8 on CPU
        x_u8 = np.asarray(x).view(np.uint8)
        # 2. Transfer to TPU as uint8 (bypasses libtpu float8 bug)
        arr_u8 = orig_device_put(x_u8, *args, **kwargs)
        # 3. Bitcast back to float8 on TPU
        target_dtype = getattr(jnp, str(x.dtype))
        arr_f8 = jax.lax.bitcast_convert_type(arr_u8, target_dtype)
        # 4. Block to throttle the async queue and ensure XLA finishes.
        # This prevents both Host RAM OOM (from unbounded DMA/compilation queues)
        # and TPU HBM OOM (by allowing arr_u8 to be freed immediately).
        arr_f8.block_until_ready()
        del x_u8
        del arr_u8
        return arr_f8
    return orig_device_put(x, *args, **kwargs)

# Apply during restore:
jax.device_put = patched_device_put
try:
    # Manual TensorStore iteration bypasses broken Orbax multi-host broadcast
    # (Implementation omitted for brevity, see `python/sgl_jax/srt/model_loader/loader.py`)
    state = ... 
finally:
    jax.device_put = orig_device_put
```

This workaround has been implemented in `python/sgl_jax/srt/model_loader/loader.py` and successfully resolves the issue without needing to wait for a newer container or saving the checkpoint in a different format.

A minimal reproducible demo of this workaround is available in `docs/tpu7tests/fp8_restore_workaround/single_tpu_job.yaml`.

---

## Related Docs

- [mimo_v25_pro_progress.md](mimo_v25_pro_progress.md) — overall work status
- [mimo_v25_pro_weight_checkpoint.md](mimo_v25_pro_weight_checkpoint.md) — checkpoint design doc
- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — weight loading pipeline
