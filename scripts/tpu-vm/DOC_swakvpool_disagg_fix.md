# Commit 607351f: SWAKVPool Support in JaxTransfer KV Disaggregation

**Commit**: `607351f0 fix(disagg): support SWAKVPool in JaxTransfer KV disaggregation path`  
**Files changed**: 3 (`memory_pool.py`, `prefill.py`, `decode.py`)  
**Lines**: +76 / -29

---

## Background

MiMo-V2-Flash is a hybrid-attention model: some layers use **full attention** and others use **sliding-window attention (SWA)**. At runtime this means the KV cache is managed by `SWAKVPool` instead of `MHATokenToKVPool`.

`SWAKVPool` wraps two independent sub-pools:
- `full_kv_pool` — an `MHATokenToKVPool` for full-attention layers
- `swa_kv_pool` — an `MHATokenToKVPool` for SWA layers (smaller, since SWA only needs a window of tokens)

The two sub-pools have different sizes, different layer counts, and different token index spaces. A mapping array (`full_to_swa_index_mapping`) translates full-pool token indices into SWA-pool token indices.

In **NonPD mode** (single VM), the KV cache never leaves the device — `SWAKVPool.get_kv_buffer()` and `set_kv_buffer()` handle the sub-pool dispatch transparently. No problem.

In **1P1D disaggregated mode**, the prefill server must **extract** KV data from its pool and send it via JaxTransfer, and the decode server must **receive** KV data and **write** it into its pool. This extraction/write code directly accesses pool attributes (`kv_sharding`, `layer_num`, `start_layer`, `kv_buffer`, `dtype`, `attention_data_partition_axis`) that exist on `MHATokenToKVPool` but **not** on `SWAKVPool`.

---

## Change 1: Add property accessors to `SWAKVPool`

**File**: `python/sgl_jax/srt/mem_cache/memory_pool.py`  
**Location**: After `replace_buffer()` (inserted before `remap_cache_loc()`)

```python
@property
def layer_num(self):
    return self.swa_layer_nums + self.full_layer_nums
```
Returns the **total** number of KV-cache layers across both sub-pools. The disaggregation code uses this to iterate over all layers when extracting/writing KV data.

```python
@property
def start_layer(self):
    return min(self.layers_mapping.keys())
```
Returns the smallest global layer ID. `layers_mapping` maps global layer IDs to `(sub_pool_layer_id, is_swa)` tuples. `start_layer` is used as the beginning of the `range(start_layer, start_layer + layer_num)` loop.

```python
@property
def kv_sharding(self):
    return self.full_kv_pool.kv_sharding
```
Returns the `NamedSharding` used for KV buffers. Both sub-pools use the same sharding layout (same mesh, same partition spec), so delegating to `full_kv_pool` is correct. This is used to determine the partition spec for gather output sharding and for building `ShapeDtypeStruct` specs for JaxTransfer pulls.

```python
@property
def dtype(self):
    return self.full_kv_pool.dtype
```
Returns the dtype of KV buffers (e.g. `bfloat16`). Both sub-pools use the same dtype. Used by `local_kv_spec_for_pool` to construct `jax.ShapeDtypeStruct`.

```python
@property
def attention_data_partition_axis(self):
    return self.full_kv_pool.attention_data_partition_axis
```
Returns the partition axis name for the token/data dimension (typically `"data"`). Used when sharding `loc` (the per-token write indices) for the Pallas write kernel.

---

## Change 2: Replace `kv_buffer[0]` with `get_kv_buffer(start_layer)`

`SWAKVPool` has no `kv_buffer` list — each sub-pool has its own. Direct `kv_pool.kv_buffer[0]` access raises `AttributeError`. The fix replaces it with `kv_pool.get_kv_buffer(kv_pool.start_layer)`, which dispatches to the correct sub-pool via `layers_mapping`.

All three call sites only use `kv_buffer[0]` to get `.shape[1:]` (the per-layer tail shape: `(page_size, num_heads, head_dim)` or similar). The shape is the same for both sub-pools (same head count and dim), so using any layer's buffer is correct.

### 2a. `prefill.py` — `local_kv_spec_for_pool()` (line 108)

```python
# Before:
per_layer_tail = kv_pool.kv_buffer[0].shape[1:]

# After:
per_layer_tail = kv_pool.get_kv_buffer(kv_pool.start_layer).shape[1:]
```

This function builds a `jax.ShapeDtypeStruct` describing the shape of KV data that each host will pull in multi-host mode. `per_layer_tail` gives the dimensions after `(padded_pages,)`.

### 2b. `decode.py` — `_build_kv_spec_for_req()` (line 985)

```python
# Before:
per_layer_tail = kv_pool.kv_buffer[0].shape[1:]

# After:
per_layer_tail = kv_pool.get_kv_buffer(kv_pool.start_layer).shape[1:]
```

Builds per-layer `ShapeDtypeStruct` specs that tell JaxTransfer what shape/dtype/sharding to expect when pulling KV from the prefill server.

### 2c. `decode.py` — `_write_kv_to_pool()` (line 1010)

```python
# Before:
per_layer_tail = kv_pool.kv_buffer[0].shape[1:]

# After (moved earlier, before the if-block):
per_layer_tail = kv_pool.get_kv_buffer(kv_pool.start_layer).shape[1:]
```

The `per_layer_tail` lookup was moved from inside the `if jax.process_count() > 1` block to before it, so it's available regardless of host count. Same purpose: get the shape of a single layer's KV buffer.

---

## Change 3: Rewrite `_write_kv_to_pool` for SWAKVPool

**File**: `python/sgl_jax/srt/disaggregation/decode.py`  
**Method**: `_write_kv_to_pool()` (the decode-side method that writes received KV data into the local paged pool)

### 3a. Pre-compute SWA location remapping

```python
from sgl_jax.srt.mem_cache.memory_pool import SWAKVPool

is_swa_pool = isinstance(kv_pool, SWAKVPool)
swa_loc = None
if is_swa_pool:
    mapping = kv_pool.full_to_swa_index_mapping
    if mapping is not None:
        swa_loc_np = np.full_like(loc_np, -1)
        valid = loc_np >= 0
        if isinstance(mapping, list):
            # DP>1: per-rank remapping
            tokens_per_rank = len(loc_np) // kv_pool.dp_size
            for rank in range(kv_pool.dp_size):
                s = rank * tokens_per_rank
                e = s + tokens_per_rank
                rank_valid = valid[s:e]
                swa_loc_np[s:e][rank_valid] = np.asarray(
                    mapping[rank]
                )[loc_np[s:e][rank_valid]]
        else:
            # DP=1: single mapping array
            swa_loc_np[valid] = np.asarray(mapping)[loc_np[valid]]
        swa_loc = jax.device_put(jnp.asarray(swa_loc_np), loc_sharding)
```

**Why this is needed**: The prefill server extracts KV using **full-pool indices** (the standard token slots). Full-attention layers can use these indices directly to write into `full_kv_pool`. But SWA layers use a **different, smaller index space** — the same token might be at index 500 in the full pool but index 42 in the SWA pool. `full_to_swa_index_mapping` is a lookup table that translates full→SWA indices.

`swa_loc` is the remapped version of `loc` (the per-token write indices) for SWA layers. For DP>1 (data parallel), each rank has its own mapping array, so the remapping is done per-rank.

`loc_np[i] == -1` means "padding, skip this token" — the remapped `swa_loc_np` preserves these as -1.

### 3b. Per-layer write with sub-pool dispatch

```python
for i, layer_id in enumerate(
    range(kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num)
):
    if is_swa_pool:
        layer_id_pool, is_swa_layer = kv_pool.layers_mapping[layer_id]
        if is_swa_layer:
            sub_pool = kv_pool.swa_kv_pool
            layer_loc = swa_loc if swa_loc is not None else loc
        else:
            sub_pool = kv_pool.full_kv_pool
            layer_loc = loc
        sub_pool.kv_buffer[layer_id_pool] = write_kv_layer(
            kv[i],
            layer_loc,
            sub_pool.kv_buffer[layer_id_pool],
            page_size,
            sub_pool.kv_partition_axis,
            sub_pool.attention_data_partition_axis,
            sub_pool.mesh,
        )
    else:
        # Original path for regular KVPool (unchanged)
        layer_idx = layer_id - kv_pool.start_layer
        kv_pool.kv_buffer[layer_idx] = write_kv_layer(
            kv[i],
            loc,
            kv_pool.kv_buffer[layer_idx],
            page_size,
            kv_pool.kv_partition_axis,
            kv_pool.attention_data_partition_axis,
            kv_pool.mesh,
        )
```

**What this does for each layer**:

1. **Look up the layer's sub-pool**: `layers_mapping[layer_id]` returns `(sub_pool_layer_id, is_swa)`. For example, global layer 5 might map to `(2, True)` meaning it's the 3rd layer in `swa_kv_pool`.

2. **Pick the right write indices**: SWA layers use `swa_loc` (remapped indices); full-attention layers use `loc` (original indices).

3. **Write to the sub-pool's buffer**: `sub_pool.kv_buffer[layer_id_pool]` addresses the correct buffer within either `swa_kv_pool` or `full_kv_pool`, using the sub-pool-local layer index.

The `else` branch (regular `KVPool`) is the original code, unchanged.

---

## Change 4: Minor cleanup

- Removed several comments that described what the code does (e.g., "Pulled KV is this host's local shard...", "page_ids_padded is only consumed by the debug verifier...", "Write via the in-place Pallas kernel..."). The code is self-explanatory.
- Extracted `loc_sharding` into a local variable since it's now used in two places (once for `loc`, once for `swa_loc`).

---

## What this commit does NOT fix

The commit fixes the `AttributeError` crashes so that the disaggregation code can correctly extract KV from `SWAKVPool` on the prefill side and write it back on the decode side.

**Update (2026-07-16):** The cross-slice JaxTransfer pull is now working. The original hang was caused by `process_allgather` silently truncating int64 room IDs to int32 (JAX issue #18385), not by JaxTransfer itself. With room IDs generated in `[0, 2^31-1]` and `np.int32` dtype in the allgather, end-to-end 1P1D disaggregated inference works correctly. See `DOC_1p1d_hang_investigation.md` and `DOC_cross_cluster_transfer_review.md` for full details.
