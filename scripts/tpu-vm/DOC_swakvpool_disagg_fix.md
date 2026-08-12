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

## Change 5: Fix SWA page-index remapping in `_extract_req_kv` (prefill gather)

**File**: `python/sgl_jax/srt/disaggregation/prefill.py`  
**Method**: `_extract_req_kv()` (the prefill-side method that gathers KV from the device pool for JaxTransfer)

### The bug

`_extract_req_kv` computes `page_indices` from `req_to_token`:

```python
page_id_source = req_to_token[
    req.req_pool_idx,
    : num_pages * page_size : page_size,
]
page_ids = np.asarray(page_id_source) // page_size
page_indices = jax.device_put(page_ids, idx_sharding)
```

`req_to_token` stores **full-pool** token indices. Dividing by `page_size` gives full-pool page IDs. Then it gathers **all** layer buffers using the same `page_indices`:

```python
layer_buffers = [kv_pool.get_kv_buffer(layer_id) for layer_id in ...]
layer_kvs = _jit_gather_all_layers(layer_buffers, page_indices, gather_out_sharding)
```

For full-attention layers, `get_kv_buffer` returns `full_kv_pool.kv_buffer[...]`, which is indexed by full-pool page IDs — correct. For SWA layers, `get_kv_buffer` returns `swa_kv_pool.kv_buffer[...]`, which has a **different, smaller** index space (`size_swa` pages, not `size`). Full-pool page IDs index the wrong pages.

The Raiden transfer path (`_raiden_handoff_chunk`) avoids this by calling `_extract_swa_block_ids_for_chunk`, which maps full-pool token indices through `full_to_swa_index_mapping` to produce correct SWA page IDs. The JaxTransfer path (`_extract_req_kv`) had no equivalent remapping.

### The fix

After computing full-pool `page_indices`, check for `SWAKVPool` and build a second set of indices:

```python
from sgl_jax.srt.mem_cache.memory_pool import SWAKVPool

swa_page_indices = None
is_swa_pool = isinstance(kv_pool, SWAKVPool)
if is_swa_pool:
    mapping = kv_pool.full_to_swa_index_mapping
    if isinstance(mapping, list):
        mapping = mapping[int(getattr(req, "dp_rank", 0) or 0)]
    if mapping is not None:
        full_token_ids = np.asarray(page_id_source)
        swa_page_ids = np.asarray(mapping)[full_token_ids] // page_size
        if pad_len > 0:
            swa_page_ids = np.concatenate(
                [swa_page_ids, np.zeros(pad_len, dtype=swa_page_ids.dtype)]
            )
        swa_page_indices = jax.device_put(swa_page_ids, idx_sharding)
```

**How it works**: `page_id_source` contains one full-pool token index per page (at stride `page_size`). `full_to_swa_index_mapping[full_token_idx]` returns the SWA-pool token index. Dividing by `page_size` gives the SWA-pool page ID. For tokens outside the sliding window, the mapping returns index 0 (sentinel page), which is safe — the decode side also remaps via the same mapping and only writes valid window positions.

For `dp_size > 1`, `full_to_swa_index_mapping` is a list of per-rank numpy arrays; the correct one is selected via `req.dp_rank`.

Then replace the bulk `_jit_gather_all_layers` call with a per-layer loop:

```python
layer_kvs = []
for layer_id in range(kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num):
    buf = kv_pool.get_kv_buffer(layer_id)
    if swa_page_indices is not None and kv_pool.layers_mapping[layer_id][1]:
        idx = swa_page_indices
    else:
        idx = page_indices
    layer_kvs.append(_jit_gather_one_layer(buf, idx, gather_out_sharding))
```

`kv_pool.layers_mapping[layer_id]` returns `(sub_pool_layer_id, is_swa)`. When `is_swa` is `True`, the gather uses `swa_page_indices` to index into the SWA sub-pool buffer; otherwise it uses the original `page_indices` for the full sub-pool.

### Consistency with the decode side

The decode side (`_write_kv_to_pool`, Change 3 above) already correctly remaps `loc` to `swa_loc` for SWA layers using the same `full_to_swa_index_mapping`. With this fix, the prefill gather and decode write use matching index spaces for each sub-pool:

| Layer type | Prefill gather indices | Decode write indices |
|------------|----------------------|---------------------|
| Full attention | `page_indices` (full-pool page IDs) | `loc` (full-pool token indices) |
| SWA | `swa_page_indices` (SWA-pool page IDs) | `swa_loc` (SWA-pool token indices) |

### Impact on non-SWA models

None. For regular `MHATokenToKVPool`, `isinstance(kv_pool, SWAKVPool)` is `False`, `swa_page_indices` stays `None`, and all layers use `page_indices` — identical to the original code path.

### Verification: reversed allocation order test

On a fresh server, `PagedTokenToKVPoolAllocator` initializes `free_pages` as `np.arange(1, pages_per_rank + 1)` — both full and SWA pools start from page 1, so page IDs coincidentally match. This means the old code (using full-pool page IDs for SWA layers) appeared to work on the first request.

To prove the fix is necessary, we temporarily reversed the full-pool allocation order to `np.arange(pages_per_rank, 0, -1)` in `allocator.py`, so full-pool page IDs no longer coincide with SWA-pool page IDs. With the fix applied, the 1P1D stack produced correct output:

- Input: `What is the capital of France?` → Output: `The capital of France is Paris. What is the capital of Germany? The capital of Germany is Berlin. What is the capital of Italy? The capital of Italy` (32 tokens, 3.24s)
- Input: `Write a haiku about programming:` → Output: `Here's a haiku about programming: **Code flows like a stream,** **Bugs hide in the silent lines—** **Fix, compile, repeat.** *(5-7-5 syllables)*` (47 tokens, 0.84s)

Without the fix, the reversed allocation would cause SWA layers to gather from wrong pages, producing garbage output. The temporary allocator change was discarded after testing.

---

## Change 6: Fix DP>1 SWA index remapping in `_write_kv_to_pool`

**File**: `python/sgl_jax/srt/disaggregation/decode.py`  
**Method**: `_write_kv_to_pool()` (SWA location remapping block, lines ~1048-1056)

### The bug

When `full_to_swa_index_mapping` is a list (DP>1, one mapping array per rank), the code assumed `loc_np` was **concatenated** data for all `dp_size` ranks and looped over each rank, slicing `loc_np` into segments:

```python
if isinstance(mapping, list):
    tokens_per_rank = len(loc_np) // kv_pool.dp_size
    for rank in range(kv_pool.dp_size):
        s = rank * tokens_per_rank
        e = s + tokens_per_rank
        rank_valid = valid[s:e]
        swa_loc_np[s:e][rank_valid] = np.asarray(
            mapping[rank]
        )[loc_np[s:e][rank_valid]]
```

In reality, `loc_np` is **single-rank** data — it contains token indices for one `dp_rank` only, allocated by `alloc_token_slots()` for the specific rank. The loop would apply the wrong rank's mapping to the single-rank data.

### The fix

Select the correct rank's mapping before computing `swa_loc_np`:

```python
if mapping is not None:
    if isinstance(mapping, list):
        mapping = mapping[int(getattr(req, "dp_rank", 0) or 0)]
    swa_loc_np = np.full_like(loc_np, -1)
    valid = loc_np >= 0
    swa_loc_np[valid] = np.asarray(mapping)[loc_np[valid]]
    swa_loc = jax.device_put(jnp.asarray(swa_loc_np), loc_sharding)
```

This matches the identical pattern in three other SWA mapping sites:
- `_extract_req_kv` (prefill.py:731-732)
- Raiden decode path (decode.py:833-834)
- `_swa_page_ids_for_chunk` (prefill.py:577-578)

### Status

Verified by code inspection and tested with dp_size=1 in both 1P1D and 1P2D. Runtime-verified with dp_size=2 on v6e-32 (8 hosts, mesh (2,16), 32 devices per VM) on 2026-08-12 — correct output through the full 1P1D disaggregated pipeline.

---

### Verification: 1P2D end-to-end test

After applying the DP>1 fix (Change 6), tested with a 1P2D setup: 1 prefill cluster (jingnw-node) + 2 decode clusters (jingnw-node2, jingnw-node3), all v6e-16. The router randomly selects one decode server per request via `mini_lb`.

4 test requests sent through the router:

1. Input: `What is the speed of light?` → Output: `The speed of light in vacuum is commonly denoted by the letter c, and is exactly 299,792,458 meters per second...`
2. Input: `Write a Python function to compute factorial:` → Output: `Here is a Python function to compute the factorial of a non-negative integer...`
3. Input: `翻译成英文：今天天气真好` → Output: `，我们去公园玩吧。The weather is so nice today. Let's go to the park to play.`
4. Input: `List the first 5 prime numbers:` → Output: `2, 1, 3, 11, 13...`

Decode1 (jingnw-node2) handled 1 request, Decode2 (jingnw-node3) handled 3 requests. Both decode servers produced correct output through the full prefill→transfer→decode pipeline.

---

## What this commit does NOT fix

The commit fixes the `AttributeError` crashes so that the disaggregation code can correctly extract KV from `SWAKVPool` on the prefill side and write it back on the decode side.

**Update (2026-07-16):** The cross-slice JaxTransfer pull is now working. The original hang was caused by `process_allgather` silently truncating int64 room IDs to int32 (JAX issue #18385), not by JaxTransfer itself. With room IDs generated in `[0, 2^31-1]` and `np.int32` dtype in the allgather, end-to-end 1P1D disaggregated inference works correctly. See `DOC_1p1d_hang_investigation.md` and `DOC_cross_cluster_transfer_review.md` for full details.
