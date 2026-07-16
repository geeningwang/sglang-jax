# Cross-Cluster JAX Transfer Code Review

**Date**: 2026-07-14 (updated 2026-07-16)  
**Context**: 1P1D disaggregated inference for MiMo-V2-Flash on TPU v6e-16.  
**Prefill VM**: jingnw-node (4 workers, w0=10.202.0.29)  
**Decode VM**: jingnw-node2 (4 workers, w0=10.202.15.227)  
**Branch**: mimo-tpu7-stage3  
**Status**: RESOLVED — 1P1D end-to-end working as of 2026-07-16

---

## 1. Problem Statement

With commit 607351f applied, 1P1D requests complete with 0 output tokens. The
`link.pull()` call on the decode side hangs indefinitely. The 30s pull timeout
reaper cannot interrupt the native C `link.pull()` — it only flips a state flag,
leaving the worker thread permanently stuck.

We needed to determine: is the hang caused by a code bug in the transfer path,
or by `jax.experimental.transfer` not supporting cross-cluster transfers?

---

## 2. Cross-Cluster Transfer Test Result

A standalone test (`scripts/tpu-vm/test_cross_cluster_transfer/`) confirmed that
`jax.experimental.transfer` **works correctly** between the two VMs:

| Worker pair | Producer → Consumer | Pull time | Result |
|---|---|---|---|
| w0 | 10.202.0.29 → 10.202.15.227 | 0.0s | SUCCESS |
| w1 | 10.202.15.197 → 10.202.15.230 | 0.0s | SUCCESS |
| w2 | 10.202.15.194 → 10.202.15.229 | 0.0s | SUCCESS |
| w3 | 10.202.15.202 → 10.202.15.228 | 0.0s | SUCCESS |

Each VM ran `jax.distributed.initialize()` with its own coordinator (separate
JAX clusters). Producer called `server.await_pull(uuid, array)`, consumer
called `link.pull(uuid, spec)`. Transfer was instant on all 4 worker pairs.

**Conclusion**: Cross-cluster transfer is fully supported. The hang is NOT
caused by a transport-layer limitation.

---

## 3. Transfer Path Code Review

### 3.1 Prefill Side (data registration)

**Flow**: `_extract_req_kv()` → `attach_payload({"kv": device_kv})` → `sender.send()` → `producer_handoff()` → `wrapper.register_pull(sub_uuid, data)` → `server.await_pull(uuid_int, data)`

Key files and lines:
- `prefill.py:691-743` — `_extract_req_kv()`: gathers KV from paged pool per layer
- `prefill.py:734-740` — Multi-host: `jnp.stack(layer_kvs, axis=0)` → `_global_to_local_shard(stacked)` → returns **single stacked array**
- `prefill.py:742-743` — Single-host: returns **list of arrays** (one per layer)
- `prefill.py:98-120` — `local_kv_spec_for_pool()`: builds matching `ShapeDtypeStruct` for multi-host
- `prefill.py:410-418` — sender init, attach payload, send
- `conn.py:315-377` — `producer_handoff()`: registers each payload entry as `f"{uuid}:{name}"` (i.e., `f"{transfer_id}:kv"`)
- `wrapper.py:132-177` — `register_pull()`: calls `server.await_pull(_uuid_to_int(uuid), data)`
- `wrapper.py:39-47` — `_uuid_to_int()`: `zlib.crc32(uuid.encode("utf-8")) & 0xFFFFFFFF`

### 3.2 Decode Side (data pull)

**Flow**: `_build_kv_spec_for_req()` → `PMetadata(specs={"kv": spec})` → `receiver.init(p_metadata)` → enqueue → `_run_pull()` → `wrapper.pull(sub_uuid, spec, remote_addr)` → `link.pull(uuid_int, spec)`

Key files and lines:
- `decode.py:971-991` — `_build_kv_spec_for_req()`:
  - Multi-host (line 981-984): calls `local_kv_spec_for_pool()` → returns **single ShapeDtypeStruct**
  - Single-host (lines 985-991): returns **list of ShapeDtypeStruct** (one per layer)
- `decode.py:707-740` — admission: builds `PMetadata`, inits receiver
- `decode.py:714-722` — `PMetadata` construction:
  - `remote_addr = f"{p_info['host']}:{p_info['transfer_port']}"`
  - `uuid = entry.req.disagg_transfer_id or entry.req.rid`
  - `specs = {"kv": spec}`
- `conn.py:1200-1235` — `_run_pull()`: iterates `specs.items()`, builds `sub_uuid = f"{uuid}:kv"`, calls `wrapper.pull()`
- `wrapper.py:189-219` — `pull()`: connects to remote via `server.connect(remote_addr)`, calls `link.pull(_uuid_to_int(uuid), spec)`

### 3.3 Bootstrap and Routing

- `runtime.py:233-250` — Each prefill worker registers with bootstrap: host IP, transfer_port, side_channel_port, `jax_process_index`, `jax_process_count`
- `host_ip.py:67-102` — `_validate()` rejects `0.0.0.0`, `127.0.0.1`, localhost — only routable IPs allowed
- `decode.py:574-608` — `_pick_prefill_peer_for_this_host()`: matches D worker to P worker by `jax_process_index`. Verifies `jax_process_count` matches. D worker N always pulls from P worker N.
- `bootstrap.py:806` — Room-based selection: `sorted(keys)[room % len]`

### 3.4 Multi-Host Synchronization

- `decode.py:610-644` — `_drain_transfer_queue_synced()`: uses `multihost_utils.process_allgather()` — all NPs must agree on which rooms succeeded/failed before KV write-back
- A room stays in TRANSFERRING until every worker's receiver reaches a terminal state
- If any worker times out → FAILED propagated to all workers for that room

---

## 4. Verification Checklist

| Check | Result |
|---|---|
| UUID string matches (P vs D) | ✅ Both use `req.disagg_transfer_id or req.rid` |
| Sub-UUID format matches | ✅ Both use `f"{transfer_id}:kv"` |
| CRC32 conversion matches | ✅ Same `_uuid_to_int()` function |
| Pytree structure matches (multi-host) | ✅ Single array ↔ single ShapeDtypeStruct |
| Pytree structure matches (single-host) | ✅ List of arrays ↔ list of ShapeDtypeStruct |
| Shape matches | ✅ Same `padded_pages` via `_pad_to_page_bucket()` |
| SWA vs full buffer shapes | ✅ Both replicate to 16 KV heads at TP=16 (4→16, 8→16) |
| Sharding matches | ✅ Same `_local` mesh construction on both sides |
| Remote address = bind address | ✅ `_validate()` prevents 0.0.0.0 mismatch |
| Worker-to-worker routing | ✅ Matched by `jax_process_index` |
| Cross-cluster transfer works | ✅ Confirmed by standalone test (0.0s latency) |

---

## 5. MiMo-V2-Flash Model Config (relevant fields)

```
num_key_value_heads: 4        → replicated to 16 at TP=16
swa_num_key_value_heads: 8    → replicated to 16 at TP=16
head_dim: 192                 → aligned to 192 (already 128-aligned)
swa_head_dim: 192             → same as full
num_hidden_layers: 48
```

Buffer shape per layer (both full and SWA at TP=16):
`(pool_pages, page_size, 16*2//2, 2, 192)` = `(pool_pages, page_size, 16, 2, 192)`

Shapes match → `jnp.stack()` in multi-host path works correctly.

---

## 6. ROOT CAUSE FOUND: int64 Truncation in `synced_terminal_rooms`

### Summary

The KV transfer itself works perfectly — `link.pull()` completes instantly on
all 4 decode workers. The bug is in the **multi-host drain synchronization**
(`synced_terminal_rooms` in `multihost_sync.py`), which silently truncates
64-bit `bootstrap_room` IDs to 32 bits.

### Mechanism

1. `bootstrap_room` is a 64-bit Python int (e.g. `5690320318231820188`)
2. `synced_terminal_rooms()` stores it in a `np.int64` array
3. `multihost_utils.process_allgather()` converts the NumPy array to JAX
4. **JAX silently truncates int64 → int32** because `jax_enable_x64` is `False`
   by default on TPU
5. The returned room ID is the lower 32 bits only:
   `5690320318231820188 & 0xFFFFFFFF = 1248147356`
6. The drain code compares `room in success` using the original 64-bit value
   (`5690320318231820188`) against the truncated set (`{1248147356}`) → no match
7. The request is **never drained** from the transfer queue
8. The scheduler loops forever calling `synced_terminal_rooms` → allgather →
   mismatch → not drained → loop

### Evidence from decode w0 log

```
synced_terminal_rooms local room=5690320318231820188 poll_state=KVPoll.SUCCESS encoded=1
synced_terminal_rooms calling allgather with local=[(5690320318231820188, 1)]
synced_terminal_rooms allgather done shape=(4, 256, 2)
synced_terminal_rooms returned success={1248147356} failed=set()
```

The room goes in as `5690320318231820188`, comes back as `1248147356`.

### Fix Attempts (2026-07-14 / 2026-07-15)

All fix attempts were tested on TPU VMs in **us-east5-b** (jingnw-node /
jingnw-node2). After attempt 4, we discovered that **SPMD desync (E0200)** was
a pre-existing issue — even the original unmodified code crashes.

#### Attempt 1: Split int64 → two int32 columns (hi/lo)

- Changed array shape from `(256, 2)` to `(256, 3)` with columns
  `[room_hi, room_lo, state]`, dtype `np.int32`
- **Result**: E0200 SPMD desync — changing the allgather's shape causes a
  different XLA collective compilation on one process. Decode w3 crashes:
  `"unexpected peer in launch group with different launch id; HLO module:
  jit__identity_fn"`

#### Attempt 2: `jax_enable_x64` context manager around allgather

- Wrapped the `process_allgather()` call with
  `jax.config.update("jax_enable_x64", True)` / restore
- **Result 1**: `TypeError: 'bool' object is not callable` — initial
  implementation tried `jax.config.jax_enable_x64(True)` which is a bool
  property, not callable. Fixed to use `jax.config.update()`.
- **Result 2**: `JaxRuntimeError: INVALID_ARGUMENT: E0102: Executable
  (jit_set_future_token_ids) expected parameter 1 of size 1024 (s64[]) but got
  buffer with incompatible size 512 (s32[])` — the x64 flag is global and
  leaked to the forward thread's JIT compilation, breaking the model.

#### Attempt 3: Keep shape (256,2), change dtype to int32

- Kept array shape `(256, 2)` but changed dtype from `np.int64` to `np.int32`,
  stored truncated room IDs directly
- **Result**: E0200 SPMD desync — even same shape but different input dtype
  (int32 vs int64 from other processes still on old code) produces different
  XLA compilation.

#### Attempt 4: `trunc_to_orig` mapping (allgather completely unchanged) ★

- Kept allgather **exactly as original**: same `np.int64` dtype, same
  `(256, 2)` shape, same compiled XLA
- Added Python-level mapping: `trunc_to_orig` dict maps truncated int32 room
  value back to original int64 value after allgather returns
- `trunc_to_orig[int(np.int32(np.int64(orig)))] = orig` precomputes the
  truncated value
- `trunc_to_orig.get(trunc, trunc)` restores original 64-bit room ID in
  output sets
- **Result**: E0200 SPMD desync — even with zero changes to the allgather.
  This proved the SPMD desync is **pre-existing**, not caused by any fix.
- **This is the current fix** — code is correct, awaiting clean TPU state to
  verify.

#### Attempt 5: Global `JAX_ENABLE_X64=True` environment variable

- Set `JAX_ENABLE_X64=True` as env var before server launch on all workers
- **Result**: `ValueError: Expected int32 dtype for page_indices.dtype=
  dtype('int64')` — the Pallas attention kernel validates that `page_indices`
  must be int32. With x64 enabled globally, `page_indices` becomes int64,
  breaking the kernel. **Not a viable approach.**

### Final Fix: int32 room IDs (2026-07-16)

The `trunc_to_orig` mapping (attempt 4) was replaced with a simpler approach:
- `generate_bootstrap_room()` now returns `[0, 2^31-1]` (fits in int32)
- `multihost_sync.py` uses `np.int32` dtype (no truncation by JAX)
- CRC32 fallback in `tokenizer_manager.py` masked with `& 0x7FFFFFFF`
- The `trunc_to_orig` workaround has been removed entirely

This avoids the truncation problem at the source. Previous attempts that
changed allgather shape/dtype mid-run caused E0200, but this change requires
a full restart of all workers (same dtype on all processes from the start).

**Verified working (2026-07-16):** End-to-end 1P1D test with chat endpoint
returned correct output (`"That's easy! 2 + 3 equals **5**."`, finish_reason=stop).

---

## 7. RESOLVED: SPMD Desync (E0200) — Corrupted TPU State

### Summary

All decode-side test attempts during fix development (2026-07-14) crashed with
TPU Error 0200 on decode worker 3:
```
"unexpected peer in launch group with different launch id;
HLO module: jit__identity_fn"
```

This happened even with the **original unmodified code** (attempt 4 proved
this). The SPMD desync was a separate issue from the int64 truncation bug.

### Root Cause: Corrupted TPU Hardware State

**Confirmed (2026-07-15)**: The SPMD desync was caused by corrupted TPU
hardware state from many `kill -9` / restart cycles during debugging. It was
**NOT** an overlap thread conflict, a code bug, or a missing dependency.

### Evidence

Three independent tests confirm the root cause after rebooting the VMs:

1. **Standalone reproducer** (`scripts/tpu-vm/test_spmd_desync_hypothesis/`):
   `process_allgather()` interleaved with background-thread collectives for
   **100 iterations on all 4 workers — zero errors**.

2. **gcsfs-isolated reproducer** (`scripts/tpu-vm/test_e0200_gcsfs_hypothesis/`):
   8 XLA compilations + 50 allgather iterations + 50 overlap iterations,
   all **WITHOUT gcsfs** installed, JAX_COMPILATION_CACHE_DIR pointing to
   inaccessible GCS bucket — **all phases passed on all 4 workers**.

3. **Full decode server without gcsfs** (2026-07-15 04:42–04:48):
   Complete MiMo-V2-Flash decode server started on all 4 workers WITHOUT
   gcsfs, ran event loop (continuous `process_allgather`) for 90+ seconds —
   **zero errors, zero E0200**.

### gcsfs is NOT related to E0200

Initial observations suggested gcsfs might prevent E0200 because:
- Run without gcsfs → E0200 on w3
- Run with gcsfs → no E0200

This was a **false correlation**. The actual sequence was:
1. Many `kill -9` cycles corrupted TPU state → E0200
2. Reboot cleared corrupted state
3. First full server run after reboot: w0 crashed due to **port 10001
   conflict** (old process still held the port), cascading to w1–w3 as
   coordination service failures — misdiagnosed as E0200
4. Installed gcsfs between runs (irrelevant change)
5. Second run: clean start, no port conflict → no E0200

Controlled experiment confirmed: full decode server runs fine without gcsfs
on clean TPU state.

**Note**: gcsfs IS recommended for production (enables JAX compilation cache
on GCS, improving startup time), but is not required for correctness.

### Lesson Learned

When debugging on TPU pods, repeated `kill -9` of JAX processes can leave
TPU hardware in a corrupted state where collectives fail with E0200. **Always
reboot the TPU VM** (or stop/start if supported) before concluding that a
collective error is a code bug. For v6e-16 (pod nodes), `stop`/`start` is not
supported — use `sudo reboot` on all workers instead.

Additionally, always verify ports are free before restarting servers — an old
process holding a port causes one worker to crash, which cascades to all other
workers as a coordination service failure.

---

## 8. Additional Issues (from decode-side trace)

### 8.1 Sharding mismatch on path B (direct HBM) with dp_size > 1

In the single-host direct-HBM path (no D2H staging):
- **Prefill** gather output sharding: `P(None, *pool_pspec[1:])` — first axis explicitly `None`
- **Decode** spec sharding: `kv_pool.kv_sharding` = `P("data", None, "tensor", None, None)` — first axis `"data"`

With `dp_size=1`, `"data"` axis has size 1, so they're semantically equivalent.
With `dp_size>1`, per-device buffer sizes differ → pull could hang or fail.

**Not triggered in our setup** (dp_size=1), but a **latent bug for DP>1 setups**.

### 8.2 No native timeout on `link.pull()`

`link.pull()` (wrapper.py:219) has no timeout parameter. If the prefill never
registers the UUID (e.g., crashed after bootstrap publish but before
`await_pull`), the pull blocks forever. The reaper (30s default) flips a state
flag but cannot interrupt the native C call. The worker thread is permanently
stuck.

### 8.3 Thread exhaustion

With `pull_worker_count` workers (default 4) and no native timeout, 4 stuck
pulls exhaust the worker pool. All subsequent pulls queue up and hang.

### 8.4 Link caching with no reconnection

`wrapper.py:235-241` caches one link per `remote_addr`. If a link becomes stale
(prefill restarts, network partition), all subsequent pulls to that address
hang. There is no reconnection logic.
