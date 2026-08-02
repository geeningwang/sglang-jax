# 1P1D Disaggregated Inference Hang — Investigation Log

**Date:** 2026-07-15 (updated 2026-07-16)
**Setup:** MiMo-V2-Flash on v6e-16, prefill on jingnw-node, decode on jingnw-node2 (us-east5-b)
**Branch:** mimo-tpu7-stage3
**Status:** RESOLVED — 1P1D end-to-end working as of 2026-07-16
**Original symptom:** Decode produces 0 output tokens. Prefill returns HTTP 200 in ~1s but decode hangs indefinitely.

---

## Phase 1: bootstrap_room tracing (RESOLVED — not the root cause)

**Hypothesis:** `bootstrap_room` is None on the prefill side, so KV extraction is skipped.

**Method:** Added debug logging to three files:
- `tokenizer_manager.py` — trace bootstrap_room from HTTP request to tokenized object
- `scheduler.py` — trace bootstrap_room from recv_req to Req object
- `prefill.py` — trace bootstrap_room inside `process_prefill_chunk()`

**Result:** bootstrap_room flows correctly through the entire pipeline. Value `4694084749967068129` appeared on both prefill and decode sides. The prefill extracts KV and calls `sender.send()` successfully.

**Conclusion:** bootstrap_room is NOT the issue.

---

## Phase 2: Cross-cluster JaxTransfer verification (RESOLVED — transport works)

**Hypothesis:** `jax.experimental.transfer` does not support cross-cluster transfers (different `jax.distributed.initialize()` coordinators).

**Method:** Built standalone test at `scripts/tpu-vm/test_cross_cluster_transfer/transfer_test.py`.
- Producer on jingnw-node (coordinator 10.202.0.29:9299)
- Consumer on jingnw-node2 (coordinator 10.202.15.227:9299)
- All 4 workers per VM participate in `jax.distributed.initialize()` (required for v6e-16)
- Only worker 0 does the actual transfer; workers 1-3 init + sleep
- Test array: (8,4) float32, sharded across 4 local devices

**Result:** Pull completed in **0.0s** with correct data. Cross-cluster transfer works.

**Key learnings:**
- v6e-16 requires all 4 workers to call `jax.distributed.initialize()`; single-process mode hangs
- Worker ID comes from GCE metadata `agent-worker-number`, NOT from `/tmp/tpu-env` (which doesn't exist on these VMs)
- Both sides must create local-only meshes (`jax.local_devices()`) for the sharding spec
- `LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7"` needed for stable TPU init

**Conclusion:** The transport layer works. The hang is in sglang's disaggregation logic.

---

## Phase 3: Root cause found — int64 room truncation in synced_terminal_rooms

### Debug method

Added targeted logging for 4 suspects across 7 files (decode.py, prefill.py, wrapper.py, conn.py, multihost_sync.py). Deployed to all workers and sent a test request.

### What the logs showed

**Suspects 1-3 eliminated:** All 4 decode workers correctly matched their prefill peers, UUID/transfer_id values matched, sharding specs matched, and the `jax.experimental.transfer` pull completed successfully on all 4 workers (state=SUCCESS).

**Suspect 4 confirmed — but not an allgather hang.** The allgather completed in 0.001s on all workers. The bug was in the **post-allgather processing**.

### ROOT CAUSE

In `multihost_sync.py`, `synced_terminal_rooms()` line 86:

```python
local = np.full((_SYNC_MAX_INFLIGHT, 2), -1, dtype=np.int64)
# ... fill local[i, 0] = room (int64), local[i, 1] = state
gathered = multihost_utils.process_allgather(local)  # truncates to int32!
for p in range(nproc):
    for i in range(_SYNC_MAX_INFLIGHT):
        room = int(gathered[p, i, 0])
        if room < 0:        # <--- BUG
            continue
```

**The problem:** `jax_enable_x64` is off by default on TPU, so `process_allgather` silently truncates int64 to int32. The sentinel value -1 (for empty entries) stays -1 after truncation. But valid room values like `4713361715480306529` truncate to negative int32 values (e.g., `-1921980575`). The check `if room < 0: continue` was meant to skip empty entries (sentinel -1), but it also skips ~50% of valid rooms whose int64 values truncate to negative int32.

**Evidence from all 4 decode workers:**
```
[DEBUG-S4] synced_terminal_rooms: pidx=1 n_entries=1 local_rooms=[4713361715480306529] local_states=[1]
[DEBUG-S4] process_allgather done: pidx=1 elapsed=0.001s gathered_shape=(4, 256, 2)
[DEBUG-S4] post-allgather: pidx=1 success=set() failed=set()
```
Every worker reports state=SUCCESS locally, but `success` is always empty because the truncated room value is negative and gets skipped.

### FIX

**Final fix (2026-07-16):** Eliminated the truncation at the source:
- `generate_bootstrap_room()` now returns `[0, 2^31-1]` (fits in int32)
- `multihost_sync.py` uses `np.int32` dtype — no truncation by JAX
- `if room < 0:` changed to `if room == -1:` (exact sentinel check)
- CRC32 fallback masked with `& 0x7FFFFFFF` to stay in int32 range

Earlier intermediate fix used a `trunc_to_orig` mapping to recover 64-bit
room IDs after allgather, but this was removed in favor of the simpler
int32 range approach.

**Verified working (2026-07-16):** End-to-end 1P1D test with chat endpoint
returned correct output (`"That's easy! 2 + 3 equals **5**."`, finish_reason=stop).

### Post-fix E0200 crash → Phase 4

After the fix, the decode server successfully recognized the transfer and started processing the batch ("Prefill batch. #new-seq: 1"), but then crashed with E0200 (TPU core halt / SPMD desync in `jit_jitted_sampler`). Initially attributed to corrupted TPU state from `kill -9`, but the crash persisted across 3 attempts including full VM reboots and fresh provisioning with `gcsfs`. See Phase 4 for the root cause.

---

## Phase 4: Root cause found — SPMD race in overlap disagg decode event loop

### Symptom

After the Phase 3 room truncation fix, decode correctly processes the KV transfer and runs the "Prefill batch" on all 4 workers. But ~1 second later, E0200 crashes with:
```
HLO module: jit_jitted_sampler
An unexpected peer shows up in the launch group with a different launch id
than the current group leader.
```

All 4 workers crash at the same callsite: `process_allgather` in `synced_terminal_rooms`, but the TPU reports the conflicting program as `jit_jitted_sampler`.

### ROOT CAUSE

The overlap disagg decode event loop (`event_loop_overlap_disagg_decode` in `decode.py`) has a thread race condition between two SPMD programs:

1. **Forward thread**: `run_batch()` dispatches the forward pass + sampler to a background thread via `ModelWorkerClient.forward_batch_generation()`. This puts the batch in an `input_queue` and returns immediately. The forward thread picks it up and executes `jit_jitted_sampler` (an SPMD collective) on the TPU.

2. **Main thread**: After `run_batch()` returns, the event loop calls `process_decode_queue()` at line 330, which calls `synced_terminal_rooms()` → `process_allgather()` — a DIFFERENT SPMD collective — while the forward thread is still executing `jit_jitted_sampler`.

The TPU sees two different SPMD programs dispatched concurrently from the same process. If the order of these programs differs across workers (due to thread scheduling), the TPU detects the mismatch and halts with E0200.

**Why this only affects disagg decode:** The non-disagg overlap event loop (`event_loop_overlap`) does NOT call any SPMD operations after `run_batch()`. It only does CPU-level work (Python sampling, memory management) until `process_batch_result()` drains the forward thread via `output_queue.get()`. The disagg decode loop introduced `process_decode_queue()` (with `process_allgather`) after `run_batch()`, creating the race.

**Comparison:**
```
# Non-disagg overlap loop (SAFE):
batch = get_next_batch()
if batch:
    run_batch(batch)        # forward thread starts SPMD
    # NO SPMD operations here — CPU-only work
if self.last_batch:
    process_batch_result()  # drains forward thread, then SPMD-safe

# Disagg decode overlap loop (RACE):
process_decode_queue()      # allgather (SPMD) — safe before run_batch
batch = get_next_batch()
if batch:
    run_batch(batch)        # forward thread starts SPMD
    process_decode_queue()  # allgather (SPMD) — RACES with forward thread!
if self.last_batch:
    process_batch_result()  # drains forward thread
```

Additionally, the allgather at the TOP of the next iteration (line 299) can also race with the forward thread from the PREVIOUS iteration, since `process_batch_result()` (which drains the forward thread) runs AFTER the allgather.

### Evidence

All 4 workers' crash tracebacks are identical:
```
multihost_sync.py:72 synced_terminal_rooms → process_allgather
  → decode.py:623 _drain_transfer_queue_synced
    → decode.py:490 process_decode_queue
      → decode.py:287 event_loop_overlap_disagg_decode
```

The forward thread on all workers was idle (`queue.get` in `forward_thread_func_`) at crash time — it had finished dispatching the sampler to the TPU but the TPU hadn't completed it when the allgather arrived.

Timeline from w3 logs:
```
14:14:33  allgather succeeds, transfer complete (state=SUCCESS)
14:14:35  "Prefill batch. #new-seq: 1" (JIT compilation delay)
14:14:36  E0200 in process_allgather (jit_jitted_sampler conflict)
```

Reproduced 3 times across VM reboots and fresh provisions — not transient TPU corruption.

### FIX

**Immediate fix:** Add `--disable-overlap-schedule` to the decode server command. This uses `event_loop_normal_disagg_decode` which runs all operations synchronously in a single thread — no race possible.

**Proper fix (future):** Restructure the overlap disagg decode event loop to move `process_batch_result()` (which drains the forward thread) to BEFORE `process_decode_queue()` (which does allgather). This ensures the forward thread is idle before any SPMD collective:
```python
# Fixed overlap loop:
if self.last_batch:
    process_batch_result()  # drain forward thread FIRST
process_decode_queue()      # allgather — forward thread idle ✓
batch = get_next_batch()
if batch:
    run_batch(batch)        # forward thread starts
```
This sacrifices some CPU/TPU overlap but eliminates the SPMD race. A more sophisticated fix could use a `threading.Event` to gate the allgather until the forward thread completes.

---

## File reference

| File | Role |
|------|------|
| `decode.py:270-338` | Overlap decode event loop |
| `decode.py:363-468` | `process_input_requests_disagg_decode` — processes PD requests |
| `decode.py:485-572` | `process_decode_queue` — drives prealloc → transfer → ready |
| `decode.py:574-608` | `_pick_prefill_peer_for_this_host` — peer lookup |
| `decode.py:610-644` | `_drain_transfer_queue_synced` — process_allgather sync |
| `decode.py:646-741` | `_admit_decode_prealloc` — creates receiver, calls init |
| `decode.py:971-991` | `_build_kv_spec_for_req` — builds pull spec |
| `prefill.py:98-120` | `local_kv_spec_for_pool` — local sharding spec |
| `prefill.py:323-468` | `process_prefill_chunk` — KV extraction + sender.send |
| `conn.py:861-870` | `JaxTransferKVReceiver.init` — stores metadata |
| `conn.py:872-942` | `receiver.poll` — enqueues pull, checks completion |
| `conn.py:1200-1235` | `_run_pull` — background pull execution |
| `wrapper.py:132-187` | `register_pull` — prefill registers data |
| `wrapper.py:189-219` | `pull` — decode pulls from remote |
| `multihost_sync.py` | `synced_terminal_rooms` — process_allgather wrapper |
