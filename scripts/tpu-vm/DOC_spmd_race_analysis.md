# SPMD Race in Overlap Disagg Decode Event Loop

**Status:** Fixed in commit 3c301255 (Fix A applied)
**Affects:** Multi-host TPU only (multiple JAX processes per pod, e.g. v6e-16 with 4 workers)
**Does NOT affect:** Single-host TPU (v7x 2x2x2, single JAX process per pod)

---

## 1. Problem Statement

The overlap disagg decode event loop runs `process_allgather` (an SPMD collective) on the scheduler thread while the forward thread is concurrently executing model SPMD operations (`jit_jitted_sampler`). On multi-host TPU, thread scheduling differences across hosts cause different JAX processes to enter these collectives in different orders, and the TPU halts with E0200:

```
An unexpected peer shows up in the launch group with a different launch id
than the current group leader.
HLO module: jit_jitted_sampler
```

The workaround `--disable-overlap-schedule` uses a fully synchronous event loop, eliminating the race but losing the CPU/TPU pipeline benefit.

---

## 2. Related Code Pieces

All paths are relative to `python/sgl_jax/srt/`.

### The racy loop

**`event_loop_overlap_disagg_decode`** — `disaggregation/decode.py:263-337`

The overlap disagg decode event loop. Calls `process_decode_queue()` TWICE per iteration:
- Line 287: before `run_batch` (safe — forward thread idle)
- Line 318: after `run_batch` but before `process_batch_result` drains the forward thread (**RACE**)

```
iteration N:
  process_decode_queue()          ← line 287 (safe, forward thread idle)
  run_batch(batch)                ← line 296 (enqueues to forward thread, returns immediately)
  process_decode_queue()          ← line 318 (RACE — forward thread running SPMD)
  process_batch_result(last_batch) ← line 331 (drains forward thread via output_queue.get())
```

The second `process_decode_queue()` at line 318 can call `process_allgather` while the forward thread is still executing the model forward pass dispatched by `run_batch` at line 296.

Additionally, the first `process_decode_queue()` at line 287 of iteration N+1 can race with the forward thread from iteration N, because `process_batch_result` (which drains the forward thread) runs AFTER both `process_decode_queue` calls.

### The safe loops (for comparison)

**`event_loop_normal_disagg_decode`** — `disaggregation/decode.py:222-261`

The non-overlap disagg decode loop. Calls `process_decode_queue()` only once (line 243), and `run_batch` + `process_batch_result` are synchronous — no forward thread, no race.

```
iteration N:
  process_decode_queue()          ← line 243 (only call, no forward thread)
  run_batch(batch)                ← line 251 (synchronous)
  process_batch_result(batch)     ← line 253 (synchronous)
```

**`event_loop_overlap`** — `managers/scheduler.py:971-1035`

The non-disagg overlap loop. Has a forward thread but does NOT call `process_decode_queue` (no disagg queue to drain). The scheduler thread does only CPU work between `run_batch` and `process_batch_result` — no SPMD collectives.

```
iteration N:
  get_next_batch_to_run()         ← CPU only
  run_batch(batch)                ← enqueues to forward thread
  process_batch_result(last_batch) ← blocks on output_queue.get(), drains forward thread
```

### The SPMD synchronization point

**`process_decode_queue`** — `disaggregation/decode.py:485-572`

Drives prealloc → transfer → ready transitions. Line 490 calls `_drain_transfer_queue_synced()`.

**`_drain_transfer_queue_synced`** — `disaggregation/decode.py:610-644`

On multi-host (`jax.process_count() > 1`), calls `synced_terminal_rooms()` at line 623 which does `process_allgather`. On single-host, bypasses allgather entirely (line 616).

**`synced_terminal_rooms`** — `disaggregation/common/multihost_sync.py:21-64`

The actual `process_allgather` call (line 48). Uses a fixed-shape `(256, 2)` int32 buffer so all processes have the same allgather shape regardless of local queue state. This is the SPMD collective that conflicts with the forward thread's model collectives.

### The forward thread

**`forward_thread_func_`** — `managers/tp_worker_overlap_thread.py:104-139`

Daemon thread that reads batches from `input_queue`, runs the model forward pass (line 124: `self.worker.forward_batch_generation`), and puts results on `output_queue`. All model SPMD operations (attention, MoE, sampler) execute here.

**`ModelWorkerClient.forward_batch_generation`** — `managers/tp_worker_overlap_thread.py:184-238`

Called from `run_batch` on the scheduler thread. Enqueues the batch to `input_queue` (line 219) and returns immediately with future token IDs. Does NOT run any SPMD operations.

### The synchronization drain

**`resolve_last_batch_result`** — `managers/tp_worker_overlap_thread.py:141-182`

Blocks on `output_queue.get()` (line 151) until the forward thread completes and deposits results. Called from `process_batch_result` via `process_batch_result_decode` (line 397 of `managers/scheduler_output_processor_mixin.py`).

**`process_batch_result`** — `managers/scheduler.py:2157-2173`

Dispatcher that calls `process_batch_result_decode` or `process_batch_result_prefill`, both of which call `resolve_last_batch_result` to drain the forward thread.

### Batch dispatch

**`run_batch`** — `managers/scheduler.py:2028-2076`

In overlap mode (line 2052), calls `self.tp_worker.forward_batch_generation()` which is the non-blocking enqueue described above. In synchronous mode (line 2063), runs the model directly on the scheduler thread.

---

## 3. Race Mechanism

Two threads, two SPMD programs:

| Thread | SPMD program | Trigger |
|---|---|---|
| Forward thread | `jit_jitted_sampler` (model forward + sampling) | `run_batch` → `input_queue.put` → `forward_thread_func_` |
| Scheduler thread | `process_allgather` (room state sync) | `process_decode_queue` → `_drain_transfer_queue_synced` → `synced_terminal_rooms` |

On multi-host, each worker (JAX process) has both threads. If worker A's scheduler thread calls `process_allgather` while worker A's forward thread is in `jit_jitted_sampler`, and worker B's threads are in the opposite order, the TPU sees mismatched SPMD programs across processes and halts.

On single-host, there's only one JAX process, so both threads always dispatch to the same set of devices in the same order — the hardware can serialize them without cross-process disagreement.

### Timeline (observed crash)

```
Worker 0                              Worker 1
────────                              ────────
forward_thread: jit_jitted_sampler    forward_thread: jit_jitted_sampler
scheduler:      process_allgather     scheduler:      (still in run_batch)
                ↑ MISMATCH            forward_thread: (still in sampler)
                                      scheduler:      process_allgather
                                                      ↑ TOO LATE
                                      → E0200
```

---

## 4. Fix Suggestions

### Fix A: Drain forward thread before `process_decode_queue` (Recommended)

Move `process_batch_result` (which blocks on `output_queue.get()`, draining the forward thread) to BEFORE `process_decode_queue` in the loop. This ensures the forward thread is idle before any `process_allgather` call.

```python
# Current (RACE):
process_decode_queue()           # allgather (safe — forward idle)
batch = get_next_batch_to_run()
if batch:
    run_batch(batch)             # forward thread starts
    process_decode_queue()       # allgather (RACE — forward running)
if self.last_batch:
    process_batch_result()       # drains forward thread

# Fixed:
if self.last_batch:
    process_batch_result()       # drain forward thread FIRST
process_decode_queue()           # allgather (safe — forward idle)
batch = get_next_batch_to_run()
if batch:
    run_batch(batch)             # forward thread starts
    # NO process_decode_queue here — forward thread running
```

**Trade-off:** The second `process_decode_queue` call (line 318) is removed, reducing how often the transfer queue is polled. Transfers that complete during `run_batch` won't be noticed until the next iteration. This adds at most one iteration of latency (~tens of ms) to transfer completion — negligible vs. the KV transfer time itself.

**Risk:** Low. This is a loop reorder, not a logic change. The safe non-disagg overlap loop (`event_loop_overlap`) already uses this structure.

### Fix B: Gate allgather behind forward-thread-idle event

Add a `threading.Event` (`forward_idle`) that the forward thread sets after completing each batch (after `output_queue.put`) and that the scheduler clears before `run_batch`. In `_drain_transfer_queue_synced`, wait on `forward_idle` before calling `synced_terminal_rooms`.

```python
# In forward_thread_func_, after output_queue.put:
self.forward_idle.set()

# In ModelWorkerClient.forward_batch_generation, before input_queue.put:
self.forward_idle.clear()

# In _drain_transfer_queue_synced, before synced_terminal_rooms:
self.tp_worker.forward_idle.wait()
```

**Trade-off:** More surgical — preserves both `process_decode_queue` calls, so transfer polling frequency is unchanged. But adds threading synchronization complexity and a new coupling between the scheduler and the forward thread.

**Risk:** Medium. Introduces a potential deadlock if the event logic is wrong, and the `wait()` blocks the scheduler if the forward pass takes a long time.

### Fix C: Move allgather into the forward thread

Run `synced_terminal_rooms` inside `forward_thread_func_` as a post-forward step, serializing all SPMD operations on one thread.

```python
# In forward_thread_func_, after output_queue.put:
terminal_entries = self._run_synced_terminal_rooms()
self.terminal_queue.put(terminal_entries)

# In process_decode_queue, replace _drain_transfer_queue_synced:
entries = self.terminal_queue.get_nowait()  # non-blocking
```

**Trade-off:** Clean SPMD serialization — all collectives on one thread, impossible to race. But requires a new queue, refactoring the data flow between scheduler and forward thread, and passing the disagg transfer queue state into the forward thread.

**Risk:** High. Significant refactoring. The forward thread currently has no knowledge of the disagg queue or bootstrap rooms.

### Recommendation

**Fix A** is the right choice. It's the simplest, lowest-risk change, matches the pattern already used by the safe non-disagg overlap loop, and the cost (one iteration of polling delay) is negligible. Fix B and C add complexity without meaningful benefit for the current use case.
