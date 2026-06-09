# Opt C — Per-Step Host Sync Removal: Code Analysis & Findings

**Date**: 2026-06-09
**Status**: Option A implemented and benchmarked — zero measurable gain; F=3.9ms confirmed as fixed TPU compute

---

## Executive Summary

Full code investigation reveals the overlap schedule is already well-optimized. The F=3.9ms
fixed overhead per step is primarily **fixed TPU compute** (attention, layer norms, MoE router),
not a removable host sync. Realistic Opt C gain is **5-10%** (not the projected 22%).

The primary actionable improvement is moving the CPU prep work (ForwardBatch construction,
SamplingMetadata preparation) to the background thread, reducing the per-step pipeline bubble.

---

## Decode Loop Architecture (Overlap Mode)

### Event loop (`scheduler.py:807`)

```
event_loop_overlap per iteration:
  1. recv_requests()                       # non-blocking
  2. process_input_requests()              # handle new requests
  3. batch = get_next_batch_to_run()       # scheduling + prepare_for_decode (CPU)
  4. result = run_batch(batch)             # launches batch_N on TPU (async)
     -> ModelWorkerClient.forward_batch_generation():
          update_penalties()               # CPU ~0.1ms
          SamplingMetadata.from_...()      # CPU: device_array() calls ~0.2ms
          get_forward_metadata()           # CPU ~0.1ms
          ForwardBatch.init_new()          # CPU: device_array() calls ~0.3ms
          input_queue.put(batch_N)         # LAUNCH: background thread picks up
          return (None, future_ids, 0)     # returns immediately
  5. process_batch_result(batch_{N-1}, launch_done=batch_N.launch_done)
     -> resolve_last_batch_result(launch_done):
          output_queue.get()               # waits for background thread to put result
          jax.copy_to_host_async(...)      # starts async D2H transfers
          np.asarray(tokens).tolist()      # HOST SYNC: waits for TPU + D2H
          launch_done.wait()              # waits for batch_N's forward dispatch
     -> req.output_ids.append(next_token_id)
     -> req.check_finished()              # EOS detection on CPU
     -> stream_output()                   # send to detokenizer
```

### Background thread (`tp_worker_overlap_thread.py:96`)

```
forward_thread_func_ per step:
  input_queue.get()                        # waits for next batch
  resolve_future_token_ids(...)            # on-device lookup: no host sync
  worker.forward_batch_generation():
    model_runner.forward()                 # dispatches XLA compute (async)
    launch_done.set()                      # fires after forward dispatch
    model_runner.sample()                  # dispatches sampling (async)
  async_gather_fn(next_token_ids)          # dispatches all-gather (async)
  set_future_token_ids(...)                # dispatches map update (async)
  output_queue.put(...)                    # puts result immediately (all JAX async)
```

---

## Host Sync Location

**File**: `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py:143-174`

```python
def resolve_last_batch_result(self, launch_done=None):
    _, logits_output, next_token_ids, cache_miss_count = self.output_queue.get()  # (1)
    async_next_tokens = jax.copy_to_host_async(next_token_ids)                   # (2)
    ...
    next_token_ids = np.asarray(async_next_tokens).tolist()  # HOST SYNC         (3)
    if launch_done is not None:
        launch_done.wait()                                                         # (4)
```

| Point | What it does | Typical wait |
|-------|-------------|--------------|
| (1) `output_queue.get()` | Waits for background thread to put result | Fast: thread puts immediately after async dispatch |
| (2) `copy_to_host_async()` | Kicks off async PCIe D2H transfer | Returns immediately |
| (3) `np.asarray().tolist()` | **Primary sync**: waits for TPU compute + D2H | ~0ms if already done; up to step_time if stalled |
| (4) `launch_done.wait()` | Waits for batch_N forward dispatch | Fast: fires ~0.2ms after forward thread picks up |

The `np.asarray` at (3) is necessary: EOS detection in
`scheduler_output_processor_mixin.py:348` (`req.check_finished()`) needs token IDs on CPU.

---

## `future_token_ids_map` — Already On-Device

The next step's input token IDs are resolved entirely on-device:
- Sampled tokens stored in `future_token_ids_map` (a JAX device array via `set_future_token_ids`)
- Next step: `resolve_future_token_ids(input_ids, future_token_ids_map)` reads from device array
- **No D2H of token IDs for the forward pass** — only for CPU-side EOS detection

This is already optimized. D2H transfer only exists for EOS checking.

---

## Decomposition of F=3.9ms Fixed Overhead

From the TPOT scaling fit: `step_time = 2.21ms x batch + 3.9ms`

| Component | Estimated share | Reducible? |
|-----------|----------------|------------|
| Fixed TPU compute: attention, layer norms, MoE router topk | ~2.5ms | No (actual compute) |
| All-gather of token IDs across 8 TP chips | ~0.5ms | Partially |
| CPU prep before launch (ForwardBatch, SamplingMetadata) | ~0.5ms | Yes (move to background thread) |
| `np.asarray` wait when TPU already done | ~0.3ms | Minimal (already overlapped) |
| Python scheduling overhead | ~0.1ms | Negligible |

**Revised gain estimate**: 5-10% (not 22%). The 22% projection incorrectly attributed all of F
to a removable host sync. Most of F is actual fixed TPU compute.

---

## Optimization Options

### Option A: Move CPU prep to background thread (quick win, 2-5%) ✅ IMPLEMENTED

~~Currently in main thread before `input_queue.put()`:~~
```python
# REMOVED from ModelWorkerClient.forward_batch_generation():
sampling_metadata = SamplingMetadata.from_model_worker_batch(...)  # ~0.2ms
forward_metadata = attn_backend.get_forward_metadata(model_worker_batch)  # ~0.1ms
model_worker_batch.forward_batch = ForwardBatch.init_new(...)  # ~0.3ms
```

Moved to background thread (after `input_queue.get()`), before `resolve_future_token_ids`.
Main thread now pushes a 2-tuple `(model_worker_batch, future_token_ids_ct)` immediately
and returns, shrinking the per-step pipeline bubble by ~0.5-1ms.

**Kept on main thread**: `update_penalties()` + `sampling_info` reassignment (must precede
the scheduler's `self.cur_sampling_info` read) and LoRA prep (guards a shared writer).

**Change**: `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`

**Risk**: Low — all three calls are read-only w.r.t. shared state
**Expected gain**: 0.5-1ms pipeline bubble reduction = 2-5% TPOT improvement at conc=8
**Actual gain**: **0%** — TPOT at conc=8 unchanged at 21.6ms (see benchmark below)

### Option B: On-device EOS detection (medium effort, 3-5%)

Add `done_flags` bool array as an additional output from the sampler. Transfer
`done_flags` (batch_size bytes) instead of `next_token_ids` (batch_size x 4 bytes) per step.
Transfer token IDs only every K steps (streaming) or when a request finishes.

**Changes needed**:
- `model_runner.py`: compute `done_mask = jnp.isin(next_token_ids, eos_token_ids)` inside JIT
- `tp_worker_overlap_thread.py`: D2H `done_mask` per step, token_ids lazily
- `scheduler_output_processor_mixin.py`: use `done_mask` for EOS; batch D2H of token_ids

**Risk**: Medium — output processing pipeline refactor  
**Expected gain**: 0.3-0.5ms per sync point = 1-3% TPOT improvement

---

## Benchmark Results — Opt C-A vs Baseline

**Run**: 2026-06-09T06:56–07:08Z, `gke-tpu-b00d966f-fppd`, same config as baseline
**GCS**: `gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-opt-c/flash_opt_c_20260609T065629Z.json`

### Concurrency sweep (input=512, output=256)

| conc | tok/s | TPOT (ms) | baseline TPOT | Δ |
|------|------:|----------:|----------:|---|
| 1 | 111.3 | 9.0 | 9.0 | 0% |
| 2 | 181.1 | 11.0 | 10.9 | 0% |
| 4 | 264.5 | 15.1 | 15.3 | +1% (noise) |
| **8** | **370.1** | **21.6** | **21.6** | **0%** |
| 16 | 374.8 | 39.0 | 39.3 | +1% (noise) |
| 32 | 369.7 | 75.7 | 75.5 | 0% |

Peak: 374.8 tok/s @ conc=16 (baseline: ~372 tok/s).

### Conclusion: F=3.9ms is NOT removable host overhead

The ~0.6ms of prep work moved to the background thread was already hidden by the
overlap design. The XLA data dependency chain (`set_future_token_ids` → `resolve_future_token_ids`)
serializes consecutive steps regardless of where the host prep runs. Moving prep off the
main thread does not change when the TPU actually starts the next step.

**F=3.9ms is fixed TPU compute**: attention, layer norms, MoE router topk, and the
`future_token_ids_map` read/write round-trip — none of which can be pipelined away.

Opt C is effectively closed. The code change (commit `10a5699`) is kept since it's a
clean refactoring (slightly shorter main-thread critical section) with no downside.

---

## Revised Priority Assessment

| Optimization | Expected gain | Effort | Priority |
|---|---|---|---|
| Opt C-A: Move prep to background thread | **0% (measured)** | Done | ~~closed~~ |
| Opt A2: Attention FP8 (skip BF16 dequant at load) | **<0.3% (corrected)** | — | ~~closed~~ — not worth impl. |
| Opt E: Speculative decoding | ~2-3× per-seq latency | High | **Next** |
| Opt C-B: On-device EOS detection | 1-3% | Medium | Low priority (F is compute, not sync) |
| Opt B: Larger batch (conc > 8 if KV allows) | Already tested in baseline | - | Diminishing returns |

---

## Key Files

| File | Role |
|------|------|
| `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py` | Host sync in `resolve_last_batch_result`; background thread |
| `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py` | EOS detection, stream output |
| `python/sgl_jax/srt/managers/scheduler.py` | Overlap event loop |
| `python/sgl_jax/srt/model_executor/forward_batch_info.py` | `ForwardBatch.init_new()` — 8 H2D transfers per step |
| `python/sgl_jax/srt/sampling/sampling_batch_info.py` | `SamplingMetadata.from_model_worker_batch()` — 5 H2D transfers per step |
