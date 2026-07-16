# 1P1D Multi-Host TPU Fixes (since d3ac424)

**Base commit:** d3ac42493745dcf4d9e141ef0a2a351029d7cc25 (primatrix upstream, single-host v7x)
**Branch:** mimo-tpu7-stage3
**Target:** Multi-host TPU v6e-16 (4 workers per VM)

---

## 1. SWAKVPool not supported in JaxTransfer disaggregation

**Problem:** MiMo-V2-Flash uses hybrid attention (full + sliding-window), so its KV cache is managed by `SWAKVPool` instead of `MHATokenToKVPool`. The disaggregation code directly accessed `kv_pool.kv_buffer`, `kv_pool.layer_num`, etc. — attributes that exist on `MHATokenToKVPool` but not `SWAKVPool`. Result: `AttributeError` crash on both prefill and decode.

**Fix:** Added property accessors (`layer_num`, `start_layer`, `kv_sharding`, `dtype`, `attention_data_partition_axis`) to `SWAKVPool`, and rewrote `_write_kv_to_pool` to dispatch per-layer writes to the correct sub-pool (`full_kv_pool` vs `swa_kv_pool`) with SWA index remapping.

**Commits:** 607351f, a1d8cecb

## 2. process_allgather int64→int32 room ID truncation

**Problem:** `generate_bootstrap_room()` returned 63-bit room IDs. On TPU, `jax_enable_x64` is off by default, so `process_allgather` silently truncates int64 arrays to int32. ~50% of valid room IDs became negative after truncation and were skipped by the `if room < 0: continue` sentinel check in `synced_terminal_rooms`. Result: decode never recognized completed transfers — 0 output tokens.

**Fix:** Generate room IDs in `[0, 2^31-1]` (fits in int32 natively), use `np.int32` dtype in the allgather array, change sentinel check to `if room == -1:`, and mask the CRC32 fallback with `& 0x7FFFFFFF`.

**Commit:** 56bfad02

## 3. SPMD race in overlap decode event loop

**Problem:** The overlap disagg decode event loop (`event_loop_overlap_disagg_decode`) calls `process_allgather` (an SPMD collective) in `process_decode_queue()` while the forward thread is still executing `jit_jitted_sampler` (a different SPMD collective). On multi-host TPU, thread scheduling differences across hosts cause mismatched SPMD program order, and the TPU halts with E0200. On single-host, all devices share the same thread scheduler, so the race is unlikely to trigger.

**Fix:** Reorder the overlap decode event loop to drain the forward thread (`process_batch_result` → `output_queue.get()`) BEFORE calling `process_decode_queue`, and remove the second `process_decode_queue` call after `run_batch`. This ensures the forward thread is idle when `process_allgather` runs. Signal `sampling_info_done` directly after `run_batch` to preserve the grammar/guided-decoding pipeline. `--disable-overlap-schedule` is no longer needed.

**Performance:** A/B tested against `--disable-overlap-schedule`. No degradation — both configurations produce ~71–73 token/s decode throughput on single-request workloads. The fix retains CPU/TPU pipeline overlap that the synchronous workaround sacrificed.

**Commit:** 3c301255

**Analysis:** See [DOC_spmd_race_analysis.md](DOC_spmd_race_analysis.md) for the full race mechanism, related code pieces, and alternative fix options considered.

---

Issue 1 (SWAKVPool) affects any setup using MiMo-V2-Flash with PD disaggregation — the upstream likely already had this fix in their remote commit `c6105f1`; ours was porting it to this branch. Issues 2 and 3 are specific to **multi-process-per-pod** setups (v6e-16 with 4 JAX processes per pod) where workers must coordinate via `process_allgather` in `synced_terminal_rooms`. The upstream benchmarks used v7x 2x2x2 (single JAX process per pod), so intra-pod multi-host coordination was never invoked — PD cross-pod transfer itself worked fine in both setups.
