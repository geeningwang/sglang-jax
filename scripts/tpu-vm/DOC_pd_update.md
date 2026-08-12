# PD Multi-Host TPU Fixes (since d3ac424)

**Base commit:** d3ac42493745dcf4d9e141ef0a2a351029d7cc25 (primatrix upstream, single-host v7x)
**Branch:** mimo-tpu7-stage3
**Target:** Multi-host TPU v6e-16/v6e-32 (4–8 workers per VM)

---

## 1. SWAKVPool not supported in JaxTransfer disaggregation

**Problem:** MiMo-V2-Flash uses hybrid attention (full + sliding-window), so its KV cache is managed by `SWAKVPool` instead of `MHATokenToKVPool`. `SWAKVPool` wraps two independent sub-pools (`full_kv_pool` and `swa_kv_pool`) with different sizes and index spaces, connected by `full_to_swa_index_mapping`. The disaggregation code had two gaps:

1. **Missing API surface:** The code directly accessed `kv_pool.kv_buffer`, `kv_pool.layer_num`, etc. — attributes that exist on `MHATokenToKVPool` but not `SWAKVPool`. Result: `AttributeError` crash on both prefill and decode.

2. **Missing page-index remapping on prefill gather:** The JaxTransfer extraction path (`_extract_req_kv`) computes `page_indices` from `req_to_token` (full-pool token indices) and uses the same indices to gather from all layers, including SWA layers. For SWA layers, `get_kv_buffer` returns the `swa_kv_pool` buffer, but full-pool page IDs are meaningless in the SWA pool — they address wrong pages or go out-of-bounds. The Raiden path handles this correctly via `_extract_swa_block_ids_for_chunk`, but the JaxTransfer path had no equivalent remapping.

**Fix:**

1. Added property accessors (`layer_num`, `start_layer`, `kv_sharding`, `dtype`, `attention_data_partition_axis`) to `SWAKVPool`, and rewrote `_write_kv_to_pool` to dispatch per-layer writes to the correct sub-pool with SWA index remapping (decode side).

2. In `_extract_req_kv`, after computing full-pool `page_indices`, detect `SWAKVPool` and compute separate `swa_page_indices` by mapping full-pool token indices through `full_to_swa_index_mapping` and dividing by `page_size`. Replace the bulk gather with a per-layer loop that selects `swa_page_indices` for SWA layers and `page_indices` for full-attention layers (prefill side).

**Commits:** 607351f, a1d8cecb (gap 1); 428a82c5 (gap 2)

**Verified:** Tested by temporarily reversing the full-pool allocation order (`pages_per_rank` down to `1`) so that full-pool and SWA-pool page IDs no longer coincide. The 1P1D stack produced correct output with the fix, confirming the remapping is necessary and working.

**Analysis:** See [DOC_swakvpool_disagg_fix.md](DOC_swakvpool_disagg_fix.md) for the detailed code walkthrough and test results.

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

## 4. DP>1 SWA index remapping in `_write_kv_to_pool`

**Problem:** In `_write_kv_to_pool` (decode side), when `full_to_swa_index_mapping` is a list (DP>1), the code incorrectly assumed `loc_np` contains concatenated data for all `dp_size` ranks and looped `for rank in range(dp_size)`, slicing `loc_np` into per-rank segments. In reality, `loc_np` is single-rank data allocated for one `dp_rank` — the loop was indexing out of bounds or into the wrong rank's mapping.

**Fix:** Select the single rank's mapping via `mapping[int(getattr(req, "dp_rank", 0) or 0)]` before computing `swa_loc_np`, eliminating the per-rank loop entirely. This matches the pattern in three other SWA mapping sites:
- `_extract_req_kv` (prefill.py:731-732)
- Raiden decode path (decode.py:833-834)
- `_swa_page_ids_for_chunk` (prefill.py:577-578)

**Status:** Tested with dp_size=1 in both 1P1D and 1P2D configurations — correct output confirmed. Runtime-verified with dp_size=2 on v6e-32 (8 hosts, mesh (2,16), 32 devices per VM) — correct output on 2026-08-12.

**Analysis:** See [DOC_swakvpool_disagg_fix.md](DOC_swakvpool_disagg_fix.md) Change 6 for the before/after code diff.

## 5. `synced_terminal_rooms` threshold too high for dp_size>1

**Problem:** `synced_terminal_rooms` (decode side) requires `len(sts) >= nproc` — all JAX processes must report SUCCESS before a transfer is considered complete. With `dp_size=2`, each request is handled by only one DP rank (half the processes). Only `nproc // dp_size` processes ever report for a given room, so the threshold is never met. Result: transfers succeed on every individual process but are never acknowledged as complete → 0 output tokens.

**Fix:** Added `dp_size: int = 1` parameter to `synced_terminal_rooms`. Compute `nproc_per_dp = nproc // max(dp_size, 1)` and use that as the success threshold. Pass `dp_size=self.dp_size` from the call site in `_drain_transfer_queue_synced` (decode.py).

**Files:** `python/sgl_jax/srt/disaggregation/common/multihost_sync.py`, `python/sgl_jax/srt/disaggregation/decode.py`

**Verified:** End-to-end dp_size=2 symmetric test on v6e-32 (8 hosts, mesh (2,16)) — request completed with correct output. Regression-tested with dp_size=1 — no change in behavior (threshold reduces to nproc).

## 6. `local_kv_spec_for_pool` shape divisor wrong for non-trivial mesh

**Problem:** `local_kv_spec_for_pool` (prefill.py) computes the local shard shape by dividing the global shape's sharded dimension by `jax.process_count()`. This is only correct when every process holds one device on the sharded axis. With mesh `(2, 16)` and 8 processes, the tensor axis has 16 devices (2 per host), but the code divides by 8 — producing a local shape that's 2× too small and doesn't match the actual per-process data.

**Fix:** Compute the divisor from the mesh: find how many devices sit on the sharded axis (`ndev_on_axis`), divide by `jax.local_device_count()` to get `nproc_on_axis`, and use that as the divisor. With mesh `(2, 16)` and 4 local devices: `ndev_on_axis=16`, `nproc_on_axis=4`, which gives the correct local shape.

**File:** `python/sgl_jax/srt/disaggregation/prefill.py`

**Verified:** Same end-to-end test as Issue 5. Regression-tested with dp_size=1 — formula reduces to `nproc_on_axis = nproc` (identical to original).

---

Issues 1 and 4 (SWAKVPool) affect any setup using MiMo-V2-Flash with PD disaggregation — the upstream likely already had the gap 1 fix in their remote commit `c6105f1`; ours was porting it to this branch. Issues 2 and 3 are specific to **multi-process-per-pod** setups (v6e-16 with 4 JAX processes per pod) where workers must coordinate via `process_allgather` in `synced_terminal_rooms`. Issues 5 and 6 are specific to **dp_size>1** configurations where the mesh shape differs from the simple `(1, ndevices)` layout. The upstream benchmarks used v7x 2x2x2 (single JAX process per pod, dp_size=1), so neither multi-host coordination nor dp_size>1 code paths were exercised.

All six issues have been verified end-to-end on v6e-32 as of 2026-08-12:

- **dp_size=1** (mesh (1,32)): Fully working. 1P1D on v6e-32 produces correct output on all requests — "2 + 2 = 4", "The capital of France is **Paris**." — with 25s first-request latency (XLA compilation) and sub-second warm latency.
- **dp_size=2** (mesh (2,16)): dp_rank=0 requests produce correct output through both 1P1D and 1P2D pipelines. dp_rank=1 requests produce garbled output (open issue below). Long-input requests (16K+ tokens) can trigger E0100 OOM during KV extraction due to the `jnp.stack(layer_kvs)` allocation — see [DOC_pd_environment.md](DOC_pd_environment.md#e0100-oom-during-kv-extraction-with-large-inputs-dp_size2).

**Open issue:** dp_rank=1 produces garbled output with dp_size=2. Requests routed to dp_rank=0 are correct; dp_rank=1 consistently returns nonsensical text. Observed in both 1P1D and 1P2D configurations (1 prefill + 2 decode clusters on 3× v6e-32). See [DOC_pd_environment.md](DOC_pd_environment.md#dp_rank1-produces-garbled-output-dp_size2). Current workaround: use dp_size=1.
