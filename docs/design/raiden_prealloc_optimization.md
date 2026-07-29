# Raiden Decode Prealloc Overhead Optimization

## 1. Problem Statement

In PD disaggregation with the Raiden transfer engine, the decode-side "prealloc
overhead" — time between a request entering the decode prealloc queue and KV
transfer actually starting — dominates TTFT (time to first token) for large
prompts.

Raiden is send-initiated: the prefill (producer) calls `register_read()` to mark
blocks readable, then the decode (consumer) discovers them via bootstrap,
allocates local KV pages, initializes a receiver, and calls `start_read()`.
This sequential chain means decode cannot begin any work until prefill finishes
its forward pass, registers the blocks, and publishes metadata to bootstrap.

Benchmark data from the existing PD setup shows prealloc_wait of
18/38/39/75 ms for 512/1024/2048/4096 input tokens — comparable to or exceeding
the actual KV transfer time. This erases Raiden's ~3x speed advantage over
path-A for the transfer itself.

### Sub-phase breakdown (decode side)

```
prealloc_wait = metadata_wait + kv_alloc + receiver_init + transfer_setup
```

- `metadata_wait`: time polling bootstrap until P publishes block metadata
- `kv_alloc`: time to allocate local KV pages from the decode pool
- `receiver_init`: time to create the receiver and build page mappings
- `transfer_setup`: time from receiver init to first `start_read` call

## 2. Optimizations

Four optimizations target different sub-phases. They compose independently and
are safe to deploy individually.

### 2.1 Speculative Preallocation

**Target sub-phase:** `kv_alloc` (1-2 ms gain)

**Idea:** Allocate local KV pages before bootstrap metadata arrives. When
metadata is not yet available (bootstrap 404), keep the allocated pages on the
prealloc entry for the next scheduler tick instead of freeing them.

**Implementation:**

- `decode.py` / `_admit_decode_prealloc()`: entries with existing `kv_indices`
  skip the capacity check and allocation, going straight to the metadata query.
  New entries get KV allocated immediately and `kv_alloc_done` marked before the
  metadata check. On metadata 404, pages are retained (not freed).
- `DecodeBookkeeping` gains a `kv_alloc_time: float | None` field that stamps
  when speculative allocation occurred.
- TTL guard (5 s): if metadata never arrives, pages are freed to prevent leaks.
  Implemented via `time.perf_counter()` comparison in the admission loop.
- Speculative entries are counted toward the inflight capacity budget
  (`n_speculative` counter) so they don't starve other requests.
- `req_time_stats.py`: phase specs reordered so `kv_alloc` measures
  `prealloc_entry -> kv_alloc_done` (before metadata_wait), reflecting the new
  ordering.

**Risk:** Memory pressure from holding speculatively allocated pages. Mitigated
by TTL-based cleanup and inclusion of speculative count in inflight/capacity
budget.

**Files changed:** `decode.py`, `req_time_stats.py`

### 2.2 Two-Phase Bootstrap Publish (Pre-publish Block IDs)

**Target sub-phase:** `metadata_wait` (10-50 ms gain, largest impact)

**Idea:** Split prefill's bootstrap publish into two phases:
1. **Pre-publish** (before forward pass): publish block IDs with
   `raiden_ready=false`. Decode can discover metadata and begin local setup.
2. **Confirm** (after forward + `register_read`): update the entry with
   `raiden_ready=true`. Decode's `_discover_and_start_chunks` then calls
   `start_read`.

This overlaps decode-side local setup (KV alloc, receiver init, page mapping)
with prefill's forward pass computation.

**Implementation:**

- `bootstrap.py`: `RegisterTransferRequest` gains `raiden_ready: bool = True`.
  Server-side registry merges (not overwrites) when updating an existing entry
  with `raiden_ready=true`, preserving any fields from the pre-publish.
- `conn.py`: `producer_pre_publish()` publishes to bootstrap with
  `raiden_ready=false`, skipping `register_read`. `pre_publish_chunk()` wraps
  this for per-chunk use. `_discover_and_start_chunks()` skips chunks where
  `raiden_ready=false` (line 1118: `if not chunk_info.get("raiden_ready", True):
  continue`).
- `prefill.py`: `_raiden_pre_publish_batch()` extracts block IDs from the
  `req_to_token` pool and pre-publishes for all PD requests in the batch BEFORE
  `run_batch()`. Called from both `event_loop_normal_disagg_prefill` and
  `event_loop_overlap_disagg_prefill`. Failures are non-fatal (debug log only);
  the normal handoff path still publishes with `raiden_ready=true`.

**Risk:** Decode calling `start_read` before `register_read` (data not yet
readable). Prevented by the `raiden_ready` flag — `_discover_and_start_chunks`
only calls `start_read` when `raiden_ready=true`.

**Files changed:** `bootstrap.py`, `conn.py`, `prefill.py`

### 2.3 Connection Pre-warming

**Target sub-phase:** `transfer_setup` (2-10 ms gain)

**Idea:** Pre-establish TCP connections to known prefill Raiden endpoints so
`start_read` doesn't pay DNS resolution and TCP handshake cost on the hot path.

**Implementation:**

- tpu-raiden fork (`DigitalWNZ/tpu-raiden`):
  `KVCacheManager.pre_connect(remote_endpoint)` added — a pure-Python TCP
  connect+close that warms the OS DNS cache and verifies reachability. Accepts
  endpoint strings (`"host:port"`), list-of-dicts, or single dict formats.
  Timeout: 2s. Failures are debug-logged and swallowed.
- `wrapper.py`: `RaidenTransferWrapper.pre_connect(remote_endpoint)` calls
  `engine.pre_connect()` if available on the tpu-raiden fork. Falls back to
  no-op if the method is unavailable (graceful degradation).
- `decode.py` / `_admit_one_raiden()`: on first encounter with a new prefill
  endpoint `(host, port)`, calls `pre_connect` before creating the receiver.
  Tracks pre-connected endpoints in `_raiden_preconnected` set (lazily
  initialized on the Scheduler instance) to avoid redundant calls.

**Dependency:** Requires tpu-raiden fork with `KVCacheManager.pre_connect()`
API. The sglang-jax side gracefully degrades (no-op) without the fork change.

**Files changed:** `wrapper.py`, `decode.py`,
`tpu-raiden/tpu_raiden/api/jax/kv_cache_manager.py`

### 2.4 Batch Bootstrap Metadata Fetch

**Target sub-phase:** `metadata_wait` at high concurrency (orthogonal gain)

**Idea:** Replace N sequential `GET /get_transfer_info` calls with a single
`POST /batch_get_transfer_info` when multiple requests are in the prealloc
queue.

**Implementation:**

- `bootstrap.py` (server): `_Registry.batch_get_transfer_chunks()` reads
  metadata for multiple rooms under one lock acquisition.
  `POST /batch_get_transfer_info` endpoint accepts a list of room ints and
  returns `{room_str: info_or_null}`.
- `bootstrap.py` (client): `BootstrapClient.batch_get_transfer_info()` sends the
  batch POST and normalizes JSON chunk-index keys back to int.
- `decode.py` / `_admit_decode_prealloc()`: before the admission loop,
  batch-fetches metadata for all pending entries (collects bootstrap_room for all
  non-started entries). Results are passed to `_admit_one_raiden()` as
  `prefetched_metadata` dict, which checks it before falling back to per-entry
  fetch on error.

**Risk:** None — purely additive. Falls back to per-entry fetch if the batch
call fails.

**Files changed:** `bootstrap.py`, `decode.py`

## 3. Measured Results

Deployed on TPU VM `a6e-wangez` (v6e-32, 4x8 topology, us-east5-b) with
DeepSeek-R1-Distill-Qwen-1.5B model, TP=4, Python 3.12 venv,
JAX 0.10.2 + libtpu 0.0.42.1, tpu-raiden built from source
(`DigitalWNZ/tpu-raiden` fork).

Worker 0 (10.202.0.127): bootstrap (port 8998) + prefill (port 10000)
Worker 1 (10.202.0.119): decode (port 10001) + router (port 30000)

### TTFT (Time To First Token)

| Concurrency | Input Length | Median TTFT | P99 TTFT |
|:-----------:|:-----------:|:-----------:|:--------:|
| 1 | 512 | 44.7 ms | 52.4 ms |
| 1 | 1024 | 53.0 ms | 55.8 ms |
| 1 | 2048 | 69.1 ms | 73.6 ms |
| 1 | 4096 | 112.4 ms | 114.5 ms |
| 4 | 512 | 61.2 ms | 107.6 ms |
| 4 | 1024 | 68.9 ms | 124.1 ms |
| 4 | 2048 | 107.4 ms | 156.2 ms |
| 4 | 4096 | 171.5 ms | 268.9 ms |

### PD-TIME-STATS Breakdown (168 entries, all input lengths combined)

| Phase | Median | P90 | P99 | Min | Max |
|:------|:------:|:---:|:---:|:---:|:---:|
| metadata_wait | 2.7 ms | 29.8 ms | 50.7 ms | 0.8 ms | 54.6 ms |
| kv_alloc | 1.2 ms | 5.5 ms | 9.9 ms | 0.8 ms | 10.5 ms |
| receiver_init | 0.1 ms | 0.1 ms | 0.2 ms | 0.0 ms | 0.5 ms |
| transfer_setup | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.1 ms |
| **prealloc_wait** | **2.8 ms** | **29.9 ms** | **50.9 ms** | **0.9 ms** | **54.7 ms** |
| first_chunk_wait | 21.7 ms | 51.3 ms | 106.5 ms | 1.0 ms | 107.4 ms |
| transfer_tail | 28.5 ms | 63.9 ms | 118.9 ms | 0.0 ms | 119.8 ms |
| kv_wait | 55.2 ms | 120.9 ms | 225.6 ms | 17.1 ms | 228.0 ms |
| total | 61.7 ms | 135.0 ms | 234.3 ms | 25.8 ms | 234.4 ms |

### Key observations

- **`metadata_wait` median: 2.7 ms** — down from baseline 18-75 ms. The two-phase
  publish (Opt 2.2) and batch fetch (Opt 2.4) are clearly effective.
- **`transfer_setup` median: 0.0 ms** — connection pre-warming (Opt 2.3)
  eliminated this sub-phase entirely.
- **`prealloc_wait` median: 2.8 ms** — the total prealloc overhead is now a small
  fraction of the overall transfer time.
- The P90 `metadata_wait` of 29.8 ms suggests some requests still hit the
  sequential path (pre-publish fails or metadata arrives after forward completes).
  This is expected tail behavior — the pre-publish is best-effort.

## 4. Profiling

### Instrumentation

All sub-phases are instrumented via `PD-TIME-STATS` (enabled with
`--enable-request-time-stats-logging`). Lines are emitted to the decode server
log for every completed KV transfer.

### Deployment setup

**Environment requirements:**
- Python >= 3.12 (install via deadsnakes PPA on Ubuntu 22.04)
- JAX 0.10.2 from Google artifact registry:
  `pip install 'jax[tpu]==0.10.2' -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/`
- tpu-raiden built from source: `HERMETIC_PYTHON_VERSION=3.12 ./build.sh jax`
- Single-host JAX on multi-host VM: set `TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"`
  and `TPU_HOST_BOUNDS="1,1,1"` for v6e-32 (gives 4 chips per process)
- Use `--tp-size 4` to match the 4 visible chips

**Launch commands (Worker 0 — prefill):**
```bash
source ~/venv312/bin/activate
export PYTHONPATH="$HOME/tpu-raiden:${PYTHONPATH:-}"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,1"

python3 -m sgl_jax.srt.disaggregation.run_bootstrap --host 0.0.0.0 --port 8998 &
python3 -m sgl_jax.launch_server \
  --model-path $MODEL --host 0.0.0.0 --port 10000 --tp-size 4 \
  --page-size 128 --disable-radix-cache --enable-request-time-stats-logging \
  --disaggregation-mode prefill --disaggregation-bootstrap-url http://localhost:8998 \
  --chunked-prefill-size 16384 --disaggregation-enable-d2h \
  --disaggregation-use-raiden --enable-metrics
```

**Launch commands (Worker 1 — decode + router):**
```bash
# Same env vars as worker 0
python3 -m sgl_jax.launch_server \
  --model-path $MODEL --host 0.0.0.0 --port 10001 --tp-size 4 \
  --page-size 128 --disable-radix-cache --enable-request-time-stats-logging \
  --disaggregation-mode decode --disaggregation-bootstrap-url http://$P_HOST:8998 \
  --chunked-prefill-size 16384 --disaggregation-enable-d2h \
  --disaggregation-use-raiden --disaggregation-max-inflight-transfers 8 \
  --mem-fraction-static 0.80 --enable-metrics &

python3 -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://$P_HOST:10000 8998 --decode http://localhost:10001 \
  --prefill-bootstrap-host $P_HOST --max-concurrent-requests 64 \
  --host 0.0.0.0 --port 30000
```

**TTFT benchmark:**
```bash
python3 -m sgl_jax.bench_serving --backend sgl-jax \
  --base-url http://localhost:30000 --model $MODEL \
  --dataset-name random --random-input-len $L --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 13 --max-concurrency 1 \
  --warmup-requests 3 --output-file /tmp/ttft.jsonl
```

### Validation

After each optimization:
1. Re-run profiling, compare sub-phase breakdown against baseline
2. Verify correctness: GSM8K accuracy unchanged
3. Stability: sustained load at conc=16 and conc=32, monitor for OOM or
   `failed_recving`

## 5. Architecture Diagram

```
                    BEFORE (sequential)
                    ===================

  Prefill:  --[forward]-----------[register_read]--[bootstrap publish]----------
  Decode:   -------------------------------------------[poll]--[alloc]--[init]--[start_read]--

                    AFTER (overlapped)
                    ==================

  Prefill:  --[pre-publish]--[forward]-----------[register_read]--[confirm ready]---
  Decode:   ---------[poll]--[alloc]--[init]-----------------------------[start_read]------
                     ^                                                   ^
                     pre-published metadata                              raiden_ready=true
                     (raiden_ready=false)                                 triggers start_read
```

The key insight: decode's local work (KV page allocation, receiver
initialization, page mapping) does not depend on the KV data being readable.
It only needs the block IDs and endpoint descriptors, which are known before
the forward pass.

## 6. File Change Summary

| File | Changes |
|---|---|
| `decode.py` | Speculative prealloc (kv_alloc_time, TTL guard, n_speculative), batch metadata fetch (prefetched_metadata), connection pre-warm (_raiden_preconnected) |
| `prefill.py` | `_raiden_pre_publish_batch()` method, calls in both event loops |
| `bootstrap.py` | `raiden_ready` field, merge-on-update, `batch_get_transfer_chunks()`, `POST /batch_get_transfer_info` endpoint, `BootstrapClient.batch_get_transfer_info()` |
| `conn.py` | `producer_pre_publish()`, `pre_publish_chunk()`, `raiden_ready` skip in `_discover_and_start_chunks` |
| `wrapper.py` | `RaidenTransferWrapper.pre_connect()` method |
| `req_time_stats.py` | Phase spec reordering for speculative prealloc |
| `tpu-raiden/.../kv_cache_manager.py` | `KVCacheManager.pre_connect()` — TCP connect+close for DNS/connection warming |

## 7. Future Work

- **Raiden receive-initiated mode** (Opt 5): add `register_receive()` /
  `notify_ready()` to raiden itself. Only pursue if profiling shows
  `metadata_wait` is still dominant after Opts 1-4.
- **P90 tail reduction**: the P90 metadata_wait of 29.8 ms suggests
  pre-publish failures or race conditions on some requests. Investigate with
  per-request tracing.
- **MiMo Flash validation**: profile with the target production model after
  baseline validation with 1.5B.
- **XLA compilation cache**: setting `JAX_COMPILATION_CACHE_DIR` would eliminate
  the ~3 min precompile time on server restart.
