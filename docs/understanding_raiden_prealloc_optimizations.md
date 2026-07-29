# Understanding the Raiden Prealloc Optimizations

A beginner-friendly guide to what we optimized, why, and how it works.

---

## Background: What Problem Are We Solving?

### What is PD Disaggregation?

When you send a prompt to a large language model (LLM), the inference happens in two stages:

1. **Prefill** — The model reads your entire prompt at once and produces an internal representation called **KV cache** (key-value cache). This stage is compute-heavy (lots of math per token) but processes all tokens in parallel. It produces the first output token.

2. **Decode** — The model generates tokens one at a time, using the KV cache from prefill. Each step is memory-bandwidth-heavy (reading lots of data) but does very little computation per step.

Because prefill and decode have opposite hardware requirements (compute vs. memory bandwidth), **PD disaggregation** runs them on separate TPU chips:

```
User prompt                                        Output tokens
     |                                                  ^
     v                                                  |
 [ Prefill TPU ]  ---(KV cache transfer)--->  [ Decode TPU ]
   "Read the prompt,                           "Generate tokens
    compute KV cache"                           one at a time"
```

The user's metric that matters most is **TTFT (Time To First Token)** — how long until they see the first response token. TTFT = prefill time + KV transfer time + decode startup overhead.

### What is Raiden?

Raiden is Google's high-performance data transfer engine for TPUs. It moves the KV cache from the prefill TPU to the decode TPU over the network. It's ~3x faster than the older "path-A" method for the raw data transfer itself.

But Raiden is **send-initiated**: the prefill side (sender) has to explicitly mark each block of KV cache as "readable" before the decode side (receiver) can pull it. This creates a coordination problem.

### What is "Prealloc Overhead"?

Before the decode TPU can start receiving KV data, it must go through several setup steps. The total time spent on these steps is the **prealloc overhead**:

```
                      prealloc overhead (what we're optimizing)
              |<-------------------------------------------------->|
              |                                                    |
  [Request enters    [metadata    [KV page    [receiver  [transfer
   decode queue]      arrives]    allocated]   created]   starts]
```

Each sub-step is:

| Sub-step | What happens | Analogy |
|----------|-------------|---------|
| **metadata_wait** | Decode polls the bootstrap server asking "has prefill published the block metadata yet?" and waits until the answer is yes. | Waiting at a restaurant for the kitchen to tell you what's on the menu |
| **kv_alloc** | Decode allocates local memory pages to receive the incoming KV data. | Setting a table before the food arrives |
| **receiver_init** | Decode creates a Raiden receiver object and maps which local pages correspond to which remote blocks. | Writing down which dish goes on which plate |
| **transfer_setup** | Decode calls `start_read()` to begin the actual data transfer. | Telling the kitchen to start sending the food |

**The problem:** In the original code, ALL of this happens sequentially AFTER prefill finishes. For a 4096-token prompt, the prealloc overhead alone was 75ms — comparable to or exceeding the actual data transfer time. This erased Raiden's speed advantage.

### The Coordination Flow (Before Optimization)

Here's what happens step by step in the original code:

```
Time ──────────────────────────────────────────────────────────────>

Prefill TPU:
  1. Receive prompt
  2. Run forward pass (compute KV cache)          ████████████
  3. Call register_read() to mark blocks readable       |
  4. POST to bootstrap: "here are the block IDs"        |──▶ publish
                                                              |
Decode TPU:                                                   |
  (waiting...)                                                |
  5. Poll bootstrap: "any metadata for me?"        ◄──────────┘
  6. Yes! Allocate local KV pages
  7. Create receiver, build page mappings
  8. Call start_read() to pull data
  9. Wait for transfer to complete
  10. Start generating tokens
```

Notice: steps 5-8 on decode cannot start until step 4 on prefill completes. The decode TPU just sits idle while prefill is computing.

---

## The Four Optimizations

### Optimization 1: Speculative Preallocation

**Sub-phase targeted:** `kv_alloc` (saving ~1-2ms)

**The insight:** Allocating local KV pages does not require any information from prefill. The decode side knows the prompt length from the request metadata, so it knows how many pages it will need.

**What we changed:** Instead of waiting for metadata to arrive before allocating pages, decode now allocates pages immediately when a request enters the prealloc queue. If metadata hasn't arrived yet (bootstrap returns 404), the allocated pages are kept on the entry and reused on the next scheduler tick.

```
BEFORE:
  [request arrives] → [wait for metadata] → [allocate pages] → [setup]
                       ~~~wasted time~~~

AFTER:
  [request arrives] → [allocate pages immediately]
                       [wait for metadata] → [setup]  (pages already allocated)
```

**Safety mechanism:** A 5-second TTL guard frees speculatively allocated pages if metadata never arrives (e.g., prefill crashed). This prevents memory leaks.

**Code location:** `decode.py` in `_admit_decode_prealloc()` — entries with existing `kv_indices` skip the capacity check and go straight to the metadata query.

---

### Optimization 2: Two-Phase Bootstrap Publish (Biggest Impact)

**Sub-phase targeted:** `metadata_wait` (saving ~10-50ms)

**The insight:** Decode's local setup work (page allocation, receiver creation, page mapping) only needs to know *which blocks* will be sent — it does NOT need the data to be readable yet. The block IDs are known before the forward pass even starts (they're determined by the memory allocator when the request is scheduled).

**What we changed:** We split the bootstrap publish into two phases:

**Phase 1 — Pre-publish (BEFORE forward pass):**
Prefill publishes block IDs to bootstrap with a flag `raiden_ready=false`. This tells decode: "here's where the KV data will be, but it's not readable yet."

**Phase 2 — Confirm (AFTER forward pass + register_read):**
Prefill updates the same bootstrap entry with `raiden_ready=true`. This tells decode: "the data is now readable, you can call start_read."

```
BEFORE (sequential — decode idle during forward pass):
  Prefill:  ──[forward ████████████]──[register_read]──[publish]──
  Decode:   ──────────────idle──────────────────────────[poll]──[alloc]──[init]──[start_read]──

AFTER (overlapped — decode works during forward pass):
  Prefill:  ──[pre-publish]──[forward ████████████]──[register_read]──[confirm]──
  Decode:   ────────[poll]──[alloc]──[init]─────────────────────────[start_read]──
                    ↑                                                ↑
                    gets metadata early                              waits for
                    (raiden_ready=false)                              raiden_ready=true
```

The decode side now overlaps its setup work with the prefill forward pass, instead of waiting until it finishes. For a 4096-token prompt where the forward pass takes ~50ms, this can save most of that 50ms from the `metadata_wait` sub-phase.

**Safety mechanism:** The `_discover_and_start_chunks()` function in `conn.py` checks the `raiden_ready` flag — it skips any chunk where `raiden_ready=false`, so `start_read()` is never called before the data is actually readable.

**Code locations:**
- `prefill.py`: `_raiden_pre_publish_batch()` — extracts block IDs and pre-publishes for all PD requests before calling `run_batch()`
- `bootstrap.py`: `raiden_ready` field on `RegisterTransferRequest`; server-side merge logic that updates (not overwrites) existing entries when confirming
- `conn.py`: `producer_pre_publish()` and `pre_publish_chunk()` for the pre-publish call; `raiden_ready` check in `_discover_and_start_chunks()`

---

### Optimization 3: Connection Pre-warming

**Sub-phase targeted:** `transfer_setup` (saving ~2-10ms)

**The insight:** The first time decode calls `start_read()` to a new prefill host, the underlying network stack must resolve the hostname (DNS lookup) and establish a TCP connection (3-way handshake). This adds latency to the first transfer.

**What we changed:** When decode encounters a new prefill endpoint for the first time (during the admission process), it does a quick TCP connect-and-close to that endpoint. This warms the OS's DNS cache and verifies network reachability. When the actual `start_read()` call happens later, the DNS result is cached and the OS TCP stack may reuse the connection path.

```
BEFORE:
  start_read() → [DNS lookup 2ms] → [TCP handshake 3ms] → [actual transfer]

AFTER:
  (earlier, during admission) pre_connect() → [DNS lookup] → [TCP connect] → [close]
  (later)  start_read() → [DNS cached, fast] → [actual transfer]
```

**Important detail:** This optimization required a change in the `tpu-raiden` library itself (in the fork at `DigitalWNZ/tpu-raiden`). We added a `pre_connect()` method to `KVCacheManager` that does the TCP connect+close. The sglang-jax side gracefully falls back to a no-op if the fork method is unavailable.

**Code locations:**
- `tpu-raiden/tpu_raiden/api/jax/kv_cache_manager.py`: `pre_connect()` — pure Python TCP connect+close
- `wrapper.py`: `RaidenTransferWrapper.pre_connect()` — delegates to the engine's `pre_connect()` if available
- `decode.py`: `_admit_one_raiden()` — calls `pre_connect` on first encounter with each `(host, port)` pair, tracked in `_raiden_preconnected` set

---

### Optimization 4: Batch Bootstrap Metadata Fetch

**Sub-phase targeted:** `metadata_wait` at high concurrency

**The insight:** When multiple requests are waiting in the prealloc queue (high concurrency), the decode side was making N separate HTTP requests to bootstrap to check metadata for each one. Each HTTP round-trip takes ~1-2ms even on localhost. With 16 concurrent requests, that's 16-32ms of serialized HTTP calls.

**What we changed:** We added a batch endpoint `POST /batch_get_transfer_info` that accepts a list of bootstrap room IDs and returns all their metadata in one HTTP call. Before the admission loop, decode now batch-fetches metadata for ALL pending entries, then uses the cached results during per-entry admission.

```
BEFORE (N requests in queue, N HTTP calls):
  for each request:
    GET /get_transfer_info?room=X    → 1-2ms
    GET /get_transfer_info?room=Y    → 1-2ms
    GET /get_transfer_info?room=Z    → 1-2ms
  Total: N × 1-2ms

AFTER (1 batch HTTP call):
  POST /batch_get_transfer_info [X, Y, Z]    → 1-2ms total
  (use cached results for per-entry admission)
```

**Scaling:** The gain increases with queue depth. At conc=1 there's no benefit (one request = one call either way). At conc=16+, this saves ~15-30ms per admission cycle.

**Code locations:**
- `bootstrap.py`: server-side `batch_get_transfer_chunks()` reads metadata for multiple rooms under one lock; `POST /batch_get_transfer_info` endpoint; client-side `batch_get_transfer_info()` method
- `decode.py`: `_admit_decode_prealloc()` calls `batch_get_transfer_info()` before the admission loop, passes results as `prefetched_metadata` dict to `_admit_one_raiden()`

---

## How They Work Together

The four optimizations are independent — each targets a different sub-phase, and each has its own fallback if something goes wrong. But they compose to attack the full prealloc overhead:

```
                         prealloc overhead breakdown
                    |<---------------------------------------->|
                    |                                          |
  BEFORE:  [metadata_wait 35ms] [kv_alloc 2ms] [recv_init] [transfer_setup 5ms]
                                                            Total: ~42ms

  AFTER:   [metadata_wait 3ms] [kv_alloc 1ms] [recv_init] [transfer_setup 0ms]
            ↑ Opt 2+4            ↑ Opt 1                    ↑ Opt 3
                                                            Total: ~4ms
```

---

## Measured Results

Deployed on TPU VM `a6e-wangez` (v6e-32, us-east5-b) with DeepSeek-R1-Distill-Qwen-1.5B model, TP=4.

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

### PD-TIME-STATS Breakdown (168 entries, all input lengths)

| Phase | Median | P90 | P99 |
|:------|:------:|:---:|:---:|
| metadata_wait | 2.7 ms | 29.8 ms | 50.7 ms |
| kv_alloc | 1.2 ms | 5.5 ms | 9.9 ms |
| receiver_init | 0.1 ms | 0.1 ms | 0.2 ms |
| transfer_setup | 0.0 ms | 0.0 ms | 0.0 ms |
| prealloc_wait | 2.8 ms | 29.9 ms | 50.9 ms |
| first_chunk_wait | 21.7 ms | 51.3 ms | 106.5 ms |
| transfer_tail | 28.5 ms | 63.9 ms | 118.9 ms |
| kv_wait | 55.2 ms | 120.9 ms | 225.6 ms |
| total | 61.7 ms | 135.0 ms | 234.3 ms |

Key observation: **median metadata_wait is 2.7ms** and **transfer_setup is 0.0ms**, confirming the optimizations are effective. The prealloc_wait (which is the sum of metadata_wait + kv_alloc + receiver_init + transfer_setup) has a median of just 2.8ms.

---

## Glossary

| Term | Meaning |
|------|---------|
| **TPU** | Tensor Processing Unit — Google's custom chip for ML workloads |
| **KV cache** | Key-Value cache — intermediate data produced during prefill that decode needs to generate tokens |
| **Raiden** | Google's high-performance transfer engine for moving data between TPU hosts |
| **Bootstrap server** | A lightweight HTTP server that acts as a rendezvous point between prefill and decode — prefill publishes metadata there, decode polls for it |
| **PD disaggregation** | Running prefill and decode on separate TPU hosts |
| **TTFT** | Time To First Token — primary latency metric (lower is better) |
| **Prealloc** | The setup work decode does before it can receive KV data (allocate pages, create receiver, etc.) |
| **register_read** | Raiden API call on prefill side that marks blocks as readable |
| **start_read** | Raiden API call on decode side that initiates the data pull |
| **TP** | Tensor Parallelism — splitting a model across multiple chips |
| **bootstrap room** | A unique ID for each KV transfer, used as a key in the bootstrap server's registry |
