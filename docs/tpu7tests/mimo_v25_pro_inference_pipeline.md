# MiMo-V2.5-Pro Inference Pipeline: Module-by-Module Guide

This document traces the full MiMo-V2.5-Pro inference data flow in sglang-jax,
from a raw HTTP request to decoded output text. Each section covers the relevant
source file, input/output specs, and practical correctness checks.

Deployment context: 4-node GKE TPU v7x (2x2x4 DWS), tp-size=32, FP8 static weights
served via gcsfuse from GCS.

---

## Pipeline Overview

```
HTTP POST /v1/chat/completions
    |
    v
1) sgl_jax/launch_server.py
   ServerArgs.parse() → http_server.launch()
    |
    v
2) srt/entrypoints/http_server.py
   FastAPI app → TokenizerManager → Scheduler (IPC)
    |
    v
3) srt/managers/tokenizer_manager.py
   - apply_chat_template(messages)
   - tokenizer.encode(prompt)
    |
    v
4) srt/managers/scheduler.py
   - schedule_batch() — KV cache slot allocation
   - ScheduleBatch → TP worker (ZMQ)
    |
    v
5) srt/managers/tp_worker_overlap_thread.py
   TpWorkerClass(server_args, mesh=mesh)
    |
    v
6) srt/model_executor/model_runner.py
   ModelRunner.forward(forward_batch)
   - prefill mode: full prompt sequence
   - decode mode: one token per step
    |
    v
7) srt/models/mimo_v2_pro.py  (MiMoV2ForCausalLM)
   MiMoV2Model.__call__(forward_batch, token_to_kv_pool)
   → 70 × MiMoV2DecoderLayer
   → logits
    |
    v
8) Sampler → token IDs
    |
    v
9) srt/managers/detokenizer_manager.py
   tokenizer.decode(token_ids)
    |
    v
HTTP response (OpenAI-compatible JSON)
```

---

## Module 0 — Weight Preparation (One-Time, Already Done)

MiMo-V2.5-Pro weights are stored **directly in HuggingFace FP8 safetensors format**
in GCS. No checkpoint conversion is required — sglang-jax reads FP8 safetensors
natively and applies dequantization during weight loading.

### GCS paths

| Resource | Path | Size |
|----------|------|------|
| HF weights (34 safetensors, FP8) | `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` | ~962 GB |
| JAX XLA compilation cache | `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/` | ~85 MB (post first run) |

### Quantization format

Weights are stored as FP8 E4M3FN with associated per-block scales
(`weight_scale_inv`). The weight loader in `weight_utils.py` applies scale
conversion at load time. Unlike MaxText PTQ, the FP8 quantization is
**static offline** (built into the HF checkpoint) — no runtime dequantization
to BF16; tensors remain FP8 in HBM.

### Critical correctness requirement

FP8 scale tensors (`wi_0_scale`, `wi_1_scale`, `wo_scale`) must be reshaped from
compact HF layout `(num_experts, k_blocks, out_blocks)` to GMM kernel layout
`(num_experts, k_blocks, 1, out_dim_padded)` before assignment. This reshape is
applied in `_maybe_convert_epmoe_scale_for_kernel()` in `weight_utils.py`. Skipping
it produces incorrect MoE expert outputs.

---

## Module 1 — Server Launch and GKE Job Setup

### Source files

- `python/sgl_jax/launch_server.py`
- `scripts/mimo_v25_pro_demo_job.yaml`

### Main responsibilities

1. Parse `ServerArgs` from CLI flags.
2. Set up JAX distributed init (`jax.distributed.initialize`) across 4 nodes.
3. Build mesh with tp-size=32 TensorCores.
4. Launch `http_server.launch()` on rank 0; worker processes on ranks 1–3.

### GKE job structure

The job uses a Kubernetes Indexed Job (4 completions, `completionMode: Indexed`)
with a DWS ProvisioningRequest for the `jingnw-dws-tpu7-16ch` node pool:

```
rank 0 (coordinator): runs HTTP server + health-check + inference loop
ranks 1-3 (workers):  participate in JAX collectives, no HTTP server
```

### Launch flags (4-node config)

| Flag | Value | Reason |
|------|-------|--------|
| `--tp-size` | 32 | 4 nodes × 4 chips × 2 TCs |
| `--nnodes` | 4 | DWS 2x2x4 slice |
| `--device` | tpu | TPU v7x backend |
| `--dtype` | bfloat16 | Compute dtype (weights stay FP8) |
| `--mem-fraction-static` | 0.75 | 72 GB static / 24 GB XLA temp per TC |
| `--page-size` | 16 | KV cache page size in tokens |
| `--chunked-prefill-size` | 512 | Max tokens per prefill chunk |
| `--max-running-requests` | 2 | Concurrent request limit |
| `--dist-init-addr` | `<rank0-dns>:6006` | JAX distributed coordinator |

### JAX distributed init

`jax.distributed.initialize()` is called with coordinator address from
`--dist-init-addr`. All 32 TensorCores across 4 nodes form a single JAX mesh:

```python
mesh = jax.sharding.Mesh(devices, axis_names=("tensor",))
```

MoE expert parallelism uses a sub-mesh `("expert", "tensor")` built dynamically
inside the weight loader for expert tensor sharding.

---

## Module 2 — Weight Loading

### Source files

- `python/sgl_jax/srt/model_loader/loader.py` (`JAXModelLoader`)
- `python/sgl_jax/srt/utils/weight_utils.py` (`SequentialSafetensorManager`)

### Loading sequence

1. **gcsfuse mount**: GCS bucket mounted as local filesystem at `/mnt/gcs`.
   800 GB LRU RAM cache (`--file-cache-max-size-mb=800000`) serves repeated
   accesses without re-downloading from GCS.
2. **Safetensors scan**: `weight_utils._scan_weight_info()` reads the header of
   each of the 34 safetensors files (byte-range, no data download) to build a
   mapping of `hf_key → (file, byte_offset, shape, dtype)`.
3. **Regular weights** (557 tensors, ~3 min): embeddings, attention projections,
   layer norms loaded sequentially via `safe_open()` + `make_array_from_callback`.
   Each tensor is sharded across the 32-TC mesh via its `NamedSharding`.
4. **MoE weights** (414 groups, ~2–2.5h): expert tensors are loaded in bulk
   per group using `_bulk_read_file()`. Each group covers all 384 experts for
   one weight matrix (wi_0, wi_0_scale, wi_1, wi_1_scale, wo, wo_scale)
   across all MoE layers. FP8 scale tensors are reshaped to GMM kernel layout
   after loading.

### Weight sharding

| Weight type | Sharding axis | Notes |
|-------------|---------------|-------|
| Attention Q/K/V proj | `tensor` (column) | Sharded along head dim |
| Attention O proj | `tensor` (row) | Sharded along hidden dim |
| MoE expert wi_0/wi_1 | `expert` × `tensor` | Expert-parallel + TP |
| MoE expert wo | `expert` × `tensor` | Expert-parallel + TP |
| Embeddings | `tensor` | Vocab sharded |
| Layer norms | replicated | Small; broadcast |

### HBM allocation after weight loading

| Config | Weights per TC | Notes |
|--------|---------------|-------|
| 4-node (tp-32) | ~30 GB | 962 GB ÷ 32 TCs |
| 2-node (tp-16) | ~60 GB | Not viable — no KV cache headroom |

---

## Module 3 — Server Initialization: KV Cache Profiling and XLA Compilation

### Source files

- `python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sgl_jax/srt/model_executor/model_runner.py`

### KV cache profiling

After weights are loaded, `profile_max_num_token()` runs a forward pass at max
batch size to measure HBM headroom:

1. Temporarily occupy static pool with a max-batch dummy forward pass.
2. Measure residual free HBM.
3. Compute max KV tokens from available bytes and token size.

At tp-size=32 with `--mem-fraction-static 0.75`:

| Metric | Value |
|--------|-------|
| Static pool per TC | 72 GB |
| Weights per TC | ~30 GB |
| KV cache per TC | ~43 GB |
| KV cache — attention layers | 286 GB / node (156,288 tokens) |
| KV cache — MLA/linear layers | 60 GB / node (195,360 tokens) |
| XLA temporaries per TC | 24 GB |

The 24 GB XLA temp budget is the minimum required for the 384-expert MoE GEMM
intermediate buffers. Using `--mem-fraction-static > 0.90` reduces XLA temp
below ~8 GB and causes OOM during KV cache profiling.

### XLA compilation

On first request (or after cache miss), JAX traces and compiles the prefill and
decode kernels via XLA. The compiled artifacts are written to GCS:

```
JAX_COMPILATION_CACHE_DIR=gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache
```

| Scenario | Duration |
|----------|----------|
| Cold compile (first run, new tp-size) | 15+ hours |
| Warm cache hit | ~55 seconds |

**The cache key encodes tp-size, model hash, and XLA version.** Changing
`--tp-size` (e.g. switching from 4-node to 2-node) invalidates the cache and
requires a full cold recompile.

---

## Module 4 — HTTP API Layer

### Source files

- `python/sgl_jax/srt/entrypoints/http_server.py`
- `python/sgl_jax/srt/managers/tokenizer_manager.py`

### Main responsibilities

1. Expose OpenAI-compatible REST endpoints (`/v1/chat/completions`, `/health`).
2. Parse incoming JSON into `GenerateReqInput`.
3. Forward to `TokenizerManager` via async queue.
4. Stream or collect response tokens and return `ChatCompletionResponse`.

### Endpoints

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/health` | GET | Returns 200 when server is ready (after XLA warmup) |
| `/v1/chat/completions` | POST | OpenAI chat completions API |
| `/v1/completions` | POST | Legacy completions API |
| `/v1/models` | GET | Returns model metadata |

---

## Module 5 — Tokenization and Chat Template

### Source files

- `python/sgl_jax/srt/managers/tokenizer_manager.py`
- `python/sgl_jax/srt/managers/tiktoken_tokenizer.py`

### Main responsibilities

1. Apply HuggingFace chat template to `messages` list.
2. Encode formatted prompt to token IDs.
3. Enforce `--chunked-prefill-size` constraint.

### Why template matters

MiMo-V2.5-Pro is a reasoning model. Using the chat template injects the correct
system prompt structure and `<think>...</think>` reasoning wrapper, enabling clean
EOS termination and structured output.

### Tokenizer

Loaded from `/mnt/gcs/hf-weights/` at startup (same GCS path as weights).
Downloaded via gcsfuse; config JSONs and tokenizer files are served from RAM cache.

---

## Module 6 — Scheduler and KV Cache Management

### Source files

- `python/sgl_jax/srt/managers/scheduler.py`
- `python/sgl_jax/srt/managers/schedule_batch.py`
- `python/sgl_jax/srt/managers/schedule_policy.py`

### Main responsibilities

1. Maintain the KV cache pool across concurrent requests.
2. Assign KV cache slots to new requests (prefill) and running requests (decode).
3. Build `ScheduleBatch` and dispatch to TP worker via ZMQ.
4. Apply chunked prefill: split long prompts into `--chunked-prefill-size` chunks.

### KV cache structure

MiMo-V2.5-Pro uses two KV cache pools:

| Pool | Covers | Token capacity (4-node) |
|------|--------|------------------------|
| Attention KV cache | Global attention layers | 156,288 tokens / node |
| MLA/linear KV cache | Sliding-window attention layers | 195,360 tokens / node |

Page size is 16 tokens (`--page-size 16`). Slots are allocated per page;
freed on request completion.

---

## Module 7 — TP Worker and Model Runner

### Source files

- `python/sgl_jax/srt/managers/tp_worker.py`
- `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`
- `python/sgl_jax/srt/model_executor/model_runner.py`

### Main responsibilities

1. Receive `ScheduleBatch` from scheduler (ZMQ).
2. Build `ForwardBatch` (token IDs, positions, KV cache indices).
3. Call `ModelRunner.forward(forward_batch)` in prefill or decode mode.
4. Return logits to scheduler for sampling.

### Forward modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| `prefill` | New request | Full prompt sequence in one pass (or chunked) |
| `decode` | Continuation | Single token per step across all running requests |
| `chunked_prefill` | Long prompt | Prompt split into ≤512 token chunks |

### JAX collective operations

All 32 TensorCores participate in every forward pass via:
- `jax.lax.psum` — reduce attention outputs across TP axis
- `jax.lax.all_gather` — gather expert outputs across EP axis
- Implicit via `NamedSharding` XLA SPMD

---

## Module 8 — MiMo-V2.5-Pro Model Architecture

### Source files

- `python/sgl_jax/srt/models/mimo_v2_pro.py` (`MiMoV2ForCausalLM`)
- `python/sgl_jax/srt/models/mimo_v2_flash.py` (base classes)

### Architecture summary

| Parameter | Value |
|-----------|-------|
| Decoder layers | 70 |
| Hidden size | 7168 |
| Attention heads (Q) | 64 |
| KV heads | 8 (GQA, 8:1 ratio) |
| Head dim | 128 |
| FFN type | Dense (layer 0) + Sparse MoE (layers 1–69) |
| Experts (total) | 384 |
| Experts per token | 8 (top-8 routing) |
| MoE intermediate size | 2048 |
| Weight dtype | FP8 E4M3FN (static) |
| Compute dtype | BF16 |
| Context length | 1,048,576 |

### Per-layer forward path

```
input hidden states
    |
    ├── RMSNorm (pre-attention)
    ├── MiMoV2Attention
    │     ├── QKV projection (FP8 linear)
    │     ├── RoPE position encoding
    │     ├── Flash / paged attention
    │     └── O projection (FP8 linear)
    ├── Residual add
    ├── RMSNorm (post-attention)
    ├── FFN:
    │     ├── Layer 0: MiMoV2MLP (dense, BF16)
    │     └── Layers 1-69: MiMoV2Moe
    │           ├── Router (sigmoid scores + correction bias)
    │           ├── Top-8 expert selection (L1-normalized weights)
    │           └── GMM expert GEMM (FP8 wi_0, wi_1, wo)
    └── Residual add
```

### MoE routing

MiMo-V2.5-Pro uses sigmoid-based routing with a `correction_bias` parameter:

1. Router logits computed from hidden states.
2. Sigmoid scores + `correction_bias` used for top-k selection.
3. Expert weights computed from **unbiased** sigmoid scores, L1-normalized.
4. Expert GEMM dispatched via GMM kernel (FP8 × FP8 with block scales).

### FP8 compute

Expert weight matrices (`wi_0`, `wi_1`, `wo`) remain in FP8 E4M3FN in HBM.
Block scales (`wi_0_scale`, `wi_1_scale`, `wo_scale`) are stored in reshaped
GMM layout. The fused FP8 GEMM kernel handles dequantization internally —
no explicit BF16 cast before the matmul.

---

## Module 9 — Prefill and Chunked Prefill

### Source file

- `python/sgl_jax/srt/model_executor/model_runner.py`

### Standard prefill

- Full prompt token sequence forwarded in one pass.
- KV cache written for all prompt tokens.
- Returns first generated token logits.

### Chunked prefill

When prompt length exceeds `--chunked-prefill-size` (512):

1. Split prompt into ≤512-token chunks.
2. Forward chunk 0; write KV cache for chunk 0 tokens.
3. Forward chunk 1 with KV cache from chunk 0 as prefix; extend cache.
4. Repeat until all prompt tokens processed.
5. First decode token sampled from final chunk output.

This avoids HBM OOM from large activation buffers during long prefill.

---

## Module 10 — Autoregressive Decode

### Source file

- `python/sgl_jax/srt/model_executor/model_runner.py`
- `python/sgl_jax/srt/managers/scheduler.py`

### Flow per decode step

1. Scheduler batches all running requests into one `ScheduleBatch`.
2. `ModelRunner.forward(batch, mode=decode)` runs one token forward pass.
3. Each request reads its KV cache slot; writes one new KV entry.
4. Logits sampled → one token per request.
5. EOS check: if EOS token, request marked complete and KV slot freed.
6. `DetokenizerManager` converts token IDs to text chunks, streams to HTTP.

### Observed throughput (validated 2026-05-28, 4-node run)

| Metric | Value |
|--------|-------|
| Decode throughput | ~10.81 tok/s |
| Time to first token | ~5 s (prefill 267 tokens) |
| Max tokens generated | 256 (demo) |

---

## Module 11 — Configuration Reference

### Server flags

```bash
python3 -m sgl_jax.launch_server \
  --model-path /mnt/gcs/hf-weights \   # gcsfuse mount of gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/
  --trust-remote-code \
  --tp-size 32 \                         # 4 nodes × 4 chips × 2 TensorCores
  --device tpu \
  --dtype bfloat16 \                     # compute dtype; weights stay FP8
  --mem-fraction-static 0.75 \           # 72 GB static / 24 GB XLA temp per TC
  --page-size 16 \                       # KV page size in tokens
  --chunked-prefill-size 512 \           # max tokens per prefill chunk
  --max-running-requests 2 \             # concurrent request limit
  --host 0.0.0.0 \
  --port 8080 \
  --nnodes 4 \
  --node-rank <rank> \
  --dist-init-addr <coordinator>:6006
```

### Environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `JAX_COMPILATION_CACHE_DIR` | `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache` | XLA kernel cache |
| `PYTHONUNBUFFERED` | `1` | Real-time log streaming |

---

## Quick Bring-Up Checklist

1. Verify gcsfuse is installed and `jingnw-mimo-v2-5-pro-us-central1` is mounted at `/mnt/gcs`.
2. Confirm 34 safetensors files are present: `ls /mnt/gcs/hf-weights/*.safetensors | wc -l`.
3. Check `JAX_COMPILATION_CACHE_DIR` points to a writable GCS path.
4. Confirm all 4 nodes have joined JAX distributed init before weight loading starts.
5. Watch for `All weights loaded successfully` in logs (~2.5h after pod start).
6. After weight loading, wait for KV cache profiling (~1 min) and XLA warmup (~55s cached / 15h+ cold).
7. Poll `/health` — only returns 200 after XLA warmup completes.
8. Send a test inference request and verify `<think>...</think>` reasoning block appears in output.
9. Check decode throughput: expect ~10 tok/s on 4-node 2x2x4 TPU v7x.

---

## Related Documents

- [gke_tpu7x_smoke_tests.md](gke_tpu7x_smoke_tests.md) — Test 4 runbook for this demo
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — Full HBM/RAM/GCS allocation
- [gke_tpu7x_env_setup.md](gke_tpu7x_env_setup.md) — DWS node pool setup and known pitfalls
- `scripts/mimo_v25_pro_demo_job.yaml` — 4-node GKE job definition
