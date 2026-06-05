# MiMo-V2.5-Pro Weight Checkpoint Conversion

## Status: Complete ✅ — Validated in Production (2026-06-05)

**Result**: End-to-end checkpoint save and restore working on 4-node 2x2x4 (tp-size=32).
Restore takes **~98.6s** (~1m38s) vs ~42 min NFS load. Total startup ~6.6 min vs ~57 min.

**Checkpoint**: `gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/95dc2640/tp32_bfloat16/`

Enabled via env var: `SGLANG_CHECKPOINT_DIR=gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint`

See `mimo_v25_pro_progress.md` §5 for full bug list and fix details.

**Previously failed approaches**:
- Orbax 0.11.28/0.12.0 + JAX 0.8.1/0.9.0: ShapeDtypeStruct ✗
- uint8 + on-device tree_map bitcast: HBM OOM (holds full tree at once) ✗
- CPU numpy per-shard view + device_put: still allocates new device buffer ✗
- Option B hybrid: 100% FP8 model, nothing non-FP8 to restore ✗

Convert HuggingFace FP8 safetensors → sharded Orbax/zarr checkpoint so each
TensorCore loads only its 30 GB TP shard instead of the full 240 GB node shard.
Expected startup reduction: ~40 min → ~5 min (MoE loading phase).

---

## Startup Timing — Checkpoint Restore Path (measured 2026-06-05)

All times from pod start (`T=0`). 4-node 2x2x4, tp-size=32, warm XLA cache.

| Phase | Wall time | Duration | Notes |
|-------|-----------|----------|-------|
| Container setup (git clone, pip install, apt, NFS mount) | T+0s → T+0s | ~0s | Warm nodes; install cached from prior run |
| JAX distributed init (4-node rendezvous on port 6006) | T+48s → T+57s | ~9s | All 4 nodes synchronize |
| Quantization structure prep (`apply_moe_quantization`) | T+57s → T+59s | ~2s | Model graph wired up in host RAM |
| **Checkpoint restore** (Orbax OCDBT from GCS) | T+59s → T+158s | **98.3s** | 5.28 GiB/s · 481.9 GiB per host |
| KV cache profiling (binary search, HBM probe) | T+158s → T+162s | ~4s | Tries 156k → 195k tokens |
| KV cache allocation (fused slabs) | T+162s → T+162s | <1s | 286.6 GB + 59.7 GB |
| XLA precompile — EXTEND (4 token-pad shapes) | T+162s → T+212s | **50s** | bs=2; tokens ∈ {64,128,256,512}; warm cache |
| XLA precompile — DECODE (2 batch sizes) | T+212s → T+231s | **19s** | bs ∈ {1,2}; warm cache |
| **Server healthy** (`/health` returns 200) | **T+235s** | | **~3m55s total** |
| Inference (276 prompt + 512 decode tokens) | T+235s → T+283s | ~48s | ~10.7 tok/s decode |

**Total startup: 235s (~3m55s)** — vs 57 min NFS slow-path, vs ~2h25m gcsfuse.

---

## Startup Timing — NFS Slow Path (reference, measured 2026-06-04)

| Phase | Duration | Notes |
|-------|----------|-------|
| Container setup + NFS mount | ~5 min | git clone, pip install, apt, mount |
| JAX distributed init | ~9s | |
| Quantization structure prep | ~2s | |
| **MoE weight loading (414 groups)** | **~40 min** | NFS RAM-backed; I/O + CPU conversion bottleneck |
| Regular weights (attn/MLP, 557 tensors) | ~3 min | Fast, small tensors |
| KV cache profiling + allocation | ~1 min | |
| XLA precompile (warm cache) | ~69s | |
| **Server healthy** | **~57 min total** | |

---

## Background — Slow-Path Loading Pipeline

### What happens during MoE weight loading

For each of the 414 MoE groups, across all 70 layers × 6 weight types:

```
1. Read raw bytes from safetensors file via NFS/gcsfuse
   → 384 expert tensors, FP8, packed in one large contiguous span

2. CPU numpy: _bulk_read_file()
   → gsutil/NFS read → np.frombuffer → reshape to per-expert arrays
   → I/O burst: ~1-3s at ~1.8 Gbps per node

3. CPU numpy: _maybe_convert_epmoe_scale_for_kernel()
   → scale tensors: (num_experts, k_blocks, out_blocks)
                  → (num_experts, k_blocks, 1, out_dim_padded)   [GMM kernel layout]

4. CPU numpy: fused QKV construction (regular weights only)
   → Q, K, V projections concatenated into single tensor

5. JAX: make_array_from_callback()
   → TP-shard the stacked expert tensor across mesh
   → Each TC receives its 1/32 slice
   → Device transfer (HBM): fast, ~ms

6. JAX: model_param.value = stacked_weight.astype(dtype)
   → Final assignment to model param

   CPU idle (~3-5s between reads)
```

### Why loading is slow

The fundamental problem: **each node reads 240 GB from the weight source**
(all 34 safetensors files), then extracts only its **30 GB TP shard**
(1/8 of what its 8 TCs need = 240/8 = 30 GB per TC). The other 210 GB
read per node is discarded.

```
Network I/O per node:  240 GB   (read)
Useful data per node:   30 GB   (kept in HBM)
Wasted I/O:            210 GB   (87.5% of reads discarded)
```

With 3 NFS servers at ~12.5 GB/s each, the 240 GB/node read takes:
`240 GB / (37.5 GB/s ÷ 4 nodes) ≈ 25 min just for I/O`
(plus ~15 min CPU conversion = ~40 min total)

### NFS network utilization

From measured data:
- Per-pod inbound: ~1.8 Gbps (10s average)
- 4 pods aggregate: ~7.2 Gbps
- 3 NFS server NICs (3 × 100 Gbps): utilization **~19%**

The NIC is not the bottleneck — the **sequential read pattern** of the weight
loader is: read one group → CPU convert → next group (NIC idle during CPU work).

---

## Analysis — Three Approaches

### Approach A: Pre-convert safetensors (CPU conversions only)

Pre-apply steps 3 and 4 (scale reshape, QKV fusion) to the safetensors files
and save back to GCS.

| | |
|---|---|
| **Saves** | CPU conversion time (~5–10 min of the ~40 min total) |
| **Does not save** | I/O time (still reads 240 GB per node) |
| **Compatible with** | Any tp-size |
| **Verdict** | Marginal gain (~10–20% speedup) |

### Approach B: Sharded Orbax/zarr checkpoint ✅ RECOMMENDED

After the first full load completes, save the model's JAX state via Orbax.
Each TC writes only its local 30 GB shard. On subsequent runs, each TC reads
only its own shard directly — no conversion, no wasted I/O.

```
First run:  read 240 GB → convert → load → save 30 GB checkpoint per TC
Later runs: read 30 GB checkpoint per TC → done (8× less I/O per node)
```

| | |
|---|---|
| **Saves** | I/O time + all CPU conversion (90%+ of loading time) |
| **Expected load time** | ~30 GB / (NFS bandwidth per TC) ≈ 2–5 min |
| **Checkpoint size** | ~962 GB total (same as source), split into 32 shards |
| **Cache key** | `tp_size + model_path + dtype` — changing tp-size invalidates |
| **Requires** | All 4 nodes present for both save AND load |
| **Verdict** | **8× I/O reduction, eliminates all CPU conversion on reload** |

### Approach C: Pre-sharded safetensors (per-TC files)

Write 32 safetensors files (one per TC), each containing only that TC's weight
shard. No JAX dependency; load is just `safe_open` → `device_put`.

| | |
|---|---|
| **Saves** | I/O time + CPU conversion |
| **Compatible with** | Only one tp-size (like Approach B) |
| **Complexity** | Requires custom shard-extraction script |
| **Verdict** | Similar benefit to B but more complex; B is better |

---

## Plan — Approach B: Sharded Orbax Checkpoint

### GCS checkpoint path

```
gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/tp{tp_size}/
```

The `tp_size` in the path ensures the correct checkpoint is used and different
tp-size runs don't conflict. Future extension: add model hash or dtype.

### Save flow (first run only)

```
model.load_weights(model_config)          # existing path, ~40 min
    ↓
save_checkpoint(model, checkpoint_path)   # NEW: ~5 min, all 4 nodes participate
    ↓
exit (or continue to serve)
```

### Load flow (subsequent runs)

```
checkpoint exists at gs://.../tp32/ ?
    YES → load_checkpoint(model, checkpoint_path)   # NEW: ~5 min
    NO  → model.load_weights(model_config)          # existing path, ~40 min
          save_checkpoint(model, checkpoint_path)   # NEW: save for next time
```

### Checkpoint format

Use `flax.nnx` state + `orbax.checkpoint`:

```python
import orbax.checkpoint as ocp
from flax import nnx

# Save
state = nnx.state(model)
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save(gcs_path, state)

# Load
abstract_state = nnx.eval_shape(lambda: model_class(...))
state = checkpointer.restore(gcs_path, item=abstract_state)
nnx.update(model, state)
```

Orbax handles GCS paths natively via `gs://` URIs. Each process (one per node)
saves/loads only its local device's shards — no all-gather needed.

---

## Implementation Notes

### Files to modify

| File | Change |
|------|--------|
| `python/sgl_jax/srt/model_loader/loader.py` | Add `_save_checkpoint()`, `_load_checkpoint()`, checkpoint detection in `_get_model()` |
| `python/sgl_jax/srt/server_args.py` | Add `--checkpoint-path` flag (default: auto-derived from model path) |

### Key implementation details

**1. Checkpoint path derivation (if not explicitly set):**
```python
# Auto-derive from model_path and tp_size
import hashlib
model_hash = hashlib.md5(model_config.model_path.encode()).hexdigest()[:8]
checkpoint_path = (
    f"gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint/"
    f"tp{mesh.size}/{model_hash}/"
)
```

**2. Multi-host save coordination:**
Orbax with GCS handles multi-host saves correctly when each process writes
its own shards. Use `ocp.CheckpointManager` with `multiprocessing_options`
for safe concurrent writes.

**3. Checkpoint validity check:**
Before loading, verify:
- Checkpoint directory exists in GCS
- `tp_size` matches current launch config
- Metadata file contains matching model hash

**4. Fallback:**
If checkpoint load fails (corrupt, wrong dtype, etc.), fall back to
`load_weights()` and re-save.

**5. Save timing:**
Save happens AFTER `load_weights()` completes, before the server starts
accepting requests. The save itself runs in parallel across all nodes
(each saves its shards concurrently) — expected ~5 min.

### Pseudocode for `_get_model` modification

```python
def _get_model(self, model_class, model_config):
    checkpoint_path = self._get_checkpoint_path(model_config)

    with jax.set_mesh(self.mesh):
        model = nnx.eval_shape(
            lambda: model_class(config, dtype=model_config.dtype, mesh=self.mesh)
        )

    # Apply quantization structure if needed
    model = self._apply_quantization(model, model_config)

    if self._checkpoint_exists(checkpoint_path):
        logger.info("Loading from checkpoint: %s", checkpoint_path)
        self._load_checkpoint(model, checkpoint_path)
    else:
        logger.info("No checkpoint found, loading from weights source")
        model.load_weights(model_config)
        logger.info("Saving checkpoint to: %s", checkpoint_path)
        self._save_checkpoint(model, checkpoint_path)

    return model
```

### Dependencies

`orbax-checkpoint` is already in the dependency tree via `flax`. Verify:
```bash
python3 -c "import orbax.checkpoint; print(orbax.checkpoint.__version__)"
```

---

## Measured Results (2026-06-05)

| Metric | NFS slow-path | Checkpoint restore |
|--------|--------------|-------------------|
| First-run load time | ~42 min | ~44.6 min (42 + 2.6 save) |
| Subsequent restore time | ~42 min | **98.3s (~1m38s)** |
| Total startup (restore + warmup) | ~57 min | **~3m55s** |
| GCS I/O per host | — | 481.9 GiB at 5.28 GiB/s |
| GCS raw block I/O per host | — | 273.9 GiB at 3.64 GiB/s |
| CPU conversion during restore | — | **None** |
| tp-size portability | Any | Fixed to saved tp-size |

First-run cost: +2.6 min (save). Subsequent savings: ~53 min per run.
Break-even: after 2 runs.

---

## Memory Allocation — Checkpoint Restore Path

### HBM (per TensorCore, 96 GB total)

| Use | Per TC | 32-TC total | Notes |
|-----|--------|-------------|-------|
| Model weights | ~30 GB | ~962 GB | FP8 MoE + BF16 attention/MLP |
| KV cache — 60 SWA layers | ~8.96 GB | 286.64 GB | 156,528 tokens, bfloat16 |
| KV cache — 10 full layers | ~1.87 GB | 59.72 GB | 195,664 tokens, bfloat16 |
| XLA temporaries | 24 GB | 768 GB | 25% reserved; needed for EPMoE GEMM |
| **Total used** | **~65 GB** | **~2,076 GB** | of 3,072 GB total |

### Host RAM (per TPU worker node, 900 Gi pod limit)

| Consumer | Per node | 4-node total | Notes |
|----------|----------|-------------|-------|
| Orbax I/O buffer (GCS read window) | up to 89.4 GB | up to 358 GB | Released after restore; `restore_concurrent_bytes=96 GB` |
| Python / JAX runtime + model graph | ~8–15 GB | ~32–60 GB | NNX module objects, sglang-jax, tokenizer |
| XLA compilation cache | ~2–5 GB | ~8–20 GB | Warm HLO modules |
| FP8 staging (uint8 shard buffers) | ~3–6 GB peak | ~12–24 GB peak | Transient; ~3 MB per shard, serial |
| OS + container runtime | ~3 GB | ~12 GB | |
| **Peak (during restore)** | **~105–115 GB** | **~420–460 GB** | |
| **Steady-state (serving)** | **~15–25 GB** | **~60–100 GB** | Buffer released; model is in HBM |

> `enable_pinned_host_transfer=False` — Orbax does **not** use pinned host memory.
> The 89.4 GiB is a streaming concurrency cap, not a reserved allocation.

### NFS weight servers (always-on, not read during fast-restore)

| VM | RAM | tmpfs | Files |
|----|-----|-------|-------|
| `jingnw-nfs-weights-1` | 384 GB | ~322 GB | 12 safetensors |
| `jingnw-nfs-weights-2` | 384 GB | ~350 GB | 12 safetensors |
| `jingnw-nfs-weights-3` | 384 GB | ~292 GB | 10 safetensors |
| **Total** | **1,152 GB** | **~964 GB** | 34 files |

NFS servers hold the HuggingFace safetensors in RAM for the slow-path (first run
only). They are **not read** during checkpoint restore — only the tokenizer and
`config.json` are fetched (KBs via NFS).

---

## Data Dependencies — Checkpoint Restore Path

```
pod start
    │
    ├─── GitHub (git clone -b tpu7)
    │       └─ python/sgl_jax/...  ← sglang-jax source code
    │
    ├─── NFS (read-only mount, tokenizer + config only)
    │       ├─ 10.128.0.92:/mnt/weights    (jingnw-nfs-weights-1)
    │       ├─ 10.128.15.231:/mnt/weights  (jingnw-nfs-weights-2)
    │       └─ 10.128.0.45:/mnt/weights    (jingnw-nfs-weights-3)
    │       → /mnt/weights/ symlink union
    │           ├─ tokenizer.json, tokenizer_config.json  ← read
    │           ├─ config.json, generation_config.json    ← read
    │           └─ model-*.safetensors (34 files, ~964 GB) ← NOT read
    │
    ├─── GCS (checkpoint restore, ~5.28 GiB/s per host, 98s)
    │       └─ gs://.../sglang-checkpoint/95dc2640/tp32_bfloat16/
    │               ├─ ocdbt.process_{0,1,2,3}/  ← each node reads its own shards
    │               ├─ d/                         ← OCDBT shared data blocks
    │               ├─ _METADATA                  ← tensor tree structure
    │               ├─ _CHECKPOINT_METADATA       ← Orbax bookkeeping
    │               └─ tp32_bfloat16_abstract_state.pkl  ← shape/sharding metadata
    │
    ├─── GCS (XLA compilation cache, warm hit, negligible I/O)
    │       └─ gs://.../jax-compilation-cache/   ← precompiled kernels (tp-size=32)
    │
    └─── Inter-node network (ICI + GKE overlay)
            ├─ JAX distributed rendezvous (port 6006, coordinator node)
            ├─ Orbax multi-host checkpoint sync (all-barrier before/after restore)
            └─ Inference: all-reduce per transformer layer (~2× per layer, 70 layers)
```

**Critical path**: GitHub clone → JAX init → checkpoint restore (GCS) → KV cache →
XLA precompile (cache hit) → serve. The only external bottleneck is GCS bandwidth.

---

## Related Documents

- [mimo_v25_pro_progress.md](mimo_v25_pro_progress.md) — overall status and bug log
- [mimo_v25_pro_inference_pipeline.md](mimo_v25_pro_inference_pipeline.md) — weight loading pipeline detail
- [mimo_v25_pro_perf_benchmark.md](mimo_v25_pro_perf_benchmark.md) — throughput benchmarks
- [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) — full HBM/RAM allocation tables
- [fp8_restore_workaround/README.md](fp8_restore_workaround/README.md) — FP8 device_put bug and fix
