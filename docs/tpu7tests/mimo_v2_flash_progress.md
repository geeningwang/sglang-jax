# MiMo-V2-Flash on TPU v7x — Progress

**Last updated**: 2026-06-08 (plan drafted)  
**Branch**: `tpu7` (`geeningwang/sglang-jax`)  
**Cluster**: `jingnw-tpu7-cluster`, zone `us-central1-c`

---

## Goal

Run MiMo-V2-Flash inference on GKE TPU v7x using the same infrastructure as
MiMo-V2.5-Pro (Orbax checkpoint, DWS nodes, GCS). Target: 2-node demo first
(tp-size=16), then 1-node if feasible for cost.

---

## What is MiMo-V2-Flash?

MiMo-V2-Flash is a **hybrid MoE model** (Xiaomi) that alternates between:
- **Sliding Window Attention (SWA)** layers — local context, lower KV cost
- **Full attention** layers — long-range context
- **MoE layers** — sparse expert routing (subset of layers)
- **Dense MLP layers** — standard dense FFN (remaining layers)

It uses FP8 static quantization for MoE experts (like V2.5-Pro) but has
**separate q/k/v projections** (not fused qkv_proj like V2.5-Pro).

HF repo: **`XiaomiMiMo/MiMo-V2-Flash`**

**Key differences from V2.5-Pro**:

| Property | V2.5-Pro | V2-Flash |
|----------|----------|----------|
| HF arch class | `MiMoV2ForCausalLM` | `MiMoV2FlashForCausalLM` |
| Attention | Full only | Hybrid (SWA + Full) |
| QKV weights | Fused `qkv_proj` | Separate `q/k/v_proj` |
| MLP | All MoE | Mix MoE + Dense |
| Model size | ~962 GB FP8 | TBD (much smaller) |
| 2-node feasible? | No (OOM at tp-16) | Likely yes |

---

## Code Status

**Model class**: `MiMoV2FlashForCausalLM` — **already implemented** in
[`python/sgl_jax/srt/models/mimo_v2_flash.py`](../../python/sgl_jax/srt/models/mimo_v2_flash.py).

Supports:
- Hybrid SWA/full attention via `hybrid_layer_pattern`
- Mixed MoE/dense layers via `moe_layer_freq`
- FP8 MoE expert quantization (same path as V2.5-Pro)
- Per-head fused KV dequantization
- `noaux_tc` topk with correction bias
- Attention sink bias for SWA layers
- `attention_value_scale` scaling

**Registry**: Auto-registered via `EntryClass = [MiMoV2FlashForCausalLM]`.
No code changes needed for model inference logic.

**Checkpoint infrastructure**: Same Orbax/OCDBT path as V2.5-Pro. The loader
auto-derives the checkpoint path from `SGLANG_CHECKPOINT_DIR` + model hash.

---

## HBM Feasibility Estimate

MiMo-V2-Flash is much smaller than V2.5-Pro. Conservative upper-bound analysis:

| Metric | V2.5-Pro (tp-32) | Flash (tp-16, estimated) |
|--------|------------------|--------------------------|
| HBM per TC | 101.73 GB | 101.73 GB |
| Model weights | ~47 GB/TC | **~3–10 GB/TC** |
| `apply_moe_quantization` scales | ~11 GB/TC | **~1–3 GB/TC** |
| XLA temp (25%) | 25.43 GB | 25.43 GB |
| KV cache available | 11.62 GB/TC | **~60–70 GB/TC** |

Flash should fit comfortably in 2 nodes (tp-16, 16 TCs × 101.73 GB).
Even 1 node (tp-8, 8 TCs) may be sufficient.

**Note**: Exact model size TBD — verify after fetching HF config.json.

---

## GCS Structure Plan

Reuse the existing bucket with Flash-specific subfolders:

```
gs://jingnw-mimo-v2-5-pro-us-central1/
  mimo-v2-flash-hf-weights/          ← HF safetensors (download once)
  sglang-checkpoint/
    {flash_model_hash}/
      tp16_bfloat16/                 ← Orbax checkpoint (2-node run)
      tp8_bfloat16/                  ← Orbax checkpoint (1-node run, if tested)
  jax-compilation-cache/             ← shared XLA cache (reuse)
```

The Flash model hash is derived from the GCS path used as `--model-path`:
`md5("gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights")[:8]`

---

## Step-by-Step Plan

### §1. Fetch HF config to confirm model spec ⬜

```bash
# Run from this VM or a GCS-connected VM
huggingface-cli download XiaomiMiMo/MiMo-V2-Flash config.json \
  --local-dir /tmp/mimo-v2-flash-config
cat /tmp/mimo-v2-flash-config/config.json | python3 -c "
import json, sys; c = json.load(sys.stdin)
print('arch:', c.get('architectures'))
print('layers:', c.get('num_hidden_layers'))
print('experts:', c.get('n_routed_experts', c.get('num_experts')))
print('hidden:', c.get('hidden_size'))
print('moe_freq:', c.get('moe_layer_freq', 'n/a')[:5], '...')
print('hybrid:', c.get('hybrid_layer_pattern', 'n/a')[:5], '...')
"
```

Confirm: model size, FP8 vs BF16 weights, expected safetensors file list.

### §2. Download HF weights to GCS ⬜

Option A — from this operator VM (most direct):
```bash
pip install huggingface-hub
huggingface-cli download XiaomiMiMo/MiMo-V2-Flash \
  --local-dir /tmp/mimo-v2-flash \
  --include "*.safetensors" "*.json" "*.txt"
gsutil -m cp -r /tmp/mimo-v2-flash/* \
  gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights/
```

Option B — from a GKE pod (if VM disk space is limited):
Use a single-pod download job that writes directly to GCS via `huggingface_hub`.

Flash weights are expected to be **much smaller than V2.5-Pro** (~20–100 GB
vs ~962 GB) so download completes in minutes.

### §3. Create job YAMLs ✅

Two YAMLs created:
- [`scripts/mimo_v2_flash_2node_nfs_demo_job.yaml`](../../scripts/mimo_v2_flash_2node_nfs_demo_job.yaml) — **primary**: NFS weight loading (fast retries)
- [`scripts/mimo_v2_flash_2node_demo_job.yaml`](../../scripts/mimo_v2_flash_2node_demo_job.yaml) — fallback: gcsfuse loading

### §3 (original). Create job YAML: `scripts/mimo_v2_flash_2node_demo_job.yaml` ✅

Key differences from `mimo_v25_pro_nfs_demo_job.yaml`:
- `--tp-size 16 --nnodes 2` (2-node, same as attempted for Pro)
- `--model-path gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights`
  (gcsfuse; Flash is small enough that gcsfuse is reasonable)
- `--mem-fraction-static 0.75` (same XLA temp headroom)
- No NFS mounts needed (gcsfuse sufficient for smaller model)
- `SGLANG_CHECKPOINT_DIR` set to same GCS path (auto-save checkpoint on first run)
- DWS node pool: `jingnw-dws-tpu7-8ch` (2x2x2 topology), 2 nodes

### §4. Run 2-node demo and create checkpoint ⬜

First run (~load from HF weights → save checkpoint):
- Expected: Flash loads much faster than V2.5-Pro (smaller model)
- Auto-saves checkpoint to `gs://.../sglang-checkpoint/{hash}/tp16_bfloat16/`
- Run demo inference and verify output quality

Second run (~restore from checkpoint):
- Expected: ~90s restore (same infrastructure as V2.5-Pro)

### §5. (Optional) 1-node run for cost efficiency ⬜

If Flash fits in 8 TCs, 1-node (tp-8) reduces DWS cost by 2×.
Same job YAML with `--tp-size 8 --nnodes 1` on `jingnw-dws-tpu7-4ch` pool.

---

## Code Adjustments (if needed)

The model class is complete. Potential gaps to verify during §3–4:

| Item | Status | Note |
|------|--------|------|
| `MiMoV2FlashForCausalLM` model class | ✅ Implemented | `mimo_v2_flash.py` |
| Registry entry | ✅ Auto | `EntryClass = [MiMoV2FlashForCausalLM]` |
| Orbax checkpoint save/restore | ✅ Inherited | Same loader as Pro |
| FP8 monkey-patch | ✅ Inherited | `loader.py` `_load_checkpoint` |
| gcsfuse weight loading path | ✅ Inherited | `_warmup_safetensors_cache` |
| SWA KV cache head dim | ⚠️ Verify | `swa_head_dim` vs `head_dim` for KV pool sizing |
| Dense MLP layer-0 dequant | ✅ Implemented | `load_weights` step 3 in Flash |
| `noaux_tc` correction bias | ✅ Implemented | `_create_layer_mappings` |

The KV cache pool sizing uses `head_dim` globally — need to verify that
SWA layers (which may have a different `swa_head_dim`) pad correctly so
a single fused KV pool works across both layer types.

---

## Infrastructure State (as of 2026-06-08)

| Resource | Status | Notes |
|----------|--------|-------|
| Flash HF weights on GCS | 🔄 IN PROGRESS | Downloading 313 GB; ~30/145 files done |
| Flash HF weights on NFS VMs | 🔄 IN PROGRESS | Copying from GCS to NFS VMs in parallel |
| Flash Orbax checkpoint | ⬜ NOT YET | Created on first run |
| DWS 8ch node pool (`jingnw-dws-tpu7-8ch`) | ✅ Available | Used for 2-node ep2 test |
| DWS 4ch node pool (`jingnw-dws-tpu7-4ch`) | ❓ Check | For 1-node if needed |
| NFS weight servers | ✅ READY (cleared + loading Flash) | Pro weights deleted; Flash loading |
| XLA compilation cache | ✅ | Shared; Flash will populate its own entries |

### NFS VM file split

| VM | IP | Files | Approx size | Pattern |
|----|-----|-------|-------------|---------|
| jingnw-nfs-weights-1 | 10.128.0.92 | 49 | ~104 GB | model_0, \*_linear_fc1 (even layers) |
| jingnw-nfs-weights-2 | 10.128.15.231 | 48 | ~104 GB | model_1, \*_linear_fc2 (even layers) |
| jingnw-nfs-weights-3 | 10.128.0.45 | 48 | ~104 GB | model_10..47 (regular attn files) |

Merged via symlinks into `/mnt/weights` on each TPU pod (same as V2.5-Pro approach).

### Checkpoint paths

| Run type | model-path | checkpoint hash | GCS checkpoint path |
|----------|-----------|-----------------|---------------------|
| NFS (primary) | `/mnt/weights` | `95dc2640` | `sglang-checkpoint/95dc2640/tp16_bfloat16/` |
| gcsfuse (fallback) | `/mnt/gcs/mimo-v2-flash-hf-weights` | `e0e89a7d` | `sglang-checkpoint/e0e89a7d/tp16_bfloat16/` |

---

## Open Questions

1. **Flash model size**: Need to fetch `config.json` to confirm layers, experts,
   hidden_size, and total weight size. Determines tp-size choice.
2. **Weight format**: Does Flash use FP8 (like V2.5-Pro) or BF16 weights?
   FP8 → `weight_q` + `weight_scale_inv` format; BF16 → `weight` only.
3. **gcsfuse vs NFS**: If Flash weights are >100 GB MoE, NFS RAM servers
   might be faster for first run. But Flash is expected small enough for gcsfuse.
4. **DWS 4ch pool existence**: Need to verify `jingnw-dws-tpu7-4ch` exists
   for potential 1-node test.

---

## References

- Model class: [`python/sgl_jax/srt/models/mimo_v2_flash.py`](../../python/sgl_jax/srt/models/mimo_v2_flash.py)
- V2.5-Pro checkpoint guide: [`mimo_v25_pro_weight_checkpoint.md`](mimo_v25_pro_weight_checkpoint.md)
- V2.5-Pro progress (HBM data, infra): [`mimo_v25_pro_progress.md`](mimo_v25_pro_progress.md)
- GKE env setup: [`gke_tpu7x_env_setup.md`](gke_tpu7x_env_setup.md)
- HBM resource allocation: [`gke_tpu7x_resource_allocation.md`](gke_tpu7x_resource_allocation.md)
