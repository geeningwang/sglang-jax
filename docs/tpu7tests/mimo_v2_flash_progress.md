# MiMo-V2-Flash on TPU v7x — Progress

**Last updated**: 2026-06-09 (optimization work in progress)
**Branch**: `tpu7` (`geeningwang/sglang-jax`)
**Cluster**: `jingnw-tpu7-cluster`, zone `us-central1-c`

---

## Goal

Run MiMo-V2-Flash inference on GKE TPU v7x using the same infrastructure as
MiMo-V2.5-Pro (Orbax checkpoint, DWS nodes, GCS). Target: 2-node demo first
(tp-size=16), then 1-node if feasible for cost.

**Status: COMPLETE** — 2-node demo (tp=16) and 1-node demo (tp=8) both done
on 2026-06-08. All runs include working Orbax checkpoint save + restore.

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
| Layers | 70 | **48** |
| Routed experts/layer | 384 | **256** |
| Hidden size | 6144 | **4096** |
| HF weights size | ~962 GB FP8 | **~313 GB FP8** |
| 2-node feasible? | No (OOM at tp-16) | **Yes** ✅ |

---

## Code Status

**Model class**: `MiMoV2FlashForCausalLM` — **implemented and verified** in
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
No code changes were needed for model inference logic.

**Checkpoint infrastructure**: Same Orbax/OCDBT path as V2.5-Pro. The loader
auto-derives the checkpoint path from `SGLANG_CHECKPOINT_DIR` + model hash.

### Code adjustments verified

| Item | Status | Note |
|------|--------|------|
| `MiMoV2FlashForCausalLM` model class | ✅ Verified | `mimo_v2_flash.py` |
| Registry entry | ✅ Verified | `EntryClass = [MiMoV2FlashForCausalLM]` |
| Orbax checkpoint save/restore | ✅ Verified | Works end-to-end |
| FP8 auto-detection | ✅ Verified | `_resolve_quantization_config()` reads HF config |
| NFS weight loading path | ✅ Verified | `_warmup_safetensors_cache` reads all 145 files |
| SWA KV cache head dim | ✅ Verified | Single fused KV pool works; head_dim=256 per-device |
| Dense MLP layer-0 dequant | ✅ Verified | `load_weights` step 3 in Flash |
| `noaux_tc` correction bias | ✅ Verified | `_create_layer_mappings` |

---

## HBM Resources

### tp=16 (2-node, actual measured)

| Metric | Value |
|--------|-------|
| HBM per TC | 101.73 GB |
| KV cache allocated | **156.40 GB per TC** |
| XLA static pool (25%) | 25.43 GB |
| Model weights + scales | ~19.9 GB/TC (FP8 experts + BF16 attn) |
| `max_total_num_tokens` | 1,139,376 |

KV cache is ~13× larger than V2.5-Pro at tp-32 (11.6 GB/TC) — huge room for long contexts.

### tp=8 (1-node, confirmed fits)

At tp=8, model weight per TC doubles (~40 GB/TC) but the HBM headroom is still
ample. Flash has only 4 KV heads; at tp=8 each TC gets fewer than 1 head, so
the loader replicates them 2× (4→8 heads per TC, logged as
`KV head replication: layer N self_attn.k_proj 4->8 heads`). This is the same
pattern as tp=16 on V2.5-Pro and is handled correctly by `load_weights`.

---

## GCS Structure

```
gs://jingnw-mimo-v2-5-pro-us-central1/
  mimo-v2-flash-hf-weights/          ← 145 HF safetensors + metadata
  sglang-checkpoint/
    9d3df1bf/
      tp16_bfloat16/                 ← Flash Orbax checkpoint tp=16 (151.5 GiB) ✅
    e0e89a7d/
      tp8_bfloat16/                  ← Flash Orbax checkpoint tp=8 (242.94 GiB) ✅
  jax-compilation-cache/             ← shared XLA cache (Flash tp=8 + tp=16 entries)
```

**Checkpoint path naming** (`loader.py:252`):
`{SGLANG_CHECKPOINT_DIR}/{md5(model_path)[:8]}/tp{tp_size}_{dtype}/`

The `bfloat16` suffix reflects `--dtype bfloat16` (the compute/activation dtype),
not the expert weight format. The checkpoint stores a **mixed format**: expert
weights as `float8_e4m3fn` (FP8), attention/norm weights as `bfloat16`.

### Checkpoint paths

| Run type | `--model-path` | hash | GCS path | Size |
|----------|----------------|------|----------|------|
| NFS tp=16 ✅ | `/mnt/flash-weights` | `9d3df1bf` | `sglang-checkpoint/9d3df1bf/tp16_bfloat16/` | 151.5 GiB |
| gcsfuse tp=8 ✅ | `/mnt/gcs/mimo-v2-flash-hf-weights` | `e0e89a7d` | `sglang-checkpoint/e0e89a7d/tp8_bfloat16/` | 242.94 GiB |

`/mnt/flash-weights` (not `/mnt/weights`) avoids hash collision with the Pro
checkpoint at `95dc2640` (Pro also uses `/mnt/weights` as its model-path).
On the NFS VMs the safetensors live at `/mnt/weights/`; the job symlinks them
into `/mnt/flash-weights/` inside the pod.

The tp=8 checkpoint is larger than tp=16 (242 vs 152 GiB) because each TC
holds a larger shard of the weight tensors when tp is smaller.

---

## Results: 2-Node Demo (2026-06-08)

### Run 1 — first run (NFS weight load + Orbax save)

| Phase | Duration | Notes |
|-------|----------|-------|
| NFS mount + git clone + install | ~60s | NFS VMs pre-loaded |
| JAX distributed init | ~5s | Both nodes synced |
| FP8 quantization auto-detected | ~1s | From `config.json` |
| Sequential NFS warmup (291.6 GB) | ~120s | Page-cache prefetch |
| Regular weights (557 tensors) | ~11s | Linear, embed, norm |
| MoE weights (282 tensors) | ~455s | 256 experts × MoE layers, FP8 |
| FP8 → BF16 attention dequant | ~1s | All 48 layers |
| Orbax checkpoint save | ~78s | 151.5 GiB at 6.28 GiB/s |
| EXTEND precompile (6 shapes) | 167s | ~28s/shape |
| DECODE precompile (3 shapes) | ~81s | ~27s/shape |
| **Total to server healthy** | **~952s (~16 min)** | |
| Demo inference (512 tokens) | ~4s | |

### Run 2 — checkpoint restore

| Phase | Duration | Notes |
|-------|----------|-------|
| NFS mount + git clone + install | ~60s | |
| JAX distributed init | ~5s | |
| Orbax checkpoint restore | **55s** | 4.26 GiB/s, 219 GiB read |
| EXTEND precompile (6 shapes) | 157s | ~10s faster (XLA cache partial hits) |
| DECODE precompile (3 shapes) | 54s | |
| **Total to server healthy** | **~331s (~5.5 min)** | **2.9× faster** than run 1 |
| Demo inference (512 tokens) | ~4s | |

### Demo inference

**Request**:
```json
{
  "model": "MiMo-V2-Flash",
  "messages": [{"role": "user", "content": "Explain mixture-of-experts and why sliding window attention helps efficiency."}],
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response** (finish_reason: `length` — hit 512-token cap mid-sentence on SWA section):

Structured explanation covering:
- MoE: router selects 1–2 experts per token, only those activate; hospital-of-specialists analogy
- Efficiency vs cost trade-off (inference speed vs memory footprint)
- SWA intro started but truncated at cap

**Usage**: prompt=40 tokens, completion=512 tokens.

---

## Results: 1-Node Demo (2026-06-08)

Pool: `jingnw-dws-tpu7-4ch` (DWS, tpu7x-standard-4t, 2x2x1, cloud-platform scope)
Weight source: gcsfuse (`/mnt/gcs/mimo-v2-flash-hf-weights` → hash `e0e89a7d`)

### Run 1 — cold load + Orbax save

| Phase | Duration | Notes |
|-------|----------|-------|
| git clone + install | ~60s | |
| gcsfuse mount + sglang launch | ~5s | cache to /dev/shm (400 GiB) |
| Regular weights | ~25s | linear, embed, norm layers |
| MoE weights (282 groups) | ~570s (~9.5 min) | gcsfuse, ~2s/group |
| KV head replication (4→8) | ~3s | all 9 full-attn layers |
| Orbax checkpoint save | ~3 min | 242.94 GiB to `e0e89a7d/tp8_bfloat16/` |
| EXTEND precompile (6 shapes) | ~160s | ~27s/shape |
| DECODE precompile (3 shapes) | ~80s | ~27s/shape |
| **Total to server healthy** | **~24 min** | |
| Demo inference (512 tokens) | ~4s | |

### Run 2 — checkpoint restore

| Phase | Duration | Notes |
|-------|----------|-------|
| git clone + install | ~60s | |
| gcsfuse mount + sglang launch | ~5s | |
| Orbax checkpoint restore | **~60s** | reads 242.94 GiB from GCS |
| EXTEND precompile (6 shapes) | **153s** | XLA cache partial hits |
| DECODE precompile (3 shapes) | **~65s** | XLA cache partial hits |
| **Total to server healthy** | **~7 min** | **3.4× faster** than run 1 |
| Demo inference (512 tokens) | ~4s | |

### Demo inference (both runs)

**Request**:
```json
{
  "model": "MiMo-V2-Flash",
  "messages": [{"role": "user", "content": "Explain mixture-of-experts and why sliding window attention helps efficiency."}],
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response** (finish_reason: `length` — hit 512-token cap):

Structured explanation covering MoE router/expert mechanics, sparse activation
efficiency (Mixtral 8x7B analogy: 47B params, 12B active), then SWA intro
truncated at token cap.

**Usage**: prompt=40 tokens, completion=512 tokens.

---

## Infrastructure State

| Resource | Status | Notes |
|----------|--------|-------|
| Flash HF weights on GCS | ✅ | 145 safetensors + all metadata in `mimo-v2-flash-hf-weights/` |
| Flash HF weights on NFS VMs | ✅ | VM-1=49, VM-2=48, VM-3=48 files + Flash `config.json` + metadata |
| Flash Orbax checkpoint (tp=16) | ✅ | `9d3df1bf/tp16_bfloat16/` (151.5 GiB, `commit_success.txt` present) |
| Flash Orbax checkpoint (tp=8) | ✅ | `e0e89a7d/tp8_bfloat16/` (242.94 GiB, `commit_success.txt` present) |
| DWS 8ch node pool (`jingnw-dws-tpu7-8ch`) | ✅ | 2x2x2 topology, cloud-platform scope, used for 2-node demo |
| DWS 4ch node pool (`jingnw-dws-tpu7-4ch`) | ✅ | 2x2x1 topology, cloud-platform scope, used for 1-node demo |
| Flex 4ch node pool (`jingnw-flex-tpu7`) | ❌ DELETED | Had devstorage.read_only scope; caused Orbax 403 — see Issues |
| GCE workload policy (`jingnw-tpu7-policy-4ch`) | ✅ | 2x2x1, HIGH_THROUGHPUT; required for single-host DWS pool |
| NFS weight servers | ✅ | Flash safetensors + Flash config on all 3 VMs |
| XLA compilation cache | ✅ | Flash tp=8 and tp=16 entries populated |

### NFS VM file split

| VM | IP | Files | Pattern |
|----|-----|-------|---------|
| jingnw-nfs-weights-1 | 10.128.0.92 | 49 safetensors | `model_0`, `*_linear_fc1` (even layers) |
| jingnw-nfs-weights-2 | 10.128.15.231 | 48 safetensors | `model_1`, `*_linear_fc2` (even layers) |
| jingnw-nfs-weights-3 | 10.128.0.45 | 48 safetensors | `model_10`..`model_47` (regular attn files) |

All 3 VMs also hold metadata: `config.json`, `tokenizer.json`, `model.safetensors.index.json`, etc.

In each pod the job mounts all 3 NFS shares and symlinks their contents into
`/mnt/flash-weights/`, which becomes the `--model-path`.

---

## Issues Encountered and Fixed

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Orbax restore crash (layer 64 not found) | Hash collision: Pro and Flash both used `--model-path /mnt/weights` → same hash `95dc2640`; server tried to restore a Pro checkpoint into Flash | Changed Flash job to use `/mnt/flash-weights` as the symlink dir → hash `9d3df1bf` |
| Wrong architecture class loaded (`mimo_v2_pro.py`) | NFS VMs had the Pro `config.json` (70 layers, 384 experts); Flash metadata was never synced when clearing Pro weights | Copied all Flash metadata files from GCS to all 3 NFS VMs |
| 1-node pod evicted (ephemeral storage OOM) | gcsfuse `--cache-dir=/mnt/gcsfuse-cache` wrote 72 GB to container overlay filesystem; node had <10 GB ephemeral → disk-pressure taint + eviction | Redirect cache to RAM: `mount -o remount,size=400g /dev/shm` then `--cache-dir=/dev/shm/gcsfuse-cache` |
| Orbax checkpoint save 403 on 1-node | `jingnw-flex-tpu7` pool had OAuth scope `devstorage.read_only`; GCS writes return HTTP 403 | Created new DWS pool `jingnw-dws-tpu7-4ch` with `cloud-platform` scope (see below) |
| Cannot recreate flex 4ch pool with cloud-platform scope | GCE rejects single-host `tpu7x-standard-4t` MIG with placement policy: "use workload policy instead" | Created GCE workload policy `jingnw-tpu7-policy-4ch` (2x2x1, HIGH_THROUGHPUT) via REST API |
| Workload-policy pool misclassified as multi-host | GKE 1.35 overrides the resource label; autoscaler requires 2-pod gang for "multi-host" pools | Abandoned flex approach; used DWS (queued-provisioning) pool instead via REST API v1beta1 |
| SA key creation blocked | `constraints/iam.disableServiceAccountKeyCreation` org policy active on project | No fix; abandoned SA key approach. DWS pool uses node's compute SA with cloud-platform scope |

---

## Job YAMLs

- [`scripts/mimo_v2_flash_2node_nfs_demo_job.yaml`](../../scripts/mimo_v2_flash_2node_nfs_demo_job.yaml) — 2-node primary (NFS, fast retries, pool `jingnw-dws-tpu7-8ch`)
- [`scripts/mimo_v2_flash_2node_demo_job.yaml`](../../scripts/mimo_v2_flash_2node_demo_job.yaml) — 2-node fallback (gcsfuse)
- [`scripts/mimo_v2_flash_1node_demo_job.yaml`](../../scripts/mimo_v2_flash_1node_demo_job.yaml) — **1-node** (gcsfuse, pool `jingnw-dws-tpu7-4ch`, DWS ProvisioningRequest)

### DWS pool creation notes (jingnw-dws-tpu7-4ch)

Single-host `tpu7x-standard-4t` (2x2x1) pools cannot be created via `gcloud`
with a placement policy — GCE returns "use workload policy instead". The
workaround is to create via REST API v1beta1:

1. Create GCE workload policy first:
   ```bash
   gcloud compute resource-policies create group-placement jingnw-tpu7-policy-4ch \
     --region=us-central1 --tpu-topology=2x2x1 --workload-type=HIGH_THROUGHPUT
   ```
2. Create node pool via REST API with `queuedProvisioning.enabled=true`,
   `placementPolicy.policyName=jingnw-tpu7-policy-4ch`,
   `oauthScopes=[cloud-platform]`, `maxNodeCount=1`, **no `maxRunDuration`**
   (DWS pools reject `maxRunDuration` at pool level — set it on the
   ProvisioningRequest instead via `maxRunDurationSeconds`).

---

## Optimization Work (2026-06-09)

Full plan: [`mimo_v2_flash_optimization/plan.md`](mimo_v2_flash_optimization/plan.md)

### Baseline performance (2026-06-08, tp=8, conc=8)

| Metric | Value |
|--------|-------|
| Peak decode throughput | **371 tok/s** @ conc=8 |
| TPOT @ conc=8 | **21.6ms** |
| TTFT @ 512 tok | 56ms |
| TPOT linear fit | `2.21ms × batch + 3.9ms` |

Throughput plateaus at conc=8 — decode is HBM bandwidth-bound (weight reads dominate).

### Findings so far

| Opt | What we learned | Gain |
|-----|----------------|------|
| A1 (expert quant) | MoE expert weights already FP8 in HBM — no dequantization at load time | 0% (already done) |
| C (host sync) | Overlap design is fully pipelined; F=3.9ms is fixed TPU compute, not host bubble | 0% measured |
| A2 (attention FP8) | Initial estimate of 4-5% was wrong by ~20×. Corrected: <0.3% gain. Flash uses GQA (1 global SWA KV head), TP=8 shards both dims, o_proj already FP8. Closed — not worth checkpoint rebuild. | **<0.3% — closed** |

### Next step: Opt E — Speculative decoding

All analyzed throughput optimizations (A1, A2, B, C) are closed. The remaining lever
for user-visible latency is speculative decoding (draft model generates K tokens per step;
full model verifies in one pass). Potential: 2-3× per-sequence latency reduction.

Candidate draft: shared-expert-only forward (skip routed MoE, use dense MLP layers) —
low complexity, reduced acceptance rate. See `opt_e_speculative/` for investigation plan.

---

## References

- Model class: [`python/sgl_jax/srt/models/mimo_v2_flash.py`](../../python/sgl_jax/srt/models/mimo_v2_flash.py)
- Job YAMLs: [`scripts/mimo_v2_flash_2node_nfs_demo_job.yaml`](../../scripts/mimo_v2_flash_2node_nfs_demo_job.yaml)
- V2.5-Pro checkpoint guide: [`mimo_v25_pro_weight_checkpoint.md`](mimo_v25_pro_weight_checkpoint.md)
- V2.5-Pro progress (HBM data, infra): [`mimo_v25_pro_progress.md`](mimo_v25_pro_progress.md)
- GKE env setup: [`gke_tpu7x_env_setup.md`](gke_tpu7x_env_setup.md)
- HBM resource allocation: [`gke_tpu7x_resource_allocation.md`](gke_tpu7x_resource_allocation.md)
