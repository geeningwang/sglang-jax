# H-1: Model Dimensions and Current EP Architecture

**Date**: 2026-06-15
**Status**: Complete
**Exit criteria**: Model dimensions confirmed; actual MoE backend and ep_size wiring identified.

---

## Model Dimensions (from config.json in GCS)

| Parameter | Value | Source |
|-----------|-------|--------|
| hidden_size | 4096 | config.json |
| moe_intermediate_size | **2048** | config.json |
| n_routed_experts | 256 | config.json |
| num_experts_per_tok | 8 (top-k) | config.json |
| num_hidden_layers | 48 | config.json |
| MoE layers | 47 (layers 1–47) | moe_layer_freq[0]=0, rest=1 |
| Dense layer | 1 (layer 0) | moe_layer_freq[0]=0 |
| weight dtype | float8_e4m3fn | config.json quantization_config |
| quant_block_size | [128, 128] (k=128, n=128) | config.json weight_block_size |
| scoring_func | sigmoid | config.json |
| topk_method | noaux_tc | config.json |
| moe_backend | **NOT PRESENT** in config.json | — |
| ep_size | **NOT PRESENT** in config.json | — |

---

## Actual MoE Backend: EPMoE (NOT FusedEPMoE)

The plan originally assumed FusedEPMoE was running. **This is wrong.**

### How the backend is determined

**Step 1** — `server_args.moe_backend = "epmoe"` (default from `server_args.py:138`).

**Step 2** — `ModelConfig.__init__` sets `self.moe_backend = MoEBackend.EPMOE`
(since "epmoe" != AUTO, the auto-select at `model_config.py:80` is skipped).

**Step 3** — `ModelRunner.load_model()` writes this back to the HF config:
```python
# model_runner.py:336
self.model_config.hf_config.moe_backend = self.model_config.moe_backend.value
# → hf_config.moe_backend = "epmoe"
```

**Step 4** — `mimo_v2_flash.py` reads the HF config:
```python
# mimo_v2_flash.py:100–101
self.moe_backend = getattr(config, "moe_backend", "epmoe")  # = "epmoe"
self.use_fused = self.moe_backend == "fused"                 # = False
```

**Result**: `EPMoE` is instantiated (line 125), not `FusedEPMoE`.

---

## Actual ep_size: 1 (TP-style sharding)

`server_args.ep_size = 1` (default from `server_args.py:80`).

`ModelRunner.load_model()` writes:
```python
# model_runner.py:332
self.model_config.hf_config.ep_size = self.ep_size  # = server_args.ep_size = 1
```

`EPMoE` is created with `ep_size=1` (`mimo_v2_flash.py:131`).

### What ep_size=1 means inside EPMoE

```python
# layers/moe.py:81–90
world_size = mesh.shape["data"] * mesh.shape["tensor"]  # = 1 × 8 = 8
self.tp_size = world_size // self.ep_size               # = 8 // 1 = 8
self.experts_per_device = self.num_experts // self.ep_size  # = 256 // 1 = 256

# Mesh reshaped to (expert=1, tensor=8):
devices.reshape(1, 8)
# Expert weights sharded: P("expert", None, "tensor")
# → "expert" axis size=1 → NOT sharded across EP groups
# → "tensor" axis size=8 → intermediate dim split 8-way across devices
```

**Every device holds all 256 experts, TP-sharded** (each device has `inter/8 = 256` columns per expert).

### Communication pattern with ep_size=1

- **No expert routing all-to-all** — `if self.ep_size > 1:` at `moe.py:563` is skipped
- **Allreduce after each expert layer** — `if self.tp_size > 1:` at `moe.py:557` runs `psum` or `psum_scatter`
- Allreduce volume: `num_tokens × hidden × BF16` per layer = small (128 tokens × 4096 × 2B ≈ 1MB)

---

## Weight Bandwidth Per Device (Current vs EP-Scaled)

For decode at conc=16 (16 tokens × top-k=8 → 128 expert activations):

| Config | Experts/device | Weight per activation | a2a volume | allreduce volume |
|--------|---------------|----------------------|------------|-----------------|
| EPMoE, ep=1 (current) | 256 (inter/8 each) | 4096×256 FP8 = 1MB | none | 16tok×4096×2B×47 ≈ 246MB/step |
| EPMoE, ep=8 (1-node) | 32 (full inter) | 4096×2048 FP8 = 8MB | ~16tok×4096×2B×47 ≈ 246MB/step | none |
| EPMoE, ep=16 (2-node) | 16 (full inter) | 4096×2048 FP8 = 8MB | DCN a2a per layer | none |

**Key insight**: On a single 8-device node, EPMoE(ep=1) and EPMoE(ep=8) have the **same total weight bandwidth per device** (256 experts × 256 cols = 32 experts × 2048 cols). Increasing ep_size on the same node does NOT reduce per-device weight reads — it only changes the communication pattern (allreduce → a2a).

For 2-node EP (ep=16, 16 total devices): per-device expert count drops from 32 to 16, genuinely halving local weight reads. BUT now DCN all-to-all is required, with its latency penalty.

---

## Implications for the Opt H Plan

### H-2 (tuned block configs) — **INVALID**

`tuned_block_configs.py` is used only by `FusedEPMoE`'s Pallas kernel. Since the current backend is `EPMoE`, this file is never consulted. H-2 as written is a no-op.

### Two distinct paths for "increasing EP"

**Path A — Switch backend to FusedEPMoE (1-node)**
```
--json-model-override-args '{"moe_backend": "fused"}'
```
- FusedEPMoE kernel derives ep_size from mesh: `dp × tp = 1 × 8 = 8`
- 32 local experts per device, ICI a2a INSIDE the Pallas kernel
- Needs tuned block configs for MiMo-V2-Flash shape (hidden=4096, inter=2048, ep=8)
- CURRENTLY: falls back to DEFAULT_FUSED_MOE_BLOCK_CONFIG → potentially suboptimal

**Path B — Keep EPMoE, increase --ep-size (same node or multi-node)**
```
--ep-size 8    # same 1-node, changes allreduce → a2a
--ep-size 16   # 2-node, halves local expert count + adds DCN a2a
```
- ep_size wiring to hf_config already exists (model_runner.py:332)
- EPMoE has ep_size > 1 code path with `jax.lax.ragged_all_to_all`
- For 1-node: same weight bandwidth, different communication — uncertain benefit
- For 2-node: genuinely fewer local experts, but DCN overhead is the risk

### Revised sub-problem plan

| Sub-problem | Goal | Exit criteria |
|-------------|------|---------------|
| **H-2a** | Benchmark FusedEPMoE (Path A) on 1 node | tok/s vs EPMoE baseline (534 tok/s) |
| **H-2b** | Add tuned block configs for FusedEPMoE (if H-2a shows FusedEPMoE is viable) | Config key: `(bf16, fp8, ntok, 256, 8, 4096, 2048, 8, False, False)` |
| **H-3** | Profile TPOT breakdown (current EPMoE and/or FusedEPMoE) | Measure allreduce vs a2a latency, weight bandwidth fraction |
| **H-4** | 2-node EP (ep=16) benchmark | tok/s vs 534 tok/s; DCN overhead per call |

**Immediate next step**: H-2a — test `--json-model-override-args '{"moe_backend": "fused"}'` on the existing 1-node bench job. This is low-risk (1 node, same hardware) and tests whether the Pallas FusedEPMoE kernel outperforms EPMoE.

---

## Summary

| Question | Answer |
|----------|--------|
| Backend | **EPMoE** (default "epmoe"; FusedEPMoE only if `moe_backend="fused"` passed) |
| Current ep_size | **1** (from server_args.ep_size default) |
| Expert sharding | **TP-style** — 256 experts per device, intermediate dim split ×8 |
| Expert a2a | **None** — psum allreduce instead |
| Tuned block configs | **Irrelevant** for current EPMoE backend |
| "Increasing EP" paths | A: switch to FusedEPMoE (1-node, ep=8 via mesh) or B: --ep-size flag (EPMoE, same or more nodes) |
| 1-node EP benefit | Uncertain — same weight bandwidth, trades allreduce for a2a |
| 2-node EP benefit | Potentially large (fewer local experts) but DCN overhead is unknown |
