# GKE TPU v7x Smoke Tests (jingnw-tpu7-cluster)

Reproduces the TPU v7x readiness tests run on `jingnw-tpu7-cluster` (us-central1-c).
Covers two node pools:

| Pool | Topology | Hosts | JAX devices |
|------|----------|-------|-------------|
| `jingnw-flex-tpu7` | 2x2x1 | 1 | 8 (4 chips × 2 cores) |
| `jingnw-flex-tpu7-8ch` | 2x2x2 | 2 | 16 (8 chips × 2 cores) |

---

## Prerequisites

```bash
# Authenticate
gcloud container clusters get-credentials jingnw-tpu7-cluster \
  --zone=us-central1-c --project=tpu-launchpad-playground
```

---

## Test 1 — Single-host JAX smoke test (`jingnw-flex-tpu7`)

**What it does:** Provisions one flex-start TPU v7x node (2x2x1, 4 chips, 8 JAX devices)
and prints the JAX device list. No distributed init needed.

**Script:** `scripts/flex_tpu7_smoke.yaml`

```bash
kubectl apply -f scripts/flex_tpu7_smoke.yaml
kubectl wait --for=condition=complete job/flex-tpu7-smoke --timeout=600s
kubectl logs -l job-name=flex-tpu7-smoke
kubectl delete -f scripts/flex_tpu7_smoke.yaml
```

**Expected output:**
```
JAX version: 0.8.1
Device count: 8
Devices: [TpuDevice(id=0, ..., core_on_chip=0), TpuDevice(id=1, ..., core_on_chip=1), ...]
```

The autoscaler handles node provisioning automatically (~3–5 min for the VM to appear).

---

## Test 2 — Qwen3-8B inference demo (`jingnw-flex-tpu7`)

**What it does:** Downloads Qwen3-8B from HuggingFace, starts the sglang-jax server
with `--tp-size 8`, sends a chat completion request, and exits.

**Script:** `scripts/qwen3_8b_demo_job.yaml`

```bash
kubectl apply -f scripts/qwen3_8b_demo_job.yaml
kubectl logs -f -l job-name=qwen3-8b-demo
kubectl delete -f scripts/qwen3_8b_demo_job.yaml
```

**Expected output (tail):**
```
=== Demo Inference ===
Input prompt: 'Explain what a transformer model is in one sentence.'
Output:
<think>...</think>
[tokens: prompt=19, completion=128]
=== Demo complete ===
```

Typical decode throughput: ~100 tok/s.

---

## Test 3 — Multi-host JAX smoke test (`jingnw-flex-tpu7-8ch`)

**What it does:** Provisions two flex-start TPU v7x nodes (2x2x2 gang, 8 chips total,
16 JAX devices), runs `jax.distributed.initialize()` across both hosts, and prints
the full global device list.

**Script:** `scripts/flex_tpu7_8ch_smoke.yaml`

### Step 1 — Provision nodes manually

> **Important:** The GKE cluster autoscaler for `jingnw-flex-tpu7-8ch` has a known bug
> where it sends `resizeBy=4` instead of the required `resizeBy=2` (gang size), causing
> every automatic scale-up to fail. Until this is fixed in GKE, nodes must be provisioned
> via a direct resize request.

```bash
# Find the MIG name for jingnw-flex-tpu7-8ch
gcloud compute instance-groups managed list \
  --filter="name~jingnw-flex-tpu7" --zones=us-central1-c \
  --format="table(name,size,targetSize)"

# Create a resize request for exactly 1 gang (2 nodes)
gcloud compute instance-groups managed resize-requests create \
  gke-jingnw-tpu7-clus-jingnw-flex-tpu7-707194bc-grp \
  --resize-by=2 \
  --resize-request=smoke-resize-1 \
  --zone=us-central1-c \
  --requested-run-duration=3600s

# Wait for SUCCEEDED (~1–2 min)
gcloud compute instance-groups managed resize-requests describe \
  gke-jingnw-tpu7-clus-jingnw-flex-tpu7-707194bc-grp \
  --resize-request=smoke-resize-1 \
  --zone=us-central1-c \
  --format="value(state)"

# Confirm nodes registered with GKE
kubectl get nodes -l cloud.google.com/gke-nodepool=jingnw-flex-tpu7-8ch
```

### Step 2 — Run the smoke test

```bash
kubectl apply -f scripts/flex_tpu7_8ch_smoke.yaml
kubectl wait --for=condition=complete job/flex-tpu7-8ch-smoke --timeout=120s
kubectl logs -l job-name=flex-tpu7-8ch-smoke --prefix
kubectl delete -f scripts/flex_tpu7_8ch_smoke.yaml
```

**Expected output (both pods):**
```
[rank0] JAX version: 0.8.1
[rank0] Global device count: 16
[rank0] Local device count: 8
[rank0] Devices: [TpuDevice(id=0, ..., coords=(0,0,0), core_on_chip=0), ...,
                  TpuDevice(id=15, ..., coords=(1,1,1), core_on_chip=1)]
[rank1] JAX version: 0.8.1
[rank1] Global device count: 16
[rank1] Local device count: 8
```

All 16 devices are visible from both ranks. Coords `(x,y,0)` are on node 0;
`(x,y,1)` are on node 1.

---

## Test 4 — MiMo-V2.5-Pro inference demo (`jingnw-dws-tpu7-16ch`)

**What it does:** Mounts FP8 weights from GCS via gcsfuse, starts the sglang-jax server
with `--tp-size 32 --nnodes 4` across a 2x2x4 DWS TPU slice, sends a chat completion
request from rank 0, and exits. Node provisioning is fully automatic via DWS
ProvisioningRequest — no manual resize needed.

**Script:** `scripts/mimo_v25_pro_demo_job.yaml`

**Key parameters:**
- Node pool: `jingnw-dws-tpu7-16ch` (2x2x4, 16 chips, 32 TensorCores)
- `--tp-size 32`, `--nnodes 4`, `--mem-fraction-static 0.75`
- Health-check timeout: 36 hours (covers ~2h MoE weight load + ~55s cached XLA compilation)
- XLA compilation cache: `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/`
  (kernels cached incrementally — each restart is faster)
- See [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) for full HBM/RAM/GCS breakdown

**Why 4 nodes instead of 2:** The model weights fill ~93% of HBM at tp-size=16 (2 nodes),
leaving insufficient room for KV cache. Doubling to tp-size=32 halves the per-TensorCore
weight footprint, making `--mem-fraction-static 0.92` safe.

### Run the demo

```bash
kubectl apply -f scripts/mimo_v25_pro_demo_job.yaml

# Watch all 4 pods
kubectl logs -f -l job-name=mimo-v25-pro-demo --prefix

# Monitor via Cloud Logging
gcloud logging read \
  'resource.type="k8s_container" AND resource.labels.cluster_name="jingnw-tpu7-cluster" AND labels."k8s-pod/job-name"="mimo-v25-pro-demo"' \
  --project=tpu-launchpad-playground --format="value(timestamp,textPayload)" --limit=30

kubectl delete -f scripts/mimo_v25_pro_demo_job.yaml
```

**Loading sequence and expected timing:**

| Phase | Duration | Notes |
|-------|----------|-------|
| gcsfuse mount + sglang-jax install | ~5 min | All 4 ranks in parallel |
| Regular weights → TPU HBM | ~3 min | 557 tensors |
| MoE weights → TPU HBM | ~1.5–2 h | 414 groups × 4 nodes |
| KV cache profiling | ~1 min | |
| XLA warmup compilation | ~55 s (cached) / 15 h+ (first run) | Cached to GCS; subsequent runs ~55s |
| `/health` passes | — | Only after XLA compilation completes |
| Inference curl | ~30 s | |

**Expected output (rank 0 tail):**
```
[rank0] PHASE: server healthy after Xs
[rank0] PHASE: sending demo inference request
Output:
Mixture-of-experts (MoE) is a neural network architecture ...
[tokens: prompt=24, completion=256]
=== Demo complete ===
```

---

## Test 5 — MiMo-V2.5-Pro inference demo, 2-node (`jingnw-dws-tpu7-8ch`)

**What it does:** Same as Test 4 but on a 2x2x2 DWS slice (2 nodes, 16 TensorCores,
tp-size=16). Weights double per TensorCore (~60 GB vs ~30 GB), so KV cache is
minimal (~12 GB/TC) — sufficient for a single-request smoke test only.

**Script:** `scripts/mimo_v25_pro_2node_demo_job.yaml`

**Key parameters:**
- Node pool: `jingnw-dws-tpu7-8ch` (2x2x2, 8 chips, 16 TensorCores)
- `--tp-size 16`, `--nnodes 2`, `--mem-fraction-static 0.75`
- `--max-running-requests 1` (minimal KV cache budget)
- XLA compilation cache: `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/`
  (**separate cache key** from the 4-node run — expect a cold-cache first run of 15+ h)
- See [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) for full HBM/RAM breakdown

**HBM budget per TensorCore (96 GB):**

| Pool | Size | Notes |
|------|------|-------|
| Model weights | ~60 GB | 962 GB ÷ 16 TCs |
| KV cache | ~12 GB | Minimal; single-request only |
| XLA temporaries | ~24 GB | 25%; required for 384-expert MoE GEMM |

### Run the demo

```bash
kubectl apply -f scripts/mimo_v25_pro_2node_demo_job.yaml

# Watch both pods
kubectl logs -f -l job-name=mimo-v25-pro-2node-demo --prefix

kubectl delete -f scripts/mimo_v25_pro_2node_demo_job.yaml
```

**Loading sequence and expected timing:**

| Phase | Duration | Notes |
|-------|----------|-------|
| gcsfuse mount + sglang-jax install | ~5 min | Both ranks in parallel |
| Regular weights → TPU HBM | ~3 min | 557 tensors |
| MoE weights → TPU HBM | ~2–2.5 h | 414 groups × 2 nodes via gcsfuse |
| KV cache profiling | ~1 min | Minimal cache; fast profile |
| XLA warmup compilation | ~55 s (cached) / 15 h+ (first run) | **Different cache key than 4-node** |
| `/health` passes | — | Only after XLA compilation completes |
| Inference curl | ~30 s | |

**Expected output (rank 0 tail):**
```
[rank0] PHASE: server healthy after Xs
[rank0] PHASE: sending demo inference request
Output:
Mixture-of-experts (MoE) is a neural network architecture ...
[tokens: prompt=24, completion=256]
=== Demo complete ===
```

---

## Test 6 — MiMo-V2.5-Pro inference demo via NFS RAM (`jingnw-dws-tpu7-16ch`)

**What it does:** Same as Test 4 but reads weights from 3 × n2-highmem-48 RAM-backed
NFS servers instead of GCS/gcsfuse. Weight loading is **2.2–3× faster** (~42 min vs
~2h25m). Uses `jax0.9.0-rev1` container.

**Prerequisites**: NFS VMs (`jingnw-nfs-weights-1/2/3`) must be running with weights
in tmpfs. See [mimo_v25_pro_weight_checkpoint.md](mimo_v25_pro_weight_checkpoint.md)
for the NFS setup procedure.

**Script:** `scripts/mimo_v25_pro_nfs_demo_job.yaml`

**Key parameters:**
- Node pool: `jingnw-dws-tpu7-16ch`, Container: `jax0.9.0-rev1`
- NFS servers: `10.128.0.92`, `10.128.15.231`, `10.128.0.45` (all `/mnt/weights`)
- `--tp-size 32`, `--nnodes 4`, `--mem-fraction-static 0.75`
- `SGLANG_CHECKPOINT_DIR=gs://jingnw-mimo-v2-5-pro-us-central1/sglang-checkpoint`

**Loading timing:**

| Phase | Duration | Notes |
|-------|----------|-------|
| Install + NFS mount | ~5 min | |
| Regular weights | ~3 min | |
| MoE weights (NFS RAM) | **~42 min** | ~5–7 s/group vs gcsfuse ~14–17 |
| KV cache profiling + XLA warmup | ~2 min | Warm cache |
| **Total** | **~52 min** | vs gcsfuse ~2h30m |

### Run the demo

```bash
kubectl apply -f scripts/mimo_v25_pro_nfs_demo_job.yaml
kubectl logs -f -l job-name=mimo-v25-pro-nfs-demo --prefix
kubectl delete -f scripts/mimo_v25_pro_nfs_demo_job.yaml
```

**Expected output:**
```
[rank0] PHASE: server healthy after Xs  [total elapsed: Xs]
[tokens: prompt=276, completion=512]
=== Demo complete ===
```

---

## Cluster reference

| Pool | Machine | Topology | Provisioning | Max nodes |
|------|---------|----------|--------------|-----------|
| `default-pool` | e2-standard-4 | — | Standard | 2 (always-on) |
| `jingnw-flex-tpu7` | tpu7x-standard-4t | 2x2x1 | Flex Start (auto) | 2 |
| `jingnw-flex-tpu7-8ch` | tpu7x-standard-4t | 2x2x2 | Flex Start (**manual resize**) | 4 |
| `jingnw-cpu-highmem` | n2-highmem-96 | — | Standard | — |
| `jingnw-dws-tpu7-8ch` | tpu7x-standard-4t | 2x2x2 | DWS (ProvisioningRequest) | 2 |
| `jingnw-dws-tpu7-16ch` | tpu7x-standard-4t | 2x2x4 | DWS (ProvisioningRequest) | 4 |

## TPU v7x architecture

- Each chip has **2 TensorCores** (cores), each core is an independent JAX device with 96 GB HBM
- `google.com/tpu` Kubernetes resource counts **chips** (not cores)
- Single-host node (2x2x1): 4 chips → 8 JAX devices (`--tp-size 8`)
- Multi-host slice (2x2x2): 8 chips across 2 nodes → 16 JAX devices (`--tp-size 16`)
