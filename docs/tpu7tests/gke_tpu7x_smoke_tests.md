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

## Test 4 — MiMo-V2.5-Pro inference demo (`jingnw-flex-tpu7-8ch`)

**What it does:** Copies FP8 weights from GCS, starts the sglang-jax server with
`--tp-size 16 --nnodes 2`, sends a chat completion request from rank 0, and exits.

**Script:** `scripts/mimo_v25_pro_demo_job.yaml`

### Step 1 — Provision nodes (same as Test 3 Step 1)

Nodes must be pre-provisioned via resize request (see above). Reuse existing nodes
if they are still Ready.

```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=jingnw-flex-tpu7-8ch
```

### Step 2 — Run the demo

```bash
kubectl apply -f scripts/mimo_v25_pro_demo_job.yaml
kubectl logs -f -l job-name=mimo-v25-pro-demo --prefix
kubectl delete -f scripts/mimo_v25_pro_demo_job.yaml
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

## TPU v7x architecture

- Each chip has **2 TensorCores** (cores), each core is an independent JAX device with 96 GB HBM
- `google.com/tpu` Kubernetes resource counts **chips** (not cores)
- Single-host node (2x2x1): 4 chips → 8 JAX devices (`--tp-size 8`)
- Multi-host slice (2x2x2): 8 chips across 2 nodes → 16 JAX devices (`--tp-size 16`)
