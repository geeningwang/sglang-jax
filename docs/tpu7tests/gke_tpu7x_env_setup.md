# GKE TPU v7x Environment Setup (jingnw-tpu7-cluster)

How to set up the GKE cluster, node pools, and supporting infrastructure for TPU v7x
readiness tests. Covers discovered pitfalls and the workarounds required.

## Current test environment (as of 2026-06)

All active tests use **DWS (Dynamic Workload Scheduler)** with `jingnw-dws-tpu7-16ch`
(4-node 2x2x4) or `jingnw-dws-tpu7-8ch` (2-node 2x2x2).

| Test | Script | Node pool | Container | Provisioning |
|------|--------|-----------|-----------|--------------|
| MiMo-V2.5-Pro smoke (gcsfuse) | `scripts/mimo_v25_pro_demo_job.yaml` | `jingnw-dws-tpu7-16ch` | `jax0.8.1-rev1` | DWS |
| MiMo-V2.5-Pro smoke (NFS RAM) | `scripts/mimo_v25_pro_nfs_demo_job.yaml` | `jingnw-dws-tpu7-16ch` | `jax0.9.0-rev1` | DWS |
| MiMo-V2.5-Pro benchmark | `scripts/mimo_v25_pro_bench_job.yaml` | `jingnw-dws-tpu7-16ch` | `jax0.9.0-rev1` | DWS |
| MiMo-V2.5-Pro smoke (2-node) | `scripts/mimo_v25_pro_2node_demo_job.yaml` | `jingnw-dws-tpu7-8ch` | `jax0.8.1-rev1` | DWS |

**Existing DWS node pools:**

| Pool name | Topology | Max nodes | Policy |
|-----------|----------|-----------|--------|
| `jingnw-dws-tpu7-8ch` | 2x2x2 | 2 | `jingnw-tpu7-policy-8ch` |
| `jingnw-dws-tpu7-16ch` | 2x2x4 | 4 | `jingnw-tpu7-workload-policy-16ch-v2` |

No setup is needed to run tests — just `kubectl apply -f scripts/YAML`. DWS handles
node provisioning automatically via the embedded `ProvisioningRequest`.

---

---

## Prerequisites

```bash
# Authenticate
gcloud auth login
gcloud config set project tpu-launchpad-playground
gcloud container clusters get-credentials jingnw-tpu7-cluster \
  --zone=us-central1-c --project=tpu-launchpad-playground
```

---

## Creating a DWS TPU node pool

DWS (Dynamic Workload Scheduler) provisions TPU capacity on-demand via a
`ProvisioningRequest` resource. Three steps are required: create a workload resource
policy, create the node pool via REST API, and verify.

### Step 1 — Create a workload resource policy

TPU v7x node pools require a `workloadPolicy` resource policy (type `HIGH_THROUGHPUT`).
The gcloud CLI only creates `groupPlacementPolicy` type, which is rejected by tpu7x.
Use the Compute Engine REST API directly:

```bash
ACCESS_TOKEN=$(gcloud auth print-access-token)

curl -X POST \
  "https://compute.googleapis.com/compute/v1/projects/tpu-launchpad-playground/regions/us-central1/resourcePolicies" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "POLICY_NAME",
    "workloadPolicy": {
      "type": "HIGH_THROUGHPUT",
      "acceleratorTopology": "TOPOLOGY"
    }
  }'
```

Replace `POLICY_NAME` and `TOPOLOGY` (e.g. `2x2x2`, `2x2x4`). Verify:

```bash
gcloud compute resource-policies describe POLICY_NAME \
  --region=us-central1 --project=tpu-launchpad-playground --format=yaml
```

Expected output includes `workloadPolicy.type: HIGH_THROUGHPUT`.

**Existing policies:**

| Policy name | Topology |
|---|---|
| `jingnw-tpu7-policy-8ch` | 2x2x2 |
| `jingnw-tpu7-workload-policy-16ch-v2` | 2x2x4 |

### Step 2 — Create the node pool via GKE REST API

The gcloud CLI flags `--enable-queued-provisioning` and `--flex-start` both fail for
TPU v7x multi-host pools in different ways:
- `--enable-queued-provisioning` → `"Queued_provisioning doesn't support TPUs"`
- `--flex-start` → creates the pool but omits `queuedProvisioning.enabled: true`,
  so DWS cannot use it

The workaround is to POST directly to the GKE node pools API:

```bash
ACCESS_TOKEN=$(gcloud auth print-access-token)
PROJECT="tpu-launchpad-playground"
CLUSTER="jingnw-tpu7-cluster"
ZONE="us-central1-c"

curl -X POST \
  "https://container.googleapis.com/v1/projects/${PROJECT}/zones/${ZONE}/clusters/${CLUSTER}/nodePools" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "nodePool": {
      "name": "POOL_NAME",
      "config": {
        "machineType": "tpu7x-standard-4t",
        "diskSizeGb": 100,
        "diskType": "hyperdisk-balanced",
        "imageType": "COS_CONTAINERD",
        "flexStart": true,
        "oauthScopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "serviceAccount": "default",
        "reservationAffinity": {"consumeReservationType": "NO_RESERVATION"},
        "resourceLabels": {
          "goog-gke-accelerator-type": "tpu7x",
          "goog-gke-node-pool-provisioning-model": "flex-start",
          "goog-gke-tpu-node-pool-type": "multi-host"
        },
        "taints": [
          {"effect": "NO_SCHEDULE", "key": "google.com/tpu", "value": "present"},
          {"effect": "NO_SCHEDULE", "key": "cloud.google.com/gke-queued", "value": "true"}
        ],
        "shieldedInstanceConfig": {"enableIntegrityMonitoring": true},
        "metadata": {"disable-legacy-endpoints": "true"}
      },
      "queuedProvisioning": {"enabled": true},
      "autoscaling": {"enabled": true, "locationPolicy": "ANY", "minNodeCount": 0, "maxNodeCount": MAX_NODES},
      "placementPolicy": {
        "policyName": "POLICY_NAME",
        "tpuTopology": "TOPOLOGY"
      },
      "maxPodsConstraint": {"maxPodsPerNode": "110"},
      "management": {"autoRepair": true, "autoUpgrade": true},
      "initialNodeCount": 0
    }
  }'
```

### Step 3 — Verify

```bash
gcloud container node-pools describe POOL_NAME \
  --cluster=jingnw-tpu7-cluster --region=us-central1-c \
  --format="yaml(name,status,queuedProvisioning,placementPolicy)"
```

Expected: `queuedProvisioning.enabled: true`, `status: RUNNING`.

---

## Submitting a DWS job

A DWS job requires three Kubernetes resources in addition to the Job itself:

1. **PodTemplate** — used by the ProvisioningRequest to estimate resource requirements
2. **ProvisioningRequest** — requests capacity from DWS for a given duration
3. **Job pods** must carry the `consume-provisioning-request` annotation

The pod annotation binds pods to the provisioned capacity:
```yaml
annotations:
  cluster-autoscaler.kubernetes.io/consume-provisioning-request: PROVISIONING_REQUEST_NAME
  cluster-autoscaler.kubernetes.io/provisioning-class-name: queued-provisioning.gke.io
  cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
```

`safe-to-evict: "false"` prevents the cluster autoscaler from evicting pods after the
DWS `BookingExpired` event fires (which happens ~10 min after provisioning). The DWS
nodes remain available for the full `maxRunDurationSeconds`.

### Monitoring

```bash
# Check provisioning status
kubectl get provisioningrequest PROVISIONING_REQUEST_NAME

# ACCEPTED = capacity confirmed, PROVISIONED = nodes joined, FAILED = no capacity
kubectl describe provisioningrequest PROVISIONING_REQUEST_NAME

# Watch pods
kubectl get pods -l job-name=JOB_NAME -o wide

# Tail logs from all pods
kubectl logs -f -l job-name=JOB_NAME --prefix

# Cloud Logging (survives pod termination)
gcloud logging read \
  'resource.type="k8s_container" AND resource.labels.cluster_name="jingnw-tpu7-cluster" AND labels."k8s-pod/job-name"="JOB_NAME"' \
  --project=tpu-launchpad-playground \
  --format="value(timestamp,textPayload)" \
  --limit=50
```

### Resubmitting a failed job

When a job fails or needs to be restarted, always delete the old PR before reapplying
(otherwise the PR stays FAILED and blocks new pods from scheduling):

```bash
kubectl delete job JOB_NAME --ignore-not-found
kubectl delete provisioningrequest PR_NAME --ignore-not-found
kubectl delete service SERVICE_NAME --ignore-not-found
kubectl apply -f scripts/JOB_YAML.yaml
```

### NodepoolSizeReached — waiting for nodes to drain

If DWS returns `NodepoolSizeReached`, old nodes from a previous run are still allocated.
Check and wait:

```bash
# Check node count — wait until it reaches 0
kubectl get nodes -l cloud.google.com/gke-nodepool=POOL_NAME --no-headers | wc -l

# Once 0: delete the FAILED PR and resubmit
kubectl delete provisioningrequest PR_NAME
kubectl apply -f scripts/JOB_YAML.yaml
```

Nodes typically drain 10–20 min after the last pod exits. GKE autoscaler does not
scale down while any pod (even Completed/Error) remains on the node — ensure pods are
fully terminated before expecting node drain.

---

## GCS bucket setup

The MiMo-V2.5-Pro demo reads weights from GCS:

| Bucket | Contents |
|---|---|
| `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` | 34 safetensors files (~962 GB FP8) |
| `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/` | XLA compiled kernels (~85 MB after first run) |

The XLA compilation cache accumulates across job restarts. The cache key encodes the
model hash, TPU topology, and XLA version — changing `--tp-size` or the container image
invalidates cached entries. With a warm cache, XLA warmup takes ~55 seconds instead of
15+ hours.

See [gke_tpu7x_resource_allocation.md](gke_tpu7x_resource_allocation.md) for full
GCS, RAM, HBM, and disk allocation details.

---

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ProvisioningRequest FAILED: pods cannot be scheduled in nodepool` | Node pool missing `queuedProvisioning.enabled: true` | Recreate pool via GKE REST API (see above) |
| `gcloud container node-pools create --enable-queued-provisioning` fails | Flag unsupported for TPU node pools via gcloud CLI | Use GKE REST API |
| `gcloud container node-pools create --flex-start` creates pool but DWS ignores it | `--flex-start` sets `flexStart: true` but not `queuedProvisioning.enabled: true` | Use GKE REST API |
| `gcloud compute resource-policies create group-placement --tpu-topology` fails | Creates `groupPlacementPolicy` (COLLOCATED), rejected by tpu7x | Use Compute Engine REST API to create `workloadPolicy` type |
| `OOM: Not enough memory. Please try to increase --mem-fraction-static` in `_profile_available_bytes` | Model weights fill >92% of HBM at current tp-size | Increase tp-size (more nodes) rather than raising mem-fraction-static |
| XLA temp OOM during KV cache profiling at higher `--mem-fraction-static` | 384-expert MoE forward pass needs ~24 GB/TensorCore scratch; `0.92` leaves only 8% (~7.7 GB) | Use `--mem-fraction-static 0.75` (25% XLA scratch = ~24 GB/TensorCore) |
| DWS nodes evicted ~10 min after provisioning | `BookingExpired` event triggers cluster autoscaler | Add `safe-to-evict: "false"` pod annotation |
| `ProvisioningRequest FAILED: NodepoolSizeReached` immediately after resubmit | Old nodes from previous run still allocated; pool at max capacity | Wait for nodes to drain (10–20 min) then delete PR and reapply (see Resubmitting section) |
| PR shows `FAILED=True` but was just created | Old FAILED PR was not deleted before `kubectl apply` — `apply` leaves existing resources unchanged | Always `kubectl delete provisioningrequest PR_NAME` before reapplying |
| `--precompile-bs-paddings` argument error: `invalid int value: '1,2,4,8,16,32'` | Flag expects space-separated integers, not comma-separated | Use `--precompile-bs-paddings 1 2 4 8 16 32` |
| Orbax checkpoint restore: `ShapeDtypeStruct is not a valid JAX type` for FP8 arrays | libtpu in `jax0.8.1-rev1` and `jax0.9.0-rev1` cannot create `float8_e4m3fn` device buffers via `make_array_from_single_device_arrays` | Blocked — requires newer libtpu; use NFS loading as workaround |
