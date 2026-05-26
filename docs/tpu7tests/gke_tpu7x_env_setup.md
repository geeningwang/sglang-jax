# GKE TPU v7x Environment Setup (jingnw-tpu7-cluster)

How to set up the GKE cluster, node pools, and supporting infrastructure for TPU v7x
readiness tests. Covers discovered pitfalls and the workarounds required.

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

# Cloud Logging (survives pod termination)
gcloud logging read \
  'resource.type="k8s_container" AND resource.labels.cluster_name="jingnw-tpu7-cluster" AND labels."k8s-pod/job-name"="JOB_NAME"' \
  --project=tpu-launchpad-playground \
  --format="value(timestamp,textPayload)" \
  --limit=50
```

---

## GCS bucket setup

The MiMo-V2.5-Pro demo reads weights from GCS:

| Bucket | Contents |
|---|---|
| `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/` | 34 safetensors files (~962 GB FP8) |
| `gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache/` | XLA compiled kernels (incremental) |

The XLA compilation cache accumulates across job restarts. The cache key encodes the
model hash, TPU topology, and XLA version — changing `--tp-size` or the container image
invalidates cached entries.

---

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ProvisioningRequest FAILED: pods cannot be scheduled in nodepool` | Node pool missing `queuedProvisioning.enabled: true` | Recreate pool via GKE REST API (see above) |
| `gcloud container node-pools create --enable-queued-provisioning` fails | Flag unsupported for TPU node pools via gcloud CLI | Use GKE REST API |
| `gcloud container node-pools create --flex-start` creates pool but DWS ignores it | `--flex-start` sets `flexStart: true` but not `queuedProvisioning.enabled: true` | Use GKE REST API |
| `gcloud compute resource-policies create group-placement --tpu-topology` fails | Creates `groupPlacementPolicy` (COLLOCATED), rejected by tpu7x | Use Compute Engine REST API to create `workloadPolicy` type |
| `OOM: Not enough memory. Please try to increase --mem-fraction-static` in `_profile_available_bytes` | Model weights fill >92% of HBM at current tp-size | Increase tp-size (more nodes) rather than raising mem-fraction-static |
| DWS nodes evicted ~10 min after provisioning | `BookingExpired` event triggers cluster autoscaler | Add `safe-to-evict: "false"` pod annotation |
