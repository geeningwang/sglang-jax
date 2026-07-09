# MiMo-V2-Flash 1P1D Disaggregated PD: Approach Log

**Goal**: Run two independent servers on two separate TPU v7x VMs simultaneously:
- Pod 0 (prefill): tp-size 8, 4 chips, 8 JAX devices, port 10000
- Pod 1 (decode): tp-size 8, 4 chips, 8 JAX devices, port 10001

Each VM needs a full 2x2x1 TPU slice (4 chips = 8 JAX devices with 2 TensorCores/chip).

**Infrastructure**:
- GKE cluster, us-central1-c (under heavy capacity pressure as of 2025-07)
- Available DWS pools: `jingnw-dws-tpu7-4ch` (2x2x1, gang size=1), `jingnw-dws-tpu7-8ch` (2x2x2, gang size=2)
- Flex Start pool: `jingnw-flex-tpu7-8ch` — unavailable (zone capacity exhausted)
- DWS provisioning wait: ~6h

---

## Attempt 1 — Two separate PRs on the same single-host pool

**Files**: `pd-prefill.yaml` + `pd-decode.yaml` (deleted)  
**Pool**: `jingnw-dws-tpu7-4ch` (2x2x1)  
**PR count**: 1 per job (matching gang size=1)

**Idea**: Submit two independent Jobs each with their own ProvisioningRequest targeting
the single-host pool. Each pod gets its own 2x2x1 VM.

**Result**: ❌ Failed immediately on apply.

**Error**:
```
Invalid value for field 'resource.resizeBy': '2'.
Requested invalid target size '2' for a Managed Instance Group in the gang mode of size '1'.
```

**Root cause**: A DWS pool is backed by a single MIG. Both PRs try to resize the same MIG
simultaneously. MIG gang size = 1 means only one resize can be active. The second PR pushes
the target to 2, which violates the gang constraint. Two simultaneous PRs on the same pool
is structurally impossible regardless of count.

---

## Attempt 2 — Single 2x2x2 IndexedJob, no init override

**File**: `pd1p1d.yaml` (original, before env var fix)  
**Pool**: `jingnw-dws-tpu7-8ch` (2x2x2, gang size=2)  
**Design**: Both pods land on the two VMs of the same 2x2x2 slice via gang scheduling.
Pod 0 = prefill, pod 1 = decode. Each pod gets one VM.

**Result**: ❌ Both pods crashed immediately after DWS provisioned.

**Error**:
```
jax.errors.JaxRuntimeError: UNKNOWN: TPU initialization failed:
Invalid --deepsea_slice_builder_worker_addresses specified.
Expected 2 worker addresses, got 1.
```

**Root cause**: In a 2x2x2 multi-host slice, libtpu requires ALL VMs in the gang to
participate in a coordinated handshake during `jax.distributed.initialize()`. GKE sets
`TPU_WORKER_HOSTNAMES=host0,host1` (both VMs) and `CLOUD_TPU_TASK_ID=0` or `1` for each
pod. But the pods start their servers at different times (pod 1 waits for pod 0's IP from
GCS), so they never handshake simultaneously. Each pod presents only 1 worker address,
but libtpu expects 2.

---

## Attempt 3 — Flex Start pool

**Pool**: `jingnw-flex-tpu7-8ch`  
**Idea**: Avoid DWS queuing via cluster autoscaler reactive provisioning.

**Result**: ❌ Failed immediately.

**Error**:
```
Node scale up in zones us-central1-c associated with this pod failed: Internal error
```

**Root cause**: us-central1-c has no available TPU v7x capacity for Flex Start. Zone is
under heavy pressure.

---

## Attempt 4 — 2x2x2 IndexedJob with env var override (ABANDONED, untested)

**File**: `pd1p1d.yaml` (superseded by Attempt 5)  
**Pool**: `jingnw-dws-tpu7-8ch` (2x2x2, gang size=2)  
**Change**: Added before each `launch_server` call:

```bash
export CLOUD_TPU_TASK_ID=0
export TPU_WORKER_HOSTNAMES=$(hostname)
```

**Idea**: Override GKE's multi-host env vars so libtpu sees only 1 worker (the local VM).
This should bypass "Expected 2 worker addresses, got 1" and initialize each pod as an
independent single-host system with 4 chips (8 JAX devices).

**Status**: ❌ Abandoned before testing — replaced by Attempt 5 (simultaneous barrier).

**Why abandoned**: If env vars are respected and each pod inits as a truly independent
single-host JAX process, the two pods are not synchronized with each other at the libtpu
level. Whether they collide at the hardware level (chip ICI mesh, driver state) is unknown.
More importantly, even if this passes libtpu init, there is no guarantee the JAX XLA
compiler correctly limits each pod to its local 8 devices without the standard
multi-host initialization path. Attempt 5 uses the proper multi-host path instead.

---

## Attempt 5 — 2x2x2 IndexedJob with simultaneous-start barrier (CURRENT, PENDING)

**File**: `pd1p1d.yaml` (current)  
**Pool**: `jingnw-dws-tpu7-8ch` (2x2x2, gang size=2)

**Idea**: libtpu requires all VMs in a 2x2x2 gang to call into the driver at the same
time. Previous attempts failed because pod 1 waited for pod 0's IP before starting its
server — so pod 0 started JAX init before pod 1 was ready. Option E fixes this with a
GCS barrier:

1. Pod 0: starts bootstrap server (no JAX) → writes its IP → **waits for pod1 IP flag**
2. Pod 1: **writes its IP immediately** → waits for pod0 IP → starts decode server
3. Pod 0: sees pod1 IP → starts prefill server
4. Both servers call into libtpu within seconds of each other → handshake succeeds

New GCS flag: `pd1p1d-pod1-ip` (pod 1 writes this as its barrier signal to pod 0).

**Status**: ⏳ Waiting for DWS provisioning (~6h). Not yet validated.

**Known risk**: After successful 2x2x2 libtpu init, both JAX processes are part of the
SAME distributed JAX program (SPMD). Each process will see:
- `jax.device_count()` = 16 (all 8 chips × 2 TensorCores across both VMs)
- `jax.local_device_count()` = 8 (local 4 chips × 2 TensorCores)

sgl_jax is launched with `--nnodes 1 --node-rank 0` on both pods, which assumes single-host.
In SPMD JAX, both processes must execute the SAME JAX ops simultaneously — they cannot
run independent server logic. If sgl_jax doesn't account for `jax.process_count() == 2`,
the two server processes will deadlock waiting for each other to issue matching JAX ops.

**If this fails**: The only clean solution is Option A (two separate DWS node pools).

---

## Approaches Not Yet Tried

### Option A — Two separate DWS pools (most promising)

Create two distinct GKE node pools, each with 2x2x1 topology and gang size=1:
- Pool `tpu7-4ch-prefill`: for pod 0 (prefill)
- Pool `tpu7-4ch-decode`: for pod 1 (decode)

Each pool has its own MIG → no gang-size conflict → two simultaneous single-host PRs work.
Submit prefill job targeting `tpu7-4ch-prefill` and decode job targeting `tpu7-4ch-decode`.

**Blocker**: Requires GKE admin access to create new node pools (gcloud container node-pools create).

### Option B — Different zone

Submit the same `pd1p1d.yaml` in us-central1-a or us-central1-b. Capacity may be available
faster. Requires verifying that the TPU v7x pool topology is available in target zone.

### Option C — Same-VM co-location (no disaggregation)

Run both prefill and decode on the same single-host VM inside one pod. Eliminates
inter-pod coordination. Loses true disaggregation isolation but validates the server logic.

### Option D — Manual JAX distributed init across both pods

Let both pods call `jax.distributed.initialize()` with the full 2-worker address list
(pod0 IP + pod1 IP), so the 2x2x2 slice initializes correctly. Then:
- Pod 0 claims device indices 0-7 (chips 0-3)
- Pod 1 claims device indices 8-15 (chips 4-7)

Use `jax.local_devices()` instead of `jax.devices()` inside each server.

**Complexity**: Requires modifying `sgl_jax.launch_server` to accept explicit device
subsets, which is a deeper code change.

### Option E — Simultaneous multi-host init with barrier

Have both pods start `launch_server` simultaneously by writing both IPs to GCS first
(pod0 writes its IP; pod1 writes its IP; both wait until both flags exist), then launch
servers at the same time. This satisfies libtpu's requirement that both workers handshake
at init time.

**Risk**: Servers may not be truly independent — XLA will see all 16 JAX devices across
2 VMs and try to use them as a single mesh. This conflicts with disaggregated design where
pod 0 is prefill-only and pod 1 is decode-only, each needing their own full tp-size 8.

---

## Key Constraints Summary

| Constraint | Detail |
|---|---|
| MIG gang size | A DWS pool's MIG enforces `count == gang_size`; only one PR active per pool at a time |
| libtpu multi-host init | 2x2x2 slice requires simultaneous init handshake from ALL VMs |
| Zone capacity | us-central1-c: DWS ~6h wait, Flex Start unavailable |
| tp-size per pod | 8 (requires 4 chips = 8 JAX devices per pod, matching 2x2x1) |
| Chip sharing | Two pods on the same 2x2x2 slice share physical silicon — concurrent independent init may conflict |
