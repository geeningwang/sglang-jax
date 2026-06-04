# FP8 Checkpoint Restore Workaround — Validated ✅

**Status**: Workaround fully validated (2026-06-03). All 4 critical tests passed.
Ready for production integration into `loader.py`.

This folder contains the investigation work, minimal demo, and validation test suite
for the workaround to the JAX/libtpu issue preventing `float8_e4m3fn` arrays from
being transferred via `jax.device_put` or `jax.make_array_from_single_device_arrays`
on TPU v7x.

## Contents

- `single_tpu_job.yaml`: A Kubernetes Job manifest that runs a minimal test case on a single TPU v7x node. It verifies that we can save a dummy FP8 array using `orbax.checkpoint`, intercept `jax.device_put` to transfer the array as `uint8`, and `bitcast` it back to FP8 on the device while avoiding Host RAM OOM limits via `block_until_ready`.

## Running the Demo

To test the workaround on your cluster:

```bash
kubectl apply -f single_tpu_job.yaml
```

Once the job is completed, you can check the logs to verify that the FP8 array was successfully restored via the `uint8` fallback path:

```bash
kubectl logs -l job-name=test-orbax-fp8-single
```

You should see:
```text
JAX devices: 1
Putting array on device...
Patched device_put called!
Saving...
Restoring...
/opt/venv/lib/python3.12/site-packages/orbax/checkpoint/_src/serialization/jax_array_handlers.py:749: UserWarning: Sharding info not provided when restoring. Populating sharding info from sharding file. Please note restoration time will be slightly increased due to reading from file. Note also that this option is unsafe when restoring on a different topology than the checkpoint was saved with.
  warnings.warn(
Patched device_put called!
Restored type: <class 'jaxlib._jax.ArrayImpl'>
Restored dtype: float8_e4m3fn
```

This confirms the underlying transfer bug is successfully bypassed on a **single-chip** setup. To implement this across a multi-host TPU slice, the `tensorstore` reads must be manually iterated to bypass Orbax `0.12.0`'s broken `SingleReplicaArrayHandler` multi-host broadcast tree logic.

---

## Validation Test Results (2026-06-03)

All 4 tests executed and passed on `jax0.9.0-rev1` + Orbax 0.12.0:

| Test | Result | Key Finding |
|------|--------|-------------|
| `test_bitcast_on_tpu.yaml` | ✅ PASS | `bitcast_convert_type(uint8→float8_e4m3fn)` works on TPU v7x |
| `test_hbm_pressure.yaml` | ✅ PASS | Patch succeeds with ~14 MB free HBM, no OOM |
| `test_concurrent_shards.yaml` | ✅ PASS | Max concurrency = 1 (Orbax is serial), no semaphore needed |
| `test_4node_multihost_intercept.yaml` | ✅ PASS | `jax.device_put` IS intercepted in the 4-node multi-host path |

The confidence level for the full production implementation rises from **25–40%** to **~90%**.
All three critical unknowns are resolved.

---

## Validation Test Suite (for re-running)

Tests in order of dependency — run Test 1 first (prerequisite), then 2 and 3 in
parallel, then Test 4 last (requires DWS 4-node slice):

### Test 1 — `test_bitcast_on_tpu.yaml` (prerequisite)

**Question**: Does `jax.lax.bitcast_convert_type(uint8_arr, float8_e4m3fn)` work on TPU?

This is the foundation of the entire approach. If bitcast fails on TPU, nothing else matters.

```bash
kubectl apply -f test_bitcast_on_tpu.yaml
kubectl logs -l job-name=test-fp8-bitcast -f
kubectl delete job test-fp8-bitcast
```

**Expected PASS**:
```
[1] uint8 on device: PASS
[2] bitcast to float8: PASS
[3] dtype correct: PASS
[4] compute works: PASS
=== ALL TESTS PASSED ===
```

**If FAIL**: The entire uint8 bitcast approach is not viable on this libtpu version. Stop here.

---

### Test 2 — `test_hbm_pressure.yaml` (HBM constraint)

**Question**: Does the patch work when HBM is drained to ~14 MB free per TC (post-model-load state)?

The single-node demo ran with fresh HBM. This test drains HBM first, then runs the patch.

```bash
kubectl apply -f test_hbm_pressure.yaml
kubectl logs -l job-name=test-fp8-hbm -f
kubectl delete job test-fp8-hbm
```

**Expected PASS**:
```
[1] HBM drained to ~14 MB free: OK
[2] Checkpoint saved
Patched device_put called! #1
[3] Restore under pressure: PASS
=== PASS: patch works under HBM pressure ===
```

**If FAIL (OOM)**: Per-shard granularity is not fine enough — bitcast still accumulates buffers.
Add a `threading.Semaphore(1)` inside the patch to serialize shard transfers.

---

### Test 3 — `test_concurrent_shards.yaml` (concurrency)

**Question**: How many shard transfers does Orbax run concurrently? Does the peak exceed 14 MB free?

With 14 MB free and 6 MB per shard (3 MB uint8 + 3 MB float8), concurrency ≥ 3 causes OOM.

```bash
kubectl apply -f test_concurrent_shards.yaml
kubectl logs -l job-name=test-fp8-concurrent -f
kubectl delete job test-fp8-concurrent
```

**Expected PASS** (concurrency ≤ 2):
```
[4] Max concurrent patch calls: 1
[4] All 20 arrays restored correctly: True
=== PASS: Concurrency is low enough ===
```

**If WARNING/FAIL** (concurrency ≥ 3): Add a semaphore to the patch:
```python
sem = threading.Semaphore(2)  # limit to 2 concurrent bitcasts
def patched_dp(x, *args, **kwargs):
    if is_fp8(x):
        with sem:
            ...  # existing patch logic
```

---

### Test 4 — `test_4node_multihost_intercept.yaml` (the critical test)

**Question**: Does `jax.device_put = patched_dp` actually intercept the restore call
in a 4-node multi-host setup? Or does `SingleReplicaArrayHandler` bypass `jax.device_put`?

This requires the DWS 4-node slice (provisioned automatically).

```bash
kubectl apply -f test_4node_multihost_intercept.yaml
kubectl logs -l job-name=test-fp8-4node --prefix -f
kubectl delete job test-fp8-4node
kubectl delete service test-fp8-4node
```

**Expected PASS**:
```
[rank0] Patched device_put called! (call #1)
[rank0] Restored type: ArrayImpl
[rank0] Restored dtype: float8_e4m3fn
[rank0] === PASS: Multi-host FP8 restore via patch works ===
```

**If FAIL** (patch never called):
```
[rank0] === FAIL: Got ShapeDtypeStruct not ArrayImpl ===
[rank0] DIAGNOSIS: patch was NEVER called — multi-host path bypasses jax.device_put
[rank0] NEXT STEP: manual TensorStore iteration required
```
→ The `SingleReplicaArrayHandler` uses `create_async_array_from_callback` internally,
which does NOT call `jax.device_put`. Manual TensorStore reads (as in `async_deserialize.py`)
would be needed to intercept at the right level.

---

## Decision Tree

```
Test 1 (bitcast on TPU)
  └─ FAIL → approach not viable, stop
  └─ PASS →
       Test 2 (HBM pressure)
         └─ FAIL → add Semaphore(1) to patch, retest
         └─ PASS →
              Test 3 (concurrency)
                └─ concurrency ≥ 3 → add Semaphore(2), retest
                └─ concurrency ≤ 2 →
                     Test 4 (4-node multi-host)
                       └─ patch NOT called → implement TensorStore manual iteration
                       └─ patch called + PASS → workaround is production-ready
```
