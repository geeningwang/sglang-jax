# FP8 Checkpoint Restore Workaround Demo

This folder contains a minimal reproducible demo that successfully tests the workaround for the JAX/libtpu issue preventing `float8_e4m3fn` arrays from being transferred via `jax.device_put` or `jax.make_array_from_single_device_arrays` on TPU v7x.

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

This confirms the underlying transfer bug is successfully bypassed. To implement this across a multi-host TPU slice, the `tensorstore` reads must be manually iterated to bypass Orbax `0.12.0`'s broken `SingleReplicaArrayHandler` multi-host broadcast tree logic.
