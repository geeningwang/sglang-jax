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
```
Patched device_put called!
Restored type: <class 'jaxlib._jax.ArrayImpl'>
Restored dtype: float8_e4m3fn
```
