import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
import orbax.checkpoint as ocp
import tempfile
import os

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()

sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec('x'))
# Save a float8 array
ckptr.save(path, {"w": jax.device_put(np.ones((10, 10), dtype=ml_dtypes.float8_e4m3fn), sharding)})

orig_device_put = jax.device_put
def patched_device_put(x, *args, **kwargs):
    if hasattr(x, "dtype") and str(x.dtype) in {"float8_e4m3fn", "float8_e5m2"}:
        print("Patching device_put!")
        x_u8 = np.asarray(x).view(np.uint8)
        arr_u8 = orig_device_put(x_u8, *args, **kwargs)
        target_dtype = getattr(jnp, str(x.dtype))
        arr_f8 = jax.lax.bitcast_convert_type(arr_u8, target_dtype)
        arr_f8.block_until_ready()
        return arr_f8
    return orig_device_put(x, *args, **kwargs)

jax.device_put = patched_device_put
abstract = {"w": jax.ShapeDtypeStruct((10, 10), ml_dtypes.float8_e4m3fn, sharding=sharding)}

# Let's intercept to see if `make_array_from_single_device_arrays` is called!
orig_make_array = jax.make_array_from_single_device_arrays
def patched_make_array(*args, **kwargs):
    print("make_array called! with elements of type:", type(args[2][0]))
    res = orig_make_array(*args, **kwargs)
    print("make_array returned:", type(res))
    return res
jax.make_array_from_single_device_arrays = patched_make_array

restored = ckptr.restore(path, item=abstract)
print("Restored type:", type(restored["w"]))
