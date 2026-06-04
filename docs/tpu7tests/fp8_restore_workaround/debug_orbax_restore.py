import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
import orbax.checkpoint as ocp
import tempfile
import os

print("Starting debug...")

orig_device_put = jax.device_put
def patched_device_put(x, *args, **kwargs):
    if hasattr(x, "dtype") and str(x.dtype) in {"float8_e4m3fn", "float8_e5m2"}:
        print("Patched device_put called!")
        x_u8 = np.asarray(x).view(np.uint8)
        arr_u8 = orig_device_put(x_u8, *args, **kwargs)
        target_dtype = getattr(jnp, str(x.dtype))
        arr_f8 = jax.lax.bitcast_convert_type(arr_u8, target_dtype)
        arr_f8.block_until_ready()
        return arr_f8
    return orig_device_put(x, *args, **kwargs)

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()

print("Saving FP8 array...")
sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec('x'))
state = {"w": jax.device_put(np.ones((10, 10), dtype=ml_dtypes.float8_e4m3fn), sharding)}
ckptr.save(path, state)

abstract_state = {"w": jax.ShapeDtypeStruct((10, 10), ml_dtypes.float8_e4m3fn, sharding=sharding)}

jax.device_put = patched_device_put
try:
    print("Restoring...")
    restored = ckptr.restore(path, item=abstract_state)
    print("Restored type:", type(restored["w"]))
finally:
    jax.device_put = orig_device_put

import shutil
shutil.rmtree(base)
