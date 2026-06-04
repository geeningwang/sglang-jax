import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

orig_device_put = jax.device_put

def patched_device_put(x, *args, **kwargs):
    if hasattr(x, "dtype") and str(x.dtype) in {"float8_e4m3fn", "float8_e5m2"}:
        print("Patched device_put called!")
        x_u8 = np.asarray(x).view(np.uint8)
        arr_u8 = orig_device_put(x_u8, *args, **kwargs)
        target_dtype = getattr(jnp, str(x.dtype))
        return jax.lax.bitcast_convert_type(arr_u8, target_dtype)
    return orig_device_put(x, *args, **kwargs)

jax.device_put = patched_device_put

# test
a = np.ones((10, 10), dtype=ml_dtypes.float8_e4m3fn)
res = jax.device_put(a)
print(res.dtype)

b = np.ones((10, 10), dtype=np.float32)
res2 = jax.device_put(b)
print(res2.dtype)
