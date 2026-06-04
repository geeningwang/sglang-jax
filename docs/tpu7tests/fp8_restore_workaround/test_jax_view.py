import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

u8 = jax.device_put(np.ones((10, 10), dtype=np.uint8))
try:
    f8 = u8.view(ml_dtypes.float8_e4m3fn)
    print("view success!", f8.dtype)
except Exception as e:
    print("view failed:", e)
