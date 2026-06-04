import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

# Try to put a float8 array on TPU
np_arr = np.zeros((1024, 1024), dtype=ml_dtypes.float8_e4m3fn)
try:
    jax_arr = jax.device_put(np_arr)
    print("device_put success!", jax_arr.dtype)
except Exception as e:
    print("device_put failed:", type(e), e)
