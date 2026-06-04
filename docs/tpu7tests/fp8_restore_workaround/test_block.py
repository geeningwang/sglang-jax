import jax
import numpy as np

def patched(x):
    # simulate
    arr = jax.device_put(x)
    return arr.block_until_ready()

a = np.ones((10, 10))
res = patched(a)
print("Success")
