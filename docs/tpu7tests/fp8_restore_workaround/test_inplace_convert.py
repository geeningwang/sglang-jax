import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

# Simulate the bitcast
# jax.lax.bitcast_convert_type
u8 = jnp.ones((10, 10), dtype=jnp.uint8)
f8 = jax.lax.bitcast_convert_type(u8, jnp.float8_e4m3fn)
print(f8.dtype)
