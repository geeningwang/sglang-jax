import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

# Check if .view() shares the same buffer
u8 = jax.device_put(np.ones((10, 10), dtype=np.uint8))
f8 = u8.view(ml_dtypes.float8_e4m3fn)
print("u8 is f8?", u8 is f8)
print("u8.unsafe_buffer_pointer() == f8.unsafe_buffer_pointer()?", u8.unsafe_buffer_pointer() == f8.unsafe_buffer_pointer())
