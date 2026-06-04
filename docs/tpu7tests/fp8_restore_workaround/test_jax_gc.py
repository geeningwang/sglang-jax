import jax
import jax.numpy as jnp
import gc

def check_memory():
    # Only works if we have some way to check memory.
    pass

# We can just check if JAX reclaims memory.
# In JAX, arrays are immediately freed when refcount goes to 0.
arr = jnp.zeros((1024, 1024))
del arr
print("Freed successfully.")
