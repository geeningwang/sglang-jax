import jax
import jax.numpy as jnp
import gc

def inplace_convert(d_u8, d_orig):
    # This assumes it's a dict. What if it's a list or tuple?
    # nnx.State is usually a dict-like or Custom class.
    # We might need to handle other types, or just use `jax.tree_util.tree_map_with_path`?
    # No, tree_map doesn't delete.
    if isinstance(d_u8, dict):
        d_f8 = type(d_u8)() # Try to keep same dict type (e.g. FrozenDict? FrozenDict has no pop)
        for k in list(d_u8.keys()):
            v_u8 = d_u8.pop(k)
            d_f8[k] = inplace_convert(v_u8, d_orig[k])
        return d_f8
    else:
        # leaf
        # check if bitcast is needed
        # simulate bitcast
        f8 = jnp.zeros_like(d_u8)
        return f8

d = {"a": {"b": jnp.zeros((10, 10))}, "c": jnp.zeros((10, 10))}
d_orig = {"a": {"b": jnp.zeros((10, 10))}, "c": jnp.zeros((10, 10))}

res = inplace_convert(d, d_orig)
print("Success", res.keys())
