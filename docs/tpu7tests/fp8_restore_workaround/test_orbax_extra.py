import jax
import orbax.checkpoint as ocp
import tempfile
import os
import numpy as np

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()

state = {"a": jax.device_put(np.ones((10,), dtype=np.float32))}
ckptr.save(path, state)

item = {
    "a": jax.ShapeDtypeStruct((10,), np.float32, sharding=None),
    "b": jax.ShapeDtypeStruct((10,), np.float32, sharding=None)
}
try:
    res = ckptr.restore(path, item=item)
    print("b type:", type(res["b"]))
except Exception as e:
    print("Threw exception:", type(e))
