import jax
import orbax.checkpoint as ocp
import numpy as np
import tempfile
import os

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()
state = {"w": jax.device_put(np.ones((10,), dtype=np.float32))}
ckptr.save(path, state)

orig_device_put = jax.device_put
def failing_device_put(*args, **kwargs):
    raise RuntimeError("I failed!")
jax.device_put = failing_device_put

try:
    sds = jax.ShapeDtypeStruct((10,), np.float32, sharding=None)
    res = ckptr.restore(path, item={"w": sds})
    print("Restore succeeded! Type:", type(res["w"]))
except Exception as e:
    print("Restore threw exception:", type(e), e)

jax.device_put = orig_device_put
