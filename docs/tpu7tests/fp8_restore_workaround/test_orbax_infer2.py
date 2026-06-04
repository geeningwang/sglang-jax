import jax
import orbax.checkpoint as ocp
import tempfile
import os
import numpy as np

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()
state = {"w": jax.device_put(np.ones((10, 10), dtype=np.float32))}
ckptr.save(path, state)

sds = jax.ShapeDtypeStruct((10, 10), np.float32, sharding=None)
restored = ckptr.restore(path, item={"w": sds})
print("Restored type:", type(restored["w"]))
