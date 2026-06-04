import jax
import numpy as np
import ml_dtypes
import orbax.checkpoint as ocp
from jax.sharding import PartitionSpec, NamedSharding, Mesh
import tempfile
import shutil
import os

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()

state = {
    "a": np.ones((10,), dtype=np.uint8),
    "b": np.ones((10,), dtype=np.uint8)
}
ckptr.save(path, state)

# Restore ONLY 'b'
sds = jax.ShapeDtypeStruct(state["b"].shape, state["b"].dtype, sharding=None)
restored = ckptr.restore(path, item={"b": sds})

print("Restored keys:", restored.keys())
print("Restored b:", restored["b"])

shutil.rmtree(base)
