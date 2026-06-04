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

# Create dummy state with CPU arrays
state = {"w": np.ones((1024, 1024), dtype=ml_dtypes.float8_e4m3fn)}
ckptr.save(path, state)

# Restore with numpy=True? 
# or with jax.ShapeDtypeStruct without sharding!
sds = jax.ShapeDtypeStruct(state["w"].shape, state["w"].dtype, sharding=None)
restored = ckptr.restore(path, item={"w": sds})

print("Restored type:", type(restored["w"]))
if isinstance(restored["w"], np.ndarray):
    print("Restored dtype:", restored["w"].dtype)

shutil.rmtree(base)
