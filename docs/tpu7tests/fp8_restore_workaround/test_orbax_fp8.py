import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
import orbax.checkpoint as ocp
from jax.sharding import PartitionSpec, NamedSharding, Mesh
import tempfile
import shutil
import os

mesh = Mesh(jax.devices(), ('x',))

with mesh:
    sharding = NamedSharding(mesh, PartitionSpec('x'))
    np_arr = np.ones((1024, 1024), dtype=ml_dtypes.float8_e4m3fn)
    jax_arr = jax.device_put(np_arr, sharding)
    
    base = tempfile.mkdtemp()
    path = os.path.join(base, "ckpt")
    ckptr = ocp.PyTreeCheckpointer()
    
    ckptr.save(path, {"w": jax_arr})
    
    sds = jax.ShapeDtypeStruct(jax_arr.shape, jax_arr.dtype, sharding=sharding)
    restored = ckptr.restore(path, item={"w": sds})
    
    print("Restored type:", type(restored["w"]))
    if not isinstance(restored["w"], jax.ShapeDtypeStruct):
        print("Restored dtype:", restored["w"].dtype)
        
    shutil.rmtree(base)
