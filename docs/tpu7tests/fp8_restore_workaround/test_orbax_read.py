import jax
import numpy as np
import ml_dtypes
import orbax.checkpoint as ocp
from jax.sharding import PartitionSpec, NamedSharding, Mesh
import tempfile
import shutil
import os
import tensorstore as ts
import asyncio

mesh = Mesh(jax.devices(), ('x',))

with mesh:
    sharding = NamedSharding(mesh, PartitionSpec('x'))
    np_arr = np.ones((1024, 1024), dtype=ml_dtypes.float8_e4m3fn)
    jax_arr = jax.device_put(np_arr, sharding)
    
    base = tempfile.mkdtemp()
    path = os.path.join(base, "ckpt")
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(path, {"w": jax_arr, "b": jax_arr})
    
    # Can we get the TSpecs from Orbax without loading data?
    # Yes! PyTreeCheckpointer.metadata returns the metadata!
    metadata = ckptr.metadata(path)
    print("Metadata keys:", metadata.keys())
    
    shutil.rmtree(base)
