import jax
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
    
    # Let's intercept device_put!
    orig_device_put = jax.device_put
    def my_device_put(*args, **kwargs):
        print("Called device_put with", type(args[0]), args[0].dtype if hasattr(args[0], 'dtype') else '')
        return orig_device_put(*args, **kwargs)
    jax.device_put = my_device_put
    
    sds = jax.ShapeDtypeStruct(jax_arr.shape, jax_arr.dtype, sharding=sharding)
    restored = ckptr.restore(path, item={"w": sds})
    
    jax.device_put = orig_device_put
    shutil.rmtree(base)
