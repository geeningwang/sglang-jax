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
    
    # Try restoring with PartitionSpec instead of NamedSharding
    try:
        sds = jax.ShapeDtypeStruct(jax_arr.shape, jax_arr.dtype, sharding=PartitionSpec('x'))
    except Exception:
        # In newer JAX, ShapeDtypeStruct rejects PartitionSpec. 
        # But we can monkeypatch or mock it if they somehow saved it (maybe they saved it in older jax)
        class MockSDS:
            shape = jax_arr.shape
            dtype = jax_arr.dtype
            sharding = PartitionSpec('x')
        sds = MockSDS()

    restored = ckptr.restore(path, item={"w": sds})
    print("Restored type:", type(restored["w"]))
    
    shutil.rmtree(base)
