import jax
import numpy as np
import ml_dtypes
from jax.sharding import PartitionSpec, NamedSharding, Mesh
import tempfile

mesh = Mesh(jax.devices(), ('x',))
with mesh:
    sharding = NamedSharding(mesh, PartitionSpec('x'))
    
    # Simulate single device array creation (what Orbax does internally)
    np_arr = np.ones((1024, 1024), dtype=ml_dtypes.float8_e4m3fn)
    
    # 1. device_put to single device
    try:
        single_device_arr = jax.device_put(np_arr, jax.devices()[0])
        print("single device put success")
    except Exception as e:
        print("single device put failed:", e)

    # 2. make_array_from_single_device_arrays
    try:
        res = jax.make_array_from_single_device_arrays((1024, 1024), sharding, [single_device_arr])
        print("make_array_from_single_device_arrays success")
    except Exception as e:
        print("make_array_from_single_device_arrays failed:", e)

