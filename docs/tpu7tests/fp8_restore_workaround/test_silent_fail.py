import jax
import numpy as np

sds = jax.ShapeDtypeStruct((10, 10), np.float32)
sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec('x'))

res = jax.make_array_from_single_device_arrays((10, 10), sharding, [sds])
print(type(res))
