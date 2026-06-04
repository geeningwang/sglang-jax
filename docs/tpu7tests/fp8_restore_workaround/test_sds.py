import jax
from jax.sharding import PartitionSpec

sds = jax.ShapeDtypeStruct((10, 10), jax.numpy.float32, sharding=PartitionSpec('x'))
print(sds)
