import sys
sys.path.append("/opt/venv/lib/python3.12/site-packages")
import jax
from flax import nnx

class MyModel(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jax.ShapeDtypeStruct((10,), jax.numpy.float32))

model = MyModel()
print("Before:", type(model.w.value))

# Update with a valid state, but missing 'w'!
state = nnx.state(model)
state = {} # Empty state!

nnx.update(model, state)
print("After empty state:", type(model.w.value))

# What if state has 'w' but it's an array?
state = nnx.State({'w': jax.numpy.ones((10,))})
nnx.update(model, state)
print("After valid state:", type(model.w.value))
