import gc
import sys

class DummyArray:
    def __init__(self, size):
        self.size = size
    def __del__(self):
        print(f"Deleted array of size {self.size}")

def convert_inplace():
    # Simulate Orbax returning a tree
    state_u8 = {"a": DummyArray(1), "b": DummyArray(2)}
    
    # Flatten it
    import jax
    flat_u8, treedef = jax.tree_util.tree_flatten(state_u8)
    
    # CLEAR the original tree!
    state_u8.clear() # or if we can't clear, just delete it?
    del state_u8
    # NOTE: The caller MUST NOT have another reference to state_u8!
    
    flat_f8 = []
    for i in range(len(flat_u8)):
        u8_arr = flat_u8[i]
        # bitcast it
        f8_arr = DummyArray(u8_arr.size * 2) # simulate float8
        flat_f8.append(f8_arr)
        
        # CLEAR the reference from flat_u8
        flat_u8[i] = None
        # Now u8_arr should be garbage collected!
        print("After clearing flat_u8[i]")
        
    return jax.tree_util.tree_unflatten(treedef, flat_f8)

res = convert_inplace()
