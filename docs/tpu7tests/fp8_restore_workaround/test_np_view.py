import numpy as np
import ml_dtypes

# Create a float8 array
a = np.ones((1024, 1024), dtype=ml_dtypes.float8_e4m3fn)
# See if view copies
b = np.asarray(a).view(np.uint8)
print("Shares memory?", np.shares_memory(a, b))

# Try from bytes (what tensorstore might return)
c = np.frombuffer(a.tobytes(), dtype=ml_dtypes.float8_e4m3fn)
d = np.asarray(c).view(np.uint8)
print("Shares memory (frombuffer)?", np.shares_memory(c, d))
