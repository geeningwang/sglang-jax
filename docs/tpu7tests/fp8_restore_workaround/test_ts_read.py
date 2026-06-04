import tensorstore as ts
import jax
import orbax.checkpoint as ocp
import tempfile
import os
import numpy as np
import ml_dtypes

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()
state = {"w": jax.device_put(np.ones((10, 10), dtype=ml_dtypes.float8_e4m3fn))}
ckptr.save(path, state)

# Get metadata
# Orbax actually uses `_get_json_tspec_read` but it's internal.
# Let's try to restore with np.ndarray!
restore_args = {"w": ocp.args.ArrayRestore(restore_type=np.ndarray)}
# Wait, in orbax 0.11/0.12, ArrayRestore doesn't take restore_type.
