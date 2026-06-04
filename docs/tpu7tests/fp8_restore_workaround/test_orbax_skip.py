import jax
import orbax.checkpoint as ocp
import tempfile
import os
import numpy as np

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()
ckptr.save(path, {"w": jax.device_put(np.ones((10,), dtype=np.float32))})

args = ocp.args.PyTreeRestore(
    item={"w": jax.ShapeDtypeStruct((10,), np.float32)},
    restore_args={"w": ocp.args.ArrayRestore(restore_args=ocp.type_handlers.RestoreArgs(restore_type=jax.ShapeDtypeStruct))}
)
res = ckptr.restore(path, args=args)
print(type(res["w"]))
