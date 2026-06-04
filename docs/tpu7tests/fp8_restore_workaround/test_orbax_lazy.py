import jax
import numpy as np
import orbax.checkpoint as ocp
import tempfile
import os
import shutil

base = tempfile.mkdtemp()
path = os.path.join(base, "ckpt")
ckptr = ocp.PyTreeCheckpointer()

state = {
    "a": np.ones((10,), dtype=np.uint8),
    "b": np.ones((10,), dtype=np.uint8)
}
ckptr.save(path, state)

restore_args = {
    "a": ocp.args.ArrayRestore(restore_args=ocp.type_handlers.RestoreArgs(restore_type=jax.ShapeDtypeStruct)),
    "b": ocp.args.ArrayRestore(restore_args=ocp.type_handlers.RestoreArgs(restore_type=np.ndarray))
}
args = ocp.args.PyTreeRestore(item=state, restore_args=restore_args)

restored = ckptr.restore(path, args=args)
print("a:", type(restored["a"]))
print("b:", type(restored["b"]))
shutil.rmtree(base)
