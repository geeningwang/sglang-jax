import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeCheckpointHandler
import numpy as np

sds = jax.ShapeDtypeStruct((10,), np.float32)
# We want to see what RestoreArgs are inferred for item={"w": sds}
# ocp.args.PyTreeRestore.restore_args
args = ocp.args.PyTreeRestore(item={"w": sds})
print(args.restore_args["w"])
