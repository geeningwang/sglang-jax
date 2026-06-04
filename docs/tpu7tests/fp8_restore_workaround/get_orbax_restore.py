import inspect
import orbax.checkpoint.type_handlers as th

print(inspect.getsource(th.ArrayHandler.restore))
