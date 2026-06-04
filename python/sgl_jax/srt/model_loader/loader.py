import copy
import dataclasses
import glob
import hashlib
import io
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import huggingface_hub
import jax
import orbax.checkpoint as ocp
from flax import nnx
from safetensors import safe_open

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.debug_utils import print_parameter_shardings

logger = logging.getLogger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        """Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model_path,
            model_config.revision,
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        pass

    def _maybe_download_from_modelscope(self, model: str, revision: str | None) -> str | None:
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return None

    def _prepare_weights(
        self, model_name_or_path: str, revision: str | None
    ) -> tuple[str, list[str]]:
        model_path = self._maybe_download_from_modelscope(model_name_or_path, revision)
        if model_path is not None:
            model_name_or_path = model_path

        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            hf_folder = model_name_or_path
        else:
            from huggingface_hub import snapshot_download

            hf_folder = snapshot_download(
                model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                cache_dir=self.load_config.download_dir,
                tqdm_class=None,
                revision=revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

        return hf_folder

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, jax.Array], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder = self._prepare_weights(source.model_or_path, source.revision)
        weights_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {hf_folder}")
        weights_files.sort()
        platform = os.getenv("JAX_PLATFORMS", None)
        backend = "cpu" if platform != "proxy" else "proxy"
        for st_file in weights_files:
            with (
                jax.default_device(jax.local_devices(backend=backend)[0]),
                safe_open(st_file, framework="flax") as f,
            ):
                for name in list(f.keys()):
                    yield source.prefix + name, f.get_tensor(name)


class JAXModelLoader(DefaultModelLoader):
    @dataclasses.dataclass
    class JAXSource:
        model_or_path: str
        revision: str | None

        @classmethod
        def init_new(cls, model_config: ModelConfig):
            return cls(
                model_config.model_path,
                model_config.revision,
            )

    def __init__(self, load_config: LoadConfig, mesh: jax.sharding.Mesh):
        super().__init__(load_config)
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> str:
        source = self.JAXSource.init_new(model_config)
        hf_folder = self._prepare_weights(source.model_or_path, source.revision)
        return hf_folder

    def load_model(
        self,
        model_config: ModelConfig,
    ) -> Any:
        # prepare model file
        hf_folder = self.download_model(model_config)

        # if sub_dir is specified, use it
        if self.load_config.sub_dir is not None:
            hf_folder = os.path.join(hf_folder, self.load_config.sub_dir)
            model_config = copy.copy(model_config)

        model_config.model_path = hf_folder
        # Initialize JAX model
        model = self._initialize_model(model_config)

        # Load weights
        jit_model = self._get_model(model, model_config)

        return jit_model

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        if not isinstance(model_config, ModelConfig) or self.load_config.model_class is not None:
            model_class = (
                model_config.model_class
                if hasattr(model_config, "model_class") and model_config.model_class is not None
                else self.load_config.model_class
            )
        else:
            model_class, _ = get_model_architecture(model_config)

        if not hasattr(model_class, "load_weights"):
            raise ValueError(
                f"Model class {model_class.__name__} does not support weights loading. "
                "Please ensure you're using a JAX-compatible model and implement load_weights method."
            )

        return model_class

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _checkpoint_path(self, model_config: ModelConfig) -> str | None:
        """Return GCS checkpoint path for this model/tp-size, or None if disabled.

        Auto-derives from model_path when SGLANG_CHECKPOINT_DIR is set.
        Format: {SGLANG_CHECKPOINT_DIR}/{model_hash}/tp{tp_size}_{dtype}/
        Example: gs://.../sglang-checkpoint/95dc2640/tp32_bfloat16/
        """
        checkpoint_dir = os.environ.get("SGLANG_CHECKPOINT_DIR", "")
        if not checkpoint_dir:
            return None
        model_path = model_config.model_path
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
        tp_size = self.mesh.size
        # Use dtype __name__ (e.g. "bfloat16") instead of str() which gives
        # the ugly "<class 'jax.numpy.bfloat16'>" representation.
        dtype_name = getattr(model_config.dtype, "__name__", None) or \
                     getattr(model_config.dtype, "name", None) or \
                     str(model_config.dtype).split(".")[-1].strip("'>")
        return f"{checkpoint_dir.rstrip('/')}/{model_hash}/tp{tp_size}_{dtype_name}/"

    def _checkpoint_exists(self, path: str) -> bool:
        """Check whether a saved checkpoint exists at `path`.

        Orbax writes `commit_success.txt` as the final step of a successful save.
        """
        try:
            if path.startswith("gs://"):
                import subprocess
                result = subprocess.run(
                    ["gsutil", "-q", "stat", f"{path}commit_success.txt"],
                    capture_output=True,
                )
                return result.returncode == 0
            else:
                import pathlib
                return (pathlib.Path(path) / "commit_success.txt").exists()
        except Exception:
            return False

    def _abstract_state_path(self, checkpoint_path: str) -> str:
        """Path for the pickled abstract state structure (small metadata file)."""
        return checkpoint_path.rstrip("/") + "_abstract_state.pkl"

    def _save_abstract_state(self, state: Any, path: str) -> None:
        """Pickle the abstract state (shapes/dtypes/shardings, no values) to GCS or local.

        Sharding info is included so Orbax can correctly restore sharded arrays
        (including FP8 dtypes) without falling back to the sharding file.
        """
        def _to_sds(x):
            # Include sharding spec (PartitionSpec/NamedSharding) so Orbax can
            # restore sharded FP8 arrays correctly. Use sharding.spec if available
            # to avoid pickling non-serializable Device objects inside NamedSharding.
            sharding = getattr(x, "sharding", None)
            if sharding is not None:
                try:
                    # NamedSharding.spec is a PartitionSpec — picklable
                    sharding = sharding.spec
                except AttributeError:
                    sharding = None
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding)

        abstract = jax.tree_util.tree_map(_to_sds, state)
        buf = pickle.dumps(abstract)
        if path.startswith("gs://"):
            import subprocess
            proc = subprocess.run(["gsutil", "cp", "-", path], input=buf, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(f"gsutil cp failed: {proc.stderr.decode()}")
        else:
            import pathlib
            pathlib.Path(path).write_bytes(buf)

    def _load_abstract_state(self, path: str) -> Any:
        """Load the pickled abstract state from GCS or local."""
        if path.startswith("gs://"):
            import subprocess
            proc = subprocess.run(["gsutil", "cp", path, "-"], capture_output=True)
            if proc.returncode != 0:
                raise FileNotFoundError(f"Abstract state not found at {path}")
            return pickle.loads(proc.stdout)
        else:
            import pathlib
            return pickle.loads(pathlib.Path(path).read_bytes())

    def _save_checkpoint(self, model: nnx.Module, path: str) -> None:
        """Save model state as sharded Orbax checkpoint to `path`.

        Saves FP8 arrays directly (requires Orbax 0.12+ for restore).
        The abstract state pickle records dtype+sharding for restore.

        Each JAX process writes only its local device shards concurrently.
        Expected time: ~5 min at NFS/GCS speeds.
        """
        logger.info("Saving checkpoint to %s (this takes ~5 min)...", path)
        checkpointer = ocp.PyTreeCheckpointer()
        state = nnx.state(model)
        try:
            self._save_abstract_state(state, self._abstract_state_path(path))
        except Exception as e:
            logger.warning("Could not save abstract state (non-fatal): %s", e)
        checkpointer.save(path, state)
        logger.info("Checkpoint saved successfully to %s", path)

    def _load_checkpoint(self, model: nnx.Module, path: str) -> None:
        """Restore model state from Orbax checkpoint at `path`.

        Requires Orbax 0.12+ for native FP8 (float8_e4m3fn) array restore.

        Each JAX process reads only its local device shards (~30 GB per TC).
        Expected time: ~90s (vs ~40 min for full weight loading).
        """
        logger.info("Loading from checkpoint %s (~2 min)...", path)
        checkpointer = ocp.PyTreeCheckpointer()
        try:
            with jax.set_mesh(self.mesh):
                abstract_state = self._load_abstract_state(self._abstract_state_path(path))
            logger.info("Restored abstract state structure from checkpoint metadata.")
        except Exception as e:
            logger.warning("Abstract state not found, falling back to model state (%s)", e)
            abstract_state = nnx.state(model)
            
        # WORKAROUND for TPU v7x libtpu float8_e4m3fn host-to-device allocation bug.
        # jax.device_put fails silently or returns ShapeDtypeStruct for float8 arrays on TPU.
        # We monkey-patch jax.device_put to transfer as uint8 and bitcast on device.
        import numpy as np
        import jax.numpy as jnp
        orig_device_put = jax.device_put
        def patched_device_put(x, *args, **kwargs):
            if hasattr(x, "dtype") and str(x.dtype) in {"float8_e4m3fn", "float8_e5m2"}:
                x_u8 = np.asarray(x).view(np.uint8)
                arr_u8 = orig_device_put(x_u8, *args, **kwargs)
                target_dtype = getattr(jnp, str(x.dtype))
                arr_f8 = jax.lax.bitcast_convert_type(arr_u8, target_dtype)
                
                # Block to throttle the async queue and ensure XLA finishes
                # so that arr_u8 can be immediately freed from HBM and Host RAM
                # doesn't OOM from unbounded compilation/DMA queues.
                arr_f8.block_until_ready()
                del x_u8
                del arr_u8
                return arr_f8
            return orig_device_put(x, *args, **kwargs)
        
        jax.device_put = patched_device_put
        try:
            state = checkpointer.restore(path, item=abstract_state)
        finally:
            jax.device_put = orig_device_put

        nnx.update(model, state)
        self._patch_narrow_blockwise(model)
        logger.info("Checkpoint loaded successfully from %s", path)

    @staticmethod
    def _patch_narrow_blockwise(model: nnx.Module) -> None:
        """Set allow_narrow_n_blockwise=True on every FP8 linear layer in the model.

        Called after checkpoint restore because apply_linear_quantization() creates
        layers with allow_narrow_n_blockwise=False, but load_weights() (used during
        the original checkpoint save) set it True for narrow-output layers.
        """
        patched = 0
        for _, module in model.iter_modules():
            if hasattr(module, "allow_narrow_n_blockwise") and not module.allow_narrow_n_blockwise:
                module.allow_narrow_n_blockwise = True
                patched += 1
        if patched:
            logger.info("Checkpoint restore: patched allow_narrow_n_blockwise=True on %d layers.", patched)

    # ── Model initialization ──────────────────────────────────────────────────

    def _get_model(self, model_class: Any, model_config: ModelConfig) -> nnx.Module:
        if not isinstance(model_config, ModelConfig):
            config = model_config
        else:
            config = model_config.hf_config

        with jax.set_mesh(self.mesh):
            model = nnx.eval_shape(
                lambda: model_class(config, dtype=model_config.dtype, mesh=self.mesh)
            )

        checkpoint_path = self._checkpoint_path(model_config)
        checkpoint_ready = checkpoint_path and self._checkpoint_exists(checkpoint_path)

        # Quantization config is already unified in model_config
        # No need for any conversion logic here
        if (
            hasattr(model_config, "quantization_config")
            and model_config.quantization_config is not None
        ):
            is_static = model_config.quantization_config.is_static_checkpoint

            if is_static:
                logger.info("Applying STATIC quantization structure preparation...")
                from sgl_jax.srt.utils.quantization.quantization_utils import (
                    apply_linear_quantization,
                    apply_moe_quantization,
                )

                if model_config.quantization_config.has_moe_quantization():
                    model = apply_moe_quantization(model_config, model, is_static_input=True)

                if model_config.quantization_config.get_linear_rules():
                    model = apply_linear_quantization(model_config, model, is_static_input=True)
            else:
                logger.info("Dynamic quantization detected. Skipping structure change in loader.")
        else:
            logger.info("No quantization config found. Skipping quantization.")

        if checkpoint_ready:
            # Fast path: load pre-converted sharded checkpoint (~5 min)
            self._load_checkpoint(model, checkpoint_path)
        else:
            # Slow path: load from raw weights (~40 min), then save checkpoint
            model.load_weights(model_config)
            if checkpoint_path:
                try:
                    self._save_checkpoint(model, checkpoint_path)
                except Exception as e:
                    logger.warning("Checkpoint save failed (non-fatal): %s", e)

        print_parameter_shardings(model)

        return model


class JAXDummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values for JAX models."""

    def __init__(self, load_config: LoadConfig, mesh: jax.sharding.Mesh):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> None:
        # Nothing to download for dummy loader
        return None

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        # Do not require a load_weights method for dummy loader
        model_class, _ = get_model_architecture(model_config)
        return model_class

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        model_class = self._initialize_model(model_config)

        with jax.set_mesh(self.mesh):
            model = nnx.eval_shape(
                lambda: model_class(
                    model_config.hf_config, dtype=model_config.dtype, mesh=self.mesh
                )
            )

        # Use model's load_weights with dummy mode to ensure correct sharding
        # Set a marker in model_config to indicate dummy mode
        model_config._dummy_mode = True
        model.load_weights(model_config)

        return model


def get_model_loader(load_config: LoadConfig, mesh: jax.sharding.Mesh) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return JAXDummyModelLoader(load_config, mesh)

    if load_config.load_format == LoadFormat.JAX:
        return JAXModelLoader(load_config, mesh)

    return JAXModelLoader(load_config, mesh)
