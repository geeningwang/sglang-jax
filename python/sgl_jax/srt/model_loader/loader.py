import copy
import dataclasses
import glob
import hashlib
import logging
import os
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
        Format: {SGLANG_CHECKPOINT_DIR}/tp{tp_size}/{model_hash}/
        """
        checkpoint_dir = os.environ.get("SGLANG_CHECKPOINT_DIR", "")
        if not checkpoint_dir:
            return None
        model_path = model_config.model_path
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
        tp_size = self.mesh.size
        dtype = str(model_config.dtype)
        return f"{checkpoint_dir.rstrip('/')}/tp{tp_size}_{dtype}/{model_hash}/"

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

    def _save_checkpoint(self, model: nnx.Module, path: str) -> None:
        """Save model state as sharded Orbax checkpoint to `path`.

        Each JAX process writes only its local device shards concurrently.
        Expected size: same as source weights (~962 GB), split across 32 files.
        Expected time: ~5 min at NFS/GCS speeds.
        """
        logger.info("Saving checkpoint to %s (this takes ~5 min)...", path)
        checkpointer = ocp.PyTreeCheckpointer()
        state = nnx.state(model)
        checkpointer.save(path, state)
        logger.info("Checkpoint saved successfully to %s", path)

    def _load_checkpoint(self, model: nnx.Module, path: str) -> None:
        """Restore model state from Orbax checkpoint at `path`.

        Each JAX process reads only its local device shards (~30 GB per TC).
        Skips all CPU conversion (scale reshape, QKV fusion, etc.).
        Expected time: ~5 min (vs ~40 min for full weight loading).
        """
        logger.info("Loading from checkpoint %s (~5 min)...", path)
        checkpointer = ocp.PyTreeCheckpointer()
        abstract_state = nnx.state(model)
        state = checkpointer.restore(path, item=abstract_state)
        nnx.update(model, state)
        logger.info("Checkpoint loaded successfully from %s", path)

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

        checkpoint_path = self._checkpoint_path(model_config)

        if checkpoint_path and self._checkpoint_exists(checkpoint_path):
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
