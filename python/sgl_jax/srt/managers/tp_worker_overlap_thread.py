"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class ModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        model_class=None,
        precompile_params: dict | None = None,
    ):
        # Load the model
        self.worker = ModelWorker(server_args, mesh=mesh)
        # overlap mode set worker need_prepare_lora_batch to False
        self.worker.need_prepare_lora_batch = False

        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32)
        self.mesh = mesh
        sharding = NamedSharding(mesh, PartitionSpec(None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding)
        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        self.async_gather_fn = jax.jit(lambda x: x, out_shardings=replicated_sharding)

    def get_model_runner(self):
        return self.worker.get_model_runner()

    @property
    def model_config(self):
        return self.worker.model_config

    @property
    def model_runner(self):
        return self.worker.model_runner

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def get_max_padded_size(self):
        return self.worker.get_max_padded_size()

    def get_precompile_paddings(self):
        return self.worker.get_precompile_paddings()

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            (
                model_worker_batch,
                future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            ) = self.input_queue.get()
            if not model_worker_batch:
                break

            # Per-batch build, moved off the scheduler thread (issue 323).
            # sampling_metadata may be pre-built by a non-overlap caller; the
            # rest is always built here now.
            if sampling_metadata is None:
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch,
                    0,
                    self.mesh,
                    self.worker.model_config.vocab_size,
                )
            if forward_metadata is None:
                forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
                    model_worker_batch
                )
            if self.worker.server_args.enable_lora:
                self.worker.prepare_lora_batch(model_worker_batch)
            if getattr(model_worker_batch, "forward_batch", None) is None:
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.worker.get_model_runner()
                )

            # Resolve future tokens in the input
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map, self.mesh
            )

            # Run forward
            with jax.profiler.TraceAnnotation(f"forward_batch_generation {model_worker_batch.bid}"):
                logits_output, next_token_ids, cache_miss_count = (
                    self.worker.forward_batch_generation(
                        model_worker_batch,
                        model_worker_batch.launch_done,
                        sampling_metadata=sampling_metadata,
                        forward_metadata=forward_metadata,
                    )
                )
            next_token_ids = self.async_gather_fn(next_token_ids)
            # Update the future token ids map
            self.future_token_ids_map = set_future_token_ids(
                self.future_token_ids_map,
                future_token_ids_ct,
                next_token_ids,
                self.mesh,
            )
            self.output_queue.put((None, logits_output, next_token_ids, cache_miss_count))

    def resolve_last_batch_result(
        self, launch_done: threading.Event | None = None, watchdog=None
    ):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.

        Uses jax.copy_to_host_async to start all device-to-host copies in
        parallel, then materializes them. This lets the four arrays we need
        overlap on PCIe rather than serializing the per-array sync that
        jax.device_get does.

        ``watchdog`` (the decode loop's EventLoopWatchdog, or None) splits the
        segment in two: the ``output_queue.get()`` above closes the caller's
        ``resolve_result`` phase (pure forward-pass/TPU wait), and everything
        after it is reattributed to ``resolve_d2h`` (the device-to-host copies).
        This answers whether the ~21 ms tick is compute-bound or PCIe-bound
        (issue 323). Called on the scheduler thread that owns the watchdog, so
        beating here is thread-safe; a None watchdog makes it a no-op.
        """
        _, logits_output, next_token_ids, cache_miss_count = self.output_queue.get()
        if watchdog is not None:
            watchdog.beat("resolve_d2h")
        # Step 1: kick off async D2H copies for everything we need
        async_next_logprobs = (
            jax.copy_to_host_async(logits_output.next_token_logprobs)
            if logits_output.next_token_logprobs is not None
            else None
        )
        async_input_logprobs = (
            jax.copy_to_host_async(logits_output.input_token_logprobs)
            if logits_output.input_token_logprobs is not None
            else None
        )
        async_hidden_states = (
            jax.copy_to_host_async(logits_output.hidden_states)
            if logits_output.hidden_states is not None
            else None
        )
        next_token_ids = jax.device_get(next_token_ids).tolist()

        # Step 2: materialize. The first np.asarray waits for that array's
        # copy; the others have been making progress in parallel.
        if async_next_logprobs is not None:
            logits_output.next_token_logprobs = np.asarray(async_next_logprobs).tolist()
        if async_input_logprobs is not None:
            logits_output.input_token_logprobs = np.asarray(async_input_logprobs).tolist()
        if async_hidden_states is not None:
            logits_output.hidden_states = np.asarray(async_hidden_states)

        if launch_done is not None:
            launch_done.wait()

        return logits_output, next_token_ids, cache_miss_count

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata = None,
    ) -> tuple[None, jax.Array, int]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        # The expensive per-batch build -- SamplingMetadata, attention
        # forward_metadata, LoRA prep, and ForwardBatch.init_new (the input
        # device_puts) -- used to run here on the scheduler thread while the
        # forward thread sat idle. It is the ~2.8ms "run_batch" segment in
        # PD-DECODE-LOOP-PROFILE. Defer it to forward_thread_func_ so this call
        # is a cheap enqueue and the build overlaps the forward thread's
        # otherwise-idle resolve window (issue 323). It stays SPMD-safe because
        # the decode loop still drains (resolve_last_batch_result) before any
        # process_allgather, so the build never runs concurrently with a
        # collective -- only inside the wait for the previous forward pass.
        #
        # sampling_info / cur_sampling_info stay on this thread: the decode loop
        # reads cur_sampling_info right after run_batch to fire
        # sampling_info_done, and moving it would race that handshake. A caller
        # that pre-builds sampling_metadata (non-overlap paths) is still honored
        # -- forward_thread_func_ only builds what arrives as None.
        self.input_queue.put(
            (
                model_worker_batch,
                self.future_token_ids_ct,
                sampling_metadata,
                None,
            )
        )

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)

        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (self.future_token_ids_ct + bs) % self.future_token_ids_limit
        return None, future_next_token_ids, 0

    def run_precompile(self):
        self.worker.run_precompile(self.future_token_ids_map)

    @property
    def page_size(self) -> int:
        return self.worker.page_size

    @property
    def sliding_window_size(self) -> int | None:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()

    def __delete__(self):
        self.input_queue.put((None, None, None, None))
