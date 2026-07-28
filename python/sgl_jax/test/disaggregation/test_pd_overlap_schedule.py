import threading
from types import SimpleNamespace


def test_dispatch_uses_overlap_loop_for_pd_prefill_and_decode():
    from sgl_jax.srt.managers.scheduler import dispatch_scheduler_event_loop

    calls = []

    class FakeScheduler:
        enable_overlap = True

        def event_loop_overlap_disagg_prefill(self):
            calls.append("prefill_overlap")

        def event_loop_normal_disagg_prefill(self):
            calls.append("prefill_normal")

        def event_loop_overlap_disagg_decode(self):
            calls.append("decode_overlap")

        def event_loop_normal_disagg_decode(self):
            calls.append("decode_normal")

        def event_loop_overlap(self):
            calls.append("null_overlap")

        def event_loop_normal(self):
            calls.append("null_normal")

    dispatch_scheduler_event_loop(
        FakeScheduler(),
        SimpleNamespace(disaggregation_mode="prefill"),
    )
    dispatch_scheduler_event_loop(
        FakeScheduler(),
        SimpleNamespace(disaggregation_mode="decode"),
    )

    assert calls == ["prefill_overlap", "decode_overlap"]


def test_prefill_chunk_resolves_overlap_result_before_handoff():
    from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    calls = []
    launch_done = threading.Event()
    req = SimpleNamespace(bootstrap_room=1, rid="rid0", pd_time_stats=None)
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    scheduler = SimpleNamespace(
        enable_overlap=True,
        tp_worker=SimpleNamespace(
            resolve_last_batch_result=lambda event=None: calls.append(("resolve", event))
        ),
        disagg_kv_manager=SimpleNamespace(use_raiden=True),
        chunked_reqs=[None],
        set_next_batch_sampling_info_done=lambda batch: calls.append(("sampling_done", batch)),
        _pd_mark_time=lambda req, name, **kwargs: calls.append(("mark", name)),
        _raiden_handoff_chunk=lambda req, req_id, is_final: calls.append(
            ("handoff", req_id, is_final)
        ),
    )

    SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
        scheduler,
        batch,
        SimpleNamespace(),
        launch_done,
    )

    assert calls[0] == ("resolve", launch_done)
    assert ("handoff", "rid0", True) in calls


def test_prefill_chunk_uses_batch_chunked_snapshot_for_final_flag():
    from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    calls = []
    req = SimpleNamespace(bootstrap_room=1, rid="rid0", pd_time_stats=None)
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        _pd_chunked_reqs=(),
    )
    scheduler = SimpleNamespace(
        enable_overlap=False,
        disagg_kv_manager=SimpleNamespace(use_raiden=True),
        # Simulate the global scheduler state already moving to the next batch.
        # The current batch snapshot must win.
        chunked_reqs=[req],
        set_next_batch_sampling_info_done=lambda batch: None,
        _pd_mark_time=lambda req, name, **kwargs: None,
        _pd_add_duration=lambda req, name, seconds: None,
        _raiden_handoff_chunk=lambda req, req_id, is_final: calls.append(is_final),
    )

    SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
        scheduler,
        batch,
        SimpleNamespace(),
    )

    assert calls == [True]


def test_prefill_chunk_snapshot_keeps_mid_chunk_when_global_state_advances():
    from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    calls = []
    req = SimpleNamespace(bootstrap_room=1, rid="rid0", pd_time_stats=None)
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        _pd_chunked_reqs=(req,),
    )
    scheduler = SimpleNamespace(
        enable_overlap=False,
        disagg_kv_manager=SimpleNamespace(use_raiden=True),
        chunked_reqs=[None],
        set_next_batch_sampling_info_done=lambda batch: None,
        _pd_mark_time=lambda req, name, **kwargs: None,
        _pd_add_duration=lambda req, name, seconds: None,
        _raiden_handoff_chunk=lambda req, req_id, is_final: calls.append(is_final),
    )

    SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
        scheduler,
        batch,
        SimpleNamespace(),
    )

    assert calls == [False]


class _StopLoop(Exception):
    """Breaks out of the (otherwise infinite) event loop under test."""


def _make_decode_loop_scheduler(calls, *, max_iterations=3):
    """A fake scheduler that models forward-thread liveness.

    ``run_batch`` launches the forward thread and ``process_batch_result``
    drains it, mirroring the real overlap loop. Anything that runs a
    cross-host collective asserts the thread is idle -- see
    :func:`test_decode_overlap_never_runs_collective_while_forward_live`.
    """

    class FakeWatchdog:
        def start(self):
            pass

        def beat(self, label):
            pass

    class FakeBatch:
        def copy(self):
            return SimpleNamespace(next_batch_sampling_info=None)

    class FakeScheduler:
        disagg_decode_watchdog = FakeWatchdog()
        _comm_backend = None
        _engine_paused = False
        # Truthy, so the loop takes the drain path instead of the
        # DUMMY_FIRST branch (which would need real memory pools).
        last_batch = object()
        forward_live = False

        def __init__(self):
            self._iterations = 0
            self._rq = None

        # The loop assigns ``self.result_queue = deque()`` on entry. Seed
        # that deque with a stand-in for the previously-launched batch so
        # the first drain has something to pop.
        @property
        def result_queue(self):
            return self._rq

        @result_queue.setter
        def result_queue(self, value):
            value.append((SimpleNamespace(next_batch_sampling_info=None), SimpleNamespace()))
            self._rq = value

        def recv_requests(self):
            self._iterations += 1
            if self._iterations > max_iterations:
                raise _StopLoop
            return []

        def select_dp_for_request(self, recv_reqs):
            return recv_reqs

        def process_input_requests_disagg_decode(self, recv_reqs):
            calls.append("process_input")

        def _admit_decode_prealloc(self):
            calls.append("admit_prealloc")

        def _reap_completed_transfers(self):
            # Stands in for _drain_transfer_queue_synced (process_allgather)
            # and _write_kv_to_pool (cross-host jit).
            assert not self.forward_live, (
                "decode overlap loop ran a cross-host collective "
                "(_reap_completed_transfers) while the forward thread was "
                "live -- this is the SPMD program-order race that produces E0200"
            )
            calls.append("reap_transfers")

        def get_next_batch_to_run(self):
            return FakeBatch()

        def run_batch(self, batch):
            calls.append("run_batch")
            # The loop itself appends to result_queue; only model liveness.
            self.forward_live = True
            return SimpleNamespace()

        def _current_sampling_info_owner(self):
            return SimpleNamespace(cur_sampling_info=None)

        def process_batch_result(self, batch, result, launch_done=None):
            calls.append("process_batch_result")
            self.forward_live = False

    return FakeScheduler()


def test_decode_overlap_never_runs_collective_while_forward_live():
    """Regression guard for the E0200 SPMD race fixed in 3c301255.

    Asserts the invariant rather than a fixed call order, so it keeps
    holding as collective-free work is moved into the overlapped region.
    """
    from sgl_jax.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin

    calls = []
    try:
        SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode(
            _make_decode_loop_scheduler(calls)
        )
    except _StopLoop:
        pass

    # The assertion lives in the fake's _reap_completed_transfers; reaching
    # here having actually reaped proves the invariant was exercised.
    assert "reap_transfers" in calls
    assert "run_batch" in calls


def test_decode_overlap_drains_forward_thread_before_reaping():
    """Every reap must be preceded by a drain of the launch before it."""
    from sgl_jax.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin

    calls = []
    try:
        SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode(
            _make_decode_loop_scheduler(calls)
        )
    except _StopLoop:
        pass

    for index, name in enumerate(calls):
        if name != "reap_transfers":
            continue
        preceding = calls[:index]
        if "run_batch" not in preceding:
            continue  # nothing launched yet, nothing to drain
        last_launch = len(preceding) - 1 - preceding[::-1].index("run_batch")
        assert "process_batch_result" in calls[last_launch:index], (
            f"reap at position {index} was not preceded by a drain of the "
            f"launch at position {last_launch}"
        )


def test_process_decode_queue_runs_both_halves():
    """The wrapper stays intact for callers that do not overlap."""
    from sgl_jax.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin

    calls = []

    class FakeScheduler:
        def _admit_decode_prealloc(self):
            calls.append("admit")

        def _reap_completed_transfers(self):
            calls.append("reap")

    SchedulerDisaggregationDecodeMixin.process_decode_queue(FakeScheduler())

    assert calls == ["admit", "reap"]
