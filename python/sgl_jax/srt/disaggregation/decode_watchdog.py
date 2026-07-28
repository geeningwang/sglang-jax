"""Diagnostic watchdog for the decode-side scheduler event loop.

The decode event loop is single-threaded: it intakes requests (with a
synchronous bootstrap HTTP lookup), polls KV receivers, and runs the
decode forward pass in the same thread. If any of those phases blocks,
token generation for *all* running requests freezes and the orphan
reaper's FAILED verdicts are never consumed (the consumer runs on the
same blocked thread).

This watchdog runs on a separate daemon thread. The loop calls
:meth:`beat` at each phase boundary (cheap: a few field writes). When
the most recent beat is older than ``stall_threshold_s``, the watchdog
logs the stuck phase, a backlog snapshot, and the main thread's
traceback so a stress run pinpoints *which* phase / line is blocking.

The same :meth:`beat` calls double as a segment profiler. Each beat closes
the previous phase and accumulates its wall-clock duration, so a run can
report where a decode tick actually goes -- which is what tells us whether
a serialized segment shrank. That accounting is independent of stall
detection: ``beat`` always accumulates, and the periodic summary is opt-in
via ``disaggregation_decode_loop_profile_seconds``.

It is pure observability: opt-in via ``disaggregation_decode_watchdog_seconds``
and ``disaggregation_decode_loop_profile_seconds``, both off by default. It
does not abort, retry, or otherwise alter loop behavior.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress

logger = logging.getLogger(__name__)


class EventLoopWatchdog:
    """Detects a stalled event loop and profiles its per-phase segment costs."""

    def __init__(
        self,
        *,
        stall_threshold_s: float,
        check_interval_s: float = 1.0,
        profile_interval_s: float = 0.0,
        snapshot_provider: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        traceback_dumper: Callable[[], None] | None = None,
    ) -> None:
        self._stall_threshold_s = stall_threshold_s
        self._check_interval_s = check_interval_s
        self._profile_interval_s = profile_interval_s
        self._snapshot_provider = snapshot_provider
        self._clock = clock
        self._traceback_dumper = traceback_dumper or self._default_traceback_dumper
        self._phase = "init"
        self._beat_ts = clock()
        self._tick = 0
        # phase name -> [total_s, count, max_s]. Written only by the loop
        # thread in ``beat``; read (and ``max`` reset) by the reporter thread.
        # Deliberately unlocked: the worst a race can do is misattribute one
        # sample in a diagnostic line, which is not worth a lock on a hot
        # path that runs several times per decode tick.
        self._phase_stats: dict[str, list[float]] = {}
        self._profile_baseline: dict[str, tuple[float, int]] = {}
        self._profile_ts = clock()
        # ``_tick`` is frozen while the loop is stuck, so reporting only
        # when it differs from the last reported tick yields exactly one
        # report per distinct stall and re-arms once the loop advances.
        self._last_reported_tick = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stall_enabled(self) -> bool:
        return self._stall_threshold_s > 0

    @property
    def profile_enabled(self) -> bool:
        return self._profile_interval_s > 0

    @property
    def enabled(self) -> bool:
        return self.stall_enabled or self.profile_enabled

    def beat(self, phase: str) -> None:
        """Close the previous phase and enter ``phase``. Hot-path cheap.

        Accumulation is unconditional (a handful of arithmetic ops) so the
        profile is available whenever the reporter thread is running,
        without the loop having to know whether profiling is on.
        """

        now = self._clock()
        stat = self._phase_stats.get(self._phase)
        if stat is None:
            self._phase_stats[self._phase] = [now - self._beat_ts, 1, now - self._beat_ts]
        else:
            delta = now - self._beat_ts
            stat[0] += delta
            stat[1] += 1
            if delta > stat[2]:
                stat[2] = delta
        self._phase = phase
        self._tick += 1
        self._beat_ts = now

    def check_once(self, now: float | None = None) -> bool:
        """Single stall check. Returns True iff a stall was reported."""

        if not self.stall_enabled:
            return False
        now = self._clock() if now is None else now
        age = now - self._beat_ts
        if age < self._stall_threshold_s:
            return False
        if self._tick == self._last_reported_tick:
            return False
        self._report(phase=self._phase, age=age, tick=self._tick)
        self._last_reported_tick = self._tick
        return True

    def _report(self, *, phase: str, age: float, tick: int) -> None:
        snapshot = ""
        if self._snapshot_provider is not None:
            with suppress(Exception):
                snapshot = self._snapshot_provider()
        logger.warning(
            "PD-DECODE-WATCHDOG stall detected: phase=%s age=%.1fs "
            "tick=%d backlog=[%s]; dumping main-thread traceback",
            phase,
            age,
            tick,
            snapshot,
        )
        with suppress(Exception):
            self._traceback_dumper()

    def report_profile_once(self, now: float | None = None) -> bool:
        """Emit one window of per-phase segment costs. True iff emitted.

        Reports the delta since the previous window, so each line describes
        a bounded interval rather than an ever-flattening lifetime average.
        Phases are ordered by their share of the window -- the serialized
        segment you are trying to shrink sorts to the front.
        """

        if not self.profile_enabled:
            return False
        now = self._clock() if now is None else now
        window = now - self._profile_ts
        if window < self._profile_interval_s:
            return False
        self._profile_ts = now

        fields: list[tuple[float, str]] = []
        total_beats = 0
        # ``list()`` over the dict is atomic under the GIL; the phase set is
        # fixed after the first few ticks, so this never races meaningfully.
        for name in list(self._phase_stats):
            stat = self._phase_stats.get(name)
            if stat is None:
                continue
            cum_total, cum_count = stat[0], stat[1]
            peak = stat[2]
            stat[2] = 0.0  # window-scoped peak
            prev_total, prev_count = self._profile_baseline.get(name, (0.0, 0))
            self._profile_baseline[name] = (cum_total, cum_count)
            d_total = cum_total - prev_total
            d_count = cum_count - prev_count
            if d_count <= 0:
                continue
            total_beats += d_count
            fields.append(
                (
                    d_total,
                    f"{name}={d_total / d_count * 1000:.2f}ms/tick"
                    f"(n={d_count},tot={d_total * 1000:.0f}ms,max={peak * 1000:.1f}ms)",
                )
            )

        if not fields:
            return False
        fields.sort(key=lambda item: item[0], reverse=True)
        # ``beats`` is the total phase transitions in the window, i.e. loop
        # iterations times the number of distinct phases -- not a tick count.
        # Per-phase ``n`` is the one to read as "iterations".
        logger.info(
            "PD-DECODE-LOOP-PROFILE window=%.1fs beats=%d %s",
            window,
            total_beats,
            " ".join(text for _, text in fields),
        )
        return True

    @staticmethod
    def _default_traceback_dumper() -> None:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._beat_ts = self._clock()
        self._profile_ts = self._beat_ts
        self._thread = threading.Thread(
            target=self._loop,
            name="PD-DecodeEventLoopWatchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self._check_interval_s + 1.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            with suppress(Exception):
                self.check_once()
            with suppress(Exception):
                self.report_profile_once()
            self._stop.wait(self._check_interval_s)
