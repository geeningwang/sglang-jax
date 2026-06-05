"""HBM snapshot utility for TPU v7x memory profiling.

Usage — standalone:
    from sgl_jax.tools.hbm.snapshot import HBMTracker
    tracker = HBMTracker()
    tracker.snap("before load")
    load_model(...)
    tracker.snap("after load")
    tracker.report()

Usage — env-var gated (add to model_runner.py):
    if os.environ.get("SGLANG_HBM_TRACE"):
        tracker.snap("after nnx.eval_shape")
"""

import gc
import logging
import os

logger = logging.getLogger(__name__)


def _bytes_in_use() -> int:
    """Return minimum bytes_in_use across all local JAX TPU devices (per-TC)."""
    try:
        import jax
        gc.collect()
        stats = [d.memory_stats() for d in jax.local_devices()]
        return min(s["bytes_in_use"] for s in stats)
    except Exception:
        return 0


def _bytes_limit() -> int:
    """Return HBM limit per TC (should be 96 GB on TPU v7x)."""
    try:
        import jax
        stats = jax.local_devices()[0].memory_stats()
        return stats["bytes_limit"]
    except Exception:
        return 0


class HBMTracker:
    """Records HBM `bytes_in_use` at named checkpoints and reports deltas.

    Enable with env var SGLANG_HBM_TRACE=1. When disabled, all methods are
    no-ops so there is zero overhead in production.
    """

    def __init__(self, enabled: bool | None = None, prefix: str = "HBM"):
        if enabled is None:
            enabled = bool(os.environ.get("SGLANG_HBM_TRACE", ""))
        self.enabled = enabled
        self.prefix = prefix
        self._snaps: list[tuple[str, int]] = []  # (label, bytes_in_use)
        self._limit: int = 0
        if self.enabled:
            self._limit = _bytes_limit()

    def snap(self, label: str) -> int:
        """Record current bytes_in_use. Returns bytes_in_use (0 if disabled)."""
        if not self.enabled:
            return 0
        used = _bytes_in_use()
        self._snaps.append((label, used))
        prev = self._snaps[-2][1] if len(self._snaps) >= 2 else 0
        delta = used - prev
        sign = "+" if delta >= 0 else "-"
        logger.info(
            "[%s] %-40s  used=%6.2f GB  delta=%s%.2f GB  free=%6.2f GB",
            self.prefix,
            label,
            used / 1e9,
            sign,
            abs(delta) / 1e9,
            (self._limit - used) / 1e9,
        )
        return used

    def report(self) -> None:
        """Print a summary table of all snapshots."""
        if not self.enabled or not self._snaps:
            return
        print(f"\n{'='*70}")
        print(f"  HBM TIMELINE REPORT  (limit per TC: {self._limit/1e9:.1f} GB)")
        print(f"{'='*70}")
        print(f"  {'Checkpoint':<40} {'Used':>8} {'Delta':>9} {'Free':>8}")
        print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*8}")
        prev = 0
        for label, used in self._snaps:
            delta = used - prev
            free = self._limit - used
            sign = "+" if delta >= 0 else ""
            print(
                f"  {label:<40} {used/1e9:>7.2f}G {sign}{delta/1e9:>8.2f}G {free/1e9:>7.2f}G"
            )
            prev = used
        total_delta = self._snaps[-1][1] - self._snaps[0][1] if self._snaps else 0
        print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*8}")
        print(f"  {'TOTAL ALLOCATED':<40} {'':>8} {total_delta/1e9:>+9.2f}G")
        print(f"{'='*70}\n")


# Module-level singleton — imported by model_runner when SGLANG_HBM_TRACE=1
_global_tracker: HBMTracker | None = None


def get_tracker() -> HBMTracker:
    """Return the global tracker (creates it on first call)."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = HBMTracker()
    return _global_tracker


def snap(label: str) -> int:
    """Convenience function: snap on the global tracker."""
    return get_tracker().snap(label)
