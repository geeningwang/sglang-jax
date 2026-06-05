"""Live array attribution for HBM profiling.

Given two snapshots of jax.live_arrays(), compute which arrays were added,
removed, or changed — and group them by shape+dtype to identify allocators.

Usage:
    from sgl_jax.tools.hbm.attribution import live_snapshot, diff_snapshots, print_diff

    snap_before = live_snapshot()
    load_model(...)
    snap_after = live_snapshot()
    print_diff(snap_before, snap_after, top_n=20)
"""

import gc
from collections import defaultdict
from typing import NamedTuple


class ArrayInfo(NamedTuple):
    shape: tuple
    dtype: str
    nbytes: int


def live_snapshot() -> list[ArrayInfo]:
    """Return a list of ArrayInfo for all live JAX arrays on local devices."""
    import jax
    gc.collect()
    infos = []
    for arr in jax.live_arrays():
        try:
            nbytes = arr.nbytes
            infos.append(ArrayInfo(
                shape=tuple(arr.shape),
                dtype=str(arr.dtype),
                nbytes=nbytes,
            ))
        except Exception:
            pass
    return infos


def _group(infos: list[ArrayInfo]) -> dict[tuple, dict]:
    """Group ArrayInfo by (shape, dtype) → {count, total_bytes}."""
    groups: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "total_bytes": 0})
    for info in infos:
        key = (info.shape, info.dtype)
        groups[key]["count"] += 1
        groups[key]["total_bytes"] += info.nbytes
    return dict(groups)


def diff_snapshots(
    before: list[ArrayInfo],
    after: list[ArrayInfo],
) -> list[dict]:
    """Compute per-(shape,dtype) delta between two live_snapshot() results.

    Returns a list of dicts sorted by |delta_bytes| descending.
    """
    before_grp = _group(before)
    after_grp = _group(after)
    all_keys = set(before_grp) | set(after_grp)
    diffs = []
    for key in all_keys:
        b = before_grp.get(key, {"count": 0, "total_bytes": 0})
        a = after_grp.get(key, {"count": 0, "total_bytes": 0})
        delta_bytes = a["total_bytes"] - b["total_bytes"]
        delta_count = a["count"] - b["count"]
        if delta_bytes != 0 or delta_count != 0:
            shape, dtype = key
            diffs.append({
                "shape": shape,
                "dtype": dtype,
                "before_count": b["count"],
                "after_count": a["count"],
                "delta_count": delta_count,
                "before_bytes": b["total_bytes"],
                "after_bytes": a["total_bytes"],
                "delta_bytes": delta_bytes,
            })
    diffs.sort(key=lambda x: -abs(x["delta_bytes"]))
    return diffs


def print_diff(
    before: list[ArrayInfo],
    after: list[ArrayInfo],
    label: str = "",
    top_n: int = 20,
) -> None:
    """Print a human-readable diff of two live_snapshots."""
    diffs = diff_snapshots(before, after)
    total_before = sum(i.nbytes for i in before)
    total_after = sum(i.nbytes for i in after)
    delta_total = total_after - total_before

    header = f"  LIVE ARRAY DIFF{': ' + label if label else ''}"
    print(f"\n{'='*72}")
    print(header)
    print(
        f"  Before: {total_before/1e9:.2f} GB total  |  "
        f"After: {total_after/1e9:.2f} GB total  |  "
        f"Delta: {delta_total/1e9:+.2f} GB"
    )
    print(f"{'='*72}")
    print(f"  {'Shape':<30} {'DType':<16} {'ΔCount':>8} {'ΔBytes':>12}")
    print(f"  {'-'*30} {'-'*16} {'-'*8} {'-'*12}")

    shown = 0
    for d in diffs:
        if shown >= top_n:
            break
        shape_str = str(d["shape"])
        if len(shape_str) > 28:
            shape_str = shape_str[:25] + "..."
        delta_str = f"{d['delta_bytes']/1e9:+.3f} GB"
        count_str = f"{d['delta_count']:+d}"
        print(f"  {shape_str:<30} {d['dtype']:<16} {count_str:>8} {delta_str:>12}")
        shown += 1

    if len(diffs) > top_n:
        rest_bytes = sum(d["delta_bytes"] for d in diffs[top_n:])
        print(f"  ... {len(diffs) - top_n} more groups  ({rest_bytes/1e9:+.3f} GB)")
    print(f"{'='*72}\n")


def print_top(
    infos: list[ArrayInfo],
    label: str = "",
    top_n: int = 20,
) -> None:
    """Print the top-N largest array groups in a live_snapshot()."""
    groups = _group(infos)
    total = sum(i.nbytes for i in infos)
    sorted_groups = sorted(groups.items(), key=lambda x: -x[1]["total_bytes"])

    print(f"\n{'='*72}")
    print(f"  LIVE ARRAYS{': ' + label if label else ''}  (total: {total/1e9:.2f} GB)")
    print(f"{'='*72}")
    print(f"  {'Shape':<30} {'DType':<16} {'Count':>8} {'Total':>12}")
    print(f"  {'-'*30} {'-'*16} {'-'*8} {'-'*12}")
    shown_bytes = 0
    for (shape, dtype), g in sorted_groups[:top_n]:
        shape_str = str(shape)
        if len(shape_str) > 28:
            shape_str = shape_str[:25] + "..."
        print(
            f"  {shape_str:<30} {dtype:<16} {g['count']:>8} {g['total_bytes']/1e9:>11.3f} GB"
        )
        shown_bytes += g["total_bytes"]
    if len(sorted_groups) > top_n:
        rest = total - shown_bytes
        print(f"  ... {len(sorted_groups) - top_n} more groups  ({rest/1e9:.3f} GB)")
    print(f"{'='*72}\n")
