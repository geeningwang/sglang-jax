#!/usr/bin/env python3
"""Parse PD-TIME-STATS logs and produce prealloc overhead profiling report.

Usage:
  python scripts/disaggregation/profile_prealloc.py --log-file /tmp/pd_time_stats.log
  python scripts/disaggregation/profile_prealloc.py --log-file /tmp/decode_server.log --output-json /tmp/profile.json

Input: log file containing PD-TIME-STATS lines (from --enable-request-time-stats-logging).
Output: per-input-length statistics table + optional JSON dump.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DECODE_PHASES = [
    "bootstrap",
    "metadata_wait",
    "kv_alloc",
    "receiver_init",
    "transfer_setup",
    "prealloc_wait",
    "first_chunk_wait",
    "chunk_start_span",
    "transfer_tail",
    "enqueue_decode",
    "kv_wait",
    "total",
]

PREALLOC_PHASES = [
    "metadata_wait",
    "kv_alloc",
    "receiver_init",
    "transfer_setup",
]

DURATION_FIELDS = [
    "start_read_call",
    "chunk_handoff",
]

LINE_RE = re.compile(r"PD-TIME-STATS\s+role=(\w+)\s+req_id=(\S+)\s+(.*)")
FIELD_RE = re.compile(r"(\w+)=([\d.]+)ms")


def parse_log(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            role, req_id, body = m.groups()
            rec = {"role": role, "req_id": req_id}
            for fm in FIELD_RE.finditer(body):
                rec[fm.group(1)] = float(fm.group(2))
            records.append(rec)
    return records


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def infer_input_len(req_id: str) -> str | None:
    for part in req_id.split("-"):
        if part.isdigit() and int(part) in (512, 1024, 2048, 4096, 8192):
            return part
    return None


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": 0, "p50": 0, "p90": 0, "p99": 0, "min": 0, "max": 0, "mean": 0}
    return {
        "count": len(values),
        "median": percentile(values, 50),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def group_by_input_len(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        ilen = infer_input_len(rec["req_id"])
        key = ilen if ilen else "unknown"
        groups[key].append(rec)
    if len(groups) == 1 and "unknown" in groups:
        groups["all"] = groups.pop("unknown")
    return dict(groups)


def analyze(records: list[dict], phases: list[str]) -> dict:
    results = {}
    groups = group_by_input_len(records)

    all_group = defaultdict(list)
    for rec in records:
        for phase in phases:
            if phase in rec:
                all_group[phase].append(rec[phase])

    results["all"] = {phase: compute_stats(all_group.get(phase, [])) for phase in phases}

    for key in sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
        recs = groups[key]
        group_data = defaultdict(list)
        for rec in recs:
            for phase in phases:
                if phase in rec:
                    group_data[phase].append(rec[phase])
        results[key] = {phase: compute_stats(group_data.get(phase, [])) for phase in phases}

    for phase in DURATION_FIELDS:
        sum_key = f"{phase}_sum"
        avg_key = f"{phase}_avg"
        count_key = f"{phase}_count"
        for rec in records:
            if sum_key in rec:
                all_group[sum_key].append(rec[sum_key])
            if avg_key in rec:
                all_group[avg_key].append(rec[avg_key])

    return results


def print_table(results: dict, phases: list[str], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    header = f"{'input_len':>12} {'n':>5}"
    for phase in phases:
        header += f" {phase:>16}"
    print(header)
    print("-" * len(header))

    for key in results:
        row = results[key]
        n = max(row[p]["count"] for p in phases if p in row) if row else 0
        line = f"{key:>12} {n:>5}"
        for phase in phases:
            stats = row.get(phase, {})
            med = stats.get("median", 0)
            line += f" {med:>13.1f}ms"
        print(line)

    print()
    print("  Percentile details (p50 / p90 / p99):")
    print("-" * 80)
    for key in results:
        row = results[key]
        print(f"  input_len={key}:")
        for phase in phases:
            stats = row.get(phase, {})
            if stats.get("count", 0) == 0:
                continue
            print(
                f"    {phase:>20}: "
                f"p50={stats['p50']:.1f}ms  "
                f"p90={stats['p90']:.1f}ms  "
                f"p99={stats['p99']:.1f}ms  "
                f"min={stats['min']:.1f}ms  "
                f"max={stats['max']:.1f}ms"
            )


def main():
    parser = argparse.ArgumentParser(description="Profile PD prealloc overhead from PD-TIME-STATS logs")
    parser.add_argument("--log-file", required=True, help="Path to log file containing PD-TIME-STATS lines")
    parser.add_argument("--output-json", help="Path to write JSON results")
    parser.add_argument("--role", default="decode", choices=["decode", "prefill", "all"], help="Filter by role")
    args = parser.parse_args()

    records = parse_log(args.log_file)
    if not records:
        print(f"No PD-TIME-STATS lines found in {args.log_file}", file=sys.stderr)
        sys.exit(1)

    decode_records = [r for r in records if r["role"] == "decode"]
    prefill_records = [r for r in records if r["role"] == "prefill"]

    output = {}

    if args.role in ("decode", "all") and decode_records:
        decode_results = analyze(decode_records, DECODE_PHASES)
        prealloc_results = analyze(decode_records, PREALLOC_PHASES)
        print_table(prealloc_results, PREALLOC_PHASES, "DECODE PREALLOC OVERHEAD BREAKDOWN (median, ms)")
        print_table(decode_results, DECODE_PHASES, "DECODE FULL PHASE BREAKDOWN (median, ms)")
        output["decode"] = decode_results
        output["decode_prealloc"] = prealloc_results
        print(f"\nDecode records: {len(decode_records)}")

    if args.role in ("prefill", "all") and prefill_records:
        prefill_phases = [
            "queue", "forward", "stage",
            "first_chunk_register_wait", "chunk_register_span",
            "sender_done_wait", "transfer", "total",
        ]
        prefill_results = analyze(prefill_records, prefill_phases)
        print_table(prefill_results, prefill_phases, "PREFILL PHASE BREAKDOWN (median, ms)")
        output["prefill"] = prefill_results
        print(f"\nPrefill records: {len(prefill_records)}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"\nJSON results written to {args.output_json}")


if __name__ == "__main__":
    main()
