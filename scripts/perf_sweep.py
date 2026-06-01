#!/usr/bin/env python3
"""
MiMo-V2.5-Pro performance benchmark sweep for sglang-jax on 4-node TPU v7x.

Phases:
  2: Concurrent request scaling  — vary N concurrent requests (fixed input=512, output=256)
  3: Prefill length sweep        — vary input tokens, output=1 (measures TTFT / prefill tok/s)
  4: Output length sweep         — vary max_tokens at optimal concurrency from Phase 2

Results are printed as tables and written to /tmp/perf_benchmark_results.json.

Usage (run inside rank0 pod after server is healthy):
  python3 /workspace/scripts/perf_sweep.py [--server http://localhost:8080] [--phase 0|2|3|4]
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


def _make_prompt(approx_tokens: int, idx: int = 0) -> str:
    """Build a prompt of approximately `approx_tokens` tokens (~3.5 chars/tok)."""
    base = (
        f"[req-{idx}] Explain in technical detail: transformer attention mechanisms, "
        "mixture-of-experts routing, tensor parallelism, expert parallelism, "
        "KV cache management, FP8 quantization, and XLA compilation for TPU inference. "
    )
    unit = "Provide equations, implementation details, and performance implications. "
    target = int(approx_tokens * 3.5)
    while len(base) < target:
        base += unit
    return base[:target]


async def _send_one(session: aiohttp.ClientSession, server: str, prompt: str, max_tokens: int):
    """Send one non-streaming chat completion. Returns (latency_s, output_tokens)."""
    payload = {
        "model": "MiMo-V2.5-Pro",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    async with session.post(
        f"{server}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as r:
        result = await r.json()
    latency = time.perf_counter() - t0
    usage = result.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)
    return latency, output_tokens


async def _run_batch(
    server: str,
    concurrency: int,
    input_tokens: int,
    max_tokens: int,
    n_requests: int,
) -> dict:
    """Drive `n_requests` requests with `concurrency` in-flight. Returns metrics dict."""
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    results = []

    async def _one(idx):
        prompt = _make_prompt(input_tokens, idx)
        async with sem:
            return await _send_one(session, server, prompt, max_tokens)

    async with aiohttp.ClientSession(connector=connector) as session:
        t_wall_start = time.perf_counter()
        tasks = [asyncio.create_task(_one(i)) for i in range(n_requests)]
        for coro in asyncio.as_completed(tasks):
            lat, out_toks = await coro
            results.append((lat, out_toks))
        t_wall = time.perf_counter() - t_wall_start

    latencies = sorted(r[0] for r in results)
    total_output = sum(r[1] for r in results)
    return {
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
        "n_requests": n_requests,
        "wall_s": round(t_wall, 2),
        "total_output_tokens": total_output,
        "decode_tok_per_s": round(total_output / t_wall, 2),
        "latency_p50_s": round(latencies[len(latencies) // 2], 3),
        "latency_p90_s": round(latencies[int(len(latencies) * 0.9)], 3),
        "latency_mean_s": round(statistics.mean(latencies), 3),
    }


async def main():
    ap = argparse.ArgumentParser(description="sglang-jax performance sweep")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--n-requests", type=int, default=20, help="requests per sweep step")
    ap.add_argument("--phase", type=int, default=0, help="0=all phases, 2/3/4=single phase")
    args = ap.parse_args()

    server = args.server
    N = args.n_requests
    all_results = {}

    # ── Phase 2: Concurrent request scaling ─────────────────────────────────
    if args.phase in (0, 2):
        print("\n" + "=" * 70)
        print("Phase 2: Concurrent request scaling  (input=512 tok, output=256 tok)")
        print("=" * 70)
        print(f"  {'conc':>4}  {'tok/s':>8}  {'lat_p50':>8}  {'lat_p90':>8}  {'vs_baseline':>12}  {'efficiency':>10}")
        print("  " + "-" * 60)

        phase2_rows = []
        baseline_tps = None
        for conc in [1, 2, 4, 8, 16, 32]:
            n = max(N, conc * 3)
            r = await _run_batch(server, conc, 512, 256, n)
            phase2_rows.append(r)
            if baseline_tps is None:
                baseline_tps = r["decode_tok_per_s"]
            vs = r["decode_tok_per_s"] / baseline_tps
            eff = r["decode_tok_per_s"] / (baseline_tps * conc) * 100
            print(
                f"  {conc:>4}  {r['decode_tok_per_s']:>8.1f}  "
                f"{r['latency_p50_s']:>8.3f}s  {r['latency_p90_s']:>8.3f}s  "
                f"{vs:>10.2f}x  {eff:>9.0f}%"
            )
        all_results["phase2"] = phase2_rows

    # ── Phase 3: Prefill length sweep ───────────────────────────────────────
    if args.phase in (0, 3):
        print("\n" + "=" * 70)
        print("Phase 3: Prefill length sweep  (output=1 tok, concurrency=1)")
        print("=" * 70)
        print(f"  {'in_tok':>7}  {'TTFT_p50':>10}  {'TTFT_p90':>10}  {'prefill_tok/s':>14}")
        print("  " + "-" * 48)

        phase3_rows = []
        for in_len in [128, 256, 512, 1024, 2048]:
            r = await _run_batch(server, 1, in_len, 1, 10)
            r["prefill_tok_per_s"] = round(in_len / r["latency_p50_s"], 1)
            phase3_rows.append(r)
            print(
                f"  {in_len:>7}  {r['latency_p50_s']:>10.3f}s  "
                f"{r['latency_p90_s']:>10.3f}s  {r['prefill_tok_per_s']:>14.1f}"
            )
        all_results["phase3"] = phase3_rows

    # ── Phase 4: Output length sweep ────────────────────────────────────────
    if args.phase in (0, 4):
        # Pick optimal concurrency from Phase 2, else default to 4
        opt_conc = 4
        if "phase2" in all_results:
            opt_conc = max(all_results["phase2"], key=lambda r: r["decode_tok_per_s"])["concurrency"]

        print("\n" + "=" * 70)
        print(f"Phase 4: Output length sweep  (input=512 tok, concurrency={opt_conc})")
        print("=" * 70)
        print(f"  {'out_tok':>7}  {'tok/s':>8}  {'lat_p50':>10}  {'lat_p90':>10}")
        print("  " + "-" * 42)

        phase4_rows = []
        for out_len in [64, 128, 256, 512]:
            n = max(N, opt_conc * 3)
            r = await _run_batch(server, opt_conc, 512, out_len, n)
            phase4_rows.append(r)
            print(
                f"  {out_len:>7}  {r['decode_tok_per_s']:>8.1f}  "
                f"{r['latency_p50_s']:>10.3f}s  {r['latency_p90_s']:>10.3f}s"
            )
        all_results["phase4"] = phase4_rows

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if "phase2" in all_results:
        best = max(all_results["phase2"], key=lambda r: r["decode_tok_per_s"])
        base = all_results["phase2"][0]["decode_tok_per_s"]
        print(f"  Peak decode throughput : {best['decode_tok_per_s']:.1f} tok/s  "
              f"at concurrency={best['concurrency']}  ({best['decode_tok_per_s']/base:.2f}x baseline)")
        print(f"  Baseline (conc=1)      : {base:.1f} tok/s")
    if "phase3" in all_results:
        best_pre = max(all_results["phase3"], key=lambda r: r["prefill_tok_per_s"])
        print(f"  Peak prefill throughput: {best_pre['prefill_tok_per_s']:.1f} tok/s  "
              f"at input={best_pre['input_tokens']} tokens  TTFT={best_pre['latency_p50_s']:.3f}s")

    out_path = "/tmp/perf_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
