#!/usr/bin/env python3
"""
MiMo-V2-Flash sglang-jax baseline performance sweep.

Measures decode throughput, TTFT, and TPOT across concurrency and length
dimensions. Run inside the server pod after the server is healthy.

Phases:
  1  Warmup              — a few requests to ensure JIT is stable
  2  Concurrency sweep   — vary concurrent requests (input=512, output=256)
  3  Prefill sweep       — vary input length, output=1 (TTFT / prefill tok/s)
  4  Output length sweep — vary max_tokens at optimal concurrency from Phase 2

Usage:
  python3 scripts/perf_sweep_flash.py [--server http://localhost:8080] \\
      [--phase 0|1|2|3|4] [--n-requests 20] [--model MiMo-V2-Flash] \\
      [--result-path /tmp/flash_baseline.json]
"""

import argparse
import asyncio
import json
import statistics
import sys
import time


try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)


MODEL = "MiMo-V2-Flash"

# Sweep configurations
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
PREFILL_LENGTHS    = [128, 256, 512, 1024, 2048, 4096]
OUTPUT_LENGTHS     = [64, 128, 256, 512, 1024]
WARMUP_REQUESTS    = 4


def _make_prompt(approx_tokens: int, idx: int = 0) -> str:
    """Build a prompt of approximately `approx_tokens` tokens (~3.5 chars/tok)."""
    base = (
        f"[req-{idx}] Explain in technical detail: transformer attention, "
        "mixture-of-experts routing, tensor parallelism, KV cache management, "
        "FP8 quantization, sliding window attention, and TPU v7x architecture. "
    )
    unit = "Include equations, implementation details, and performance analysis. "
    target = max(10, int(approx_tokens * 3.5))
    while len(base) < target:
        base += unit
    return base[:target]


async def _send_one(
    session: aiohttp.ClientSession,
    server: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Send one non-streaming chat completion. Returns timing + token counts."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    async with session.post(
        f"{server}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as r:
        result = await r.json()
    e2e_s = time.perf_counter() - t0
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "e2e_s": e2e_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def _run_batch(
    server: str,
    model: str,
    concurrency: int,
    input_tokens: int,
    max_tokens: int,
    n_requests: int,
) -> dict:
    """Drive n_requests with up to concurrency in-flight. Return metrics dict."""
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    results = []

    async def _one(idx: int):
        prompt = _make_prompt(input_tokens, idx)
        async with sem:
            return await _send_one(session, server, model, prompt, max_tokens)

    async with aiohttp.ClientSession(connector=connector) as session:
        t_wall_start = time.perf_counter()
        tasks = [asyncio.create_task(_one(i)) for i in range(n_requests)]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
        t_wall_s = time.perf_counter() - t_wall_start

    e2e_latencies = sorted(r["e2e_s"] for r in results)
    total_output = sum(r["completion_tokens"] for r in results)
    total_input  = sum(r["prompt_tokens"] for r in results)
    n = len(results)
    p50 = e2e_latencies[n // 2]
    p90 = e2e_latencies[int(n * 0.9)]
    p99 = e2e_latencies[min(int(n * 0.99), n - 1)]

    decode_tok_s = total_output / t_wall_s if t_wall_s > 0 else 0
    # TPOT: mean time per output token per request
    tpot_ms = statistics.mean(
        r["e2e_s"] / max(r["completion_tokens"], 1) * 1000 for r in results
    )
    # Prefill tok/s (TTFT proxy at output=1)
    prefill_tok_s = total_input / t_wall_s if t_wall_s > 0 else 0

    return {
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
        "n_requests": n_requests,
        "wall_s": round(t_wall_s, 3),
        "total_output_tokens": total_output,
        "decode_tok_per_s": round(decode_tok_s, 2),
        "prefill_tok_per_s": round(prefill_tok_s, 2),
        "tpot_ms": round(tpot_ms, 2),
        "e2e_latency_p50_s": round(p50, 3),
        "e2e_latency_p90_s": round(p90, 3),
        "e2e_latency_p99_s": round(p99, 3),
        "e2e_latency_mean_s": round(statistics.mean(e2e_latencies), 3),
    }


async def main():
    ap = argparse.ArgumentParser(description="MiMo-V2-Flash sglang-jax perf sweep")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n-requests", type=int, default=20,
                    help="Requests per sweep step (min=concurrency×3 for concurrent phases)")
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all phases, 1=warmup only, 2=concurrency, 3=prefill, 4=output-len")
    ap.add_argument("--result-path", default="/tmp/flash_baseline.json",
                    help="Write JSON results to this path")
    args = ap.parse_args()

    server  = args.server
    model   = args.model
    N       = args.n_requests
    all_results: dict = {
        "model": model,
        "server": server,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── Phase 1: Warmup ─────────────────────────────────────────────────────
    if args.phase in (0, 1):
        print("\n" + "=" * 70)
        print(f"Phase 1: Warmup  ({WARMUP_REQUESTS} requests, input=128, output=32)")
        print("=" * 70)
        r = await _run_batch(server, model, 1, 128, 32, WARMUP_REQUESTS)
        print(f"  Warmup done — {r['decode_tok_per_s']:.1f} tok/s  "
              f"e2e_p50={r['e2e_latency_p50_s']:.3f}s")
        all_results["warmup"] = r

    # ── Phase 2: Concurrency sweep ───────────────────────────────────────────
    if args.phase in (0, 2):
        print("\n" + "=" * 70)
        print("Phase 2: Concurrency sweep  (input=512 tok, output=256 tok)")
        print("=" * 70)
        print(f"  {'conc':>5}  {'tok/s':>8}  {'TPOT_ms':>8}  "
              f"{'e2e_p50':>8}  {'e2e_p90':>8}  {'vs_c1':>7}  {'efficiency':>10}")
        print("  " + "-" * 66)

        phase2_rows = []
        baseline_tps = None
        for conc in CONCURRENCY_LEVELS:
            n = max(N, conc * 3)
            r = await _run_batch(server, model, conc, 512, 256, n)
            phase2_rows.append(r)
            if baseline_tps is None:
                baseline_tps = r["decode_tok_per_s"] or 1.0
            vs = r["decode_tok_per_s"] / baseline_tps
            eff = vs / conc * 100
            print(
                f"  {conc:>5}  {r['decode_tok_per_s']:>8.1f}  "
                f"{r['tpot_ms']:>8.1f}  "
                f"{r['e2e_latency_p50_s']:>8.3f}s  {r['e2e_latency_p90_s']:>8.3f}s  "
                f"{vs:>6.2f}x  {eff:>9.0f}%"
            )
        all_results["phase2_concurrency"] = phase2_rows

    # ── Phase 3: Prefill length sweep ───────────────────────────────────────
    if args.phase in (0, 3):
        print("\n" + "=" * 70)
        print("Phase 3: Prefill length sweep  (output=1 tok, concurrency=1)")
        print("=" * 70)
        print(f"  {'in_tok':>7}  {'TTFT_p50':>10}  {'TTFT_p90':>10}  {'prefill_tok/s':>14}")
        print("  " + "-" * 50)

        phase3_rows = []
        for in_len in PREFILL_LENGTHS:
            r = await _run_batch(server, model, 1, in_len, 1, max(10, N // 2))
            # TTFT = e2e latency when output=1
            r["ttft_p50_s"] = r["e2e_latency_p50_s"]
            r["ttft_p90_s"] = r["e2e_latency_p90_s"]
            r["prefill_tok_per_s"] = round(in_len / r["e2e_latency_p50_s"], 1)
            phase3_rows.append(r)
            print(
                f"  {in_len:>7}  {r['ttft_p50_s']:>10.3f}s  "
                f"{r['ttft_p90_s']:>10.3f}s  {r['prefill_tok_per_s']:>14.1f}"
            )
        all_results["phase3_prefill"] = phase3_rows

    # ── Phase 4: Output length sweep ────────────────────────────────────────
    if args.phase in (0, 4):
        # Use throughput-optimal concurrency from Phase 2, else default to 4
        opt_conc = 4
        if "phase2_concurrency" in all_results:
            opt_conc = max(
                all_results["phase2_concurrency"],
                key=lambda r: r["decode_tok_per_s"]
            )["concurrency"]

        print("\n" + "=" * 70)
        print(f"Phase 4: Output length sweep  (input=512 tok, concurrency={opt_conc})")
        print("=" * 70)
        print(f"  {'out_tok':>7}  {'tok/s':>8}  {'TPOT_ms':>8}  "
              f"{'e2e_p50':>10}  {'e2e_p90':>10}")
        print("  " + "-" * 52)

        phase4_rows = []
        for out_len in OUTPUT_LENGTHS:
            n = max(N, opt_conc * 3)
            r = await _run_batch(server, model, opt_conc, 512, out_len, n)
            phase4_rows.append(r)
            print(
                f"  {out_len:>7}  {r['decode_tok_per_s']:>8.1f}  "
                f"{r['tpot_ms']:>8.1f}  "
                f"{r['e2e_latency_p50_s']:>10.3f}s  {r['e2e_latency_p90_s']:>10.3f}s"
            )
        all_results["phase4_output_len"] = phase4_rows

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if "phase2_concurrency" in all_results:
        rows = all_results["phase2_concurrency"]
        best = max(rows, key=lambda r: r["decode_tok_per_s"])
        base = rows[0]["decode_tok_per_s"]
        print(f"  Peak decode throughput : {best['decode_tok_per_s']:.1f} tok/s"
              f"  @ concurrency={best['concurrency']}"
              f"  ({best['decode_tok_per_s']/base:.2f}x conc=1)")
        print(f"  conc=1 decode          : {base:.1f} tok/s"
              f"  TPOT={rows[0]['tpot_ms']:.1f} ms")
    if "phase3_prefill" in all_results:
        rows = all_results["phase3_prefill"]
        best_pre = max(rows, key=lambda r: r["prefill_tok_per_s"])
        row_512 = next((r for r in rows if r["input_tokens"] == 512), None)
        if row_512:
            print(f"  TTFT @ 512 tok input   : {row_512['ttft_p50_s']:.3f}s"
                  f"  ({row_512['prefill_tok_per_s']:.0f} tok/s prefill)")
        print(f"  Peak prefill throughput: {best_pre['prefill_tok_per_s']:.1f} tok/s"
              f"  @ input={best_pre['input_tokens']} tok"
              f"  TTFT={best_pre['ttft_p50_s']:.3f}s")
    if "phase4_output_len" in all_results:
        rows = all_results["phase4_output_len"]
        print("  Decode tok/s by output length:")
        for r in rows:
            print(f"    output={r['max_tokens']:>5} tok  →  {r['decode_tok_per_s']:>7.1f} tok/s"
                  f"  TPOT={r['tpot_ms']:.1f} ms")

    # ── Save results ─────────────────────────────────────────────────────────
    with open(args.result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to: {args.result_path}")


if __name__ == "__main__":
    asyncio.run(main())
