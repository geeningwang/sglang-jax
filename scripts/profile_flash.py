#!/usr/bin/env python3
"""
MiMo-V2-Flash sglang-jax profiling driver.

Flow:
  1. Warmup (8 requests, conc=1, input=512, output=32)
  2. POST /start_profile  → JAX XPlane trace begins
  3. Decode burst (64 requests at conc=8, input=512, output=256) — captures steady-state
  4. POST /stop_profile   → trace written to --trace-dir
  5. Print GCS path for download / Perfetto UI

Usage:
  python3 scripts/profile_flash.py \
    --server http://localhost:8080 \
    --trace-dir gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-baseline/trace

Trace analysis: open in TensorBoard (tensorboard --logdir=<gcs-path>) or Perfetto UI.
Look for: host stall gaps (Opt C target), FP8 MXU op names (Opt A target),
routing/allreduce overhead (dispatch target).
"""

import argparse
import asyncio
import sys
import time


try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)

MODEL = "MiMo-V2-Flash"


def _make_prompt(approx_tokens: int, idx: int = 0) -> str:
    base = (
        f"[req-{idx}] Describe in detail: transformer attention mechanisms, "
        "mixture-of-experts routing, tensor parallelism, KV cache management, "
        "FP8 quantization, sliding window attention, and TPU v7x architecture. "
    )
    unit = "Include equations, implementation details, and performance characteristics. "
    target = max(10, int(approx_tokens * 3.5))
    while len(base) < target:
        base += unit
    return base[:target]


async def _send_one(session, server, prompt, max_tokens):
    payload = {
        "model": MODEL,
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
    e2e = time.perf_counter() - t0
    usage = result.get("usage", {})
    return e2e, usage.get("completion_tokens", 0)


async def _batch(server, concurrency, input_tokens, max_tokens, n_requests):
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    results = []

    async def _one(idx):
        prompt = _make_prompt(input_tokens, idx)
        async with sem:
            return await _send_one(session, server, prompt, max_tokens)

    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(_one(i)) for i in range(n_requests)]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        wall = time.perf_counter() - t0

    total_out = sum(r[1] for r in results)
    return wall, total_out


async def _post(session, url, body=None, timeout=180):
    async with session.post(
        url,
        json=body or {},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as r:
        text = await r.text()
        return r.status, text


async def main():
    ap = argparse.ArgumentParser(description="MiMo-V2-Flash profiling driver")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument(
        "--trace-dir",
        default="gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/flash-1node-tp8-baseline/trace",
        help="GCS or local path for JAX XPlane trace output",
    )
    ap.add_argument("--warmup-n", type=int, default=8)
    ap.add_argument("--burst-n", type=int, default=64)
    ap.add_argument("--burst-conc", type=int, default=8)
    ap.add_argument("--host-tracer-level", type=int, default=2,
                    help="1=user events only, 2=+XLA ops, 3=+low-level TPU ops")
    ap.add_argument("--num-profile-steps", type=int, default=None,
                    help="Stop profile after N steps (None = manual stop_profile)")
    args = ap.parse_args()

    server = args.server

    async with aiohttp.ClientSession() as session:
        # ── Phase 1: Warmup ──────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"Phase 1: Warmup  ({args.warmup_n} requests, input=512, output=32)")
        print(f"{'='*70}")
        wall, out_tok = await _batch(server, 1, 512, 32, args.warmup_n)
        print(f"  Warmup done — {out_tok/wall:.1f} tok/s  (wall={wall:.1f}s)")

        # ── Phase 2: Start profile ───────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"Phase 2: Starting JAX profiler trace")
        print(f"  output_dir = {args.trace_dir}")
        print(f"  host_tracer_level = {args.host_tracer_level}")
        print(f"{'='*70}")
        profile_body = {
            "output_dir": args.trace_dir,
            "host_tracer_level": args.host_tracer_level,
        }
        if args.num_profile_steps is not None:
            profile_body["num_steps"] = args.num_profile_steps

        status, text = await _post(session, f"{server}/start_profile", profile_body)
        if status not in (200, 201):
            print(f"  ERROR starting profile: HTTP {status}  {text}")
            sys.exit(1)
        print(f"  Profile started (HTTP {status}): {text[:120]}")
        t_profile_start = time.perf_counter()

        # ── Phase 3: Decode burst (the thing we want to profile) ─────────────
        print(f"\n{'='*70}")
        print(f"Phase 3: Decode burst (conc={args.burst_conc}, input=512, output=256, n={args.burst_n})")
        print(f"{'='*70}")
        wall, out_tok = await _batch(server, args.burst_conc, 512, 256, args.burst_n)
        tok_s = out_tok / wall
        print(f"  Burst done — {tok_s:.1f} tok/s  (wall={wall:.1f}s, out={out_tok} tok)")

        # ── Phase 4: Stop profile ────────────────────────────────────────────
        elapsed = time.perf_counter() - t_profile_start
        print(f"\n{'='*70}")
        print(f"Phase 4: Stopping profile  (elapsed={elapsed:.1f}s)")
        print(f"{'='*70}")
        status, text = await _post(session, f"{server}/stop_profile")
        if status not in (200, 201):
            print(f"  WARNING: stop_profile returned HTTP {status}: {text}")
        else:
            print(f"  Profile stopped (HTTP {status})")

    print(f"\n{'='*70}")
    print("Trace saved to:")
    print(f"  {args.trace_dir}")
    print("")
    print("To analyze:")
    print(f"  tensorboard --logdir={args.trace_dir}")
    print("  OR open https://ui.perfetto.dev and upload the .pb.gz file")
    print("")
    print("What to look for in the trace:")
    print("  - Host stall gaps (CPU gaps → Opt C target: remove effects_barrier)")
    print("  - 'custom-call: ..fp8..' op names (confirms FP8 MXU for expert matmul)")
    print("  - 'all-reduce' / psum time (metadata allreduce overhead)")
    print("  - MoE routing time (topk + score normalization)")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
