"""Minimal PD router: fans out /generate to both prefill and decode servers.

The prefill server does the prefill and transfers KV to the decode server.
The decode server receives KV, does decoding, and returns the response.
This router ensures both servers get the same bootstrap_room so they can
coordinate the KV transfer via the bootstrap server.

Usage:
    python simple_pd_router.py \
        --prefill-url http://PREFILL_IP:10000 \
        --decode-url http://DECODE_IP:10001 \
        --bootstrap-host BOOTSTRAP_IP \
        --bootstrap-port 8998 \
        --port 30000
"""

import argparse
import asyncio
import json
import uuid
import zlib

import aiohttp
from aiohttp import web


async def handle_generate(request: web.Request) -> web.Response:
    body = await request.json()

    rid = body.get("rid") or str(uuid.uuid4())
    body.setdefault("rid", rid)
    bootstrap_room = zlib.crc32(str(rid).encode("utf-8"))
    body["bootstrap_room"] = bootstrap_room
    body["bootstrap_host"] = request.app["bootstrap_host"]
    body["bootstrap_port"] = request.app["bootstrap_port"]

    prefill_url = request.app["prefill_url"] + "/generate"
    decode_url = request.app["decode_url"] + "/generate"
    payload = json.dumps(body)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1800)
    ) as session:
        prefill_task = asyncio.create_task(
            session.post(prefill_url, data=payload, headers={"Content-Type": "application/json"})
        )
        decode_task = asyncio.create_task(
            session.post(decode_url, data=payload, headers={"Content-Type": "application/json"})
        )

        decode_resp = await decode_task
        decode_body = await decode_resp.text()

        prefill_task.cancel()
        try:
            await prefill_task
        except (asyncio.CancelledError, Exception):
            pass

    return web.Response(
        text=decode_body,
        status=decode_resp.status,
        content_type="application/json",
    )


async def handle_health(request: web.Request) -> web.Response:
    decode_url = request.app["decode_url"] + "/health"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(decode_url) as resp:
            return web.Response(status=resp.status, text=await resp.text())


async def handle_get_server_info(request: web.Request) -> web.Response:
    decode_url = request.app["decode_url"] + "/get_server_info"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(decode_url) as resp:
            return web.Response(
                status=resp.status,
                text=await resp.text(),
                content_type="application/json",
            )


async def handle_get_model_info(request: web.Request) -> web.Response:
    decode_url = request.app["decode_url"] + "/get_model_info"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(decode_url) as resp:
            return web.Response(
                status=resp.status,
                text=await resp.text(),
                content_type="application/json",
            )


def main():
    parser = argparse.ArgumentParser(description="Simple PD router")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    app = web.Application()
    app["prefill_url"] = args.prefill_url.rstrip("/")
    app["decode_url"] = args.decode_url.rstrip("/")
    app["bootstrap_host"] = args.bootstrap_host
    app["bootstrap_port"] = args.bootstrap_port

    app.router.add_post("/generate", handle_generate)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/get_server_info", handle_get_server_info)
    app.router.add_get("/get_model_info", handle_get_model_info)

    print(f"PD Router starting on {args.host}:{args.port}")
    print(f"  prefill -> {args.prefill_url}")
    print(f"  decode  -> {args.decode_url}")
    print(f"  bootstrap -> {args.bootstrap_host}:{args.bootstrap_port}")
    web.run_app(app, host=args.host, port=args.port, print=print)


if __name__ == "__main__":
    main()
