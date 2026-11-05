"""ASGI middleware that rejects oversized request bodies *while streaming them in*.

Checking `len(await request.body())` is too late: by then a 2 GB upload has
already been buffered into memory. This middleware checks the Content-Length
header up front and also counts bytes as they arrive (for chunked uploads that
send no Content-Length), aborting with 413 as soon as the limit is crossed.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > self.max_bytes:
            await _send_413(send, self.max_bytes)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge:
            if not response_started:
                await _send_413(send, self.max_bytes)


async def _send_413(send: Send, limit: int) -> None:
    body: dict[str, Any] = {"error": {"message": f"request body exceeds {limit} bytes", "type": "request_too_large"}}
    raw = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
    await send({"type": "http.response.body", "body": raw})
