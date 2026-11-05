"""In-memory token-bucket rate limiter.

Picture a bucket holding up to `burst` tokens. Each request takes one token.
Tokens drip back in at `rate_per_minute / 60` per second. A client can burst
up to `burst` requests at once, but its long-run average can't exceed the
refill rate. If the bucket is empty, we answer 429 with a Retry-After header.

Limitation: state lives in this process's memory. With several workers or
serverless instances, each has its own buckets, so the effective limit is
(limit x number of instances). Production fix: keep the buckets in Redis
(atomic Lua script / INCR+EXPIRE) or enforce limits at the edge (e.g. Vercel
Firewall rate-limit rules, Cloudflare, an API gateway).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, rate_per_minute: float, burst: int, max_clients: int = 10_000) -> None:
        if rate_per_minute <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate_per_s = rate_per_minute / 60.0
        self.burst = float(burst)
        self.max_clients = max_clients
        # OrderedDict as an LRU: bounded memory even if attackers rotate client ids.
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def check(self, client: str, now: float | None = None) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        b = self._buckets.get(client)
        if b is None:
            b = _Bucket(tokens=self.burst, updated=now)
            self._buckets[client] = b
        else:
            # Refill for the time elapsed since we last saw this client.
            b.tokens = min(self.burst, b.tokens + (now - b.updated) * self.rate_per_s)
            b.updated = now
            self._buckets.move_to_end(client)
        if len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - b.tokens) / self.rate_per_s
