"""Client API-key authentication for the proxy."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request


def client_id_for(key: str) -> str:
    """A stable, non-secret identifier for a key (safe to log and to rate-limit by)."""
    return "key_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.headers.get("x-api-key") or None


def authenticate(request: Request, valid_keys: frozenset[str]) -> str:
    """Return the caller's client id, or raise 401.

    `hmac.compare_digest` compares in constant time: a normal `==` returns as
    soon as one character differs, and an attacker measuring response times
    could guess a key one character at a time (a *timing attack*).
    """
    key = extract_bearer(request)
    if not key or not valid_keys:
        raise HTTPException(status_code=401, detail="missing or invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})
    ok = False
    for valid in valid_keys:  # check all keys, no early exit
        ok |= hmac.compare_digest(key.encode(), valid.encode())
    if not ok:
        raise HTTPException(status_code=401, detail="missing or invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})
    return client_id_for(key)


def client_ip(request: Request) -> str:
    """Best-effort client IP. Behind Vercel/most proxies the first X-Forwarded-For hop is the client.

    Only trust X-Forwarded-For when running behind a proxy you control (Vercel
    sets it). Exposed directly, a client could spoof it to dodge rate limits.
    """
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"
