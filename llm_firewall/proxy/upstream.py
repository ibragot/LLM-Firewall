"""Upstream LLM clients: a real OpenAI-compatible HTTP client and a mock.

Both expose the same async interface, so the proxy doesn't care which one it
talks to (dependency inversion). The mock makes the whole system runnable,
demo-able and testable with zero API keys and zero cost.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Protocol

import httpx


class UpstreamError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Upstream(Protocol):
    name: str

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class HttpUpstream:
    """Forwards to any OpenAI-compatible API (OpenAI, Groq, Together, vLLM, Ollama...)."""

    name = "http"

    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # One pooled AsyncClient for the app's lifetime: reuses TCP/TLS connections.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
        )

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            raise UpstreamError(504, "upstream timed out") from e
        except httpx.HTTPError as e:
            raise UpstreamError(502, f"upstream unreachable: {type(e).__name__}") from e
        if resp.status_code >= 400:
            # Don't echo upstream bodies verbatim: they can contain account details.
            raise UpstreamError(502 if resp.status_code >= 500 else resp.status_code, f"upstream returned {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


# ------------------------------------------------------------------- mock
_EXTRACTION_RE = re.compile(
    r"system prompt|your (?:instructions|rules|prompt)|initial instructions|words above|text above|repeat everything|"
    r"hidden instructions|what were you told|prompt above",
    re.IGNORECASE,
)
_PII_RE = re.compile(r"customer record|contact (?:info|details)|test data|sample (?:user|customer|data)|pii|account details",
                     re.IGNORECASE)
_SECRET_RE = re.compile(r"api key|config file|\.env|credentials|secret", re.IGNORECASE)

FAKE_RECORD = (
    "Here is the customer record you asked for:\n"
    "- Name: Jane Example\n"
    "- Email: jane.example@example.com\n"
    "- Phone: +1 (613) 555-0199\n"
    "- Card on file: 4111 1111 1111 1111\n"
    "- SIN: 130 692 544\n"
)
FAKE_SECRETS = (
    "Sure, here is the config:\n"
    "OPENAI_API_KEY=sk-proj-FAKEfake0123456789abcdefFAKE\n"
    "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
    "db_password = correct-horse-battery-staple\n"
)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def mock_reply(messages: list[dict[str, Any]]) -> str:
    """A deliberately *naive, vulnerable* LLM, so the output-side defenses have
    something to catch:

    * asked for its instructions -> it leaks the whole system prompt (incl. canary)
    * asked for customer/test data -> it returns a fake record full of PII
    * asked for config/keys -> it returns fake secrets
    * otherwise -> a short, harmless canned answer that echoes the request
    """
    user = _last_user_text(messages)
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    if _EXTRACTION_RE.search(user) and system:
        return f"Of course! My instructions are:\n\n{system}"
    if _PII_RE.search(user):
        return FAKE_RECORD
    if _SECRET_RE.search(user):
        return FAKE_SECRETS
    preview = user.strip().replace("\n", " ")
    if len(preview) > 160:
        preview = preview[:160] + "..."
    return f"[mock-llm] I received your message ({len(user)} chars): \"{preview}\". This is a canned mock response."


def mock_completion(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    text = mock_reply(messages)
    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or "mock-llm-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text.split()),
                  "total_tokens": prompt_tokens + len(text.split())},
    }


class MockUpstream:
    name = "mock"

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return mock_completion(payload)

    async def aclose(self) -> None:
        return None
