"""Privacy-safe structured audit logging.

Every verdict is logged as one JSON object per line ("JSON Lines"), which log
pipelines (CloudWatch, Datadog, Loki, Vercel logs ...) can index and query.

The log must be useful for incident response *without itself becoming a data
leak*. So we never write raw prompts or responses. Instead:

  * prompt_hmac   HMAC-SHA256(secret salt, text). Lets you answer "did this exact
                  prompt appear before?" and correlate events, but can't be reversed.
                  A *plain* SHA-256 of a phone number can be brute-forced (there are
                  only ~10^10 phone numbers); a keyed HMAC can't without the salt.
  * snippets      short matched rule snippets, passed through the PII redactor.
  * counts        how many emails/cards/... were redacted, never the values.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
import time
from typing import Any

from llm_firewall.output_scan import find_sensitive, redact

_logger = logging.getLogger("llm_firewall.audit")


def configure(level: str = "INFO") -> None:
    if _logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.propagate = False


def hmac_text(text: str, salt: str) -> str:
    return hmac.new(salt.encode(), text.encode("utf-8", "replace"), hashlib.sha256).hexdigest()[:32]


def safe_snippet(text: str, max_len: int = 80) -> str:
    """Redact PII/secrets, then truncate. Used for the only free text we log."""
    return redact(text, find_sensitive(text))[:max_len]


class AuditLogger:
    def __init__(self, salt: str) -> None:
        self.salt = salt
        self.records: list[dict[str, Any]] = []  # last N records, used by tests and /api/demo
        self._max_records = 200

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"ts": round(time.time(), 3), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        _logger.info(line)
        self.records.append(record)
        if len(self.records) > self._max_records:
            self.records = self.records[-self._max_records :]
        return record

    def fingerprint(self, text: str) -> str:
        return hmac_text(text, self.salt)
