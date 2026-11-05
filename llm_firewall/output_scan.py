"""Output-side scanning: find and redact sensitive data in the LLM's answer.

Why scan outputs at all if inputs are filtered? Defense in depth: some
injections will get past the input filter, and some leaks need no attack at all
(the model was given a customer record as context and repeats it). The output
scanner is the last line before data reaches the user.

Detectors (all regex + validation):
  email, phone, credit_card (+Luhn +issuer prefix), ca_sin (+Luhn),
  api_key / private_key (common provider formats), canary_leak, system_prompt_leak.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from llm_firewall.luhn import luhn_valid

# ---------------------------------------------------------------- patterns
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone numbers. Two shapes, both refusing to start/end in the middle of a longer
# digit run (so card numbers and IDs are not mistaken for phones):
#   NANP:          +1 (613) 555-0199, 613-555-0199, 613.555.0199, 6135550199
#   international: +44 20 7946 0958, +33 1 23 45 67 89
PHONE_RE = re.compile(
    r"(?<![\w+])(?<!\d[ .-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?![\w-])(?![ .-]\d)"
    r"|(?<![\w+])\+\d{1,3}(?:[\s.-]?\d{1,4}){2,5}(?![\w-])(?![ .-]\d)"
)

# 13-19 digits, optionally grouped by spaces or dashes.
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# Canadian Social Insurance Number: 9 digits, often written 123-456-789 / 123 456 789.
SIN_RE = re.compile(r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)")

API_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "openai": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    "groq": re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b"),
    "slack": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "google": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "stripe": re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "generic_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-/+=.]{12,})"
    ),
}
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----|$)")

# Issuer prefixes (IIN ranges) of the major card networks.
_CARD_PREFIX_RE = re.compile(r"^(?:4|5[1-5]|2[2-7]|3[47]|6(?:011|5|4[4-9])|35|3(?:0[0-5]|[68]))")

# Priority when two findings overlap (e.g. a SIN inside a longer card number).
_PRIORITY = {"private_key": 0, "api_key": 1, "credit_card": 2, "ca_sin": 3, "email": 4, "phone": 5}


@dataclass(frozen=True)
class Finding:
    type: str
    start: int
    end: int
    detail: str = ""  # e.g. api key provider; never the secret itself


@dataclass
class OutputScanResult:
    findings: list[Finding]
    redacted_text: str
    canary_leaked: bool = False
    system_prompt_overlap: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)


def _card_candidates(text: str) -> list[Finding]:
    out: list[Finding] = []
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _CARD_PREFIX_RE.match(digits) and luhn_valid(digits):
            out.append(Finding("credit_card", m.start(), m.end()))
    return out


def _sin_candidates(text: str) -> list[Finding]:
    out: list[Finding] = []
    for m in SIN_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # SINs never start with 0; 8 is reserved for business numbers.
        if digits[0] not in "08" and luhn_valid(digits):
            out.append(Finding("ca_sin", m.start(), m.end()))
    return out


def find_sensitive(text: str) -> list[Finding]:
    """All non-overlapping sensitive spans, highest-priority type wins overlaps."""
    cands: list[Finding] = []
    cands += [Finding("private_key", m.start(), m.end()) for m in PRIVATE_KEY_RE.finditer(text)]
    for provider, pat in API_KEY_PATTERNS.items():
        for m in pat.finditer(text):
            # For "password = xyz" redact only the value, keep the label readable.
            s, e = (m.start(1), m.end(1)) if pat.groups else (m.start(), m.end())
            cands.append(Finding("api_key", s, e, provider))
    cands += _card_candidates(text)
    cands += _sin_candidates(text)
    cands += [Finding("email", m.start(), m.end()) for m in EMAIL_RE.finditer(text)]
    cands += [Finding("phone", m.start(), m.end()) for m in PHONE_RE.finditer(text)
              if len(re.sub(r"\D", "", m.group(0))) >= 10]

    cands.sort(key=lambda f: (_PRIORITY[f.type], f.start))
    chosen: list[Finding] = []
    for f in cands:
        if all(f.end <= c.start or f.start >= c.end for c in chosen):
            chosen.append(f)
    return sorted(chosen, key=lambda f: f.start)


def redact(text: str, findings: list[Finding], types: set[str] | None = None) -> str:
    """Replace each finding (optionally only some types) with [REDACTED:TYPE]."""
    out, last = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        if types is not None and f.type not in types:
            continue
        out.append(text[last : f.start])
        out.append(f"[REDACTED:{f.type.upper()}]")
        last = f.end
    out.append(text[last:])
    return "".join(out)


# ------------------------------------------------------------ canary tokens
def new_canary() -> str:
    """A random, unguessable marker hidden in the system prompt.

    The model has no legitimate reason to ever output it, so seeing it in a
    response proves the system prompt leaked — with zero false positives.
    """
    return f"CANARY-{secrets.token_hex(8)}"


def canary_leaked(text: str, canary: str) -> bool:
    if not canary:
        return False
    squashed = re.sub(r"[\s​-‍﻿`*_~]", "", text).lower()
    # Also catch partial echo of the random part (the model may drop the prefix).
    return canary.lower() in squashed or canary.split("-", 1)[-1].lower() in squashed


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def system_prompt_overlap(response: str, system_prompt: str, n: int = 6) -> float:
    """Fraction of the system prompt's word 6-grams that appear in the response.

    Catches paraphrase-free leaks even if the canary was stripped. 6 words in a
    row is long enough that ordinary answers rarely share them by chance.
    """
    sp = _word_ngrams(system_prompt, n)
    if not sp:
        return 0.0
    return len(sp & _word_ngrams(response, n)) / len(sp)


def scan_output(text: str, canary: str = "", system_prompt: str = "") -> OutputScanResult:
    findings = find_sensitive(text)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.type] = counts.get(f.type, 0) + 1
    return OutputScanResult(
        findings=findings,
        redacted_text=redact(text, findings),
        canary_leaked=canary_leaked(text, canary),
        system_prompt_overlap=round(system_prompt_overlap(text, system_prompt), 4) if system_prompt else 0.0,
        counts=counts,
    )
