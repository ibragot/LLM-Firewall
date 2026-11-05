"""Undo common obfuscation tricks before any detector looks at the text.

Attackers know that filters match on text, so they disguise the text:

    "1gn0re previous instructions"           leetspeak
    "ig​nore previous instructions"     zero-width characters
    "іgnore previous instructions"           Cyrillic 'і' (U+0456) homoglyph
    "i g n o r e previous instructions"      spaced-out letters
    "aWdub3JlIHByZXZpb3VzIGluc3RydWN0aW9ucw=="  base64

The model behind the proxy can often still *read* all of these, so the filter
must read them too. `normalize()` produces several "views" of the input; the
rule engine matches against every view and the ML models score the most
canonical one. The rule is: **normalize once, detect many times**, and use the
exact same function at training time and at serving time.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass, field

# --- 1. Invisible characters ----------------------------------------------
# Zero-width space/joiners, word joiner, BOM, soft hyphen, bidi controls, …
_INVISIBLE_RE = re.compile(r"[­᠎​-‏‪-‮⁠-⁤﻿]")

# Unicode "tag" characters (U+E0020..U+E007E) mirror printable ASCII and render
# as nothing. They have been used to smuggle hidden instructions to LLMs.
_TAG_RE = re.compile(r"[\U000E0020-\U000E007E]")

# --- 2. Homoglyphs ------------------------------------------------------------
# Characters from other scripts that look identical to Latin letters.
HOMOGLYPHS: dict[str, str] = {
    # Cyrillic lowercase
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d",
    "ɡ": "g", "ԛ": "q", "ԝ": "w", "ӏ": "l",
    # Cyrillic uppercase
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "τ": "t", "ι": "i", "κ": "k", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# --- 3. Leetspeak ---------------------------------------------------------------
LEET: dict[str, str] = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
# A "word" made of letters plus at least one leet symbol, e.g. "1gn0r3", "h@ck".
_LEET_WORD_RE = re.compile(r"(?<![\w@$])(?=[\w@$]*[A-Za-z])(?=[\w@$]*[013457@$])[A-Za-z013457@$]{3,}(?![\w@$])")

# --- 4. Spaced-out letters: "i g n o r e" / "i.g.n.o.r.e" / "i-g-n-o-r-e" ---------
_SPACED_RE = re.compile(r"\b(?:[a-z][ .\-_*]){3,}[a-z]\b")

# --- 5. Encoded payloads -------------------------------------------------------
_B64_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}[ :]?){10,}\b")
_ROT13_HINT_RE = re.compile(r"\brot[- ]?13\b", re.IGNORECASE)

_WS_RE = re.compile(r"\s+")


@dataclass
class NormalizationResult:
    """All views of one input. `signals` records which tricks were found."""

    original: str
    clean: str  # invisible chars removed, NFKC, homoglyphs fixed. Case preserved.
    normalized: str  # `clean`, lower-cased, whitespace collapsed, spaced letters joined
    deleeted: str  # `normalized` with leetspeak words decoded
    decoded: list[str] = field(default_factory=list)  # decoded base64/hex/rot13/tag payloads
    signals: list[str] = field(default_factory=list)

    def views(self) -> list[str]:
        """Every distinct text a rule should be matched against."""
        seen: list[str] = []
        for v in [self.normalized, self.deleeted, *(d.lower() for d in self.decoded)]:
            if v and v not in seen:
                seen.append(v)
        return seen

    def model_text(self) -> str:
        """The single canonical string fed to the ML classifiers."""
        return " ".join([self.deleeted, *(d.lower() for d in self.decoded)]).strip()


def _fix_homoglyphs(text: str) -> tuple[str, bool]:
    """Replace look-alike letters, but only inside *mixed-script* words.

    A word that is entirely Cyrillic ("привет") is genuine Russian and must not
    be mangled; a word that mixes Latin and Cyrillic ("іgnore") is a disguise.
    """
    changed = False

    def fix_word(m: re.Match[str]) -> str:
        nonlocal changed
        word = m.group(0)
        if any(ch in HOMOGLYPHS for ch in word) and _LATIN_RE.search(word):
            changed = True
            return "".join(HOMOGLYPHS.get(ch, ch) for ch in word)
        return word

    return _WORD_RE.sub(fix_word, text), changed


def _decode_leet(text: str) -> tuple[str, bool]:
    """Decode words like "1gn0r3" -> "ignore" without breaking "4pm", "mp3" or emails."""
    changed = False

    def fix(m: re.Match[str]) -> str:
        nonlocal changed
        word = m.group(0)
        letters = sum(c.isalpha() for c in word)
        leet_positions = [i for i, c in enumerate(word) if c in LEET]
        interior = any(0 < i < len(word) - 1 and word[i - 1].isalpha() and word[i + 1].isalpha() for i in leet_positions)
        # "4pm", "mp3", "b2b": a single leet symbol at the edge is a number/unit, not leetspeak.
        if letters < 2 or not (interior or len(leet_positions) >= 2):
            return word
        # "name@example.com": an @ followed by a domain is an email address, leave it alone.
        if "@" in word and m.end() < len(text) and text[m.end()] == ".":
            return word
        changed = True
        return "".join(LEET.get(c, c) for c in word)

    return _LEET_WORD_RE.sub(fix, text), changed


def _join_spaced_letters(text: str) -> tuple[str, bool]:
    changed = False

    def fix(m: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return re.sub(r"[ .\-_*]", "", m.group(0))

    return _SPACED_RE.sub(fix, text), changed


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(ch.isprintable() or ch in "\n\t" for ch in s)
    return ok / len(s)


def _looks_like_text(s: str) -> bool:
    """Decoded bytes count as a hidden message only if they read like language."""
    return len(s) >= 4 and _printable_ratio(s) > 0.95 and sum(ch.isalpha() for ch in s) / len(s) > 0.5


def _decode_base64_segments(text: str) -> list[str]:
    out: list[str] = []
    for m in _B64_RE.finditer(text):
        token = m.group(0)
        # Real words like "Internationalization" match the charset; require some
        # base64-specific evidence (digit, +, /, =, or mixed case inside the token).
        if not (re.search(r"[0-9+/=]", token) or (re.search(r"[a-z]", token) and re.search(r"[A-Z].*[A-Z]", token))):
            continue
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if _looks_like_text(decoded):
            out.append(decoded)
    return out


def _decode_hex_segments(text: str) -> list[str]:
    out: list[str] = []
    for m in _HEX_RE.finditer(text):
        token = re.sub(r"[ :]", "", m.group(0))
        if len(token) % 2:
            token = token[:-1]
        try:
            decoded = bytes.fromhex(token).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _looks_like_text(decoded):
            out.append(decoded)
    return out


def normalize(text: str) -> NormalizationResult:
    """Run the full normalization pipeline and return every view + signals."""
    signals: list[str] = []
    decoded: list[str] = []

    # Hidden ASCII smuggled in Unicode tag characters: decode, then remove.
    tags = _TAG_RE.findall(text)
    if tags:
        signals.append("unicode_tag_chars")
        decoded.append("".join(chr(ord(c) - 0xE0000) for c in tags))
        text_wo_tags = _TAG_RE.sub("", text)
    else:
        text_wo_tags = text

    # NFKC folds compatibility forms: fullwidth "ｉｇｎｏｒｅ" -> "ignore", ligatures, etc.
    nfkc = unicodedata.normalize("NFKC", text_wo_tags)
    if nfkc != text_wo_tags:
        signals.append("unicode_compat_folded")

    clean = _INVISIBLE_RE.sub("", nfkc)
    if clean != nfkc:
        signals.append("invisible_chars_removed")

    clean, homoglyphs = _fix_homoglyphs(clean)
    if homoglyphs:
        signals.append("homoglyphs_mapped")

    normalized = _WS_RE.sub(" ", clean.lower()).strip()
    normalized, spaced = _join_spaced_letters(normalized)
    if spaced:
        signals.append("spaced_letters_joined")

    deleeted, leet = _decode_leet(normalized)
    if leet:
        signals.append("leetspeak_decoded")

    b64 = _decode_base64_segments(clean)
    if b64:
        signals.append("base64_decoded")
        decoded.extend(b64)
    hx = _decode_hex_segments(clean)
    if hx:
        signals.append("hex_decoded")
        decoded.extend(hx)
    if _ROT13_HINT_RE.search(clean):
        signals.append("rot13_decoded")
        decoded.append(codecs.decode(clean, "rot_13"))

    # One level of recursion: base64 inside base64, leetspeak inside base64, etc.
    nested: list[str] = []
    for d in decoded:
        inner = _decode_base64_segments(d)
        nested.extend(inner)
    if nested:
        decoded.extend(nested)

    decoded = [_WS_RE.sub(" ", _decode_leet(d.lower())[0]).strip() for d in decoded]

    return NormalizationResult(
        original=text,
        clean=clean,
        normalized=normalized,
        deleeted=deleeted,
        decoded=decoded,
        signals=signals,
    )
