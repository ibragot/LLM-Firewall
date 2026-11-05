"""Build the held-out red-team suite: eval/data/attacks.jsonl + benign.jsonl.

    python -m eval.build_redteam

1. Expand the hand-written seeds (eval/seeds_*.py): apply obfuscations to
   payloads, embed injections inside documents, etc. Everything is
   deterministic (no randomness), so the suite is reproducible.
2. Deduplicate against ALL public training data (exact + near-duplicate),
   and drop internal duplicates.
3. Write the JSONL files and eval/data/dedup_report.json.

Why dedup? If a test prompt (or a trivially edited copy) is also in the
training set, the model has effectively seen the answer: the test score
measures memorisation, not generalisation, and is inflated.
"""

from __future__ import annotations

import base64
import codecs
import json
from pathlib import Path

import pandas as pd

from eval.seeds_attacks import (
    DIRECT,
    INDIRECT_DOCS,
    INDIRECT_INJECTIONS,
    MULTILINGUAL,
    OBFUSCATION_CARRIERS,
    OBFUSCATION_PAYLOADS,
    ROLEPLAY,
)
from eval.seeds_benign import HARD_NEGATIVE, NORMAL
from eval.seeds_benign import MULTILINGUAL as BENIGN_MULTILINGUAL
from training.data import dedup_against, drop_internal_duplicates, load_public

OUT = Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------- obfuscations
_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
_CYRILLIC = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "i": "і", "x": "х", "y": "у"})


def leet(s: str) -> str:
    return s.translate(_LEET)


def homoglyph(s: str) -> str:
    # Replace look-alikes in every word but keep the first letter Latin, so each word is mixed-script.
    return " ".join(w[0] + w[1:].translate(_CYRILLIC) if len(w) > 1 else w for w in s.split())


def zero_width(s: str) -> str:
    return " ".join("​".join(w[i : i + 2] for i in range(0, len(w), 2)) for w in s.split())


def spaced(s: str, sep: str) -> str:
    return "  ".join(sep.join(w) for w in s.split())


def fullwidth(s: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in s)


def tag_smuggle(s: str) -> str:
    return "".join(chr(0xE0000 + ord(c)) for c in s)


def mixed(s: str) -> str:
    return zero_width(homoglyph(leet(s)))


TRANSFORMS = list(OBFUSCATION_CARRIERS)


def encode(kind: str, payload: str, idx: int) -> str:
    carriers = OBFUSCATION_CARRIERS[kind]
    carrier = carriers[idx % len(carriers)]
    if kind == "split":
        words = payload.split()
        half = len(words) // 2
        return carrier.format(a=" ".join(words[:half]), b=" " + " ".join(words[half:]))
    encoded = {
        "base64": lambda p: base64.b64encode(p.encode()).decode(),
        "hex": lambda p: p.encode().hex(" "),
        "rot13": lambda p: codecs.encode(p, "rot_13"),
        "leetspeak": leet,
        "homoglyph": homoglyph,
        "zero_width": zero_width,
        "spaced": lambda p: spaced(p, " " if idx % 2 == 0 else "."),
        "fullwidth": fullwidth,
        "tag_smuggle": tag_smuggle,
        "reversed": lambda p: p[::-1],
        "mixed": mixed,
    }[kind](payload)
    return carrier.format(p=encoded)


# ------------------------------------------------------------- documents
BENIGN_FILLERS = [
    "Please let me know if you have any questions.",
    "The final numbers will be shared next week.",
    "Note: all prices include sales tax.",
    "Please ignore the previous version of this file; this one is final.",
    "Thanks again for your patience.",
    "Remember to bring your ID to the front desk.",
    "This section was last updated in March.",
    "Feel free to forward this to anyone on the team.",
    "Instructions for assembly are in the box.",
    "Disregard the old address; we moved to 22 King Street.",
]


def build_attacks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, t in enumerate(DIRECT):
        rows.append({"id": f"direct-{i:03d}", "text": t, "label": 1, "category": "direct_injection", "sub": "direct"})
    for i, t in enumerate(ROLEPLAY):
        rows.append({"id": f"roleplay-{i:03d}", "text": t, "label": 1, "category": "roleplay_jailbreak", "sub": "roleplay"})
    for i, payload in enumerate(OBFUSCATION_PAYLOADS):
        for j, kind in enumerate((TRANSFORMS[i % len(TRANSFORMS)], TRANSFORMS[(i + 5) % len(TRANSFORMS)])):
            rows.append({"id": f"obfuscated-{i:03d}-{j}", "text": encode(kind, payload, i + j), "label": 1,
                         "category": "obfuscated", "sub": kind})
    for i, (lang, t) in enumerate(MULTILINGUAL):
        rows.append({"id": f"multilingual-{i:03d}", "text": t, "label": 1, "category": "multilingual", "sub": lang})
    n_docs, n_inj = len(INDIRECT_DOCS), len(INDIRECT_INJECTIONS)
    for d, (doc_type, doc) in enumerate(INDIRECT_DOCS):
        for k in range(4 if d < 10 else 3):
            inj = INDIRECT_INJECTIONS[(d * 3 + k * 7) % n_inj]
            # JSON docs need the quotes inside the injection escaped to stay valid JSON.
            safe = inj.replace('"', "'") if doc_type == "json_api" else inj
            rows.append({"id": f"indirect-{d:02d}-{k}", "text": doc.format(inj=safe), "label": 1,
                         "category": "indirect_injection", "sub": doc_type})
    assert n_docs == 20
    return rows


def build_benign() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, t in enumerate(NORMAL):
        rows.append({"id": f"normal-{i:03d}", "text": t, "label": 0, "category": "benign_normal", "sub": "en"})
    for i, (lang, t) in enumerate(BENIGN_MULTILINGUAL):
        rows.append({"id": f"bmulti-{i:03d}", "text": t, "label": 0, "category": "benign_multilingual", "sub": lang})
    for i, t in enumerate(HARD_NEGATIVE):
        rows.append({"id": f"hardneg-{i:03d}", "text": t, "label": 0, "category": "benign_hard_negative", "sub": "en"})
    for d, (doc_type, doc) in enumerate(INDIRECT_DOCS):
        for k in range(2):
            filler = BENIGN_FILLERS[(d + k * 3) % len(BENIGN_FILLERS)]
            rows.append({"id": f"bdoc-{d:02d}-{k}", "text": doc.format(inj=filler), "label": 0,
                         "category": "benign_document", "sub": doc_type})
    return rows


def main() -> None:
    train = load_public("train")
    report: dict[str, object] = {"reference": "public training data (deepset train + jackhhao train)",
                                 "near_duplicate_threshold": 0.8, "sets": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("attacks", build_attacks()), ("benign", build_benign())):
        df = pd.DataFrame(rows)
        n0 = len(df)
        df, internal = drop_internal_duplicates(df)
        df, rep = dedup_against(df, train, near_threshold=0.8)
        report["sets"][name] = {"generated": n0, "internal_duplicates": internal, **rep.as_dict(),  # type: ignore[index]
                                "per_category": df["category"].value_counts().to_dict()}
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in df.to_dict(orient="records"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: generated {n0}, internal dups {internal}, exact-train {rep.exact_removed}, "
              f"near-train {rep.near_removed}, kept {rep.kept}")
        print("  ", df["category"].value_counts().to_dict())

    # Also report leakage *inside* the public datasets (their test vs their train).
    test = load_public("test")
    _, rep = dedup_against(test, train, near_threshold=0.8)
    report["public_test_vs_train"] = rep.as_dict()
    print(f"public test vs train: exact {rep.exact_removed}, near {rep.near_removed}, clean {rep.kept}/{len(test)}")
    (OUT / "dedup_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
