"""Load the public datasets and guard against train/test leakage.

Sources (copied into data/raw/, see data/raw/README.md):
  * deepset/prompt-injections        546 train / 116 test, label 1 = injection
  * jackhhao/jailbreak-classification 1598 train / 400 test, type "jailbreak"/"benign"

Every text goes through the same `normalize()` used at serving time, so the
model never sees a different "dialect" of text in production than in training.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_firewall.normalize import normalize

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EVAL_DATA = ROOT / "eval" / "data"


def model_text(text: str) -> str:
    """The exact string the classifiers see (train AND serve)."""
    return normalize(text).model_text()


def load_public(split: str) -> pd.DataFrame:
    """Return a DataFrame(text, label, source, category) for split in {'train','test'}."""
    ds = pd.read_parquet(RAW / "deepset" / f"{split}.parquet")
    ds = pd.DataFrame({"text": ds["text"], "label": ds["label"].astype(int), "source": "deepset"})
    jh = pd.read_csv(RAW / "jackhhao" / f"{split}.csv")
    jh = pd.DataFrame(
        {"text": jh["prompt"], "label": (jh["type"] == "jailbreak").astype(int), "source": "jackhhao"}
    )
    df = pd.concat([ds, jh], ignore_index=True)
    df["text"] = df["text"].astype(str)
    df["category"] = df["source"] + "_" + df["label"].map({1: "attack", 0: "benign"})
    return df


def load_redteam() -> pd.DataFrame:
    """Our hand-built red-team suite (eval/data/*.jsonl), already deduplicated."""
    rows: list[dict[str, object]] = []
    for name in ("attacks.jsonl", "benign.jsonl"):
        with open(EVAL_DATA / name, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- dedup
_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


def canonical(text: str) -> str:
    """Aggressive canonical form for duplicate detection: normalized, no punctuation."""
    return _NON_ALNUM.sub(" ", model_text(text)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> frozenset[str]:
    c = canonical(text).replace(" ", "_")
    if len(c) <= k:
        return frozenset({c})
    return frozenset(c[i : i + k] for i in range(len(c) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class DedupReport:
    exact_removed: int
    near_removed: int
    kept: int
    examples: list[tuple[str, str, float]]  # (removed text, matching train text, similarity)

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_removed": self.exact_removed,
            "near_removed": self.near_removed,
            "kept": self.kept,
            "examples": [{"eval": a[:160], "train": b[:160], "jaccard": round(s, 3)} for a, b, s in self.examples[:10]],
        }


def dedup_against(
    candidates: pd.DataFrame, reference: pd.DataFrame, near_threshold: float = 0.8
) -> tuple[pd.DataFrame, DedupReport]:
    """Drop rows of `candidates` that duplicate (exactly or nearly) any row of `reference`.

    Exact  = same canonical text (case, punctuation, obfuscation removed).
    Near   = Jaccard similarity of character 5-gram sets >= near_threshold.
    """
    ref_hashes = {text_hash(t) for t in reference["text"]}
    ref_texts = list(reference["text"])
    ref_shingles = [shingles(t) for t in ref_texts]
    # Bucket reference shingle sets by size so we only compare plausible pairs:
    # Jaccard(a,b) <= min(|a|,|b|)/max(|a|,|b|), so very different sizes can't match.
    keep_mask: list[bool] = []
    exact = near = 0
    examples: list[tuple[str, str, float]] = []
    for text in candidates["text"]:
        if text_hash(text) in ref_hashes:
            exact += 1
            keep_mask.append(False)
            continue
        s = shingles(text)
        best, best_j = None, 0.0
        for rt, rs in zip(ref_texts, ref_shingles, strict=True):
            lo, hi = sorted((len(s), len(rs)))
            if hi == 0 or lo / hi < near_threshold:
                continue
            j = jaccard(s, rs)
            if j > best_j:
                best, best_j = rt, j
        if best is not None and best_j >= near_threshold:
            near += 1
            examples.append((text, best, best_j))
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    kept = candidates[keep_mask].reset_index(drop=True)
    return kept, DedupReport(exact, near, len(kept), examples)


def drop_internal_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicates inside one dataset (keeps the first occurrence).

    Must happen *before* a train/validation split, otherwise one copy can land
    in train and the other in validation and the validation score is inflated.
    """
    hashes = df["text"].map(text_hash)
    dup = hashes.duplicated(keep="first")
    return df[~dup].reset_index(drop=True), int(dup.sum())


def load_train_val(val_fraction: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Deduplicated public training data, split into train/validation.

    The split is stratified by `category` (source x label) so both halves have
    the same mix of deepset/jackhhao and attack/benign. Both the baseline and
    the transformer call this function, so they train/validate on identical rows.
    """
    from sklearn.model_selection import train_test_split  # training-only dependency

    df = load_public("train")
    n_raw = len(df)
    df, n_dup = drop_internal_duplicates(df)
    # Same text with conflicting labels would be noise in training; drop all copies.
    hashes = df["text"].map(text_hash)
    conflicting = hashes.map(df.groupby(hashes)["label"].nunique()) > 1
    n_conflict = int(conflicting.sum())
    df = df[~conflicting].reset_index(drop=True)
    train, val = train_test_split(df, test_size=val_fraction, random_state=seed, stratify=df["category"])
    stats = {"raw": n_raw, "internal_duplicates": n_dup, "conflicting_labels": n_conflict,
             "train": len(train), "val": len(val)}
    return train.reset_index(drop=True), val.reset_index(drop=True), stats
