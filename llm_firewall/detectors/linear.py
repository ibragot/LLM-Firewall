"""Layer B: TF-IDF + logistic regression, inferred in *plain Python*.

The model is trained with scikit-learn (`training/train_baseline.py`), but
serving it does not need scikit-learn or numpy. A trained logistic regression
is just numbers:

    p(attack | text) = sigmoid( b + sum_i  w_i * x_i )

where x_i is the TF-IDF value of feature i (a word or character n-gram) and
w_i is the learned weight for that feature. TF-IDF itself is also simple
arithmetic over a fixed vocabulary + IDF table. So we export the vocabulary,
IDF values and weights to JSON, and this file recomputes exactly what
scikit-learn would. `tests/test_linear_parity.py` proves the two agree.

Why bother? The Vercel Python runtime has a bundle size limit, and
scikit-learn + numpy + scipy are ~100 MB. This file is 0 MB of dependencies.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# scikit-learn's default token pattern: words of 2+ "word characters".
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_MULTI_WS_RE = re.compile(r"\s\s+")


def word_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    """Replicates sklearn TfidfVectorizer(analyzer='word')."""
    tokens = _TOKEN_RE.findall(text.lower())
    lo, hi = ngram_range
    out: list[str] = []
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


def char_wb_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    """Replicates sklearn TfidfVectorizer(analyzer='char_wb').

    Each whitespace-separated word is padded with one space on each side and
    sliced into character n-grams that never cross word boundaries:
    "hack" with n=3 -> " ha", "hac", "ack", "ck ".
    """
    text = _MULTI_WS_RE.sub(" ", text.lower())
    lo, hi = ngram_range
    out: list[str] = []
    for w in text.split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(lo, hi + 1):
            offset = 0
            out.append(w[offset : offset + n])
            while offset + n < w_len:
                offset += 1
                out.append(w[offset : offset + n])
            if offset == 0:  # word shorter than n: sklearn emits it once and stops
                break
    return out


@dataclass
class _Vectorizer:
    analyzer: str
    ngram_range: tuple[int, int]
    sublinear_tf: bool
    vocab: dict[str, tuple[float, float]]  # term -> (idf, logistic-regression weight)

    def contribution(self, text: str) -> tuple[float, list[tuple[str, float]]]:
        """Return this block's share of the logit and the per-term contributions."""
        grams = word_ngrams(text, self.ngram_range) if self.analyzer == "word" else char_wb_ngrams(text, self.ngram_range)
        counts = Counter(g for g in grams if g in self.vocab)
        if not counts:
            return 0.0, []
        tfidf: dict[str, float] = {}
        for term, c in counts.items():
            tf = 1.0 + math.log(c) if self.sublinear_tf else float(c)
            tfidf[term] = tf * self.vocab[term][0]
        norm = math.sqrt(sum(v * v for v in tfidf.values()))  # L2 normalisation
        terms = [(t, (v / norm) * self.vocab[t][1]) for t, v in tfidf.items()]
        return sum(c for _, c in terms), terms


class LinearModel:
    """Loads `models/linear_model.json` and scores text in pure Python."""

    def __init__(self, blocks: list[_Vectorizer], intercept: float, meta: dict[str, Any]) -> None:
        self.blocks = blocks
        self.intercept = intercept
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path) -> LinearModel:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        blocks = [
            _Vectorizer(
                analyzer=b["analyzer"],
                ngram_range=(int(b["ngram_range"][0]), int(b["ngram_range"][1])),
                sublinear_tf=bool(b["sublinear_tf"]),
                vocab={term: (float(v[0]), float(v[1])) for term, v in b["vocab"].items()},
            )
            for b in data["blocks"]
        ]
        return cls(blocks, float(data["intercept"]), data.get("meta", {}))

    def logit(self, text: str) -> float:
        return self.intercept + sum(b.contribution(text)[0] for b in self.blocks)

    def predict_proba(self, text: str) -> float:
        z = self.logit(text)
        # Numerically stable sigmoid.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def explain(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Top features pushing the score towards 'attack' — shown on the demo page."""
        contribs: list[tuple[str, float]] = []
        for b in self.blocks:
            contribs.extend(b.contribution(text)[1])
        merged: dict[str, float] = {}
        for term, c in contribs:
            merged[term] = merged.get(term, 0.0) + c
        return sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
