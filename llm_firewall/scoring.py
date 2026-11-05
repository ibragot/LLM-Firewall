"""Combine per-layer scores into one risk score in [0, 1]."""

from __future__ import annotations

from llm_firewall.detectors.heuristics import noisy_or
from llm_firewall.policy import Policy

ML_LAYERS = ("linear", "transformer")


def ml_score(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Weighted mean of the ML layers that actually ran (None = layer unavailable).

    If the transformer is not installed, its weight simply drops out and the
    linear model carries the full ML vote — the system degrades gracefully.
    """
    pairs = [(scores[k], weights.get(k, 0.0)) for k in ML_LAYERS if scores.get(k) is not None]
    total_w = sum(w for _, w in pairs)
    if not pairs or total_w <= 0:
        return None
    return sum(float(s) * w for s, w in pairs if s is not None) / total_w


def combine(scores: dict[str, float | None], policy: Policy) -> float:
    h = float(scores.get("heuristics") or 0.0)
    m = ml_score(scores, policy.weights)
    if m is None:
        return h
    if policy.combination == "max":
        return max(h, m)
    if policy.combination == "weighted_mean":
        parts = [(h, policy.weights.get("heuristics", 1.0))]
        parts += [(float(s), policy.weights.get(k, 0.0)) for k in ML_LAYERS if (s := scores.get(k)) is not None]
        tw = sum(w for _, w in parts)
        return sum(s * w for s, w in parts) / tw if tw else 0.0
    return noisy_or([h, m])
