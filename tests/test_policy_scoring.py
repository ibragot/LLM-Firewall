from __future__ import annotations

from pathlib import Path

import pytest

from llm_firewall.policy import Action, Policy, strictest
from llm_firewall.scoring import combine, ml_score

ROOT = Path(__file__).resolve().parent.parent


def test_repo_policy_loads() -> None:
    p = Policy.load(ROOT / "config" / "policy.yaml")
    assert p.output_action("credit_card") == Action.REDACT
    assert p.output_action("canary_leak") == Action.BLOCK
    assert p.output_action("something_new") == Action.LOG


def test_thresholds_map_to_actions() -> None:
    p = Policy.from_dict({"input": {"thresholds": {"block": 0.8, "log": 0.5}}})
    assert p.input_action(0.95) == Action.BLOCK
    assert p.input_action(0.80) == Action.BLOCK
    assert p.input_action(0.60) == Action.LOG
    assert p.input_action(0.10) == Action.ALLOW


def test_invalid_policy_rejected() -> None:
    with pytest.raises(ValueError):
        Policy.from_dict({"input": {"thresholds": {"block": 0.3, "log": 0.6}}})
    with pytest.raises(ValueError):
        Policy.from_dict({"input": {"combination": "average_vibes"}})


def test_strictest_action_wins() -> None:
    assert strictest([Action.ALLOW, Action.REDACT, Action.LOG]) == Action.REDACT
    assert strictest([Action.REDACT, Action.BLOCK]) == Action.BLOCK
    assert strictest([]) == Action.ALLOW


def test_missing_transformer_weight_renormalises() -> None:
    w = {"linear": 0.4, "transformer": 0.6}
    assert ml_score({"linear": 0.7, "transformer": None}, w) == pytest.approx(0.7)
    assert ml_score({"linear": 0.5, "transformer": 1.0}, w) == pytest.approx(0.8)
    assert ml_score({"linear": None, "transformer": None}, w) is None


@pytest.mark.parametrize(
    ("combination", "expected"),
    [("noisy_or", 1 - 0.5 * 0.4), ("max", 0.6), ("weighted_mean", (0.5 * 1.0 + 0.6 * 0.4) / 1.4)],
)
def test_combinations(combination: str, expected: float) -> None:
    p = Policy.from_dict({"input": {"combination": combination,
                                    "weights": {"heuristics": 1.0, "linear": 0.4, "transformer": 0.6}}})
    assert combine({"heuristics": 0.5, "linear": 0.6, "transformer": None}, p) == pytest.approx(expected)
