"""Policy engine: turn detector outputs into actions, driven by config/policy.yaml.

Keeping the *decisions* in YAML (not code) lets a security team tighten or relax
behaviour — e.g. switch emails from `redact` to `log` — without a code change or
redeploy of the detectors, and makes the policy reviewable in a pull request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Action(StrEnum):
    ALLOW = "allow"
    LOG = "log"
    REDACT = "redact"
    BLOCK = "block"


# Severity order: when several detections disagree, the strictest action wins.
_STRICTNESS = {Action.ALLOW: 0, Action.LOG: 1, Action.REDACT: 2, Action.BLOCK: 3}


def strictest(actions: list[Action]) -> Action:
    return max(actions, key=lambda a: _STRICTNESS[a], default=Action.ALLOW)


@dataclass
class Policy:
    combination: str = "noisy_or"
    weights: dict[str, float] = field(default_factory=lambda: {"heuristics": 1.0, "linear": 0.4, "transformer": 0.6})
    block_threshold: float = 0.8
    log_threshold: float = 0.5
    action_high: Action = Action.BLOCK
    action_medium: Action = Action.LOG
    action_low: Action = Action.ALLOW
    input_block_message: str = "Request blocked by LLM-Firewall."
    output_actions: dict[str, Action] = field(default_factory=dict)
    system_prompt_overlap_threshold: float = 0.3
    output_block_message: str = "Response withheld by LLM-Firewall."

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Policy:
        inp = cfg.get("input", {})
        out = cfg.get("output", {})
        combination = inp.get("combination", "noisy_or")
        if combination not in {"noisy_or", "max", "weighted_mean"}:
            raise ValueError(f"unknown combination: {combination}")
        th = inp.get("thresholds", {})
        block_t, log_t = float(th.get("block", 0.8)), float(th.get("log", 0.5))
        if not 0.0 <= log_t <= block_t <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= log <= block <= 1")
        return cls(
            combination=combination,
            weights={k: float(v) for k, v in inp.get("weights", {}).items()} or cls().weights,
            block_threshold=block_t,
            log_threshold=log_t,
            action_high=Action(inp.get("action_high", "block")),
            action_medium=Action(inp.get("action_medium", "log")),
            action_low=Action(inp.get("action_low", "allow")),
            input_block_message=inp.get("block_message", cls.input_block_message),
            output_actions={k: Action(v) for k, v in out.get("actions", {}).items()},
            system_prompt_overlap_threshold=float(out.get("system_prompt_overlap_threshold", 0.3)),
            output_block_message=out.get("block_message", cls.output_block_message),
        )

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def input_action(self, risk: float) -> Action:
        if risk >= self.block_threshold:
            return self.action_high
        if risk >= self.log_threshold:
            return self.action_medium
        return self.action_low

    def output_action(self, detection_type: str) -> Action:
        # Unknown detection types default to `log` (fail visible, not silent).
        return self.output_actions.get(detection_type, Action.LOG)
