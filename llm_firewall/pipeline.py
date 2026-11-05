"""The firewall pipeline: normalize -> Layer A/B/C -> combine -> policy.

This module has no web code in it. The proxy, the CLI, the demo page and the
evaluation suite all call the same `Firewall.scan_input` / `scan_output`, so
what we measure in the eval is exactly what runs in production.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from llm_firewall.config import Settings
from llm_firewall.detectors.heuristics import heuristic_score
from llm_firewall.detectors.linear import LinearModel
from llm_firewall.detectors.transformer import TransformerDetector
from llm_firewall.normalize import normalize
from llm_firewall.output_scan import redact, scan_output
from llm_firewall.policy import Action, Policy, strictest
from llm_firewall.scoring import combine


@dataclass
class InputVerdict:
    action: Action
    risk: float
    scores: dict[str, float | None]  # per layer; None = layer not available
    rule_hits: list[dict[str, Any]]
    obfuscation_signals: list[str]
    normalized_text: str
    decoded_payloads: list[str]
    top_features: list[tuple[str, float]]
    latency_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


@dataclass
class OutputVerdict:
    action: Action
    text: str  # what the client receives (redacted, or the block message)
    findings: dict[str, int]  # type -> count
    canary_leaked: bool
    system_prompt_overlap: float
    latency_ms: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class Firewall:
    def __init__(self, settings: Settings, policy: Policy | None = None, load_transformer: bool = True) -> None:
        self.settings = settings
        self.policy = policy or Policy.load(settings.policy_path)
        self.linear: LinearModel | None = (
            LinearModel.load(settings.linear_model_path) if settings.linear_model_path.exists() else None
        )
        self.transformer: TransformerDetector | None = None
        if load_transformer:
            det = TransformerDetector(settings.transformer_model_dir)
            self.transformer = det if det.available else None

    @property
    def layers(self) -> dict[str, bool]:
        return {"heuristics": True, "linear": self.linear is not None, "transformer": self.transformer is not None}

    # ------------------------------------------------------------ input
    def scan_input(self, text: str) -> InputVerdict:
        lat: dict[str, float] = {}
        t0 = time.perf_counter()
        norm = normalize(text)
        lat["normalize"] = (time.perf_counter() - t0) * 1000

        t = time.perf_counter()
        heur = heuristic_score(norm)
        lat["heuristics"] = (time.perf_counter() - t) * 1000

        model_text = norm.model_text()
        scores: dict[str, float | None] = {"heuristics": heur.score, "linear": None, "transformer": None}
        top: list[tuple[str, float]] = []
        if self.linear is not None:
            t = time.perf_counter()
            scores["linear"] = round(self.linear.predict_proba(model_text), 4)
            top = [(term, round(c, 4)) for term, c in self.linear.explain(model_text, top_k=6) if c > 0]
            lat["linear"] = (time.perf_counter() - t) * 1000
        if self.transformer is not None:
            t = time.perf_counter()
            scores["transformer"] = round(self.transformer.predict_proba(model_text), 4)
            lat["transformer"] = (time.perf_counter() - t) * 1000

        risk = round(combine(scores, self.policy), 4)
        lat["total"] = (time.perf_counter() - t0) * 1000
        return InputVerdict(
            action=self.policy.input_action(risk),
            risk=risk,
            scores=scores,
            rule_hits=[{"rule": h.rule_id, "category": h.category, "severity": h.severity, "matched": h.matched}
                       for h in heur.hits],
            obfuscation_signals=heur.obfuscation_signals,
            normalized_text=norm.normalized[:2000],
            decoded_payloads=[d[:500] for d in norm.decoded],
            top_features=top,
            latency_ms={k: round(v, 3) for k, v in lat.items()},
        )

    def scan_messages(self, messages: list[dict[str, Any]]) -> InputVerdict:
        """Scan every untrusted message and return the riskiest verdict.

        System messages come from the application developer and are trusted.
        User messages and tool/function results (web pages, files, API output —
        the usual carrier of *indirect* injection) are not.
        """
        verdicts = [
            self.scan_input(_content_text(m.get("content")))
            for m in messages
            if m.get("role") in {"user", "tool", "function"} and _content_text(m.get("content")).strip()
        ]
        if not verdicts:
            return self.scan_input("")
        return max(verdicts, key=lambda v: v.risk)

    # ----------------------------------------------------------- output
    def scan_output(self, text: str, canary: str = "", system_prompt: str = "") -> OutputVerdict:
        t0 = time.perf_counter()
        res = scan_output(text, canary=canary, system_prompt=system_prompt)
        reasons: list[str] = []
        actions: list[Action] = []
        to_redact: set[str] = set()
        for ftype in res.counts:
            a = self.policy.output_action(ftype)
            actions.append(a)
            reasons.append(f"{ftype}:{a.value}")
            if a == Action.REDACT:
                to_redact.add(ftype)
        if res.canary_leaked:
            a = self.policy.output_action("canary_leak")
            actions.append(a)
            reasons.append(f"canary_leak:{a.value}")
        if system_prompt and res.system_prompt_overlap >= self.policy.system_prompt_overlap_threshold:
            a = self.policy.output_action("system_prompt_leak")
            actions.append(a)
            reasons.append(f"system_prompt_leak:{a.value}")

        action = strictest(actions)
        if action == Action.BLOCK:
            out_text = self.policy.output_block_message
        else:
            out_text = redact(text, res.findings, types=to_redact) if to_redact else text
        return OutputVerdict(
            action=action,
            text=out_text,
            findings=res.counts,
            canary_leaked=res.canary_leaked,
            system_prompt_overlap=res.system_prompt_overlap,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            reasons=reasons,
        )


def _content_text(content: Any) -> str:
    """OpenAI messages may carry a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)
