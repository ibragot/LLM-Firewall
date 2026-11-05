"""Print every intermediate value the firewall computes for one prompt.

    python -m eval.trace "Ignore previous instructions"
    python -m eval.trace --json "..."       # machine-readable

Used to produce the "Full trace" section of the README.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

from llm_firewall.config import Settings
from llm_firewall.detectors.heuristics import OBFUSCATION_SIGNAL_WEIGHTS, heuristic_score
from llm_firewall.normalize import normalize
from llm_firewall.pipeline import Firewall
from llm_firewall.scoring import ml_score


def trace(text: str) -> dict[str, Any]:
    fw = Firewall(Settings.from_env(), load_transformer=True)
    norm = normalize(text)
    heur = heuristic_score(norm)
    out: dict[str, Any] = {
        "1_input": text,
        "2_normalization": {
            "clean (invisible chars removed, NFKC, homoglyphs fixed)": norm.clean,
            "normalized (lower-cased, spaced letters joined)": norm.normalized,
            "deleeted": norm.deleeted,
            "decoded_payloads": norm.decoded,
            "signals": norm.signals,
            "model_text (fed to ML layers)": norm.model_text(),
        },
        "3_layer_A_heuristics": {
            "rule_hits": [{"rule": h.rule_id, "severity": h.severity, "matched": h.matched} for h in heur.hits],
            "obfuscation_signal_weights": {s: OBFUSCATION_SIGNAL_WEIGHTS[s] for s in heur.obfuscation_signals},
            "noisy_or_score": heur.score,
        },
    }
    if fw.linear is not None:
        mt = norm.model_text()
        blocks = []
        for b in fw.linear.blocks:
            total, terms = b.contribution(mt)
            top = sorted(terms, key=lambda kv: abs(kv[1]), reverse=True)[:6]
            blocks.append({"analyzer": b.analyzer, "n_features_present": len(terms), "logit_contribution": round(total, 4),
                           "largest_terms": [(t, round(c, 4)) for t, c in top]})
        z = fw.linear.logit(mt)
        out["4_layer_B_linear"] = {"intercept": round(fw.linear.intercept, 4), "blocks": blocks, "logit_z": round(z, 4),
                                   "sigmoid(z)": round(1 / (1 + math.exp(-z)), 4)}
    v = fw.scan_input(text)
    out["5_layer_C_transformer"] = v.scores["transformer"] if fw.transformer else "not loaded (no trained model)"
    m = ml_score(v.scores, fw.policy.weights)
    out["6_combination"] = {
        "strategy": fw.policy.combination,
        "ml_score (weighted mean of available ML layers)": None if m is None else round(m, 4),
        "risk": v.risk,
        "formula": f"1 - (1 - {v.scores['heuristics']}) * (1 - {None if m is None else round(m, 4)})",
    }
    out["7_policy"] = {"block_threshold": fw.policy.block_threshold, "log_threshold": fw.policy.log_threshold,
                       "action": v.action.value}
    out["latency_ms"] = v.latency_ms
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    t = trace(" ".join(args.text))
    if args.json:
        print(json.dumps(t, indent=2, ensure_ascii=False))
        return
    for k, v in t.items():
        print(f"\n## {k}")
        print(json.dumps(v, indent=2, ensure_ascii=False) if not isinstance(v, str) else v)


if __name__ == "__main__":
    main()
