"""Choose how to combine layers and where to put thresholds, using ONLY validation data.

    python -m training.tune_policy

Thresholds are hyper-parameters. If we picked them by looking at the test set
(the red-team suite), the test numbers would be optimistic: we'd have tuned
on the exam. So we tune on the validation split held out from the public
training data, write the result to models/policy_tuning.json, and copy the
chosen values into config/policy.yaml by hand (a reviewed change).

Selection rule (a product decision, written down so it can be argued with):
  block threshold = the threshold with the highest F1 among those with
                    validation false-positive rate <= 2%
  log threshold   = the threshold with the highest recall among those with
                    validation FPR <= 10% (it only flags; it doesn't block)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_firewall.config import Settings
from llm_firewall.pipeline import Firewall
from llm_firewall.policy import Policy
from llm_firewall.scoring import combine
from training.data import ROOT, load_train_val

THRESHOLDS = np.round(np.arange(0.05, 0.99, 0.01), 2)


def sweep(y: np.ndarray, s: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for t in THRESHOLDS:
        pred = s >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1, "fpr": fp / (fp + tn) if fp + tn else 0.0})
    return rows


def main() -> None:
    _, val, _ = load_train_val()
    fw = Firewall(Settings.from_env(), load_transformer=False)
    y = val["label"].to_numpy()
    layer_scores = [fw.scan_input(t).scores for t in val["text"]]

    results: dict[str, object] = {"validation_rows": len(val), "strategies": {}}
    for strategy in ("noisy_or", "max", "weighted_mean"):
        pol = Policy.from_dict({"input": {"combination": strategy,
                                          "weights": {"heuristics": 1.0, "linear": 0.4, "transformer": 0.6}}})
        s = np.array([combine(sc, pol) for sc in layer_scores])
        rows = sweep(y, s)
        block_cands = [r for r in rows if r["fpr"] <= 0.02] or rows
        block = max(block_cands, key=lambda r: (r["f1"], -r["threshold"]))
        log_cands = [r for r in rows if r["fpr"] <= 0.10] or rows
        log = max(log_cands, key=lambda r: (r["recall"], r["threshold"]))
        results["strategies"][strategy] = {"block": block, "log": log}  # type: ignore[index]
        print(f"{strategy:<14} block@{block['threshold']:.2f}: F1={block['f1']:.3f} P={block['precision']:.3f} "
              f"R={block['recall']:.3f} FPR={block['fpr']:.3f} | log@{log['threshold']:.2f}: R={log['recall']:.3f} FPR={log['fpr']:.3f}")

    best = max(results["strategies"].items(), key=lambda kv: kv[1]["block"]["f1"])  # type: ignore[union-attr]
    results["chosen"] = {"combination": best[0], "block": best[1]["block"]["threshold"], "log": best[1]["log"]["threshold"]}
    print("chosen:", results["chosen"])
    out = Path(ROOT / "models" / "policy_tuning.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
