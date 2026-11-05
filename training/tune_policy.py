"""Choose how to combine layers and where to put thresholds, using validation + red-team DEV data only.

    python -m training.tune_policy

Thresholds are hyper-parameters. If we picked them by looking at the test set,
the test numbers would be optimistic: we'd have tuned on the exam. So we tune on
  * the validation split held out from the public training data, and
  * the DEV half of the red-team suite (the TEST half is never looked at),
write the result to models/policy_tuning.json, and copy the chosen values into
config/policy.yaml by hand (a reviewed change).

History: v1 tuned on public validation only and chose block=0.43. That threshold
did not transfer: on red-team dev it flagged 13% of benign prompts, because the
TF-IDF model gives ~0.4-0.6 to unfamiliar benign text (distribution shift). v2
adds the dev half so the threshold reflects the traffic we care about.

Selection rule (a product decision, written down so it can be argued with):
  block threshold = highest F1 among thresholds with FPR <= 3% on the tuning data
  log threshold   = highest recall among thresholds with FPR <= 10% (it only flags)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_firewall.config import Settings
from llm_firewall.pipeline import Firewall
from llm_firewall.policy import Policy
from llm_firewall.scoring import combine
from training.data import ROOT, load_redteam, load_train_val

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


def redteam_dev() -> pd.DataFrame:
    rt = load_redteam()
    is_dev = rt["id"].map(lambda i: int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 == 0)
    return rt[is_dev].reset_index(drop=True)


def main() -> None:
    _, val, _ = load_train_val()
    dev = redteam_dev()
    tune = pd.concat([val[["text", "label"]], dev[["text", "label"]]], ignore_index=True)
    fw = Firewall(Settings.from_env(), load_transformer=False)
    y = tune["label"].to_numpy()
    layer_scores = [fw.scan_input(t).scores for t in tune["text"]]

    results: dict[str, object] = {"validation_rows": len(val), "redteam_dev_rows": len(dev), "strategies": {}}
    for strategy in ("noisy_or", "max", "weighted_mean"):
        pol = Policy.from_dict({"input": {"combination": strategy,
                                          "weights": {"heuristics": 1.0, "linear": 0.4, "transformer": 0.6}}})
        s = np.array([combine(sc, pol) for sc in layer_scores])
        rows = sweep(y, s)
        block_cands = [r for r in rows if r["fpr"] <= 0.03] or rows
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
