"""Train Layer B: TF-IDF (word + char n-grams) + logistic regression.

    python -m training.train_baseline

Outputs
  models/linear_model.json        weights for pure-Python inference (llm_firewall/detectors/linear.py)
  models/linear_model_meta.json   data stats, chosen hyper-parameters, validation metrics

Why a baseline first? It trains in seconds, it is fully interpretable (every
feature has one weight you can print), and it tells you how hard the problem
is. A fancy model is only worth its cost if it beats this.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from training.data import ROOT, load_train_val, model_text

SEED = 42


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion(
        [
            # Word unigrams + bigrams: "ignore previous", "system prompt".
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20000,
                                     sublinear_tf=True)),
            # Character 3-5 grams inside word boundaries: robust to typos ("igmre"),
            # morphology ("ignoriere"/"ignorieren") and leftover obfuscation.
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000,
                                     sublinear_tf=True)),
        ]
    )
    # class_weight="balanced" re-weights the loss so the minority class (attacks,
    # ~1/3 of the data) counts as much as the majority class in total.
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def export_json(pipe: Pipeline, path: Path, meta: dict[str, object]) -> None:
    """Write vocabulary, IDF and weights so inference needs no scikit-learn."""
    union: FeatureUnion = pipe.named_steps["features"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    coef = clf.coef_[0]
    blocks = []
    offset = 0
    for _, vec in union.transformer_list:
        vocab = vec.vocabulary_
        idf = vec.idf_
        block_vocab = {}
        for term, idx in vocab.items():
            block_vocab[term] = [round(float(idf[idx]), 6), round(float(coef[offset + idx]), 6)]
        blocks.append({
            "analyzer": vec.analyzer,
            "ngram_range": list(vec.ngram_range),
            "sublinear_tf": vec.sublinear_tf,
            "vocab": block_vocab,
        })
        offset += len(vocab)
    payload = {"format": "tfidf-logreg-v1", "intercept": float(clf.intercept_[0]), "blocks": blocks, "meta": meta}
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "models" / "linear_model.json"))
    args = ap.parse_args()

    train, val, stats = load_train_val()
    print(f"data: {stats}")
    print(f"train class balance: {train['label'].mean():.3f} attacks; val: {val['label'].mean():.3f}")
    x_train = [model_text(t) for t in train["text"]]
    x_val = [model_text(t) for t in val["text"]]
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    # Small grid over regularisation strength C (smaller C = simpler model).
    # Selection metric: average precision on the validation set (threshold-free,
    # and more informative than accuracy when classes are imbalanced).
    grid = []
    for C in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        pipe = build_pipeline(C).fit(x_train, y_train)
        p = pipe.predict_proba(x_val)[:, 1]
        ap_score = average_precision_score(y_val, p)
        grid.append({"C": C, "val_average_precision": round(float(ap_score), 4),
                     "val_roc_auc": round(float(roc_auc_score(y_val, p)), 4)})
        print(f"C={C:<5} val AP={ap_score:.4f}  ROC-AUC={grid[-1]['val_roc_auc']:.4f}")
    best_C = max(grid, key=lambda g: (g["val_average_precision"], -g["C"]))["C"]

    t0 = time.perf_counter()
    pipe = build_pipeline(best_C).fit(x_train, y_train)
    fit_s = time.perf_counter() - t0
    p_val = pipe.predict_proba(x_val)[:, 1]
    meta = {
        "data": stats,
        "chosen_C": best_C,
        "grid": grid,
        "fit_seconds": round(fit_s, 2),
        "val_roc_auc": round(float(roc_auc_score(y_val, p_val)), 4),
        "val_average_precision": round(float(average_precision_score(y_val, p_val)), 4),
        "val_f1_at_0.5": round(float(f1_score(y_val, (p_val >= 0.5).astype(int))), 4),
        "n_features": int(sum(len(v.vocabulary_) for _, v in pipe.named_steps["features"].transformer_list)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_json(pipe, out, meta)
    (out.parent / "linear_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in meta.items() if k != "grid"}, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
