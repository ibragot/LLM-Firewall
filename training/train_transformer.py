"""Train Layer C: fine-tune a small pretrained transformer as a binary classifier.

    # real run (needs internet for the pretrained weights; GPU recommended -> see notebooks/)
    python -m training.train_transformer --model distilbert-base-uncased --epochs 3

    # offline smoke test: tiny randomly-initialised DistilBERT + a WordPiece
    # tokenizer trained on our own data. Proves the whole pipeline runs end to
    # end (data -> tokenize -> train -> evaluate -> save -> load in the firewall).
    # Its scores are meaningless; it is NOT the real Layer C.
    python -m training.train_transformer --smoke

Transfer learning in one paragraph: `distilbert-base-uncased` was pretrained by
Hugging Face on English Wikipedia + books to predict masked words, so its 66M
weights already encode a lot about language. We add a fresh 2-way
classification head on top and train *all* weights for a few epochs at a small
learning rate (fine-tuning), so the general language knowledge is adapted to
"is this an attack?" using only ~1.8k labelled examples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.data import ROOT, load_train_val, model_text


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_smoke_model(texts: list[str], max_length: int) -> tuple[Any, Any]:
    """Tiny DistilBERT (2 layers, dim 64) + WordPiece tokenizer trained on `texts`. No downloads."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train_from_iterator(texts, vocab_size=3000, min_frequency=2)
    # Wrap the trained `tokenizers` object directly. (Passing vocab_file= to
    # DistilBertTokenizerFast is silently ignored by recent transformers versions,
    # which left a 5-token vocabulary where every word was [UNK].)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=wp._tokenizer, model_max_length=max_length, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    if len(tokenizer) < 100:
        raise RuntimeError(f"smoke tokenizer vocabulary is only {len(tokenizer)} tokens; tokenizer construction failed")
    config = DistilBertConfig(vocab_size=len(tokenizer), n_layers=2, n_heads=2, dim=64, hidden_dim=128,
                              max_position_embeddings=max_length)
    config.num_labels = 2
    return tokenizer, DistilBertForSequenceClassification(config)


def evaluate(model: Any, loader: Any, device: Any) -> tuple[float, np.ndarray]:
    import torch

    model.eval()
    losses, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss))
            probs.append(torch.softmax(out.logits, dim=-1)[:, 1].float().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilbert-base-uncased",
                    help="Hugging Face model id, e.g. distilbert-base-uncased or microsoft/deberta-v3-small")
    ap.add_argument("--out", default=None, help="output dir (default models/transformer or models/transformer-smoke)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps for linear LR warm-up")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny offline model on a small sample (pipeline test)")
    ap.add_argument("--limit", type=int, default=0, help="use only N training rows (debugging)")
    args = ap.parse_args()

    import torch
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or (ROOT / "models" / ("transformer-smoke" if args.smoke else "transformer")))

    train, val, stats = load_train_val()
    if args.smoke:
        # Small stratified sample so the smoke test runs in about a minute on a laptop CPU.
        def per_class(df: pd.DataFrame, n: int) -> pd.DataFrame:
            return pd.concat([g.sample(min(len(g), n), random_state=args.seed) for _, g in df.groupby("label")])

        train, val = per_class(train, 96), per_class(val, 48)
        args.epochs = min(args.epochs, 2)
    elif args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=args.seed)
    x_train = [model_text(t) for t in train["text"]]
    x_val = [model_text(t) for t in val["text"]]
    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()
    print(f"device={device} train={len(x_train)} val={len(x_val)} attack-rate={y_train.mean():.3f}")

    if args.smoke:
        tokenizer, model = build_smoke_model(x_train, args.max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, id2label={0: "benign", 1: "attack"}, label2id={"benign": 0, "attack": 1})
    model.to(device)

    def collate(rows: list[tuple[str, int]]) -> dict[str, Any]:
        texts, labels = zip(*rows, strict=True)
        # Truncation: anything past max_length tokens is cut. Long jailbreaks usually
        # reveal themselves early, and attention cost grows with length^2.
        enc = tokenizer(list(texts), truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return dict(enc)

    # A plain list of (text, label) pairs works as a map-style Dataset (it has __len__ and __getitem__).
    train_pairs: Any = list(zip(x_train, y_train.tolist(), strict=True))
    val_pairs: Any = list(zip(x_val, y_val.tolist(), strict=True))
    train_loader: Any = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader: Any = DataLoader(val_pairs, batch_size=args.batch_size * 2, collate_fn=collate)

    # Class imbalance: weight the loss by inverse class frequency, so a mistake on
    # the rarer "attack" class costs ~2x as much as one on a benign prompt.
    counts = np.bincount(y_train, minlength=2)
    class_weights = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"class weights: benign={class_weights[0]:.3f} attack={class_weights[1]:.3f}")

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    lr = 1e-3 if args.smoke else args.lr  # random init needs a bigger step than fine-tuning
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = math.ceil(len(train_loader)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    history: list[dict[str, float]] = []
    best_ap, best_state = -1.0, None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")
        val_loss, p_val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": round(running / max(1, len(train_loader)), 4), "val_loss": round(val_loss, 4),
               "val_roc_auc": round(float(roc_auc_score(y_val, p_val)), 4),
               "val_average_precision": round(float(average_precision_score(y_val, p_val)), 4),
               "val_f1_at_0.5": round(float(f1_score(y_val, (p_val >= 0.5).astype(int))), 4)}
        history.append(row)
        print(json.dumps(row))
        # Early-stopping style model selection: keep the epoch with the best validation AP.
        if row["val_average_precision"] > best_ap:
            best_ap = row["val_average_precision"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {"base_model": "tiny-random-distilbert (smoke test)" if args.smoke else args.model, "smoke": args.smoke,
            "data": stats, "train_rows": len(x_train), "val_rows": len(x_val), "epochs": args.epochs, "lr": lr,
            "batch_size": args.batch_size, "max_length": args.max_length, "device": str(device),
            "train_seconds": round(time.time() - t_start, 1), "history": history,
            "best_val_average_precision": best_ap}
    (out_dir / "training_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in out_dir.glob("*")) / 1e6
    print(f"saved to {out_dir} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
