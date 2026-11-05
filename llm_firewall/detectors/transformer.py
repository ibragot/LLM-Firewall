"""Layer C: fine-tuned transformer classifier (optional, local/Docker only).

Loaded lazily from TRANSFORMER_MODEL_DIR (produced by
`training/train_transformer.py` or the Colab notebook). If torch/transformers
are not installed, or the directory does not exist, the layer reports itself as
unavailable and the pipeline runs on Layers A + B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TransformerDetector:
    def __init__(self, model_dir: str | Path, max_length: int = 256) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self.error: str | None = None
        self._load()

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if not (self.model_dir / "config.json").exists():
            self.error = f"no trained model at {self.model_dir}"
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            self.error = "torch/transformers not installed (pip install -r requirements-transformer.txt)"
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.eval()

    def predict_proba(self, text: str) -> float:
        return self.predict_proba_batch([text])[0]

    def predict_proba_batch(self, texts: list[str]) -> list[float]:
        if not self.available:
            raise RuntimeError(self.error or "transformer unavailable")
        import torch

        enc = self._tokenizer(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        return [float(p) for p in probs]
