"""Re-download the public training datasets from Hugging Face into data/raw/.

    python -m training.download_data

The repo already ships copies (see data/raw/README.md); this script exists so
anyone can verify they came from the official sources.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    ("deepset", "train"): "hf://datasets/deepset/prompt-injections/data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    ("deepset", "test"): "hf://datasets/deepset/prompt-injections/data/test-00000-of-00001-701d16158af87368.parquet",
    ("jackhhao", "train"): "hf://datasets/jackhhao/jailbreak-classification/jailbreak_dataset_train.csv",
    ("jackhhao", "test"): "hf://datasets/jackhhao/jailbreak-classification/jailbreak_dataset_test.csv",
}


def main() -> None:
    # pandas reads hf:// paths through `huggingface_hub` (pip install huggingface_hub).
    for (name, split), url in SOURCES.items():
        out_dir = RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if url.endswith(".parquet"):
            pd.read_parquet(url).to_parquet(out_dir / f"{split}.parquet", index=False)
        else:
            pd.read_csv(url).to_csv(out_dir / f"{split}.csv", index=False)
        print(f"downloaded {name}/{split}")


if __name__ == "__main__":
    main()
