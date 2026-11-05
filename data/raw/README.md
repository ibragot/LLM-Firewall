# Training data (public datasets)

These files are copies of two public prompt-injection / jailbreak datasets, committed so that
training is reproducible offline and in CI. They are **only used for training and validation**;
the red-team evaluation suite in `eval/data/` is deduplicated against them.

| Folder | Original source | Files | License |
|---|---|---|---|
| `deepset/` | Hugging Face [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) (the dataset's parquet files, obtained via the GitHub mirror `sinanw/llm-security-prompt-injection` because Hugging Face was not reachable from the build environment) | `train.parquet` (546 rows), `test.parquet` (116 rows); columns `text`, `label` (1 = injection) | Apache-2.0 (per the dataset card) |
| `jackhhao/` | Hugging Face [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification), obtained from the author's own GitHub repo `jackhhao/llm-warden` (`datasets/default/`) | `train.csv` (1598 rows), `test.csv` (400 rows); columns `prompt`, `type` (`jailbreak`/`benign`) | MIT (see `jackhhao/LICENSE`) |

SHA-256 checksums (verify with `sha256sum data/raw/*/*`):

```
2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04  deepset/train.parquet
39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e  deepset/test.parquet
a5cb24b0865ef37a248d6d82bd21026ce39c89f11abae5d26d4ae58aad3ffe40  jackhhao/train.csv
a546ce540973d8f5ef87ce3be4bd21ee5923154b9dda86eba5a12c7fa00829da  jackhhao/test.csv
```

To re-download from Hugging Face instead (needs network access to huggingface.co):

```bash
python -m training.download_data
```
