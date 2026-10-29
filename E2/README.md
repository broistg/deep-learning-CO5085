# E2: Multi-Head Self-Attention and tokenization (CIFAR-10)

Own MSA (`Linear` + `einsum` + `softmax`) vs `nn.MultiheadAttention`, four tokenizers (patch 4x4, patch 8x8, row, CNN-stem), shallow Transformer classifier, hand-written training loop. Report: the E2 chapter and the appendix "Tái lập kết quả E2" of `reports/exercises.pdf`; web page: `docs/exercises/e2.html`.

Run from the repo root. Environment as in A1 (Python 3.10+, PyTorch with CUDA, torchvision preinstalled; `pip install -r requirements/e.txt` for the rest; `pip install matplotlib` is in base).
```
export PYTHONPATH=.:E2        # or: pip install -e .
python -m unittest discover -s E2/tests -v          # 1. correctness (MSA scratch == torch, token shapes, split)
python -m e2lib.bench                               # 2. smoke benchmark: ms/step, memory, estimated hours for all 30 runs
python -m e2lib.train --tokenizer row --attn torch --fake --smoke   # 3. dry run, synthetic data (no download)
python -m e2lib.plan --list                         # which of the 30 runs are done
python -m e2lib.plan                                # 4. run everything not finished (resumable; Ctrl-C and rerun is safe)
python -m e2lib.analyze                             # 5. tables (results/tab_*.tex) and figures (figures/)
python -m e2lib.viz                                 # 6. attention maps (needs artifacts/ from step 4)
```
Runs: 4 tokenizers x {scratch, torch} x seeds 0/1/2 = 24, plus positional-encoding ablation (patch4, row; scratch; 3 seeds) = 6; 30 epochs each. Outputs: `E2/results/<run_id>.json` (commit), `E2/artifacts/` (weights, logits, checkpoints; not committed).

Split: `common/cifar.py` (stratified 45k/5k via `common/split.py`, seed 42) writes `common/cifar10_split.json` with a SHA-256 on first use; E1 and E3 use the same file. Channel mean/std are computed from the 45k train images (`channel_stats`). The CIFAR download is shared: `data/` at the repo root. Check `python -m common.cifar` prints the same split hash on every machine.

Status: all 30 runs finished (RTX 3050 Laptop, about 2.5 h); the replication notebook was checked on Colab.

## Replication notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E2/notebooks/E2.ipynb)

`E2/notebooks/E2.ipynb` is a thin notebook (no logic inside; it calls `e2lib` and `common`). `MODE = "load"` (default) regenerates the tables and figures from the committed `E2/results/` and checks that they equal the committed (= reported) ones; it runs in minutes on a free CPU runtime. `MODE = "rerun"` retrains everything into a fresh `results/` (the published results are kept in `results_published/` and compared with the rerun; resumable; GPU needed). The confusion matrix and error gallery need the test logits from the Release (`FETCH_WEIGHTS = True`).
Weights and logits are downloaded from the GitHub Release `weights-v1` and verified by SHA-256 (see `common/README.md`). CIFAR-10 comes from the Hugging Face Hub through the `datasets` library, verified against a fingerprint of the TorchVision data (`python -m common.cifar`).

AI use: Claude assisted with code, tests and report draft; the authors reviewed and are responsible for it.
