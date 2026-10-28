# E2: Multi-Head Self-Attention and tokenization (CIFAR-10)

Own MSA (`Linear` + `einsum` + `softmax`) vs `nn.MultiheadAttention`, four tokenizers (patch 4x4, patch 8x8, row, CNN-stem), shallow Transformer classifier, hand-written training loop. Report chapter: `E2/report/chapter-e2.tex`; web page: `docs/e2.html`.

Run from the repo root. Environment is the same (Python 3.10+, PyTorch with CUDA, torchvision preinstalled; `pip install -r requirements/e.txt` for the rest).
```
export PYTHONPATH=.:E2        # Windows PowerShell: $env:PYTHONPATH = ".;E2"   (or: pip install -e .)
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

Status: torch-dependent code was written without a GPU in the authoring sandbox; run steps 1 to 3 first.
AI use: Claude assisted with code, tests and report draft; the authors reviewed and are responsible for it.
