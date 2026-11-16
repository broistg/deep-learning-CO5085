# E3: LSTM and GRU on CIFAR-10, with E1 baselines

E3 classifies CIFAR-10 images as sequences of 32 image rows or 64 non-overlapping 4×4 patches. Each representation is paired with an LSTM and a GRU. All E3 runs use `embed_dim=64`, `hidden_size=128`, and one recurrent layer. The MLP and MiniVGG baselines come directly from the six completed E1 runs in `E1/results/`; E3 does not train them again.

Run from the repository root with Python 3.10+, PyTorch and the dependencies in `requirements/e.txt`. Use the local venv if available:

```bash
export PYTHONPATH=.:E3    # Windows PowerShell: $env:PYTHONPATH = ".;E3"
venv/bin/python -m unittest discover -s E3/tests -v
venv/bin/python -m unittest discover -s common/tests -v
venv/bin/python -m e3lib.train --model lstm --tokenizer row --seed 0 --smoke
venv/bin/python -m e3lib.plan --list
venv/bin/python -m e3lib.plan
venv/bin/python -m e3lib.analyze --out E3/regenerated
venv/bin/python -m e3lib.diagnostics --out E3/regenerated
```

The E3 plan has four recurrent configurations × seeds 0, 1 and 2: **12 runs of 30 epochs**. All 12 results are published in `E3/results/`. `e3lib.plan` verifies and skips completed runs or resumes interrupted checkpoints. A smoke run uses synthetic data for one epoch and saves nothing. You can also select a run with `venv/bin/python -m e3lib.train --model gru --tokenizer patch4 --seed 1`.

`e3lib.train.Cfg` uses Adam with `lr=1e-3`, no weight decay and a constant learning rate, matching the E1 optimizer. Training uses batch size 64, standard cross-entropy, FP32 and global gradient clipping at 1.0 for the recurrent models. Evaluation batch size is 500. The same stratified 45,000/5,000 split, 10,000 test images, training-only channel statistics and training-only crop/flip augmentation are shared through `common/cifar.py`. E3's data batching implementation is independent, so comparing against E1 is a model-level comparison on the same split and training settings, not a bitwise-identical training trace.

Each real run writes `E3/results/<model>_<tokenizer>_s<seed>.json` and keeps checkpoints, final weights and test logits under `E3/artifacts/<run_id>/`. Validation is measured after every epoch; reported validation and test accuracies use the final epoch's weights. If final files exist but the JSON is missing, rerunning the plan rebuilds the final evaluation from a valid checkpoint without starting at epoch 1. `--force` restarts only the selected run.

For comparison with E1, both sets use the same CIFAR-10 split, train-only channel statistics, crop/flip augmentation recipe, seeds 0–2, 30 epochs, batch size 64, Adam at `1e-3`, and no weight decay. E3 additionally clips recurrent gradients at 1.0 and generates augmentation randomness independently. Model sizes and compute are not matched; report parameter counts and training times alongside accuracy. The hidden-128 choice came from an E3 validation pilot, so evaluate the fixed configuration across all three seeds and avoid selecting architectures by test accuracy.

`e3lib.analyze` reads completed E3 LSTM and GRU runs only. It verifies their configurations, split hash and channel statistics, then writes `summary.json`, one LaTeX table and learning curves. The summary includes mean ± sample standard deviation over seeds for each model and tokenizer, plus the mean peak GPU allocation when all three seeds are available. At least one completed E3 run is required.

`e3lib.diagnostics` reads saved E3 test logits and CIFAR-10 test images. By default it chooses the completed run with the highest test accuracy; `--run-id` lets you choose another run. One call writes `diagnostics.json`, the confusion matrix, and a grid of its 12 most confident test errors. For the published results, the default is `gru_patch4_s2` (68.61% test accuracy), and the figures are `E3/figures/confusion_e3.png` and `E3/figures/error_grid_gru_patch4.png` when `--out` is omitted. The experiment summary still reports all three seeds.

## Replication notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E3/notebooks/E3.ipynb)

`E3/notebooks/E3.ipynb` has the same two modes as E2. `MODE = "load"` (default) runs tests and regenerates tables and curves from available hidden-128 E3 results; it checks saved E3 tables when present. `MODE = "rerun"` requires a GPU, smoke-tests one recurrent configuration, trains/resumes all 12 E3 runs in `E3/results_rerun/`, compares with published E3 runs when available, and regenerates analysis and diagnostics. It never trains MLP or MiniVGG or overwrites `E1/results/`.

The notebook also displays the confusion matrix and GRU/patch4 error grid. In load mode it regenerates them when test logits are available, or displays the saved published PNGs. In rerun mode it makes them from the rerun results.
