# E1: Softmax, MLP and CNN on CIFAR-10

Three classifiers (softmax on the flattened image, MLP 3072-64-32-10, MiniVGG) with a hand-written training loop (forward, loss, backward, step). Code: Nguyễn Lê Thảo Ly (original notebook kept in `notebooks/archive/E1_original.ipynb`); moved into modules with the changes listed below.

Run from the repo root. Environment as in A1 (Python 3.10+, PyTorch with CUDA, torchvision preinstalled; `pip install -r requirements/e.txt` for the rest).
```
export PYTHONPATH=.:E1        # Windows PowerShell: $env:PYTHONPATH = ".;E1"   (or: pip install -e .)
python -m unittest discover -s E1/tests -v           # 1. models, data pipeline, aggregation
python -m common.cifar                               # 2. downloads CIFAR-10 to data/, writes/checks the split, prints its hash
python -m e1lib.run --model mlp --seed 0 --fake --smoke   # 3. dry run on synthetic data
python -m e1lib.run --all                            # 4. 3 models x seeds 0,1,2, 30 epochs (skips matching finished runs)
python -m e1lib.summary                              # 5. three-seed table -> regenerated/summary.json, regenerated/tab_e1_summary.tex
python -m e1lib.analysis --seed 0                    # 6. figures -> regenerated/figures/*.pdf and *.png
```
Optional: `python -m e1lib.run --all --no-aug` repeats the study without augmentation (the original notebook's setting); results get the suffix `_noaug`.
The default `--workers 0` matches the committed runs. A saved result is reused only when its full configuration, data type (real or fake), and prediction file match. To change a setting such as `--epochs` or `--workers`, use a separate output directory through `run()` or pass `--force` to replace that run's result and predictions.

Protocol (shared by E1-E3): `common/cifar.py` freezes the split (stratified 45,000 train / 5,000 validation, 500 per class, seed 42, SHA-256 in `common/cifar10_split.json`) and computes the channel mean/std from the 45,000 train images. Test = the 10,000 official test images, scored once per run. Train augmentation: RandomCrop(32, padding=4) + horizontal flip, applied after normalisation, train only (same as E2). The main number is the test accuracy after the LAST epoch; the checkpoint with the lowest validation loss (the notebook's rule) is reported as well.

Outputs: `E1/results/<model>_s<seed>.json` (config, split hash, history, final, best, parameters, timing, environment; commit), `<model>_s<seed>_preds.npz` (test predictions for the confusion matrices; commit), `E1/artifacts/` (weights; not committed).

Analysis reads all three saved seeds per model without retraining. `regenerated/figures/curves.pdf` shows mean ± sample standard deviation (`ddof=1`) at each epoch for train/validation loss and accuracy; train metrics use randomly augmented images and are not a direct generalization-gap estimate. The row-normalized confusion matrices and high-confidence error galleries use seed 0, named `confusion_<model>_s0` and `errors_<model>_s0`. Figures are saved as vector PDF and notebook-friendly PNG. The LaTeX table distinguishes final-epoch test accuracy from the supplementary best-validation-loss checkpoint and includes trainable parameters, mean of per-run median epoch times, amortized test prediction-pipeline time per image, and maximum peak GPU memory across seeds. The published `E1/results/` files are left untouched by the default analysis commands. EDA and error galleries need the original CIFAR-10 pixels; no inference or model weights are needed.

Changes with respect to the original notebook (details and reasons in `E1_ghi_chu_thay_doi.md`): shared split instead of `random_split`; shared train-only normalisation; augmentation on train only, three separate Dataset objects; last-epoch weights are the main result (best-val-loss checkpoint returned separately); seed before each model, seeds 0-2; 30 epochs; GPU-synchronised timing; no global `device` in the layers; results in JSON with the E2 schema. The architectures, `train_one_epoch`, `evaluate_one_epoch`, `predict` and the plotting functions are unchanged.

Status: all runs finished; the replication notebook was checked on Colab.

## Replication notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E1/notebooks/E1.ipynb)

`E1/notebooks/E1.ipynb` is a thin notebook (no logic inside; it calls `e1lib` and `common`). `MODE = "load"` (default) regenerates the table and figures into `E1/regenerated/`, checks the legacy statistics against committed `E1/results/summary.json`, and verifies that every PDF/PNG was created; it runs in minutes on a free CPU runtime. `MODE = "rerun"` retrains everything into a fresh `results/` (the published results are kept in `results_published/` and compared with the rerun; resumable; GPU needed).
Weights and logits are downloaded from the GitHub Release `weights-v1` and verified by SHA-256 (see `common/README.md`). CIFAR-10 comes from the Hugging Face Hub through the `datasets` library, verified against a fingerprint of the TorchVision data (`python -m common.cifar`).

AI use: Claude assisted with the refactoring and tests; the authors reviewed and are responsible for it.
