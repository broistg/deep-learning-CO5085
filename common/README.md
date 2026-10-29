# common/ and the replication notebooks

| Notebook | Open in Colab | Status |
|---|---|---|
| `E1/notebooks/E1.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E1/notebooks/E1.ipynb) | ready |
| `E2/notebooks/E2.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E2/notebooks/E2.ipynb) | ready |
| E3, A1, A2 notebooks | | upcoming (A1 is not finished; A2 after the `rerun` discussion) |

Each notebook has `MODE = "load"` (default: regenerate tables and figures from the committed results and assert they equal the report; minutes, free CPU runtime)
and `MODE = "rerun"` (retrain everything into a fresh `results/`; the published results stay in `results_published/`; resumable; GPU).

## Release weights (one-time setup, after the runs are finished)

Weights and test logits are too large for git, so they live in a GitHub Release; their SHA-256 is committed in `common/release_manifest.json`.

1. Make sure `E1/artifacts/` and `E2/artifacts/` exist on the machine that ran the experiments, and that `E1/results`, `E2/results` (JSON, `*_preds.npz`, `tab_*.tex`, `summary.json`) are committed.
2. `python common/pack_release.py` creates `release_assets/e1_artifacts.zip` and `e2_artifacts.zip`, fills the manifest and prints the `gh release create ...` command.
3. Run that command (or upload the two zips by hand: GitHub > Releases > Draft a new release > tag `weights-v1`).
4. Commit `common/release_manifest.json` and push. Never edit a published tag: for new weights use `--tag weights-v2`.

## CIFAR-10 source

`python -m common.cifar` loads CIFAR-10 from the Hugging Face Hub (`datasets`, `uoft-cs/cifar10`) and falls back to TorchVision. The arrays must match `common/cifar10_fingerprint.json` (SHA-256 of the TorchVision data). The Hugging Face train split has the same images in a different order, so it is put back in the TorchVision order with `common/cifar10_train_order.json`; any changed pixel or label is rejected. One-time setup on a machine that has the TorchVision data: `python -m common.cifar --write-fingerprint`, then commit both JSON files. `python -m common.compare_cifar` prints how the two copies differ.
