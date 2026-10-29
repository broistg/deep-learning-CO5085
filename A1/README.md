# A1: CNN vs Transformer on Food-101

Code for assignment A1 (CO5085). Entry points (run from the repo root with `PYTHONPATH=.:A1`, or after `pip install -e .`):

| Command | Purpose |
|---|---|
| `python -m a1lib.plan make --owners dat,ly` | create configs and the run plan (coordinator, once) |
| `python -m a1lib.worker run --owner NAME` | run my runs, resumable (Colab) |
| `python -m a1lib.worker status` | progress of all runs |
| `python -m a1lib.plan choose-lr` | pick LRs after the sweep |
| `python -m a1lib.collect results` / `bundle` | copy results into the repo / build the Release files |
| `python -m a1lib.train --config X.yaml --smoke [--fake]` | 2-epoch smoke test (`--fake` = synthetic data, no download) |
| `python -m unittest discover -s A1/tests` | torch-free unit tests |

Team workflow: **TEAM_PROTOCOL.md**. Layout: `A1/a1lib/` (A1-only package), `common/` (reusable across assignments), `A1/configs/` (one YAML per run), `A1/results/` (result JSONs), `A1/notebooks/worker.ipynb` (Colab).

Tested environment (Colab): Python 3.13, torch 2.11 (CUDA 13), timm 1.0.30, Tesla T4. torch/torchvision are preinstalled and deliberately not listed in the requirements files.

## Reproducibility
Recipe: AdamW, 224 px, batch 64, fp16 AMP, linear warmup (5%) + cosine to 1%, RandAugment(2, 9) + Mixup/CutMix + label smoothing 0.1 (applied once, inside timm `Mixup`), seeds 0/1/2, final epoch reported. Every run stores its config hash, split hash (`75ed81bfe380c089`), library versions and GPU in `result.json`.

## Status of the code
The torch-dependent modules were written without a GPU in the authoring sandbox. Pure-Python parts are unit-tested; the trainer has been checked only for syntax. **Run the smoke test on Colab before the first real run.**

## AI-use declaration
Claude (Anthropic) assisted in writing this code, the experiment plan and the protocol. The authors reviewed it and are responsible for it.
