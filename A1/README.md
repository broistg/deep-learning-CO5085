# A1: CNN vs Transformer on Food-101

Code for assignment A1 (CO5085). Entry points (run from the repo root with `PYTHONPATH=.:A1`, or after `pip install -e .`):

| Command | Purpose |
|---|---|
| `python -m a1lib.plan make --owners dat,ly` | create configs and the run plan (coordinator, once) |
| `python -m a1lib.worker launch --owner NAME --gpus 2 --max-hours 8.5` | one Kaggle session: copy the data, one worker per GPU (the normal way) |
| `python -m a1lib.worker run --owner NAME` | a single worker on the visible GPU, resumable |
| `python -m a1lib.store stats` / `squash` | store size and pushes / guarded Hugging Face history squash (coordinator) |
| `python -m a1lib.worker status` | progress of all runs |
| `python -m a1lib.plan choose-lr` | pick LRs after the sweep |
| `python -m a1lib.collect results` / `bundle` | copy results into the repo / build the Release files |
| `python -m a1lib.train --config X.yaml --smoke [--fake]` | 2-epoch smoke test (`--fake` = synthetic data, no download) |
| `python -m a1lib.cost_bench` | GPU-only training cost per architecture/mode and inference cost (one T4, one session) |
| `python -m a1lib.posthoc robustness\|gradcam\|toploss` | GPU analyses on the released weights (notebook `kaggle_posthoc.ipynb`, about 1 h) |
| `python -m a1lib.eda {counts,samples,augment,sizes}` | EDA for report Ch. 2: split sizes per class (`counts`, needs only `split.json`; also run by the replication notebook); sample grid, augmentation grid, original image sizes (need the images; run once on Kaggle/locally, commit the small figures) |
| `python -m a1lib.analyze` | tables, figures, calibration/ensemble/error analyses from the committed results |
| `python -m unittest discover -s A1/tests` | torch-free unit tests |

Team workflow: **TEAM_PROTOCOL.md**. Layout: `A1/a1lib/` (A1-only package), `common/` (reusable across assignments), `A1/configs/` (one YAML per run), `A1/results/` (result JSONs), `A1/notebooks/kaggle_worker.ipynb` (Kaggle).

Environment variables: `A1_STORE` (`hf:<org>/<repo>` or a folder), `HF_TOKEN` (Kaggle secret), `A1_LOCAL` (default `/kaggle/temp/a1_local`), `A1_PUSH_EVERY_MIN` (45), `A1_DATA_SRC` (dataset folder, default: search `/kaggle/input`).

Environment: Kaggle notebooks (2 x T4), Python 3.12+, torch 2.x, timm 1.0.30. torch/torchvision are preinstalled and deliberately not listed in the requirements files.

## Reproducibility
Recipe: AdamW, 224 px, batch 64, fp16 AMP, linear warmup (5%) + cosine to 1%, RandAugment(2, 9) + Mixup/CutMix + label smoothing 0.1 (applied once, inside timm `Mixup`), seeds 0/1/2, final epoch reported. Every run stores its config hash, split hash (`75ed81bfe380c089`), library versions and GPU in `result.json`.

## Replication (A1.2)
`A1/notebooks/A1.ipynb` (Colab badge inside) has three modes; all logic is in `a1lib` and `common`:
- `load` (CPU, minutes): regenerates every table and figure of the report from `A1/results/*.json` + the released logits (`a1_logits.zip`) with `python -m a1lib.analyze --out A1/regenerated`, and asserts that the tables equal the committed (= reported) ones.
- `evaluate` (GPU, about 40 min including the data): downloads the released weights of the 8 core runs and re-evaluates them on the 25,250 test images (`python -m a1lib.evaluate --all-core`).
- `rerun` (GPU): retrains selected runs (default: ResNet-50 linear probing) from their committed configs into a local folder store and compares with the published numbers; same GPU type gives the same numbers, other GPUs agree within the seed spread. The whole experiment (88 protocol runs plus 25 follow-up runs = 113 runs) took about 64 GPU-hours of Kaggle T4 session time (52 h for the 88 protocol runs; the original budget for the committed part alone was 24.6 h, the stretch runs about 20 h more).

After the runs finish, the coordinator prepares the committed material once:
1. `python -m a1lib.collect results` (store -> `A1/results/*.json`), `python -m a1lib.collect bundle --out release` (store -> zips)
2. `python -m a1lib.collect artifacts --bundle release --out A1/artifacts` (unpack; logits -> compact `.npz`)
3. `python -m a1lib.analyze` (tables `A1/results/tab_*.tex`, figures `A1/figures/`, `summary.json`); commit these, never `A1/artifacts`
4. `python common/pack_release.py --only a1`, create or extend the Release, commit `common/release_manifest.json`

## Status of the code
All 113 runs of A1 were trained with this code on Kaggle (2 x T4) through the Hugging Face store; the published numbers come from it. The pure-Python parts, the store backends (folder, locks, squash) and the per-sample seeding are covered by `python -m unittest discover -s A1/tests`. For a new account or a code change, run the smoke test cell of `kaggle_worker.ipynb` first.

## AI-use declaration
Claude (Anthropic) assisted in writing this code, the experiment plan and the protocol. The authors reviewed it and are responsible for it.
