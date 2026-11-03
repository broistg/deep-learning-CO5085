# A2: semantic segmentation on ADE20K, SegFormer-B0 vs DeepLabV3+ MobileNetV2
Code in `a2lib/`, plan in `plan.json` (20 runs = 12 LR-sweep + 8 full), configs in `configs/`, tests in `tests/`.

## Protocol
- **Data:** ADE20K SceneParsing (150 classes). The official train set (20,210) is split by seed 42 into 19,210 train + 1,000 dev (`split.json`, hash `af0a2b48189c8707`); the official 2,000 validation images are our test set and are never used for choices. The data is downloaded inside each session and checked against its SHA-256 (the licence forbids redistribution, so it is not stored in any dataset).
- **Models:** SegFormer-B0 (HF `nvidia/mit-b0` encoder only, new decoder; never the ADE20K-finetuned checkpoints) and DeepLabV3+ MobileNetV2-D8 (re-implemented, torchvision ImageNet backbone). Each is trained from ImageNet-pretrained and from scratch.
- **Training:** 20 epochs (24,000 iterations, batch 16; DeepLab uses micro-batch 8 x accumulation 2). SegFormer: AdamW, weight decay 0.01 (0 on norms), decoder LR x10, poly power 1.0, 6.25% linear warmup. DeepLab: SGD momentum 0.9, weight decay 5e-4, poly power 0.9, min LR 1e-4, auxiliary head weight 0.4. Cross-entropy (ignore 255) on logits upsampled to the label size.
- **Augmentation:** the mmseg ADE20K pipeline (random resize 0.5-2.0 of 2048x512, crop 512 with cat_max_ratio 0.75, flip, photometric distortion, pad).
- **Evaluation:** short side 512, single scale, no flip, batch 1; mIoU, pixel accuracy, mean Dice from a confusion matrix.
- **LR selection:** 3 epochs on 20% of train, seed 0, 3 LRs per (model, init), best dev mIoU; if the best is at the grid edge, extend one step.
- **Seeds:** main runs seed 0 for all four (model, init) pairs; seeds 1-2 for the two pretrained models.

## How to run
0. Once: create the private Hugging Face repo `dl4dl/a2-store` (type: model) in the `dl4dl` organisation, then `A1_STORE=hf:dl4dl/a2-store python -m a1lib.store init`. Both members' tokens need write access.
1. Smoke test: `notebooks/kaggle_worker_a2.ipynb` with `RUN_SMOKE = True` (cells 1-4).
2. Sweep runs (`a2_sw_*`) first. Then ONE person runs `python -m a2lib.plan choose-lr`, commits `A2/results/lr_choice_a2.json` (if an EDGE is reported: run the printed `extend` command, commit, both `git pull`).
3. Main runs unlock automatically.

Tests (torch-free): `PYTHONPATH=.:A1:A2 python -m unittest discover -s A2/tests`.
