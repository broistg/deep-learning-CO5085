"""Evaluate every E1 MLP/MiniVGG and E3 LSTM/GRU checkpoint on CIFAR-10.

Run from the repository root:
    venv/bin/python E3/compare_checkpoints.py

The default comparison uses final-epoch weights for all 18 runs. E1 weights
must be regenerated or restored to E1/artifacts/.
This script loads checkpoints and the shared CIFAR-10 data; it does not train.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "E1"), str(ROOT / "E3")]

from e3lib.train import Cfg as E3Cfg
from e3lib.train import make_model as make_e3_model

from common.cifar import DATA_ROOT, SPLIT_PATH, channel_stats, get_splits

from e1lib.models import MODELS

E1_MODELS = {"mlp": "MLP", "minivgg": "MiniVGG (CNN)"}
E3_MODELS = ("lstm", "gru")
TOKENIZERS = ("row", "patch4")
SEEDS = (0, 1, 2)


def planned_runs():
    for model, label in E1_MODELS.items():
        for seed in SEEDS:
            yield {"experiment": "E1", "model": model, "label": label,
                   "seed": seed, "run_id": f"{model}_s{seed}"}
    for model in E3_MODELS:
        for tokenizer in TOKENIZERS:
            for seed in SEEDS:
                yield {"experiment": "E3", "model": model,
                       "label": f"{model.upper()} / {tokenizer}",
                       "tokenizer": tokenizer, "seed": seed,
                       "run_id": f"{model}_{tokenizer}_s{seed}"}


def checkpoint_path(job, args):
    if job["experiment"] == "E1":
        name = "model_final.pt" if args.e1_kind == "final" else "model_best_val_loss.pt"
        return args.e1_artifacts / job["run_id"] / name
    return args.e3_artifacts / job["run_id"] / "model_final.pt"


@torch.inference_mode()
def evaluate(model, images, labels, mean, std, batch_size, device):
    """Use the same test image order and train-only normalization for every run."""
    model.to(device).eval()
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    n_correct, loss_sum = 0, 0.0
    for start in range(0, len(labels), batch_size):
        end = start + batch_size
        x = torch.from_numpy(np.ascontiguousarray(images[start:end])).permute(0, 3, 1, 2)
        x = (x.to(device=device, dtype=torch.float32) / 255.0 - mean) / std
        y = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)
        logits = model(x)
        n_correct += logits.argmax(dim=1).eq(y).sum().item()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
    return {"test_acc": n_correct / len(labels), "test_loss": loss_sum / len(labels)}


def summarize(rows):
    groups = {}
    for label in (list(E1_MODELS.values())
                  + [f"{model.upper()} / {tok}"
                     for model in E3_MODELS for tok in TOKENIZERS]):
        items = [row for row in rows if row["label"] == label]
        values = np.asarray([row["test_acc"] for row in items], dtype=float)
        groups[label] = {
            "seeds": [row["seed"] for row in items],
            "n_params": items[0]["n_params"],
            "test_acc_mean": float(values.mean()),
            "test_acc_std": float(values.std(ddof=1)),
        }
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-kind", choices=("final", "best"), default="final",
                        help="best selects E1's lowest validation-loss weights")
    parser.add_argument("--e1-artifacts", type=Path, default=ROOT / "E1/artifacts")
    parser.add_argument("--e3-artifacts", type=Path, default=ROOT / "E3/artifacts")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "E3/baseline_comparison/checkpoint_results.json",
                        help="JSON report path (default: E3/baseline_comparison/checkpoint_results.json)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    jobs = list(planned_runs())
    missing = [checkpoint_path(job, args) for job in jobs
               if not checkpoint_path(job, args).is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(map(str, missing)))

    splits, split_hash = get_splits(args.data_root, SPLIT_PATH)
    mean, std = channel_stats(splits["train"][0])
    images, labels = splits["test"]
    rows = []
    print(f"Test: {len(labels)} images | device: {device} | split: {split_hash}")
    print(f"{'Run':24} {'Params':>10} {'Test acc':>10} {'Test loss':>10}")
    for job in jobs:
        model = (MODELS[job["model"]]() if job["experiment"] == "E1"
                 else make_e3_model(E3Cfg(model=job["model"],
                                           tokenizer=job["tokenizer"],
                                           seed=job["seed"])))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        weights = torch.load(checkpoint_path(job, args), map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        metrics = evaluate(model, images, labels, mean, std, args.batch_size, device)
        row = {"experiment": job["experiment"], "label": job["label"],
               "run_id": job["run_id"], "seed": job["seed"],
               "n_params": n_params, **metrics}
        rows.append(row)
        print(f"{row['run_id']:24} {n_params:10,d} {metrics['test_acc']:9.2%} "
              f"{metrics['test_loss']:10.4f}", flush=True)
        del model

    groups = summarize(rows)
    print("\nTest accuracy across 3 seeds (mean ± sample std):")
    for label, group in groups.items():
        print(f"{label:20} {100 * group['test_acc_mean']:.2f} ± "
              f"{100 * group['test_acc_std']:.2f}%  ({group['n_params']:,} params)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"split_sha256_16": split_hash,
                                      "e1_checkpoint": args.e1_kind,
                                      "runs": rows, "groups": groups}, indent=2) + "\n",
                           encoding="utf8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
