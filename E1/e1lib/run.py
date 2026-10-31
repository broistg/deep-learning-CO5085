"""Train and evaluate one E1 model: python -m e1lib.run --model mlp --seed 0     (run from the repo root, PYTHONPATH=.:E1)
    --all      softmax, mlp, minivgg x seeds 0,1,2 (matching finished runs are skipped)
    --fake     synthetic data, no download          --smoke   1 epoch on --fake data, nothing saved
    --no-aug   no augmentation (the original notebook's setting; run id gets the suffix _noaug)
Outputs: E1/results/<run_id>.json (commit), <run_id>_preds.npz (commit, ~0.5 MB), E1/artifacts/<run_id>/ (weights; not committed)."""
import argparse
import copy
import dataclasses
import itertools
from pathlib import Path

import numpy as np
import torch

from common.cifar import DATA_ROOT, SPLIT_PATH, channel_stats, get_splits
from common.repro import config_hash, env_info, read_json, set_seed, write_json

from .data import make_loaders
from .models import MODELS
from .train import predict, train_model

E1_DIR = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class Cfg:
    model: str = "mlp"          # softmax | mlp | minivgg
    seed: int = 0
    epochs: int = 30
    batch_size: int = 64        # as in the original notebook
    lr: float = 1e-3            # Adam, as in the original notebook
    augment: bool = True        # RandomCrop(32, padding=4) + horizontal flip, train only (same as E2)
    workers: int = 0          # matches the committed runs; set explicitly to change the loader count

    @property
    def run_id(self):
        return f"{self.model}_s{self.seed}{'' if self.augment else '_noaug'}"


def ce_from_probs(y_true, y_prob):
    return float(-np.log(np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean())


def run(cfg, out_dir=None, data_root=None, fake=False, smoke=False, force=False):
    out_dir = Path(out_dir or E1_DIR / "results")
    result_path = out_dir / f"{cfg.run_id}.json"
    preds_path = out_dir / f"{cfg.run_id}_preds.npz"
    if result_path.exists() and not force and not smoke:
        saved = read_json(result_path)
        expected_cfg = dataclasses.asdict(cfg)
        if (not isinstance(saved, dict) or saved.get("run_id") != cfg.run_id
                or saved.get("config") != expected_cfg
                or saved.get("config_hash") != config_hash(expected_cfg)
                or saved.get("fake") is not fake
                or saved.get("smoke") is not False
                or not preds_path.is_file()):
            raise ValueError(
                f"{cfg.run_id}: saved result does not match the requested configuration/data "
                f"or its predictions are missing: {result_path}. "
                "Use matching options, a different output directory, or --force to rerun."
            )
        print(f"{cfg.run_id}: matching result exists -> skip")
        return saved
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)                                                  # before EACH model: its init does not depend on run order
    splits, split_hash = get_splits(data_root or DATA_ROOT, SPLIT_PATH, fake=fake)
    mean, std = channel_stats(splits["train"][0])                       # from the 45,000 train images only
    workers = 0 if (smoke or fake) else cfg.workers
    train_loader, val_loader, test_loader, _ = make_loaders(splits, mean, std, cfg.batch_size, workers, cfg.seed, cfg.augment)

    model = MODELS[cfg.model]().to(device)
    epochs = 1 if smoke else cfg.epochs
    model, history, best = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=cfg.lr)

    # main number: the weights of the LAST epoch; also report the best-validation-loss checkpoint (the notebook's rule)
    y_true, y_pred, y_prob, inf_s = predict(model, test_loader, device)
    final = {"val_acc": history["val_accuracies"][-1], "val_loss": history["val_losses"][-1],
             "test_acc": float((y_pred == y_true).mean()), "test_loss": ce_from_probs(y_true, y_prob)}
    model_best = copy.deepcopy(model)
    model_best.load_state_dict(best["state"])
    yb_true, yb_pred, yb_prob, _ = predict(model_best, test_loader, device)
    best_res = {"epoch": best["epoch"], "val_loss": best["val_loss"],
                "val_acc": history["val_accuracies"][best["epoch"] - 1],
                "test_acc": float((yb_pred == yb_true).mean()), "test_loss": ce_from_probs(yb_true, yb_prob)}

    rows = [{"epoch": i + 1, "train_loss": history["train_losses"][i], "train_acc": history["train_accuracies"][i],
             "val_loss": history["val_losses"][i], "val_acc": history["val_accuracies"][i], "lr": cfg.lr,
             "train_s": history["epoch_times"][i], "val_s": history["val_times"][i]} for i in range(epochs)]
    result = {"run_id": cfg.run_id, "config": dataclasses.asdict(cfg), "config_hash": config_hash(dataclasses.asdict(cfg)),
              "split_sha256_16": split_hash, "channel_mean": mean, "channel_std": std,
              "n_params": history["params_trainable"], "history": rows, "final": final, "best": best_res,
              "train_s_per_epoch_median": float(np.median(history["epoch_times"])),
              "inference_s_test": inf_s, "inference_ms_per_image": 1000 * inf_s / len(y_true),
              "peak_mem_mb": history["peak_mem_mb"], "env": env_info(), "smoke": smoke, "fake": fake}
    print(f"{cfg.run_id}: DONE  final test {100 * final['test_acc']:.2f}%  (best-val-loss epoch {best['epoch']}: "
          f"{100 * best_res['test_acc']:.2f}%)  {result['train_s_per_epoch_median']:.1f}s/epoch")
    if not smoke:
        out_dir.mkdir(parents=True, exist_ok=True)
        art = E1_DIR / "artifacts" / cfg.run_id
        art.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), art / "model_final.pt")
        torch.save(best["state"], art / "model_best_val_loss.pt")
        np.savez_compressed(preds_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob.astype(np.float32))
        write_json(result_path, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="3 models x seeds 0,1,2")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = itertools.product(sorted(MODELS), (0, 1, 2)) if a.all else [(a.model, a.seed)]
    for model, seed in jobs:
        cfg = Cfg(model=model, seed=seed, epochs=a.epochs, augment=not a.no_aug, workers=a.workers)
        run(cfg, data_root=a.data_root, fake=a.fake, smoke=a.smoke, force=a.force)


if __name__ == "__main__":
    main()
