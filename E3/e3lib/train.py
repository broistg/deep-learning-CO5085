"""E3: reproducible CIFAR-10 training for LSTM and GRU.

From the repository root:
    PYTHONPATH=.:E3 python -m e3lib.train --smoke
    PYTHONPATH=.:E3 python -m e3lib.train --model gru --tokenizer patch4 --seed 0
    PYTHONPATH=.:E3 python -m e3lib.train --model gru --tokenizer patch4 --force
"""

import argparse
import dataclasses
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.ckpt import atomic_save, build_state, restore_state, verify_ckpt
from common.repro import config_hash, env_info, read_json, set_seed, write_json

from .data import make_datasets
from .models import GRUClassifier, LSTMClassifier

E3_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = E3_DIR / "results"
ARTIFACTS_DIR = E3_DIR / "artifacts"
MODEL_NAMES = ("lstm", "gru")
TOKENIZERS = ("row", "patch4")


@dataclass(frozen=True)
class Cfg:

    model: str = "lstm"
    tokenizer: str = "row"
    seed: int = 0

    epochs: int = 30
    batch_size: int = 64
    eval_batch_size: int = 500
    lr: float = 1e-3
    grad_clip: float = 1.0


    embed_dim: int = 64
    hidden_size: int = 128
    num_layers: int = 1

    @property
    def run_id(self):
        return f"{self.model}_{self.tokenizer}_s{self.seed}"


def validate_cfg(cfg):
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {cfg.model}")
    
    if cfg.tokenizer not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {cfg.tokenizer}")
    
    if cfg.seed < 0:
        raise ValueError("seed must be nonnegative")
    
    for key in ("epochs", "batch_size", "eval_batch_size", "embed_dim",
                "hidden_size", "num_layers"):
        if getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive")
        
    for key in ("lr", "grad_clip"):
        value = getattr(cfg, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")

def make_model(cfg):
    validate_cfg(cfg)
    cls = LSTMClassifier if cfg.model == "lstm" else GRUClassifier
    return cls(tokenizer=cfg.tokenizer, dim=cfg.embed_dim,
               hidden_dim=cfg.hidden_size, num_layers=cfg.num_layers)


def make_optimizer(model, cfg):
    """Match the E1 baseline optimizer: Adam, lr=1e-3, no weight decay."""
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, dataset, optimizer, cfg, epoch,
                    max_batches=None):
    """One training epoch. The data iterator is seeded by (seed, epoch)."""
    model.train()
    loss_sum = torch.zeros((), device=dataset.device)
    correct = torch.zeros((), device=dataset.device)
    seen = 0

    _synchronize(dataset.device)
    start = time.perf_counter()
    for images, labels in dataset.train_batches(
            cfg.batch_size, cfg.seed, epoch, max_batches=max_batches):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()

        n = len(labels)
        loss_sum += loss.detach() * n
        correct += (logits.detach().argmax(1) == labels).sum()
        seen += n

    if not seen:
        raise RuntimeError("No training batches (batch_size too large?)")
    _synchronize(dataset.device)
    seconds = time.perf_counter() - start
    loss_mean = (loss_sum / seen).item()
    if not math.isfinite(loss_mean):
        raise RuntimeError(f"Non-finite train loss at epoch {epoch + 1}")
    return {"loss": loss_mean, "accuracy": (correct / seen).item(),
            "time_s": seconds}


@torch.no_grad()
def evaluate(model, dataset, batch_size=500, max_batches=None,
             return_predictions=False):
    """No augmentation. Metric loss uses standard CE, matching E2 evaluation."""
    model.eval()
    loss_sum = torch.zeros((), device=dataset.device)
    correct = torch.zeros((), device=dataset.device)
    seen = 0
    logits_chunks, label_chunks = [], []

    _synchronize(dataset.device)
    start = time.perf_counter()
    for i, (images, labels) in enumerate(dataset.eval_batches(batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        n = len(labels)
        loss_sum += loss * n
        correct += (logits.argmax(1) == labels).sum()
        seen += n
        if return_predictions:
            logits_chunks.append(logits.float().cpu())
            label_chunks.append(labels.cpu())

    if not seen:
        raise RuntimeError("No evaluation batches")
    _synchronize(dataset.device)
    result = {
        "loss": (loss_sum / seen).item(),
        "accuracy": (correct / seen).item(),
        "time_s": time.perf_counter() - start,
    }
    if return_predictions:
        result["logits"] = torch.cat(logits_chunks)
        result["labels"] = torch.cat(label_chunks)
    return result


def _check_completed(result_path, weights_path, logits_path, cfg, cfg_dict,
                     cfg_hash):
    """A JSON file alone is not evidence of a completed run."""
    saved = read_json(result_path)
    if not isinstance(saved, dict):
        raise TypeError(f"Invalid result JSON: {result_path}; use --force to restart")
    if (saved.get("run_id") != cfg.run_id or saved.get("config") != cfg_dict
            or saved.get("config_hash") != cfg_hash):
        raise RuntimeError(f"Result config mismatch: {result_path}; use --force to restart")
    history = saved.get("history")
    final = saved.get("final")
    model = make_model(cfg)
    tokens = getattr(getattr(model, "tokenizer", None), "n_steps", None)
    mean, std = saved.get("channel_mean"), saved.get("channel_std")
    if (not isinstance(history, list) or len(history) != cfg.epochs
            or not all(isinstance(row, dict) and row.get("epoch") == i
                       for i, row in enumerate(history, 1))
            or not isinstance(final, dict)
            or not isinstance(saved.get("split_sha256_16"), str)
            or not saved["split_sha256_16"]
            or not all(isinstance(values, list) and len(values) == 3
                       and all(isinstance(x, (int, float)) and math.isfinite(x)
                               for x in values) for values in (mean, std))
            or saved.get("n_params") != sum(p.numel() for p in model.parameters())
            or saved.get("n_tokens") != tokens
            or not all(isinstance(final.get(key), (int, float))
                       and math.isfinite(final[key])
                       for key in ("val_acc", "val_loss", "test_acc", "test_loss"))
            or final["val_acc"] != history[-1].get("val_acc")
            or final["val_loss"] != history[-1].get("val_loss")):
        raise RuntimeError(f"Incomplete result JSON: {result_path}; use --force to restart")
    for path in (weights_path, logits_path):
        if not path.is_file() or not verify_ckpt(path):
            raise RuntimeError(f"Missing or corrupt artifact: {path}; use --force to restart")
    try:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        predictions = torch.load(logits_path, map_location="cpu", weights_only=True)
        logits, labels = predictions["logits"], predictions["labels"]
        if (logits.shape != (10000, 10) or labels.shape != (10000,)
                or not torch.isfinite(logits).all()
                or not 0 <= final["test_acc"] <= 1
                or abs((logits.argmax(1) == labels).float().mean().item()
                       - final["test_acc"]) > 1e-6):
            raise ValueError("invalid test predictions")
    except (OSError, RuntimeError, TypeError, KeyError, ValueError, AttributeError) as exc:
        raise RuntimeError(
            f"Incompatible or corrupt completed artifacts for {cfg.run_id}: {exc}; "
            "use --force to restart") from exc
    return saved


def _load_checkpoint(ckpt_dir, cfg, cfg_hash):
    if not ckpt_dir.exists():
        return None
    path = ckpt_dir / "last.pt"
    if not path.is_file() or not verify_ckpt(path):
        raise RuntimeError(f"Missing or corrupt checkpoint: {path}; use --force to restart")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        extra = state["extra"]
        if extra["run_id"] != cfg.run_id or extra["config_hash"] != cfg_hash:
            raise ValueError("config mismatch")
        epoch, history = state["epoch"], extra["history"]
        if (not isinstance(epoch, int) or epoch < 1 or epoch > cfg.epochs
                or not isinstance(history, list) or len(history) != epoch
                or any(not isinstance(row, dict) or row.get("epoch") != i
                       for i, row in enumerate(history, 1))):
            raise ValueError("epoch/history mismatch")
        if (not isinstance(state["model"], dict)
                or not isinstance(state["opt"], dict)
                or state.get("sched") is not None
                or not isinstance(state["rng"], dict)):
            raise TypeError("missing model, optimizer or RNG state, or unexpected scheduler")
        return state
    except (OSError, RuntimeError, TypeError, KeyError, ValueError, AttributeError) as exc:
        raise RuntimeError(f"Incompatible checkpoint {path}: {exc}; use --force to restart") from exc


def run(cfg, out_dir=None, data_root=None, smoke=False, force=False):
    """
    Run one model/seed; resume safely or skip complete matching runs.

    """
    validate_cfg(cfg)

    output_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    artifact_root = (output_dir / "artifacts" if out_dir is not None else
                     ARTIFACTS_DIR)
    artifact_dir = artifact_root / cfg.run_id
    ckpt_dir = artifact_dir / "ckpt"
    result_path = output_dir / f"{cfg.run_id}.json"
    weights_path = artifact_dir / "model_final.pt"
    logits_path = artifact_dir / "logits_test.pt"
    cfg_dict = dataclasses.asdict(cfg)
    cfg_hash = config_hash(cfg_dict)

    # Smoke bypasses every official result and checkpoint read or write.
    state = None
    if not smoke and not force:
        if result_path.exists():
            saved = _check_completed(result_path, weights_path, logits_path,
                                     cfg, cfg_dict, cfg_hash)
            print(f"{cfg.run_id}: matching completed result -> skip", flush=True)
            return saved
        if artifact_dir.exists() and not ckpt_dir.exists() and any(artifact_dir.iterdir()):
            raise RuntimeError(
                f"Missing checkpoint for existing artifacts in {cfg.run_id}; "
                "use --force to restart")
        state = _load_checkpoint(ckpt_dir, cfg, cfg_hash)
        if weights_path.exists() or logits_path.exists():
            if state is None:
                raise RuntimeError(
                    f"Incomplete final artifacts for {cfg.run_id} without a "
                    "checkpoint; use --force to restart")
            print(f"{cfg.run_id}: incomplete final artifacts -> rebuild from "
                  "checkpoint", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed, deterministic=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train, val, test, metadata = make_datasets(
        device=device, data_root=data_root, fake=smoke)
    model = make_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, cfg)

    # A smoke test uses 2 batches/1 epoch; real runs use all full train batches.
    epochs = 1 if smoke else cfg.epochs
    max_batches = 2 if smoke else None
    steps_per_epoch = train.n // cfg.batch_size
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    if steps_per_epoch == 0:
        raise ValueError("No full training batch; reduce Cfg.batch_size")
    history, start_epoch = [], 0
    if state is not None:
        extra = state["extra"]
        expected = {
            "split_sha256_16": metadata["split_sha256_16"],
            "channel_mean": list(metadata["channel_mean"]),
            "channel_std": list(metadata["channel_std"]),
            "device_type": device.type,
        }
        for key, value in expected.items():
            if extra.get(key) != value:
                raise RuntimeError(
                    f"Checkpoint {key} mismatch in {cfg.run_id}; use --force to restart")
        try:
            start_epoch = restore_state(
                state, model, optimizer, sched=None, scaler=None)
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Incompatible checkpoint for {cfg.run_id}: {exc}; use --force to restart") from exc
        history = extra["history"]
        print(f"{cfg.run_id}: resume from epoch {start_epoch}", flush=True)

    if not smoke and force:
        print(f"WARNING: --force resets {cfg.run_id} from epoch 1", flush=True)
        result_path.unlink(missing_ok=True)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

    tokens = getattr(getattr(model, "tokenizer", None), "n_steps", None)
    print(f"{cfg.run_id} | device={device} | params={n_params:,} "
          f"| tokens={tokens if tokens is not None else '2D CNN'} "
          f"| split={metadata['split_sha256_16']} "
          f"| {'SMOKE' if smoke else 'REAL'}",
          flush=True)

    for epoch in range(start_epoch, epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tr = train_one_epoch(model, train, optimizer, cfg, epoch,
                             max_batches=max_batches)
        val_batch_size = (max(1, min(cfg.eval_batch_size, val.n // 2))
                          if smoke else cfg.eval_batch_size)
        va = evaluate(model, val, val_batch_size, max_batches=max_batches)
        peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == "cuda" else None)
        record = {
            "epoch": epoch + 1,
            "train_loss": tr["loss"], "train_acc": tr["accuracy"],
            "val_loss": va["loss"], "val_acc": va["accuracy"],
            "lr": cfg.lr, "train_s": tr["time_s"],
            "val_s": va["time_s"], "peak_mem_mb": peak_mb,
        }
        history.append(record)
        print(f"  epoch {epoch + 1:02d}/{epochs:02d} "
              f"train_loss={tr['loss']:.4f} "
              f"train_acc={tr['accuracy'] * 100:.2f}% "
              f"val_loss={va['loss']:.4f} "
              f"val_acc={va['accuracy'] * 100:.2f}% "
              f"time={tr['time_s']:.1f}s", flush=True)

        if not smoke:
            extra = {
                "config_hash": cfg_hash,
                "run_id": cfg.run_id,
                "split_sha256_16": metadata["split_sha256_16"],
                "channel_mean": list(metadata["channel_mean"]),
                "channel_std": list(metadata["channel_std"]),
                "device_type": device.type,
                "history": history,
            }
            atomic_save(build_state(model, optimizer, None, None,
                                    epoch + 1, extra), ckpt_dir / "last.pt")

    if smoke:
        print("Smoke passed; no results or checkpoints saved.", flush=True)
        return {"smoke": True, "history": history, "n_params": n_params}

    te = evaluate(model, test, cfg.eval_batch_size, return_predictions=True)
    last = history[-1]
    result = {
        "run_id": cfg.run_id,
        "config": cfg_dict,
        "config_hash": cfg_hash,
        "split_sha256_16": metadata["split_sha256_16"],
        "channel_mean": list(metadata["channel_mean"]),
        "channel_std": list(metadata["channel_std"]),
        "n_params": n_params,
        "n_tokens": tokens,
        "history": history,
        "final": {
            "val_acc": last["val_acc"], "val_loss": last["val_loss"],
            "test_acc": te["accuracy"], "test_loss": te["loss"],
        },
        "train_s_per_epoch_median": float(np.median(
            [row["train_s"] for row in history])),
        "inference_s_test": te["time_s"],
        "inference_ms_per_image": 1000 * te["time_s"] / len(te["labels"]),
        "peak_mem_mb": max((row["peak_mem_mb"] or 0 for row in history),
                           default=0.0),
        "env": env_info(), "smoke": False,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    torch.save({"logits": te["logits"], "labels": te["labels"]}, logits_path)
    write_json(result_path, result)
    print(f"{cfg.run_id}: DONE test_acc={te['accuracy'] * 100:.2f}% "
          f"-> {result_path}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="E3 CIFAR-10 training")
    parser.add_argument("--model", choices=MODEL_NAMES, default="lstm")
    parser.add_argument("--tokenizer", choices=TOKENIZERS, default="row")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="fake data, 1 epoch, 2 train/val batches; save nothing")
    parser.add_argument("--force", action="store_true",
                        help="discard checkpoint/result for the selected run")
    args = parser.parse_args(argv)
    cfg = Cfg(model=args.model, tokenizer=args.tokenizer, seed=args.seed)
    return run(cfg, smoke=args.smoke, force=args.force)


if __name__ == "__main__":
    main()
