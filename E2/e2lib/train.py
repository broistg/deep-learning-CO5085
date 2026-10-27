"""One E2 run: explicit training loop (forward -> loss -> backward -> optimizer step), per-epoch validation, test at the end.
    python -m e2lib.train --tokenizer patch4 --attn scratch --seed 0            # full run
    python -m e2lib.train --tokenizer row --attn torch --fake --smoke           # laptop dry run, synthetic data, 1 epoch x 20 batches
Resumable: a checkpoint is written after every epoch; rerunning the same command continues from it."""
import argparse
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.ckpt import atomic_save, build_state, find_checkpoint, make_scaler, restore_state
from common.repro import config_hash, env_info, read_json, set_seed, write_json

from .data import DATA_ROOT, SPLIT_PATH, GPUSet, channel_stats, get_splits
from .model import TinyTransformer

E2_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
    tokenizer: str = "patch4"        # patch4 | patch8 | row | cnn
    attn: str = "scratch"            # scratch | torch
    use_pos: bool = True
    seed: int = 0
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    amp: bool = True

    @property
    def run_id(self):
        return f"{self.tokenizer}_{self.attn}{'' if self.use_pos else '_nopos'}_s{self.seed}"


def lr_factor(step, total, warm=0.05, minf=0.01, start=0.01):
    w = max(1, int(warm * total))
    if step < w:
        return start + (1 - start) * step / w                          # linear warmup
    p = (step - w) / max(1, total - w)
    return minf + (1 - minf) * 0.5 * (1 + math.cos(math.pi * p))       # cosine decay to minf


@torch.no_grad()
def evaluate(model, dset, device, amp, bs=500, max_batches=None):
    model.eval()
    logits = []
    for i, (xb, _) in enumerate(dset.eval_batches(bs)):
        if max_batches is not None and i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits.append(model(xb).float())
    logits = torch.cat(logits)
    y = dset.y[:len(logits)]
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item(), logits.cpu(), y.cpu()


def run(cfg, out_dir=None, data_root=None, fake=False, smoke=False, force=False):
    out_dir = Path(out_dir or E2_DIR / "results")
    art = Path(out_dir).parent / "artifacts" / cfg.run_id          # checkpoints, logits, weights (not committed)
    result_path = out_dir / f"{cfg.run_id}.json"
    if result_path.exists() and not force and not smoke:
        print(f"{cfg.run_id}: result exists -> skip")
        return read_json(result_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    set_seed(cfg.seed)
    splits, split_hash = get_splits(data_root or DATA_ROOT, SPLIT_PATH, fake=fake)
    mean, std = channel_stats(splits["train"][0])                      # from the 45,000 train images only
    d_train, d_val, d_test = (GPUSet(*splits[k], device, mean, std) for k in ("train", "val", "test"))

    model = TinyTransformer(cfg.tokenizer, cfg.attn, cfg.dim, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout, cfg.use_pos,
                            seed=cfg.seed).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and n not in ("pos", "cls")]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and (p.ndim < 2 or n in ("pos", "cls"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.999))
    max_b = 20 if smoke else None
    steps_per_epoch = d_train.n // cfg.batch_size if max_b is None else max_b
    epochs = 1 if smoke else cfg.epochs
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, cfg.warmup_frac, cfg.min_lr_frac))
    scaler = make_scaler(use_amp)

    history, start_epoch = [], 0
    if not smoke:
        _, state = find_checkpoint(art / "ckpt")
        if state is not None:
            start_epoch = restore_state(state, model, opt, sched, scaler)
            history = state["extra"]["history"]
            print(f"{cfg.run_id}: resumed after epoch {start_epoch}")
    print(f"{cfg.run_id}: {n_params / 1e6:.3f} M params, {model.n_tokens} tokens, device {device}")

    for epoch in range(start_epoch, epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        loss_sum, correct, seen = torch.zeros((), device=device), torch.zeros((), device=device), 0
        for xb, yb in d_train.train_batches(cfg.batch_size, cfg.seed, epoch, max_b):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xb)                                                        # forward
                loss = F.cross_entropy(logits.float(), yb, label_smoothing=cfg.label_smoothing)   # loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()                                                 # backward
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)                                                              # optimizer step
            scaler.update()
            sched.step()
            loss_sum += loss.detach() * len(yb)
            correct += (logits.argmax(1) == yb).sum()
            seen += len(yb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.time() - t0
        train_loss = loss_sum.item() / seen
        if not math.isfinite(train_loss):
            raise RuntimeError(f"{cfg.run_id}: loss is {train_loss} at epoch {epoch + 1} (NaN/inf) - stop and report")
        t1 = time.time()
        v_loss, v_acc, _, _ = evaluate(model, d_val, device, use_amp, max_batches=max_b)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": correct.item() / seen, "val_loss": v_loss, "val_acc": v_acc,
               "lr": sched.get_last_lr()[0], "train_s": train_s, "val_s": time.time() - t1,
               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
        history.append(row)
        print(f"  ep {epoch + 1:2d}/{epochs} loss {train_loss:.3f} val_acc {100 * v_acc:.2f}% ({train_s:.1f}s)", flush=True)
        if not smoke:
            atomic_save(build_state(model, opt, sched, scaler, epoch + 1, {"history": history}), art / "ckpt" / "last.pt")

    t_loss, t_acc, t_logits, t_y = evaluate(model, d_test, device, use_amp, max_batches=max_b)
    result = {"run_id": cfg.run_id, "config": dataclasses.asdict(cfg), "config_hash": config_hash(dataclasses.asdict(cfg)),
              "split_sha256_16": split_hash, "channel_mean": mean, "channel_std": std, "n_params": n_params, "n_tokens": model.n_tokens, "history": history,
              "final": {"val_acc": history[-1]["val_acc"], "val_loss": history[-1]["val_loss"], "test_acc": t_acc, "test_loss": t_loss},
              "train_s_per_epoch_median": float(np.median([h["train_s"] for h in history])),
              "peak_mem_mb": max((h["peak_mem_mb"] or 0) for h in history), "env": env_info(), "smoke": smoke, "fake": fake}
    print(f"{cfg.run_id}: DONE  val {100 * history[-1]['val_acc']:.2f}%  test {100 * t_acc:.2f}%")
    if not smoke:
        art.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": t_logits.half(), "labels": t_y}, art / "logits_test.pt")
        torch.save(model.state_dict(), art / "model_final.pt")
        write_json(result_path, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="patch4", choices=["patch4", "patch8", "row", "cnn"])
    ap.add_argument("--attn", default="scratch", choices=["scratch", "torch"])
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--fake", action="store_true", help="synthetic data, no download")
    ap.add_argument("--smoke", action="store_true", help="1 epoch x 20 batches, nothing saved")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = Cfg(tokenizer=a.tokenizer, attn=a.attn, use_pos=not a.no_pos, seed=a.seed, epochs=a.epochs)
    run(cfg, data_root=a.data_root, fake=a.fake, smoke=a.smoke, force=a.force)


if __name__ == "__main__":
    main()
