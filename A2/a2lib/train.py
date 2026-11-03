"""Resumable segmentation run: forward -> loss -> backward -> step is written out explicitly below (no trainer framework).

    python -m a2lib.train --config A2/configs/<run_id>.yaml --owner dat            # a real run (resumes automatically)
    python -m a2lib.train --config A2/configs/<run_id>.yaml --smoke                # 2 tiny epochs on the real data, nothing written to the shared store
    python -m a2lib.train --config A2/configs/<run_id>.yaml --smoke --fake         # synthetic data (no download); weights are random

Same safety rules as A1 (see A1/TEAM_PROTOCOL.md): checkpoint every epoch (atomic, verified) on the LOCAL disk; pushed to the store every
A1_PUSH_EVERY_MIN minutes and at every clean exit (time limit, Ctrl-C, SIGTERM); resume pulls the store's checkpoint when it is ahead; a
per-run lock with a heartbeat keeps two workers off the same run. Effective batch 16 = micro_batch x gradient accumulation; BatchNorm
statistics (DeepLab) are therefore computed over the micro-batch (8 images), a documented deviation from the paper's SyncBN over 16."""
import argparse
import os
import signal
import threading
import time
import traceback
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# harmless: the dataset returns read-only numpy arrays that the collate function copies into tensors
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
# harmless with fp16 GradScaler: the first optimizer step(s) are skipped on overflow, the scheduler still advances
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
import torch.nn.functional as F

from common.ckpt import atomic_save, build_state, find_checkpoint, make_scaler, restore_state
from common.repro import append_jsonl, env_info, read_json, set_seed, write_json

from a1lib.status import Status
from a1lib.store import HEARTBEAT_S, LocalStore, RunSync, worker_id

from a2lib import metrics as M
from a2lib.config import dump_config, load_config, resolve_lr, stable_hash
from a2lib.data import ADEDataset, ensure_data, load_split, make_fake_ade, subset_nested
from a2lib.models import count_params, make_model, make_optimizer, total_loss
from a2lib.paths import Paths
from a2lib.schedule import make_scheduler

SPLIT_PATH = Path(__file__).resolve().parents[1] / "split.json"


class TimeUp(Exception):
    """The session's time budget (--max-hours) is nearly used up; the checkpoint has been pushed, the run continues elsewhere."""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Heartbeat(threading.Thread):
    def __init__(self, sync, every=HEARTBEAT_S):
        super().__init__(daemon=True)
        self.sync, self.every, self._stop_evt = sync, every, threading.Event()

    def run(self):
        while not self._stop_evt.wait(self.every):
            try:
                self.sync.beat()
            except Exception as e:  # noqa: BLE001
                log(f"  ! heartbeat failed: {e!r}")

    def stop(self):
        self._stop_evt.set()


def _worker_init(_wid):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def make_loader(ds, cfg, shuffle, epoch=0, batch_size=None, drop_last=False):
    kw = {"prefetch_factor": 4, "worker_init_fn": _worker_init, "persistent_workers": False} if cfg.workers > 0 else {}
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
    g = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(cfg.seed * 1000 + epoch)          # data order depends only on (seed, epoch); augmentation on (seed, epoch, sample)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size or cfg.micro_batch, shuffle=shuffle, generator=g, num_workers=cfg.workers,
                                       drop_last=drop_last, pin_memory=torch.cuda.is_available(), **kw)


def gpu_normalize(x, device, mean, std):
    """uint8 [N,H,W,3] -> float32 [N,3,H,W], normalised on the GPU (padding pixels were filled with the mean, so they become 0)."""
    x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


@torch.inference_mode()
def evaluate(model, ds, cfg, device, max_images=None):
    """Single scale, batch 1, logits upsampled bilinearly to the ORIGINAL label size; confusion accumulated over the whole set."""
    model.eval()
    use_cuda = device.type == "cuda"
    C = cfg.num_classes
    conf = torch.zeros(C * C, dtype=torch.int64, device=device)
    aug = cfg.augmentation
    loader = make_loader(ds, cfg, shuffle=False, batch_size=1)
    for i, (x, y) in enumerate(loader):
        x = gpu_normalize(x, device, aug["mean"], aug["std"])
        y = y.to(device)
        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=use_cuda):
            out = model(x)
        pred = F.interpolate(out.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        ok = y != 255
        conf += torch.bincount(y[ok].long() * C + pred[ok], minlength=C * C)
        if max_images and i + 1 >= max_images:
            break
    return conf.view(C, C).cpu().numpy()


def train_one_epoch(model, loader, opt, sched, scaler, cfg, device, steps, max_steps=None):
    use_cuda = device.type == "cuda"
    model.train()
    accum = cfg.accum
    n_steps = steps if not max_steps else min(steps, max_steps)
    loss_sum = torch.zeros((), device=device)
    n_micro, wait = 0, 0.0
    aug = cfg.augmentation
    opt.zero_grad(set_to_none=True)
    t_start = t_iter = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        wait += time.perf_counter() - t_iter
        if i >= n_steps * accum:
            break
        x = gpu_normalize(x, device, aug["mean"], aug["std"])
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=use_cuda):
            out = model(x)
        loss = total_loss(out, y, cfg.aux_weight)           # cross-entropy in fp32 on logits upsampled to the label size
        scaler.scale(loss / accum).backward()
        loss_sum += loss.detach()
        n_micro += 1
        if n_micro % accum == 0:
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
        t_iter = time.perf_counter()
    if use_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t_start
    mean_loss = float(loss_sum) / max(1, n_micro)
    if not np.isfinite(mean_loss):
        raise FloatingPointError("training loss is NaN/inf; stopping this run (do not rerun with a different seed to hide it; report it)")
    return {"train_loss": mean_loss, "train_s": dt, "data_wait_s": wait, "train_ips": n_micro * cfg.micro_batch / dt,
            "lr_end": opt.param_groups[0]["lr"]}


def run_training(cfg, paths, owner, worker=None, data_root=None, max_steps=None, force=False, fake=False, smoke=False, deadline=None):
    """Runs (or resumes) one configuration to completion. Returns the result dict. Raises TimeUp when `deadline` (epoch seconds) is near."""
    store = paths.store
    worker = worker or worker_id(owner)
    cfg = resolve_lr(cfg, store.read_json("lr_choice_a2.json", {}, fresh=True))
    rid = cfg.run_id
    rel = paths.run_rel(rid)
    ldir = paths.local_run(rid)
    ldir.mkdir(parents=True, exist_ok=True)
    done = store.read_json(f"{rel}/result.json", None, fresh=True)
    if done and not force:
        log(f"{rid}: already finished (result.json exists) -> skipping")
        return done
    cfg_dict = cfg.to_dict()
    chash = stable_hash(cfg_dict)
    remote_cfg = store.read_bytes_fresh(f"{rel}/config.yaml")
    local_cfg = ldir / "config.yaml"
    if remote_cfg is not None:
        (ldir / "config.remote.yaml").write_bytes(remote_cfg)
        old = load_config(ldir / "config.remote.yaml")
        if stable_hash(old.to_dict()) != chash and not force:
            raise RuntimeError(f"{rid}: config.yaml in the store differs from this config (hash). Same run_id must mean the same settings. "
                               f"Use a new run_id, or --force if you are sure.")
    dump_config(cfg, local_cfg)
    sync = RunSync(store, rid, owner, worker, ldir)
    sync.lock.acquire(force=force, extra={f"{rel}/config.yaml": local_cfg} if remote_cfg is None else None)
    hb = Heartbeat(sync)
    hb.start()
    status = Status(ldir, rid, owner, cfg.epochs, worker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = env_info()
    progress = {"epoch": 0}

    def _term(*_):
        raise KeyboardInterrupt()
    try:
        old_term = signal.signal(signal.SIGTERM, _term)
    except ValueError:
        old_term = None
    status.set(state="running", gpu=env["gpu"], message="starting")
    try:
        return _train(cfg, chash, paths, owner, sync, ldir, status, device, env, data_root, max_steps, fake, smoke, deadline, progress)
    except (KeyboardInterrupt, TimeUp) as e:
        timeup = isinstance(e, TimeUp)
        status.set(state="paused" if timeup else "interrupted",
                   message="time budget used up; another session continues from the pushed checkpoint" if timeup else "stopped; rerun the same command to resume")
        _exit_push(sync, ldir, progress)
        raise
    except BaseException:
        status.set(state="failed", message=traceback.format_exc()[-1500:])
        _exit_push(sync, ldir, progress)
        raise
    finally:
        hb.stop()
        sync.release()
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


def _exit_push(sync, ldir, progress):
    ep = progress["epoch"] if progress["epoch"] > sync.last_pushed_epoch and (ldir / "ckpt" / "last.pt").exists() else None
    try:
        sync.release(push_epoch=ep)
        if ep is not None:
            log(f"pushed checkpoint (epoch {ep}) before exit")
    except Exception as e:  # noqa: BLE001
        log(f"  ! could not push on exit ({e!r}); the checkpoint stays on the local disk of this session only")


def _data(cfg, paths, data_root, fake):
    """-> (root, split dict). fake: synthetic images and a split built from their listing."""
    if fake:
        root = paths.data_local / "fake_ade"
        if not (root / "images").exists():
            make_fake_ade(root)
        stems = lambda sub: sorted(p.stem for p in (root / "images" / sub).glob("*.jpg"))  # noqa: E731
        tr = stems("training")
        return root, {"train": tr[:-4], "dev": tr[-4:], "test": stems("validation")}
    root = Path(data_root) if data_root else ensure_data(paths.data_local)
    return root, load_split(SPLIT_PATH, expect=cfg.split_sha256_16 or None)


def _train(cfg, chash, paths, owner, sync, ldir, status, device, env, data_root, max_steps, fake, smoke, deadline, progress):
    use_cuda = device.type == "cuda"
    root, split = _data(cfg, paths, data_root, fake)
    aug = cfg.augmentation
    train_names = subset_nested(split["train"], cfg.data_fraction)
    ds_train = ADEDataset(root, train_names, "training", aug, True, seed=cfg.seed)
    ds_dev = ADEDataset(root, split["dev"], "training", aug, False)          # dev images are official TRAINING images (held out by our split)
    ds_test = ADEDataset(root, split["test"], "validation", aug, False)
    steps_per_epoch = len(ds_train) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    log(f"{cfg.run_id}: {len(ds_train)} train / {len(ds_dev)} dev images, {steps_per_epoch} steps/epoch x {cfg.epochs} epochs = {total_steps}, "
        f"lr_peak={cfg.lr_peak:g}, micro-batch {cfg.micro_batch} x accum {cfg.accum}")

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = use_cuda
    model = make_model(cfg).to(device)
    trainable, total = count_params(model)
    lr = float(cfg.lr_peak)
    opt = make_optimizer(model, cfg, lr)
    sched = make_scheduler(opt, total_steps, cfg, lr)
    scaler = make_scaler(use_cuda)
    log(f"parameters: {total / 1e6:.3f}M (trainable {trainable / 1e6:.3f}M)")

    history, sessions, epoch0 = [], [], 0
    _, st = find_checkpoint(ldir / "ckpt")
    local_epoch = st["epoch"] if st is not None else 0
    pulled = sync.pull_ckpt(local_epoch)
    if pulled is not None:
        log(f"pulled the store's checkpoint (epoch {pulled}; local was {local_epoch})")
        _, st = find_checkpoint(ldir / "ckpt")
    if st is not None:
        if st["extra"].get("config_hash") != chash:
            raise RuntimeError("checkpoint was made with a different config (hash mismatch); refusing to resume")
        epoch0 = restore_state(st, model, opt, sched, scaler, with_rng=True)
        history, sessions = st["extra"].get("history", []), st["extra"].get("sessions", [])
        sync.last_pushed_epoch = epoch0 if (pulled is not None or sync.remote_epoch() == epoch0) else -1
        log(f"resumed at epoch {epoch0}/{cfg.epochs}")
    progress["epoch"] = epoch0
    sessions.append({"owner": owner, "worker": sync.worker, "host": env["host"], "gpu": env["gpu"], "start_epoch": epoch0,
                     "started": time.time(), "torch": env["torch"], "git_commit": env["git_commit"]})
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    push_failures = 0
    for epoch in range(epoch0, cfg.epochs):
        tr = train_one_epoch(model, make_loader(ds_train, cfg, True, epoch, drop_last=True), opt, sched, scaler, cfg, device, steps_per_epoch, max_steps)
        t0 = time.perf_counter()
        conf = evaluate(model, ds_dev, cfg, device, max_images=8 if max_steps else None)
        v = M.summarize(conf)
        row = {"epoch": epoch + 1, **{k: round(val, 6) for k, val in tr.items()}, "val_s": round(time.perf_counter() - t0, 2),
               "dev_mIoU": round(v["mIoU"], 5), "dev_pixel_acc": round(v["pixel_acc"], 5), "dev_mDice": round(v["mDice"], 5),
               "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_cuda else None,
               "owner": owner, "worker": sync.worker, "gpu": env["gpu"]}
        history.append(row)
        append_jsonl(ldir / "history.jsonl", row)
        log(f"epoch {epoch + 1}/{cfg.epochs} loss {row['train_loss']:.4f} | dev mIoU {100 * v['mIoU']:.2f}% pixAcc {100 * v['pixel_acc']:.2f}% | "
            f"train {row['train_s']:.0f}s (data wait {row['data_wait_s']:.0f}s) {row['train_ips']:.1f} img/s, eval {row['val_s']:.0f}s")
        state = build_state(model, opt, sched, scaler, epoch + 1, extra={"history": history, "config_hash": chash, "sessions": sessions})
        atomic_save(state, ldir / "ckpt" / "last.pt")
        progress["epoch"] = epoch + 1
        if sync.lock.lost:
            raise RuntimeError("the lock of this run was taken over by another worker; stopping without pushing")
        epoch_s = row["train_s"] + row["val_s"]
        out_of_time = deadline is not None and time.time() + 1.15 * epoch_s > deadline and epoch + 1 < cfg.epochs
        if sync.should_push_ckpt() or out_of_time:
            try:
                sync.push_ckpt(epoch + 1)
                push_failures = 0
            except Exception as e:  # noqa: BLE001
                push_failures += 1
                log(f"  ! could not push the checkpoint ({e!r}); local copy is safe; failures in a row: {push_failures}")
                if push_failures >= 3:
                    raise
        status.set(epoch=epoch + 1, message=f"dev mIoU {100 * v['mIoU']:.2f}%", eta_min=round((cfg.epochs - epoch - 1) * epoch_s / 60, 1))
        if out_of_time:
            raise TimeUp(f"{cfg.run_id}: stopping after epoch {epoch + 1}: the next epoch would not finish before the time limit")

    # ---- final: dev + test (the test set is not used for LR sweeps), confusion matrices, weights, result.json ------------------
    conf_dev = evaluate(model, ds_dev, cfg, device, max_images=8 if max_steps else None)
    final = {"dev": M.summarize(conf_dev)}
    out = ldir / "final"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "confusion_dev.npy", conf_dev)
    artifacts, push_files = {"confusion_dev": "confusion_dev.npy"}, {}
    if cfg.kind != "sweep":
        conf_test = evaluate(model, ds_test, cfg, device, max_images=8 if max_steps else None)
        final["test"] = M.summarize(conf_test)
        np.save(out / "confusion_test.npy", conf_test)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_final.pt")
        artifacts.update({"confusion_test": "confusion_test.npy", "weights": "model_final.pt"})
    push_files = {f.name: f for f in out.iterdir()}
    result = {
        "run_id": cfg.run_id, "kind": cfg.kind, "config": cfg.to_dict(), "config_hash": chash, "split_sha256_16": cfg.split_sha256_16,
        "n_train_images": len(ds_train), "n_dev": len(ds_dev), "n_test": len(ds_test), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "trainable_M": round(trainable / 1e6, 3), "total_M": round(total / 1e6, 3), "final": final, "history": history,
        "timing": {"train_s_total": round(sum(h["train_s"] for h in history), 1), "val_s_total": round(sum(h["val_s"] for h in history), 1),
                   "note": "wall-clock on the session; use cost_bench for architecture comparison"},
        "peak_mem_mb_max": max((h["peak_mem_mb"] or 0) for h in history), "sessions": sessions, "artifacts": artifacts,
        "env": env, "finished_at": time.time(), "finished_by": owner, "worker": sync.worker, "smoke": bool(smoke), "fake": bool(fake),
    }
    write_json(ldir / "result.json", result)
    status.set(state="done", epoch=cfg.epochs, message="finished")
    sync.push_final(push_files, (ldir / "result.json").read_text(encoding="utf8"))
    log(f"DONE {cfg.run_id}: dev mIoU {100 * final['dev']['mIoU']:.2f}%" + (f" | test mIoU {100 * final['test']['mIoU']:.2f}%" if "test" in final else ""))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--owner", default=os.environ.get("A1_OWNER", "unknown"))
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 4 optimizer steps, writes only to the local disk")
    ap.add_argument("--fake", action="store_true", help="synthetic dataset, random weights (laptop dry run)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths = Paths.from_env()
    max_steps = None
    if a.smoke:
        cfg = replace(cfg, run_id="smoke_" + cfg.run_id, epochs=2, lr_peak=(1e-4 if cfg.lr_peak == "auto" else cfg.lr_peak),
                      pretrained=(cfg.pretrained and not a.fake), workers=0 if a.fake else cfg.workers)
        max_steps = 4
        if not a.fake and a.data_root is None:
            a.data_root = str(ensure_data(paths.data_local))
        paths = Paths(LocalStore(paths.local / "smoke_store"), paths.local)
    run_training(cfg, paths, a.owner, None, a.data_root, max_steps, a.force, a.fake, a.smoke)


if __name__ == "__main__":
    main()
