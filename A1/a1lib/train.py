"""Resumable training run: forward -> loss -> backward -> step is written out explicitly below (no trainer framework).

    python -m a1lib.train --config A1/configs/<run_id>.yaml --owner dat            # a real run (resumes automatically)
    python -m a1lib.train --config A1/configs/<run_id>.yaml --smoke                # 2 tiny epochs, nothing written to the shared store
    python -m a1lib.train --config A1/configs/<run_id>.yaml --smoke --fake         # laptop dry run, no dataset needed

Safety rules (see A1/TEAM_PROTOCOL.md): checkpoint every epoch (atomic, verified, last-good backup) on the LOCAL disk; push it to the
store every A1_PUSH_EVERY_MIN minutes and at every clean exit (time limit, Ctrl-C, SIGTERM); resume pulls the store's checkpoint when it
is ahead of the local one; a per-run lock with a heartbeat keeps two workers (even of the same person) off the same run.
"""
import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import signal
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from common.ckpt import atomic_save, build_state, find_checkpoint, make_scaler, restore_state
from common.repro import append_jsonl, config_hash, env_info, read_json, set_seed, write_json

from a1lib import metrics as M
from a1lib.augment import eval_transform, gpu_normalize, make_mixup, soft_target_ce, train_transform
from a1lib.config import dump_config, from_dict, load_config, resolve_lr
from a1lib.data import FoodDataset, N_CLASSES, SPLIT_SHA16, ensure_data, load_split, make_fake_cache, subset_per_class
from a1lib.models import FAMILY, apply_freeze, count_params, make_model, param_groups, set_train_mode
from a1lib.paths import Paths
from a1lib.schedule import make_scheduler
from a1lib.status import Status
from a1lib.store import HEARTBEAT_S, LocalStore, RunSync, worker_id


class TimeUp(Exception):
    """The session's time budget (--max-hours) is nearly used up; the checkpoint has been pushed, the run continues elsewhere."""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Heartbeat(threading.Thread):
    """Every `every` seconds: refresh the lock and push status.json (one commit). If the lock was taken over, flag it."""

    def __init__(self, sync, every=HEARTBEAT_S):
        super().__init__(daemon=True)
        self.sync, self.every, self._stop_evt = sync, every, threading.Event()

    def run(self):
        while not self._stop_evt.wait(self.every):
            try:
                self.sync.beat()
            except Exception as e:  # noqa: BLE001  (a network hiccup must not kill training; a lost lock is checked after each epoch)
                log(f"  ! heartbeat failed: {e!r}")

    def stop(self):
        self._stop_evt.set()


def _worker_init(_wid):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    import random
    random.seed(seed)


def make_loader(ds, cfg, shuffle, epoch=0, batch_size=None, drop_last=False):
    kw = {}
    if cfg.workers > 0:
        kw = {"prefetch_factor": 4, "worker_init_fn": _worker_init}
    g = None
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(cfg.seed * 1000 + epoch)  # data order depends only on (seed, epoch); augmentation on (seed, epoch, sample)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size or cfg.batch_size, shuffle=shuffle, generator=g,
                                       num_workers=cfg.workers, drop_last=drop_last, pin_memory=torch.cuda.is_available(), **kw)


@torch.inference_mode()
def predict(model, loader, device, channels_last, max_batches=None):
    """Hard-label evaluation: returns (logits float32 [N,101] on CPU, labels int64 [N])."""
    model.eval()
    use_cuda = device.type == "cuda"
    outs, ys = [], []
    for i, (x, y) in enumerate(loader):
        x = gpu_normalize(x, device, channels_last)
        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=use_cuda):
            out = model(x)
        outs.append(out.float().cpu())
        ys.append(y)
        if max_batches and i + 1 >= max_batches:
            break
    return torch.cat(outs), torch.cat(ys)


def train_one_epoch(model, loader, opt, sched, scaler, mix, device, channels_last, max_batches=None):
    use_cuda = device.type == "cuda"
    set_train_mode(model)  # frozen parts stay in eval (BatchNorm!)
    loss_sum = torch.zeros((), device=device)
    n_batches, n_imgs, wait = 0, 0, 0.0
    t_start = t_iter = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        wait += time.perf_counter() - t_iter
        x = gpu_normalize(x, device, channels_last)
        y = y.to(device, non_blocking=True)
        x, target = mix(x, y)                              # Mixup OR CutMix (50/50), label smoothing 0.1 applied once, here
        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=use_cuda):
            logits = model(x)
        loss = soft_target_ce(logits, target)              # soft-target CE in fp32, no extra smoothing
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.detach()
        n_batches += 1
        n_imgs += len(x)
        t_iter = time.perf_counter()
        if max_batches and i + 1 >= max_batches:
            break
    if use_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t_start
    mean_loss = float(loss_sum) / max(1, n_batches)
    if not np.isfinite(mean_loss):
        raise FloatingPointError("training loss is NaN/inf; stopping this run (do not rerun with a different seed to hide it; report it)")
    return {"train_loss": mean_loss, "train_s": dt, "data_wait_s": wait, "train_ips": n_imgs / dt, "lr_end": opt.param_groups[0]["lr"]}


def run_training(cfg, paths, owner, worker=None, data_root=None, max_batches=None, force=False, fake=False, smoke=False, deadline=None):
    """Runs (or resumes) one configuration to completion. Returns the result dict. Raises TimeUp when `deadline` (epoch seconds) is near;
    the checkpoint is then already in the store and another session continues the run."""
    store = paths.store
    worker = worker or worker_id(owner)
    cfg = resolve_lr(cfg, store.read_json("lr_choice.json", {}, fresh=True))
    rid = cfg.run_id
    rel = paths.run_rel(rid)
    ldir = paths.local_run(rid)
    ldir.mkdir(parents=True, exist_ok=True)
    done = store.read_json(f"{rel}/result.json", None, fresh=True)
    if done and not force:
        log(f"{rid}: already finished (result.json exists) -> skipping")
        return done

    cfg_dict = cfg.to_dict()
    chash = config_hash({k: v for k, v in cfg_dict.items() if k != "notes"})
    remote_cfg = store.read_bytes_fresh(f"{rel}/config.yaml")
    local_cfg = ldir / "config.yaml"
    if remote_cfg is not None:
        (ldir / "config.remote.yaml").write_bytes(remote_cfg)
        old = load_config(ldir / "config.remote.yaml")
        if config_hash({k: v for k, v in old.to_dict().items() if k != "notes"}) != chash and not force:
            raise RuntimeError(f"{rid}: config.yaml in the store differs from this config (hash). "
                               f"Same run_id must mean the same settings. Use a new run_id, or --force if you are sure.")
    dump_config(cfg, local_cfg)

    sync = RunSync(store, rid, owner, worker, ldir)
    sync.lock.acquire(force=force)
    if remote_cfg is None:
        store.put({f"{rel}/config.yaml": local_cfg}, f"config {rid}")
    hb = Heartbeat(sync)
    hb.start()
    status = Status(ldir, rid, owner, cfg.epochs, worker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = env_info()
    progress = {"epoch": 0}

    def beat():
        try:
            sync.beat()
        except Exception as e:  # noqa: BLE001
            log(f"  ! could not push status ({e!r})")

    def _term(*_):
        raise KeyboardInterrupt()
    try:
        old_term = signal.signal(signal.SIGTERM, _term)
    except ValueError:                                   # not the main thread (tests)
        old_term = None
    status.set(state="running", gpu=env["gpu"], message="starting")
    beat()
    try:
        result = _train(cfg, chash, paths, owner, sync, ldir, status, device, env, data_root, max_batches, fake, smoke, deadline, progress)
        status.set(state="done", epoch=cfg.epochs, message="finished")
        beat()
        return result
    except (KeyboardInterrupt, TimeUp) as e:
        _push_on_exit(sync, ldir, progress)
        timeup = isinstance(e, TimeUp)
        status.set(state="paused" if timeup else "interrupted",
                   message="time budget used up; another session continues from the pushed checkpoint" if timeup
                   else "stopped; rerun the same command to resume")
        beat()
        raise
    except BaseException:
        _push_on_exit(sync, ldir, progress)
        status.set(state="failed", message=traceback.format_exc()[-1500:])
        beat()
        raise
    finally:
        hb.stop()
        sync.lock.release()
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


def _push_on_exit(sync, ldir, progress):
    """Whatever the reason we stop, hand the newest local checkpoint to the store (unless it is already there)."""
    try:
        if progress["epoch"] > sync.last_pushed_epoch and (ldir / "ckpt" / "last.pt").exists():
            sync.push_ckpt(progress["epoch"])
            log(f"pushed checkpoint (epoch {progress['epoch']}) before exit")
    except Exception as e:  # noqa: BLE001
        log(f"  ! could not push the checkpoint on exit ({e!r}); it stays on the local disk of this session only")


def _train(cfg, chash, paths, owner, sync, ldir, status, device, env, data_root, max_batches, fake, smoke, deadline, progress):
    use_cuda = device.type == "cuda"
    # ---- data ----------------------------------------------------------------------------------------------
    if fake:
        root = make_fake_cache(paths.data_local)
        split = load_split(root / "split.json", expect=None)
    else:
        root = Path(data_root) if data_root else ensure_data(paths.data_local)
        split = load_split(root / "split.json", expect=cfg.split_json_sha256_16)
    classes = split["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    train_entries = subset_per_class(split["train"], cfg.data_fraction)
    tfm_train, tfm_eval = train_transform(cfg.augmentation), eval_transform(cfg.augmentation)
    ds_train = FoodDataset(root / "images", train_entries, cidx, tfm_train, seed=cfg.seed)   # per-sample augmentation seeds
    ds_val = FoodDataset(root / "images", split["val"], cidx, tfm_eval)
    ds_test = FoodDataset(root / "images", split["test"], cidx, tfm_eval)
    steps_per_epoch = len(ds_train) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    log(f"{cfg.run_id}: {len(ds_train)} train / {len(ds_val)} val images, {steps_per_epoch} steps/epoch, lr_peak={cfg.lr_peak:g}")

    # ---- model, optimiser, schedule ------------------------------------------------------------------------
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = use_cuda
    model = make_model(cfg.model, cfg.pretrained, cfg.drop_rate, cfg.drop_path_rate)
    apply_freeze(model, cfg.mode)
    channels_last = use_cuda and FAMILY.get(cfg.model) == "cnn"
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    trainable, total = count_params(model)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=float(cfg.lr_peak), betas=tuple(cfg.betas), eps=cfg.eps)
    sched = make_scheduler(opt, total_steps, cfg)
    scaler = make_scaler(use_cuda)
    mix = make_mixup(cfg.augmentation, N_CLASSES)
    log(f"trainable parameters: {trainable / 1e6:.2f}M of {total / 1e6:.2f}M ({100 * trainable / total:.1f}%)")

    # ---- resume: local checkpoint, or the store's when it is ahead ------------------------------------------
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
                     "started": time.time(), "torch": env["torch"], "timm": env["timm"], "git_commit": env["git_commit"]})
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    val_loader = make_loader(ds_val, cfg, shuffle=False, batch_size=128)

    # ---- epochs --------------------------------------------------------------------------------------------
    push_failures = 0
    for epoch in range(epoch0, cfg.epochs):
        tr = train_one_epoch(model, make_loader(ds_train, cfg, True, epoch, drop_last=True), opt, sched, scaler, mix, device,
                             channels_last, max_batches)
        t0 = time.perf_counter()
        v_logits, v_y = predict(model, val_loader, device, channels_last, max_batches)
        v = M.summarize(v_logits.numpy(), v_y.numpy(), N_CLASSES)
        row = {"epoch": epoch + 1, **{k: round(val, 5) for k, val in tr.items()}, "val_s": round(time.perf_counter() - t0, 2),
               "val_top1": round(v["top1"], 5), "val_top5": round(v["top5"], 5), "val_nll": round(v["nll"], 5),
               "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_cuda else None,
               "owner": owner, "worker": sync.worker, "gpu": env["gpu"]}
        history.append(row)
        append_jsonl(ldir / "history.jsonl", row)
        log(f"epoch {epoch + 1}/{cfg.epochs} loss {row['train_loss']:.4f} | val top1 {100 * v['top1']:.2f}% nll {v['nll']:.3f} | "
            f"train {row['train_s']:.0f}s (data wait {row['data_wait_s']:.0f}s) {row['train_ips']:.0f} img/s")
        state = build_state(model, opt, sched, scaler, epoch + 1,
                            extra={"history": history, "config_hash": chash, "sessions": sessions})
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
        remaining = (cfg.epochs - epoch - 1) * epoch_s / 60
        status.set(epoch=epoch + 1, message=f"val top1 {100 * v['top1']:.2f}%", eta_min=round(remaining, 1))
        if out_of_time:
            raise TimeUp(f"{cfg.run_id}: stopping after epoch {epoch + 1}: the next epoch would not finish before the time limit")

    # ---- final: test (not for LR sweeps), logits, weights, result.json -----------------------------------
    v_logits, v_y = predict(model, val_loader, device, channels_last, max_batches)
    final = {"val": M.summarize(v_logits.numpy(), v_y.numpy(), N_CLASSES)}
    artifacts, push_files = {}, {}
    if cfg.kind != "sweep":
        t_logits, t_y = predict(model, make_loader(ds_test, cfg, shuffle=False, batch_size=128), device, channels_last, max_batches)
        final["test"] = M.summarize(t_logits.numpy(), t_y.numpy(), N_CLASSES)
        out = ldir / "final"
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": v_logits.half(), "labels": v_y, "split": "val"}, out / "logits_val.pt")
        torch.save({"logits": t_logits.half(), "labels": t_y, "split": "test"}, out / "logits_test.pt")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_final.pt")
        artifacts = {"logits_val": "logits_val.pt", "logits_test": "logits_test.pt", "weights": "model_final.pt"}
        push_files = {f.name: f for f in out.iterdir()}
    result = {
        "run_id": cfg.run_id, "kind": cfg.kind, "config": cfg.to_dict(), "config_hash": chash,
        "split_json_sha256_16": cfg.split_json_sha256_16, "n_train_images": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "trainable_M": round(trainable / 1e6, 3), "total_M": round(total / 1e6, 3), "trainable_pct": round(100 * trainable / total, 2),
        "final": final, "history": history,
        "timing": {"train_s_total": round(sum(h["train_s"] for h in history), 1), "val_s_total": round(sum(h["val_s"] for h in history), 1),
                   "note": "wall-clock on the session (loader-bound for head/partial); use cost_bench.py for architecture comparison"},
        "peak_mem_mb_max": max((h["peak_mem_mb"] or 0) for h in history), "sessions": sessions, "artifacts": artifacts,
        "env": env, "finished_at": time.time(), "finished_by": owner, "worker": sync.worker, "smoke": bool(smoke), "fake": bool(fake),
    }
    write_json(ldir / "result.json", result)
    sync.push_final(push_files, (ldir / "result.json").read_text(encoding="utf8"))
    log(f"DONE {cfg.run_id}: val top1 {100 * final['val']['top1']:.2f}%" +
        (f" | test top1 {100 * final['test']['top1']:.2f}%" if "test" in final else ""))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--owner", default=os.environ.get("A1_OWNER", "unknown"))
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 6 batches, writes only to the local disk")
    ap.add_argument("--fake", action="store_true", help="synthetic dataset (laptop dry run)")
    ap.add_argument("--force", action="store_true", help="ignore a live lock / config mismatch / existing result (ask the team first)")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths = Paths.from_env()
    max_batches = None
    if a.smoke:
        cfg = replace(cfg, run_id="smoke_" + cfg.run_id, epochs=2, lr_peak=(1e-4 if cfg.lr_peak == "auto" else cfg.lr_peak))
        max_batches = 6
        if not a.fake and a.data_root is None:
            a.data_root = str(ensure_data(paths.data_local))
        paths = Paths(LocalStore(paths.local / "smoke_store"), paths.local)    # every OUTPUT goes to the local disk
    run_training(cfg, paths, a.owner, None, a.data_root, max_batches, a.force, a.fake, a.smoke)


if __name__ == "__main__":
    main()
