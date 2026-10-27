"""Crash-safe checkpoints: atomic save with a last-good backup, verification, push/pull to a second folder
(Google Drive), and an advisory single-writer lock. The logic mirrors the smoke test that passed 23/23 on Colab."""
import json
import os
import shutil
import socket
import time
import zipfile
from pathlib import Path

import torch

from common.repro import atomic_write_text, capture_rng, restore_rng


def log(*a):
    print(*a, flush=True)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_state(model, opt, sched, scaler, epoch, extra=None):
    """Call at an EPOCH boundary. `epoch` = number of epochs COMPLETED."""
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng(),
        "epoch": epoch,
        "extra": extra or {},
        "saved_by": socket.gethostname(),
        "saved_at": time.time(),
    }


def restore_state(state, model, opt, sched, scaler, with_rng=True):
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    if sched is not None and state.get("sched"):
        sched.load_state_dict(state["sched"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if with_rng:
        restore_rng(state["rng"])
    return state["epoch"]


def verify_ckpt(path):
    """A torch checkpoint is a zip: checks the central directory (truncation) and every member's CRC32 (bit flips)."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:  # noqa: BLE001
        return False


def atomic_save(obj, path):
    """tmp -> fsync -> verify -> (verified last.pt becomes last_good.pt) -> rename tmp over last.pt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if not verify_ckpt(tmp):
        tmp.unlink(missing_ok=True)
        raise IOError(f"checkpoint failed verification right after writing: {path}")
    if path.exists() and verify_ckpt(path):
        os.replace(path, path.with_name("last_good.pt"))
    os.replace(tmp, path)


def find_checkpoint(ckpt_dir, stale_tmp_s=120, verbose=True):
    """(path, state) of the newest valid checkpoint (last.pt, then last_good.pt), or (None, None)."""
    d = Path(ckpt_dir)
    if not d.exists():
        return None, None
    now = time.time()
    for t in d.glob("*.tmp"):
        try:
            if now - t.stat().st_mtime >= stale_tmp_s:
                t.unlink()
        except OSError:
            pass
    for name in ("last.pt", "last_good.pt"):
        p = d / name
        if not p.exists():
            continue
        if not verify_ckpt(p):
            if verbose:
                log(f"    ! {p} is corrupt/truncated -> skipping")
            continue
        try:
            return p, torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            if verbose:
                log(f"    ! {p} verified but failed to load ({e}) -> skipping")
    return None, None


def push_ckpt(local_dir, remote_dir):
    """Copy local last.pt to the remote folder with the same safety rules: copy -> verify -> rotate -> rename."""
    src = Path(local_dir) / "last.pt"
    dst_dir = Path(remote_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / "last.pt.tmp"
    shutil.copyfile(src, tmp)
    if not verify_ckpt(tmp):
        tmp.unlink(missing_ok=True)
        raise IOError("remote copy failed verification")
    if (dst_dir / "last.pt").exists() and verify_ckpt(dst_dir / "last.pt"):
        os.replace(dst_dir / "last.pt", dst_dir / "last_good.pt")
    os.replace(tmp, dst_dir / "last.pt")


def best_checkpoint(dirs):
    """Newest valid checkpoint (highest completed epoch) among several folders. Returns (dir, path, state)."""
    best = (None, None, None)
    for d in dirs:
        p, st = find_checkpoint(d)
        if st is not None and (best[2] is None or st["epoch"] > best[2]["epoch"]):
            best = (Path(d), p, st)
    return best


class RunLock:
    """Advisory single-writer lock (small JSON next to the run's files). Protects against a 'zombie' Colab session that
    looks disconnected but is still training. A different owner can take over only after the heartbeat is stale
    (or with force=True, after a human has agreed in the team chat)."""

    def __init__(self, run_dir, owner, stale_s=1200):
        self.path = Path(run_dir) / "run.lock.json"
        self.owner, self.stale_s = owner, stale_s

    def info(self):
        try:
            return json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def _write(self):
        atomic_write_text(self.path, json.dumps({"owner": self.owner, "host": socket.gethostname(), "heartbeat": time.time()}))

    def acquire(self, force=False):
        if self.path.exists() and not force:
            info = self.info()
            age = time.time() - info.get("heartbeat", 0)
            if info.get("owner") != self.owner and age < self.stale_s:
                raise RuntimeError(f"run is locked by '{info.get('owner')}' (heartbeat {age:.0f}s ago)")
        self._write()

    heartbeat = _write

    def release(self):
        info = self.info()
        if info.get("owner") in (None, self.owner):
            self.path.unlink(missing_ok=True)
