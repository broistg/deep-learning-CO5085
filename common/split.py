"""Frozen train/val split shared by E1-E3. Pure numpy (no torch): stratified, seeded, hashed.
CIFAR-10: 50,000 official train images -> 45,000 train + 5,000 val (500 per class); the official 10,000 test images stay the test set."""
import hashlib
import json
from pathlib import Path

import numpy as np


def stratified_split(labels, val_per_class=500, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        val.extend(idx[:val_per_class].tolist())
        train.extend(idx[val_per_class:].tolist())
    return {"seed": seed, "val_per_class": val_per_class, "train": sorted(train), "val": sorted(val)}


def split_hash(split, n=16):
    blob = json.dumps({"train": split["train"], "val": split["val"]}, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def save_split(split, path):
    split = dict(split, sha256_16=split_hash(split))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split), encoding="utf8")
    return split["sha256_16"]


def load_split(path, expect=None):
    s = json.loads(Path(path).read_text(encoding="utf8"))
    h = split_hash(s)
    if h != s.get("sha256_16") or (expect and h != expect):
        raise ValueError(f"split hash mismatch: file says {s.get('sha256_16')}, computed {h}, expected {expect}")
    return s
