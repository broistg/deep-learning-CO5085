"""CIFAR-10 helpers shared by E1-E3: loading, the frozen train/val split, channel statistics.
Numpy only at import time (torchvision is imported inside `load_cifar`), so tests and planning commands run without torch.

    python -m common.cifar          # downloads CIFAR-10 if needed, writes/checks the split, prints its hash and the statistics
Everybody must see the SAME split hash (it depends only on the labels and seed 42)."""
from pathlib import Path

import numpy as np

from .split import load_split, save_split, stratified_split

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"                         # one shared download for E1-E3 (git-ignored)
SPLIT_PATH = REPO / "common" / "cifar10_split.json"   # frozen 45,000 / 5,000 split (commit it after the first run)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SPLIT_SEED, VAL_PER_CLASS = 42, 500


def load_cifar(root=None):
    """-> (x_train uint8 (50000, 32, 32, 3), y_train, x_test uint8 (10000, 32, 32, 3), y_test); downloads on first use."""
    from torchvision.datasets import CIFAR10
    root = root or DATA_ROOT
    tr, te = CIFAR10(root, train=True, download=True), CIFAR10(root, train=False, download=True)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets)


def fake_cifar(n_train=2000, n_test=500, seed=0):
    """Random images with the right shapes: for smoke tests without a download."""
    r = np.random.default_rng(seed)
    return (r.integers(0, 256, (n_train, 32, 32, 3), dtype=np.uint8), r.integers(0, 10, n_train),
            r.integers(0, 256, (n_test, 32, 32, 3), dtype=np.uint8), r.integers(0, 10, n_test))


def get_splits(root=None, split_path=None, fake=False):
    """-> ({'train': (x, y), 'val': (x, y), 'test': (x, y)} with x uint8 NHWC numpy, split hash).
    The real split is written to `split_path` on first use and verified against its hash on every later use."""
    xtr, ytr, xte, yte = fake_cifar() if fake else load_cifar(root)
    if fake:
        s = stratified_split(ytr, val_per_class=20, seed=SPLIT_SEED)
    else:
        split_path = Path(split_path or SPLIT_PATH)
        if not split_path.exists():
            save_split(stratified_split(ytr, VAL_PER_CLASS, SPLIT_SEED), split_path)
        s = load_split(split_path)
    tr, va = np.array(s["train"]), np.array(s["val"])
    return {"train": (xtr[tr], ytr[tr]), "val": (xtr[va], ytr[va]), "test": (xte, yte)}, s.get("sha256_16", "fake")


def channel_stats(x_nhwc, chunk=2500):
    """Per-channel mean and (population) std of a uint8 NHWC array, pixels scaled to [0, 1]. Call it on the TRAIN images only."""
    n = 0
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    for i in range(0, len(x_nhwc), chunk):
        b = x_nhwc[i:i + chunk].reshape(-1, 3).astype(np.float64) / 255.0
        n += len(b)
        s += b.sum(0)
        ss += (b * b).sum(0)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    return tuple(float(m) for m in mean), tuple(float(v) for v in std)


def main():
    splits, h = get_splits()
    mean, std = channel_stats(splits["train"][0])
    print(f"split hash {h}  (file {SPLIT_PATH})")
    print(f"train {len(splits['train'][1])}  val {len(splits['val'][1])}  test {len(splits['test'][1])}")
    print(f"train mean {tuple(round(m, 4) for m in mean)}  std {tuple(round(v, 4) for v in std)}")
    print("val per class:", np.bincount(splits["val"][1], minlength=10).tolist())


if __name__ == "__main__":
    main()
