"""CIFAR-10 helpers shared by E1-E3: loading, the frozen train/val split, channel statistics.
Numpy only at import time (torchvision is imported inside `load_cifar`), so tests and planning commands run without torch.

    python -m common.cifar          # downloads CIFAR-10 if needed, writes/checks the split, prints its hash and the statistics
Everybody must see the SAME split hash (it depends only on the labels and seed 42)."""
import json
from pathlib import Path

import numpy as np

from .split import load_split, save_split, stratified_split

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"                         # one shared download for E1-E3 (git-ignored)
SPLIT_PATH = REPO / "common" / "cifar10_split.json"   # frozen 45,000 / 5,000 split (commit it after the first run)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SPLIT_SEED, VAL_PER_CLASS = 42, 500


FINGERPRINT_PATH = REPO / "common" / "cifar10_fingerprint.json"   # SHA-256 of the exact arrays (commit it once; see below)
HF_REPO = "uoft-cs/cifar10"


def fingerprint(xtr, ytr, xte, yte):
    """SHA-256 over the four arrays (images as uint8 NHWC, labels as int64). Identical data -> identical hash, whatever the download source."""
    import hashlib
    h = hashlib.sha256()
    for a in (xtr, ytr, xte, yte):
        a = np.ascontiguousarray(a, dtype=np.uint8 if a.ndim == 4 else np.int64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _load_torchvision(root):
    from torchvision.datasets import CIFAR10
    tr, te = CIFAR10(root, train=True, download=True), CIFAR10(root, train=False, download=True)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets)


def _load_hf(root):
    """The same CIFAR-10 through the Hugging Face `datasets` library (`uoft-cs/cifar10`, ~145 MB, CDN: fast where the Toronto server
    is slow). Only accepted if the fingerprint matches the torchvision data (see `load_cifar`)."""
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        ds = load_dataset(HF_REPO, split=split, cache_dir=str(Path(root) / "hf"))
        out += [np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds["img"]]), np.asarray(ds["label"], dtype=np.int64)]
    return tuple(out)


def load_cifar(root=None, source="auto"):
    """-> (x_train uint8 (50000, 32, 32, 3), y_train, x_test uint8 (10000, 32, 32, 3), y_test).
    source: "auto" = cached .npz, else Hugging Face, else torchvision; "hf" / "torchvision" force one.
    The arrays must match `cifar10_fingerprint.json` (made from the torchvision data); otherwise it raises, because the frozen split
    is defined on the torchvision order. The verified arrays are cached in <root>/cifar10_arrays.npz."""
    root = Path(root or DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cifar10_arrays.npz"
    expected = json.loads(FINGERPRINT_PATH.read_text(encoding="utf8"))["sha256"] if FINGERPRINT_PATH.exists() else None
    if cache.exists() and source == "auto":
        z = np.load(cache)
        arrs = z["xtr"], z["ytr"], z["xte"], z["yte"]
        if expected is None or fingerprint(*arrs) == expected:
            return arrs
        print("cached arrays do not match the fingerprint: reloading")
    loaders = {"hf": [_load_hf], "torchvision": [_load_torchvision], "auto": [_load_hf, _load_torchvision]}[source]
    err = None
    for fn in loaders:
        try:
            arrs = fn(root)
        except Exception as e:  # noqa: BLE001  (offline, library missing, slow host...)
            err = e
            print(f"{fn.__name__} failed: {e}")
            continue
        got = fingerprint(*arrs)
        if expected is not None and got != expected:
            err = RuntimeError(f"{fn.__name__}: data fingerprint {got[:16]} != committed {expected[:16]}")
            print(err)
            continue
        if expected is None:
            print(f"WARNING: no {FINGERPRINT_PATH.name}: data from {fn.__name__} is not verified (fingerprint {got[:16]})")
        np.savez(cache, xtr=arrs[0], ytr=arrs[1], xte=arrs[2], yte=arrs[3])
        return arrs
    raise RuntimeError(f"could not load a verified CIFAR-10: {err}")


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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Download/verify CIFAR-10, check the frozen split")
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "torchvision"], help="auto: Hugging Face first (fast), torchvision as fallback")
    ap.add_argument("--write-fingerprint", action="store_true", help="once, on a machine with the torchvision data: write common/cifar10_fingerprint.json")
    a = ap.parse_args(argv)
    if a.write_fingerprint:
        arrs = load_cifar(source="torchvision")
        FINGERPRINT_PATH.write_text(json.dumps({"sha256": fingerprint(*arrs), "source": "torchvision.datasets.CIFAR10",
                                                "shapes": [list(x.shape) for x in arrs]}, indent=1) + "\n", encoding="utf8")
        print(f"wrote {FINGERPRINT_PATH}")
    elif a.source != "auto":
        load_cifar(source=a.source)
    splits, h = get_splits()
    mean, std = channel_stats(splits["train"][0])
    print(f"split hash {h}  (file {SPLIT_PATH})")
    print(f"train {len(splits['train'][1])}  val {len(splits['val'][1])}  test {len(splits['test'][1])}")
    print(f"train mean {tuple(round(m, 4) for m in mean)}  std {tuple(round(v, 4) for v in std)}")
    print("val per class:", np.bincount(splits["val"][1], minlength=10).tolist())


if __name__ == "__main__":
    main()
