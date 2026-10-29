"""Food-101: split, nested class-balanced subsets, pre-resized cache, datasets.

Layout of the pre-resized cache (one tar on the shared Drive, extracted to fast local disk on every Colab VM):
    food101_256/images/<class>/<id>.jpg   (short side 256, bicubic, JPEG q90, never upscaled)
    food101_256/classes.txt, food101_256/split.json
"""
import json
import os
import random
import shutil
import subprocess
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

from common.repro import atomic_write_text, sha256_file

N_CLASSES = 101
SPLIT_SHA16 = "75ed81bfe380c089"   # sha256[:16] of the team's split.json (seed 42, 50 val images per class)
N_IMAGES = 101_000
CACHE_NAME = "food101_256"


# ----------------------------------------------------------------------------------------------------------
# split (pure Python; reproduces exactly the split made by smoke_test.py)
# ----------------------------------------------------------------------------------------------------------
def make_split(food_root, seed=42, val_per_class=50):
    """food_root = the official `food-101` folder (contains meta/ and images/). Deterministic."""
    base = Path(food_root)
    classes = (base / "meta" / "classes.txt").read_text().split()
    read = lambda n: [l.strip() for l in (base / "meta" / n).read_text().splitlines() if l.strip()]  # noqa: E731
    train_all, test = read("train.txt"), read("test.txt")
    rng = random.Random(seed)
    by_class = {c: [] for c in classes}
    for e in sorted(train_all):
        by_class[e.split("/")[0]].append(e)
    train, val = [], []
    for c in classes:
        items = by_class[c][:]
        rng.shuffle(items)
        val += items[:val_per_class]
        train += items[val_per_class:]
    train.sort(), val.sort(), test.sort()
    return {"seed": seed, "val_per_class": val_per_class, "train_per_class": 0,
            "classes": classes, "train": train, "val": val, "test": test}


def save_split(split, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return sha256_file(path)[:16]


def load_split(path, expect=SPLIT_SHA16):
    h = sha256_file(path)[:16]
    if expect and h != expect:
        raise ValueError(f"split.json hash {h} != expected {expect}. Wrong file or a different split; do not train.")
    return json.loads(Path(path).read_text())


def subset_per_class(entries, fraction, seed=42):
    """Class-balanced subset; NESTED: the 10% subset is contained in the 25% subset, and so on.
    `fraction=0.2` gives 140 images per class from 700."""
    if fraction >= 1.0:
        return sorted(entries)
    by_class = {}
    for e in sorted(entries):
        by_class.setdefault(e.split("/")[0], []).append(e)
    out = []
    for c in sorted(by_class):
        items = by_class[c][:]
        random.Random(f"{seed}|{c}").shuffle(items)
        out += items[: max(1, round(fraction * len(items)))]
    return sorted(out)


# ----------------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------------
class FoodDataset:
    """Returns (uint8 CHW tensor from `transform`, label). `transform` must end in PILToTensor (normalisation happens on the GPU)."""

    def __init__(self, images_root, entries, class_to_idx, transform):
        self.root, self.entries, self.cidx, self.tfm = Path(images_root), entries, class_to_idx, transform
        self.labels = [class_to_idx[e.split("/")[0]] for e in entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.root / f"{self.entries[i]}.jpg") as im:
            x = im.convert("RGB")
        return self.tfm(x), self.labels[i]


# ----------------------------------------------------------------------------------------------------------
# pre-resized cache
# ----------------------------------------------------------------------------------------------------------
def _resize_one(job):
    from PIL import Image
    src, dst, short = job
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = short / min(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=90)
        return os.path.getsize(dst)
    except Exception:  # noqa: BLE001
        return -1


def build_cache(food_root, shared_dir, work_dir, n_proc=None):
    """One-time job (one team member): resize all 101,000 images, tar them, copy the tar to the shared folder."""
    base, shared, work = Path(food_root), Path(shared_dir), Path(work_dir)
    split = make_split(base)
    out = work / CACHE_NAME
    entries = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    jobs = [(str(base / "images" / f"{e}.jpg"), str(out / "images" / f"{e}.jpg"), 256) for e in entries
            if not (out / "images" / f"{e}.jpg").exists()]
    print(f"resizing {len(jobs)} of {len(entries)} images (about 9 min on 2 cores)", flush=True)
    bad = 0
    with Pool(n_proc or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())) as p:
        for sz in p.imap_unordered(_resize_one, jobs, chunksize=64):
            bad += sz < 0
    if bad:
        raise RuntimeError(f"{bad} images failed to convert")
    h = save_split(split, out / "split.json")
    if h != SPLIT_SHA16:
        raise RuntimeError(f"split hash {h} != expected {SPLIT_SHA16}")
    shutil.copyfile(base / "meta" / "classes.txt", out / "classes.txt")
    tar_local = work / f"{CACHE_NAME}.tar"
    with tarfile.open(tar_local, "w") as tf:
        tf.add(out, arcname=CACHE_NAME)
    shared.mkdir(parents=True, exist_ok=True)
    dst = shared / "cache" / f"{CACHE_NAME}.tar"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(tar_local, tmp)
    assert tmp.stat().st_size == tar_local.stat().st_size, "copy to shared folder is incomplete"
    os.replace(tmp, dst)
    atomic_write_text(shared / "cache" / f"{CACHE_NAME}.manifest.json", json.dumps(
        {"n_images": len(entries), "bytes": tar_local.stat().st_size, "split_json_sha256_16": h,
         "resize": "short side 256, bicubic, JPEG q90, never upscaled", "built_at": time.time()}, indent=1))
    print(f"cache written to {dst}", flush=True)


def count_images(images_dir):
    n = 0
    for _r, _d, files in os.walk(images_dir):
        n += sum(1 for f in files if f.endswith(".jpg"))
    return n


def ensure_data(shared_dir, local_dir, expect_images=N_IMAGES, allow_fake=False):
    """Make `<local_dir>/food101_256` available on fast local disk. Returns its path. Idempotent."""
    local, root = Path(local_dir), Path(local_dir) / CACHE_NAME
    marker = root / ".complete"
    if marker.exists():
        return root
    tar = Path(shared_dir) / "cache" / f"{CACHE_NAME}.tar"
    if not tar.exists():
        raise FileNotFoundError(f"{tar} not found. One member must run `python -m a1lib.data build-cache` first.")
    local.mkdir(parents=True, exist_ok=True)
    print(f"extracting {tar} -> {local} (1-2 min)", flush=True)
    subprocess.run(["tar", "-xf", str(tar), "-C", str(local)], check=True)
    n = count_images(root / "images")
    if n != expect_images:
        raise RuntimeError(f"extracted {n} images, expected {expect_images}")
    load_split(root / "split.json")  # verifies the hash
    marker.write_text(str(n))
    return root


def make_fake_cache(local_dir, per_class=12, size=96, seed=0):
    """Tiny synthetic Food-101-shaped cache for dry runs on a laptop (no download). NOT the real split (hash check skipped)."""
    import numpy as np
    from PIL import Image
    root = Path(local_dir) / (CACHE_NAME + "_fake")
    if (root / ".complete").exists():
        return root
    rng = np.random.default_rng(seed)
    classes = [f"class_{i:03d}" for i in range(N_CLASSES)]
    n_test = max(1, per_class // 4)
    (root / "images").mkdir(parents=True, exist_ok=True)
    train, test = [], []
    for c in classes:
        (root / "images" / c).mkdir(exist_ok=True)
        for j in range(per_class):
            arr = rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)
            Image.fromarray(arr).resize((size + 32, size), Image.BILINEAR).save(root / "images" / c / f"{j}.jpg", quality=85)
            (test if j < n_test else train).append(f"{c}/{j}")
    (root / "classes.txt").write_text("\n".join(classes) + "\n")
    rg = random.Random(42)
    by = {c: [e for e in train if e.startswith(c + "/")] for c in classes}
    tr, va = [], []
    for c in classes:
        items = by[c][:]
        rg.shuffle(items)
        va += items[:2]
        tr += items[2:]
    (root / "split.json").write_text(json.dumps({"seed": 42, "val_per_class": 2, "train_per_class": 0, "classes": classes,
                                                 "train": sorted(tr), "val": sorted(va), "test": sorted(test)}))
    (root / ".complete").write_text("fake")
    return root


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Data utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-cache", help="one-time: resize Food-101 and put the tar on the shared Drive folder")
    b.add_argument("--food-root", required=True, help="official 'food-101' folder (downloaded by torchvision)")
    b.add_argument("--shared-dir", required=True)
    b.add_argument("--work-dir", default="/content/a1_work")
    s = sub.add_parser("make-split", help="write split.json from the official meta files and print its hash")
    s.add_argument("--food-root", required=True)
    s.add_argument("--out", default="A1/split.json")
    a = ap.parse_args()
    if a.cmd == "build-cache":
        build_cache(a.food_root, a.shared_dir, a.work_dir)
    else:
        h = save_split(make_split(a.food_root), a.out)
        print(f"wrote {a.out}  sha256[:16]={h}  (expected {SPLIT_SHA16}: {'OK' if h == SPLIT_SHA16 else 'MISMATCH'})")
