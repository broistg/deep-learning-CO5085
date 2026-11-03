"""ADE20K SceneParsing (ADEChallengeData2016): split, download (never redistributed), and the mmseg-style training / evaluation transforms.
Torch-free on purpose: the Dataset returns numpy arrays (the DataLoader's default collate turns them into tensors), so everything here is
testable without a GPU. Pipeline = mmsegmentation configs/_base_/datasets/ade20k.py, verified from source (protocol v3 section 4):
  train: labels 0->255 / 1..150->0..149, RandomResize((2048,512), ratio 0.5-2.0, keep ratio), RandomCrop 512 (cat_max_ratio 0.75),
         flip 0.5, PhotoMetricDistortion, pad to 512 (image with the mean colour = 0 after normalisation, label with 255)
  eval:  short side 512 (long side <= 2048), single scale, no flip; the labels keep their original size (logits are upsampled to it)."""
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from a2lib.config import IGNORE

ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE_ZIP_SHA256 = "7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29"   # recorded by us (benchmark 2026-10-26); the host publishes none
ADE_ZIP_BYTES = 967_382_037
N_TRAIN_OFFICIAL, N_VAL_OFFICIAL, N_DEV, SPLIT_SEED = 20_210, 2_000, 1_000, 42
ROOT_NAME = "ADEChallengeData2016"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- split
def official_names():
    """The official file stems are contiguous: ADE_train_00000001..00020210 and ADE_val_00000001..00002000 (checked against the archive
    by check_layout() every time the data is extracted)."""
    return ([f"ADE_train_{i:08d}" for i in range(1, N_TRAIN_OFFICIAL + 1)], [f"ADE_val_{i:08d}" for i in range(1, N_VAL_OFFICIAL + 1)])


def make_split(seed=SPLIT_SEED, n_dev=N_DEV):
    """19,210 train + 1,000 dev (random, from the official training set) and the official 2,000 validation images as our TEST set."""
    tr, va = official_names()
    perm = np.random.default_rng(seed).permutation(len(tr))
    dev = sorted(tr[i] for i in perm[:n_dev])
    train = sorted(tr[i] for i in perm[n_dev:])
    return {"seed": seed, "train": train, "dev": dev, "test": va}


def split_hash(s, n=16):
    blob = json.dumps({k: s[k] for k in ("train", "dev", "test")}, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def save_split(s, path):
    s = dict(s, sha256_16=split_hash(s))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(s), encoding="utf8")
    return s["sha256_16"]


def load_split(path, expect=None):
    s = json.loads(Path(path).read_text(encoding="utf8"))
    h = split_hash(s)
    if h != s.get("sha256_16") or (expect and h != expect):
        raise ValueError(f"split hash mismatch: file says {s.get('sha256_16')}, computed {h}, expected {expect}")
    return s


def subset_nested(train, fraction):
    """Nested random subsets: the first round(f*N) images of one fixed permutation, so the 25% subset is contained in the 50% one."""
    if fraction >= 1.0:
        return list(train)
    perm = np.random.default_rng(SPLIT_SEED + 1).permutation(len(train))
    return [train[i] for i in sorted(perm[: round(fraction * len(train))])]


# ---------------------------------------------------------------------------------------------------------------- download
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def download(url, dest, tries=3):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                n, last = 0, time.time()
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    n += len(b)
                    if time.time() - last > 30:
                        log(f"  downloaded {n / 2**20:.0f} MiB")
                        last = time.time()
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"  download attempt {i + 1}/{tries} failed: {e!r}")
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"could not download {url}. Enable internet in the Kaggle notebook settings, or put the zip on the local disk and pass --zip")


def ensure_data(local, zip_path=None, verify_sha=True):
    """Returns the folder that contains images/ and annotations/ (extracted on the session's local disk). Idempotent."""
    local = Path(local)
    root = local / ROOT_NAME
    ok = root / ".complete"
    if ok.exists():
        return root
    z = Path(zip_path) if zip_path else local / "ADEChallengeData2016.zip"
    if not z.exists():
        log(f"downloading ADE20K SceneParsing to {z} (about 920 MB)")
        download(ADE_URL, z)
    sha = sha256_file(z)
    log(f"zip sha256 {sha}")
    if verify_sha and sha != ADE_ZIP_SHA256:
        raise RuntimeError(f"zip sha256 differs from the recorded one ({ADE_ZIP_SHA256}); the host changed the file. Do not train on it until the "
                           f"team has checked (verify_sha=False only after agreeing)")
    if root.exists():
        shutil.rmtree(root)
    log("extracting ...")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(local)
    check_layout(root)
    ok.write_text(sha, encoding="utf8")
    return root


def check_layout(root):
    tr, va = official_names()
    for sub, names in (("training", tr), ("validation", va)):
        got = {p.stem for p in (Path(root) / "images" / sub).glob("*.jpg")}
        if got != set(names):
            raise RuntimeError(f"images/{sub}: expected the {len(names)} official stems, found {len(got)} (missing {len(set(names) - got)})")
        gt = {p.stem for p in (Path(root) / "annotations" / sub).glob("*.png")}
        if gt != set(names):
            raise RuntimeError(f"annotations/{sub} do not match the images")


# ---------------------------------------------------------------------------------------------------------------- transforms
def reduce_zero_label(lab):
    """ADE PNG values 0..150 (0 = 'other objects') -> 255 (ignored) and 0..149."""
    out = lab.astype(np.int16) - 1
    out[out < 0] = IGNORE
    return out.astype(np.uint8)


def rescale_size(w, h, max_long, max_short):
    """mmcv.rescale_size with keep_ratio: the largest scale that keeps long <= max_long and short <= max_short."""
    s = min(max_long / max(w, h), max_short / min(w, h))
    return max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))


def random_resize(img, lab, rng, scale=(2048, 512), ratio_range=(0.5, 2.0)):
    r = rng.uniform(*ratio_range)
    nw, nh = rescale_size(img.shape[1], img.shape[0], int(scale[0] * r), int(scale[1] * r))
    img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    lab = np.asarray(Image.fromarray(lab).resize((nw, nh), Image.NEAREST))
    return img, lab


def crop_box(lab, rng, crop, cat_max_ratio=0.75, tries=10):
    """mmseg RandomCrop: up to 10 draws; accept a crop in which no class covers >= 75% of the labelled pixels (and >= 2 classes occur)."""
    h, w = lab.shape

    def draw():
        y = int(rng.integers(0, max(h - crop, 0) + 1))
        x = int(rng.integers(0, max(w - crop, 0) + 1))
        return y, x
    y, x = draw()
    if cat_max_ratio < 1.0:
        for _ in range(tries):
            c = lab[y:y + crop, x:x + crop]
            c = c[c != IGNORE]
            if c.size:
                cnt = np.bincount(c)
                cnt = cnt[cnt > 0]
                if len(cnt) > 1 and cnt.max() / cnt.sum() < cat_max_ratio:
                    break
            y, x = draw()
    return y, x


def photometric(img, rng, brightness=32, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=18):
    """mmseg PhotoMetricDistortion on an RGB uint8 image (OpenCV hue 0..179 -> PIL hue 0..255)."""
    x = img.astype(np.float32)
    if rng.integers(2):
        x = np.clip(x + rng.uniform(-brightness, brightness), 0, 255)
    mode = int(rng.integers(2))
    if mode == 1 and rng.integers(2):
        x = np.clip(x * rng.uniform(*contrast), 0, 255)
    hsv = np.asarray(Image.fromarray(x.astype(np.uint8)).convert("HSV")).astype(np.float32)
    if rng.integers(2):
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*saturation), 0, 255)
    if rng.integers(2):
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue) * 256.0 / 180.0) % 256.0
    x = np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")).astype(np.float32)
    if mode == 0 and rng.integers(2):
        x = np.clip(x * rng.uniform(*contrast), 0, 255)
    return x.astype(np.uint8)


def pad_to(img, lab, size, mean):
    h, w = lab.shape
    if h == size and w == size:
        return img, lab
    pi = np.empty((size, size, 3), np.uint8)
    pi[:] = np.round(np.asarray(mean)).astype(np.uint8)
    pl = np.full((size, size), IGNORE, np.uint8)
    pi[:h, :w], pl[:h, :w] = img, lab
    return pi, pl


def train_sample(img, lab, rng, aug):
    """img uint8 HWC RGB, lab uint8 HW already reduce-zero-labelled -> (img uint8 [S,S,3], lab uint8 [S,S])."""
    img, lab = random_resize(img, lab, rng, tuple(aug["scale"]), tuple(aug["ratio_range"]))
    S = aug["crop"]
    y, x = crop_box(lab, rng, S, aug["cat_max_ratio"])
    img, lab = img[y:y + S, x:x + S], lab[y:y + S, x:x + S]
    if rng.random() < aug["flip_p"]:
        img, lab = img[:, ::-1], lab[:, ::-1]
    img = np.ascontiguousarray(img)
    if aug["photometric"]:
        img = photometric(img, rng)
    return pad_to(img, np.ascontiguousarray(lab), S, aug["mean"])


def eval_image(img, scale=(2048, 512)):
    nw, nh = rescale_size(img.shape[1], img.shape[0], scale[0], scale[1])
    return np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))


# ---------------------------------------------------------------------------------------------------------------- dataset
class ADEDataset:
    """train=True: random augmentation, every sample [512,512]; train=False: one image per item (use batch_size 1), the label keeps its size.
    The augmentation RNG depends only on (seed, epoch, sample index), so the data does not depend on the number of workers or on resuming."""

    def __init__(self, root, names, split_dir, aug, train, seed=0):
        self.root, self.names, self.sub, self.aug, self.train, self.seed, self.epoch = Path(root), list(names), split_dir, aug, train, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.names)

    def load(self, i):
        n = self.names[i]
        with Image.open(self.root / "images" / self.sub / f"{n}.jpg") as im:
            img = np.asarray(im.convert("RGB"))
        with Image.open(self.root / "annotations" / self.sub / f"{n}.png") as lb:
            lab = reduce_zero_label(np.asarray(lb))
        return img, lab

    def __getitem__(self, i):
        img, lab = self.load(i)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            return train_sample(img, lab, rng, self.aug)
        return eval_image(img, tuple(self.aug["eval_scale"])), lab


def make_fake_ade(root, n_train=24, n_val=8, size=(160, 120), seed=0):
    """Tiny synthetic stand-in (laptop dry run). Same folder layout and file names as the real data."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": []}
    for sub, n, tag in (("training", n_train, "train"), ("validation", n_val, "val")):
        (root / "images" / sub).mkdir(parents=True, exist_ok=True)
        (root / "annotations" / sub).mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            stem = f"ADE_{tag}_{i:08d}"
            w, h = int(size[0] * rng.uniform(0.8, 1.6)), int(size[1] * rng.uniform(0.8, 1.6))
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(root / "images" / sub / f"{stem}.jpg")
            lab = np.zeros((h, w), np.uint8)
            for _ in range(4):
                y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
                lab[y0:y0 + h // 3, x0:x0 + w // 3] = rng.integers(1, 151)
            Image.fromarray(lab).save(root / "annotations" / sub / f"{stem}.png")
            out[tag].append(stem)
    return out
