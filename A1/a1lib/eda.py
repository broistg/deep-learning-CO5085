"""Exploratory data analysis for Chapter 2 of the report.

    python -m a1lib.eda counts   --split A1/split.json          # torch-free; split sizes per class (needs no images)
    python -m a1lib.eda samples  --data <food101_256 folder>    # fixed grid of training images (needs the resized cache)
    python -m a1lib.eda augment  --data <food101_256 folder>    # the SAME images after the training augmentation (needs torch)
    python -m a1lib.eda sizes    --raw <food-101 folder>        # original image sizes (needs the raw download; optional)
    python -m a1lib.eda all      --data ... [--raw ...]

Outputs: A1/results/eda_counts.json (+ eda_sizes.json), A1/figures/eda_{counts,samples,augment,sizes}.png|pdf. The small JSON files are
committed; the figures contain a few training images and are only used as small illustrations in the report (see the licence note).
Everything is chosen by a fixed rule (below), not by looking at the images:
  * classes: every 17th class (0, 17, 34, 51, 68, 85); images: the first 6 entries of a seeded shuffle;
  * augmentation: image k of the grid is augmented with sample_seed(0, epoch=0, index=k), i.e. the exact random stream the trainer uses.
`counts` runs in the replication notebook (it needs only split.json). `samples` / `augment` need the images and are skipped there."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

A1 = Path(__file__).resolve().parents[1]
GRID_CLASSES = list(range(0, 101, 17))[:6]
N_PER_CLASS = 6


def _split(path):
    from a1lib.data import load_split
    return load_split(path)


def _names(split):
    return [c.replace("_", " ") for c in split["classes"]] if isinstance(split["classes"], list) else None


def cmd_counts(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = _split(a.split)
    classes = sorted({e.split("/")[0] for e in sp["train"]})
    out = {"n_classes": len(classes), "sets": {}}
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    for name in ("train", "val", "test"):
        cnt = {c: 0 for c in classes}
        for e in sp[name]:
            cnt[e.split("/")[0]] += 1
        v = np.array([cnt[c] for c in classes])
        out["sets"][name] = {"total": int(v.sum()), "min": int(v.min()), "max": int(v.max()), "balanced": bool(v.min() == v.max())}
        ax.plot(range(len(classes)), v, label=f"{name} ({int(v.sum()):,})".replace(",", " "), lw=2)
    disjoint = len(set(sp["train"]) | set(sp["val"]) | set(sp["test"])) == len(sp["train"]) + len(sp["val"]) + len(sp["test"])
    out["disjoint"] = disjoint
    ax.set_yscale("log"), ax.set_xlabel("chỉ số lớp"), ax.set_ylabel("số ảnh / lớp"), ax.legend(frameon=False, ncol=3)
    ax.set_ylim(30, 400)
    fig.tight_layout()
    fd = Path(a.figures)
    fd.mkdir(parents=True, exist_ok=True)
    Path(a.results).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"eda_counts.{ext}", dpi=150)
    (Path(a.results) / "eda_counts.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps(out))


def _pick(sp):
    by = {}
    for e in sorted(sp["train"]):
        by.setdefault(e.split("/")[0], []).append(e)
    classes = sorted(by)
    rows = []
    for ci in GRID_CLASSES:
        c = classes[ci]
        items = by[c][:]
        random.Random(f"eda|{c}").shuffle(items)
        rows.append((c, items[:N_PER_CLASS]))
    return rows


def _grid(rows, loader, title_fn, path_base, figures, cell=1.35):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(rows), len(rows[0][1])
    fig, axs = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr * 1.1))
    for i, (c, items) in enumerate(rows):
        for j, e in enumerate(items):
            ax = axs[i][j]
            ax.imshow(loader(i * nc + j, e))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(title_fn(c), fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
    fig.tight_layout(pad=0.3)
    fd = Path(figures)
    fd.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"{path_base}.{ext}", dpi=170)
    plt.close(fig)


def _open(root, e):
    from PIL import Image
    with Image.open(Path(root) / "images" / f"{e}.jpg") as im:
        return im.convert("RGB")


def cmd_samples(a):
    sp = _split(a.split)
    rows = _pick(sp)
    _grid(rows, lambda k, e: np.asarray(_open(a.data, e).resize((224, 224))), lambda c: c.replace("_", " "), "eda_samples", a.figures)
    print("wrote eda_samples")


def cmd_augment(a):
    """Original (left) and five augmented views per class; the views use the trainer's own per-sample seeds."""
    import torch
    from a1lib.augment import train_transform
    from a1lib.config import default_augmentation
    from a1lib.seeding import sample_seed
    tfm = train_transform(default_augmentation())
    sp = _split(a.split)
    rows = []
    for c, items in _pick(sp):
        rows.append((c, [items[0]] * N_PER_CLASS))

    def load(k, e):
        im = _open(a.data, e)
        j = k % N_PER_CLASS
        if j == 0:
            return np.asarray(im.resize((224, 224)))
        s = sample_seed(0, 0, k)
        random.seed(s)
        torch.manual_seed(s)
        return tfm(im).permute(1, 2, 0).numpy()
    _grid(rows, load, lambda c: c.replace("_", " "), "eda_augment", a.figures)
    print("wrote eda_augment")


def cmd_sizes(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    root = Path(a.raw) / "images"
    rng = random.Random(42)
    files = sorted(root.glob("*/*.jpg"))
    sample = rng.sample(files, min(a.n, len(files)))
    wh = []
    for f in sample:
        with Image.open(f) as im:
            wh.append(im.size)
    w, h = np.array(wh).T
    short, long_ = np.minimum(w, h), np.maximum(w, h)
    out = {"n": len(wh), "short_median": float(np.median(short)), "long_max": int(long_.max()), "long_median": float(np.median(long_)),
           "aspect_median": float(np.median(long_ / short)), "aspect_p95": float(np.percentile(long_ / short, 95)),
           "short_below_256_pct": float(100 * (short < 256).mean())}
    (Path(a.results) / "eda_sizes.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.4))
    axs[0].hist(short, bins=40), axs[0].set_xlabel("cạnh ngắn (px)"), axs[0].set_ylabel("số ảnh")
    axs[1].hist(long_ / short, bins=40), axs[1].set_xlabel("tỷ lệ khung hình (cạnh dài / cạnh ngắn)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(a.figures) / f"eda_sizes.{ext}", dpi=150)
    print(json.dumps(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["counts", "samples", "augment", "sizes", "all"])
    ap.add_argument("--split", default=str(A1 / "split.json"))
    ap.add_argument("--data", default=None, help="folder that contains images/ (the resized cache, e.g. /kaggle/temp/a1_local/data/food101_256)")
    ap.add_argument("--raw", default=None, help="original Food-101 folder (contains images/), only for `sizes`")
    ap.add_argument("--results", default=str(A1 / "results"))
    ap.add_argument("--figures", default=str(A1 / "figures"))
    ap.add_argument("--n", type=int, default=2000, help="images sampled for `sizes`")
    a = ap.parse_args(argv)
    todo = ["counts", "samples", "augment", "sizes"] if a.cmd == "all" else [a.cmd]
    for c in todo:
        if c in ("samples", "augment") and not a.data:
            print(f"skip {c}: --data not given (needs the resized cache)")
            continue
        if c == "sizes" and not a.raw:
            print("skip sizes: --raw not given (needs the original Food-101 download)")
            continue
        {"counts": cmd_counts, "samples": cmd_samples, "augment": cmd_augment, "sizes": cmd_sizes}[c](a)


if __name__ == "__main__":
    main()
