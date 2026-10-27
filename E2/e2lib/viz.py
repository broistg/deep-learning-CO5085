"""Attention maps from trained models (artifacts/<run_id>/model_final.pt).   python -m e2lib.viz [--seed 0]
Two views per tokenizer, on the same 4 test images:
  rollout   : attention rollout (Abnar & Zuidema 2020) CLS -> input tokens, mean over heads, residual added, layers multiplied
  last-layer: CLS attention of every head in the last block
Maps are drawn on the token grid and upsampled to 32x32 (row tokenizer: one value per image row)."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import CLASSES, DATA_ROOT, SPLIT_PATH, channel_stats, get_splits, load_cifar  # noqa: E402
from .model import TinyTransformer  # noqa: E402
from .train import Cfg, E2_DIR  # noqa: E402

TOK = ["patch4", "patch8", "row", "cnn"]
NAMES = {"patch4": "Patch 4×4", "patch8": "Patch 8×8", "row": "Hàng", "cnn": "CNN-stem"}


def rollout(attns):
    """attns: list of (B, h, N, N). Returns (B, N) = CLS row of the product of (0.5 A_mean + 0.5 I) over layers."""
    n = attns[0].shape[-1]
    eye = torch.eye(n)
    r = eye.expand(attns[0].shape[0], n, n)
    for a in attns:
        a = 0.5 * a.mean(1) + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        r = a @ r
    return r[:, 0, 1:]


def to_img(v, grid):
    h, w = grid
    m = v.reshape(1, 1, h, w) if h > 1 else v.reshape(1, 1, 1, -1)
    if w == 1:                                           # row tokenizer: tokens are rows
        m = v.reshape(1, 1, h, 1)
    if w == 1:                                           # nearest has no align_corners option (passing it raises ValueError)
        m = F.interpolate(m, size=(32, 32), mode="nearest")[0, 0]
    else:
        m = F.interpolate(m, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    return (m - m.min()) / (m.max() - m.min() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    _, _, xte, yte = load_cifar(DATA_ROOT)
    MEAN, STD = channel_stats(get_splits(DATA_ROOT, SPLIT_PATH)[0]["train"][0])           # the statistics the models were trained with
    idx = [int(np.flatnonzero(yte == c)[0]) for c in (0, 3, 7, 8)][:a.n]          # airplane, cat, horse, ship
    x = torch.from_numpy(xte[idx]).permute(0, 3, 1, 2).float() / 255
    xn = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    fig, axs = plt.subplots(len(TOK) * 2 + 1, len(idx), figsize=(1.5 * len(idx), 1.5 * (len(TOK) * 2 + 1)))
    for j, i in enumerate(idx):
        axs[0, j].imshow(xte[i])
        axs[0, j].set_title(CLASSES[yte[i]], fontsize=7)
    axs[0, 0].set_ylabel("ảnh", fontsize=7)
    for r, t in enumerate(TOK):
        cfg = Cfg(tokenizer=t, attn="scratch", seed=a.seed)
        p = E2_DIR / "artifacts" / cfg.run_id / "model_final.pt"
        if not p.exists():
            continue
        m = TinyTransformer(t, "scratch", cfg.dim, cfg.depth, cfg.heads, cfg.mlp_ratio, 0.0, True, seed=a.seed)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        with torch.no_grad():
            _, attns = m(xn, return_attn=True)
        roll = rollout(attns)
        last = attns[-1][:, :, 0, 1:].mean(1)              # CLS->tokens, mean over heads of the last block
        for j in range(len(idx)):
            for k, (v, lab) in enumerate(((roll[j], "rollout"), (last[j], "block cuối"))):
                ax = axs[1 + 2 * r + k, j]
                ax.imshow(xte[idx[j]], extent=(0, 32, 32, 0))
                ax.imshow(to_img(v, m.tok.grid).numpy(), cmap="magma", alpha=0.6, extent=(0, 32, 32, 0))
                if j == 0:
                    ax.set_ylabel(f"{NAMES[t]}\n{lab}", fontsize=6.5)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    out = E2_DIR / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "attention_maps.pdf")
    fig.savefig(out / "attention_maps.png", dpi=150)
    print("saved", out / "attention_maps.png")


if __name__ == "__main__":
    main()