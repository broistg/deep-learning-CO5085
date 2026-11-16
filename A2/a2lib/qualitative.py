"""Qualitative figure for the report: image | ground truth | SegFormer-B0 | DeepLabV3+ on test images (pretrained, seed 0).

    python -m a2lib.qualitative --data-root <ADEChallengeData2016> --split A2/split.json \
        --sf release/a2_weights/a2_m_sf_pretrained_s0.pt --dl release/a2_weights/a2_m_dl_pretrained_s0.pt --out qualitative.jpg

Selection rule (fixed, not cherry-picked): test images 0, 500, 1000, 1500, plus the image where SegFormer's pixel accuracy exceeds
DeepLabV3+'s by the most and the one where DeepLabV3+ exceeds SegFormer's by the most (computed over the whole test set).
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from a2lib.config import N_CLASSES
from a2lib.data import ADEDataset, load_split
from a2lib.models import make_model

MEAN, STD = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
FIXED = (0, 500, 1000, 1500)


def palette(n=N_CLASSES, seed=0):
    pal = np.random.default_rng(seed).integers(40, 230, size=(n + 1, 3), dtype=np.uint8)
    pal[n] = 0
    return pal


def load(kind, path, dev):
    m = make_model(SimpleNamespace(model=kind, pretrained=False, num_classes=N_CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(dev).eval()


@torch.no_grad()
def predict(m, img, size, dev):
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(dev)
    x = (x - torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        out = m(x)
    return F.interpolate(out.float(), size=size, mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sf", required=True)
    ap.add_argument("--dl", required=True)
    ap.add_argument("--out", default="qualitative.jpg")
    ap.add_argument("--width", type=int, default=320)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(a.split)
    aug = {"eval_scale": [2048, 512]}
    ds = ADEDataset(a.data_root, split["test"], "validation", aug, False)
    sf, dl = load("segformer_b0", a.sf, dev), load("deeplabv3p_mnv2", a.dl, dev)
    acc, preds = [], {}
    for i in range(len(ds)):
        img, lab = ds[i]
        ps, pd_ = predict(sf, img, lab.shape, dev), predict(dl, img, lab.shape, dev)
        ok = lab != 255
        acc.append(((ps[ok] == lab[ok]).mean() - (pd_[ok] == lab[ok]).mean()) if ok.any() else 0.0)
        if i in FIXED:
            preds[i] = (ps, pd_)
    best_sf, best_dl = int(np.argmax(acc)), int(np.argmin(acc))
    rows = list(FIXED) + [best_sf, best_dl]
    tags = [f"test #{i}" for i in FIXED] + [f"SegFormer hơn nhất (+{100 * acc[best_sf]:.1f} pt acc)", f"DeepLabV3+ hơn nhất ({100 * acc[best_dl]:.1f} pt acc)"]
    pal, W = palette(), a.width
    canvas = []
    for i, tag in zip(rows, tags):
        img, lab = ds[i]
        ps, pd_ = preds.get(i) or (predict(sf, img, lab.shape, dev), predict(dl, img, lab.shape, dev))
        col = lambda m: pal[np.where(m == 255, N_CLASSES, m)]
        tiles = [ds.load(i)[0], col(lab), col(ps), col(pd_)]
        h = int(W * lab.shape[0] / lab.shape[1])
        row = Image.new("RGB", (4 * W, h + 18), "white")
        for k, t in enumerate(tiles):
            row.paste(Image.fromarray(np.ascontiguousarray(t)).resize((W, h), Image.NEAREST if k else Image.BILINEAR), (k * W, 18))
        ImageDraw.Draw(row).text((4, 2), tag, fill="black")
        canvas.append(row)
    H = sum(r.height for r in canvas)
    out = Image.new("RGB", (4 * W, H + 18), "white")
    d = ImageDraw.Draw(out)
    for k, t in enumerate(["Image", "Ground truth", "SegFormer-B0", "DeepLabV3+ (MobileNetV2)"]):
        d.text((k * W + 4, 3), t, fill="black")
    y = 18
    for r in canvas:
        out.paste(r, (0, y)); y += r.height
    out.save(a.out, quality=90)
    print("saved", a.out, "rows", rows)


if __name__ == "__main__":
    main()
