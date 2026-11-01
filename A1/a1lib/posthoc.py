"""Post-hoc analyses that need the trained weights and a GPU (Kaggle/Colab, about 1 h in total). No training.

    python -m a1lib.posthoc robustness   # 8 core models x 6 corruptions x 3 severities on a fixed 50-images-per-class test subset
    python -m a1lib.posthoc gradcam      # Grad-CAM of ResNet-50 and DeiT-S (full fine-tuning) on 12 images chosen by a fixed rule
    python -m a1lib.posthoc toploss      # training images with the highest loss under the fine-tuned models (label-noise check)
    python -m a1lib.posthoc all

Inputs : A1/artifacts/<run>/model_final.pt (Release a1_weights_core.zip), A1/results/<run>.json (configs), the Food-101 cache.
Outputs: A1/results/posthoc_{robustness,gradcam,toploss}.json (small, committed) and A1/figures/{gradcam,toploss}.png|pdf.
The 12 Grad-CAM images and the 50-per-class subset are fixed by rules written below, BEFORE looking at any result."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from a1lib.augment import eval_transform, gpu_normalize
from a1lib.config import from_dict
from a1lib.corruptions import LEVELS, TYPES, corrupt
from a1lib.data import FoodDataset, N_CLASSES, ensure_data, load_split, subset_per_class
from a1lib.models import FAMILY, make_model
from a1lib.paths import default_local
from a1lib.train import make_loader, predict

A1 = Path(__file__).resolve().parents[1]
CORE = [f"m_{a}_{m}_s0" for a in ("resnet50", "deit_s") for m in ("head", "partial", "full", "scratch")]
CAM_RUNS = {"ResNet-50": "m_resnet50_full_s0", "DeiT-S": "m_deit_s_full_s0"}
CAM_CLASSES = list(range(0, N_CLASSES, 9))[:12]          # rule: every 9th class (0, 9, ..., 99), the first test image of each
TOPLOSS_RUNS = CAM_RUNS
SUBSET_FRACTION = 0.2                                     # 50 of the 250 test images per class (nested, seed 42: the same as subset_per_class)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_model(rid, results, artifacts, device):
    pub = json.loads((Path(results) / f"{rid}.json").read_text(encoding="utf8"))
    cfg = from_dict(pub["config"])
    model = make_model(cfg.model, False, cfg.drop_rate, cfg.drop_path_rate)
    model.load_state_dict(torch.load(Path(artifacts) / rid / "model_final.pt", map_location="cpu"))
    channels_last = device.type == "cuda" and FAMILY.get(cfg.model) == "cnn"
    model.to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return cfg, model, channels_last


def setup(a):
    root = Path(a.data_root) if a.data_root else ensure_data(default_local() / "data")
    split = load_split(root / "split.json")
    return root, split, torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------------------ robustness
class CorruptedSet(torch.utils.data.Dataset):
    """Resize 256 -> centre crop 224 (as the eval transform), then one corruption (or none), as uint8 CHW."""

    def __init__(self, root, entries, labels, ctype, severity):
        self.root, self.entries, self.labels, self.ctype, self.sev = Path(root), entries, labels, ctype, severity

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        from PIL import Image
        from torchvision import transforms as T
        with Image.open(self.root / f"{self.entries[i]}.jpg") as im:
            x = T.CenterCrop(224)(T.Resize(256, interpolation=T.InterpolationMode.BILINEAR)(im.convert("RGB")))
        arr = np.asarray(x).copy()
        if self.ctype:
            arr = corrupt(arr, self.ctype, self.sev, i)
        return torch.from_numpy(arr).permute(2, 0, 1), self.labels[i]


@torch.inference_mode()
def robustness(a):
    root, split, device = setup(a)
    cidx = {c: i for i, c in enumerate(split["classes"])}
    entries = subset_per_class(split["test"], SUBSET_FRACTION)
    labels = [cidx[e.split("/")[0]] for e in entries]
    runs = a.runs.split(",") if a.runs else CORE
    models = {rid: load_model(rid, a.results, a.artifacts, device) for rid in runs}
    log(f"{len(entries)} test images ({len(entries) // N_CLASSES} per class), {len(models)} models, {len(TYPES)} corruptions x 3 severities")
    out = {"subset": {"n": len(entries), "per_class": len(entries) // N_CLASSES, "rule": "subset_per_class(test, 0.2, seed 42)"},
           "levels": {k: v for k, v in LEVELS.items()}, "models": {rid: {} for rid in runs}}
    conds = [(None, 0)] + [(t, s) for t in TYPES for s in (1, 2, 3)]
    for ctype, sev in conds:
        ds = CorruptedSet(root, entries, labels, ctype, sev)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=a.workers, pin_memory=device.type == "cuda")
        hit = {rid: 0 for rid in runs}
        for x, y in dl:
            for rid, (cfg, m, cl) in models.items():
                xb = gpu_normalize(x, device, cl)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    p = m(xb).argmax(1).cpu()
                hit[rid] += int((p == y).sum())
        for rid in runs:
            acc = hit[rid] / len(ds)
            if ctype is None:
                out["models"][rid]["clean"] = acc
            else:
                out["models"][rid].setdefault(ctype, {})[str(sev)] = acc
        log(f"{ctype or 'clean'} {sev or ''}: " + "  ".join(f"{rid.split('_s0')[0][2:]} {100 * hit[rid] / len(ds):.1f}" for rid in runs))
    (Path(a.results) / "posthoc_robustness.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    log("wrote posthoc_robustness.json")


# ------------------------------------------------------------------------------------------------------------ Grad-CAM
def vit_reshape(t, h=14, w=14):
    r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))      # drop the class token
    return r.permute(0, 3, 1, 2)


def gradcam(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from PIL import Image
    from torchvision import transforms as T
    root, split, device = setup(a)
    cidx = {c: i for i, c in enumerate(split["classes"])}
    pick = []
    for c in CAM_CLASSES:
        name = split["classes"][c]
        pick.append(next(e for e in sorted(split["test"]) if e.startswith(name + "/")))
    tfm = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224)])
    imgs = []
    for e in pick:
        with Image.open(Path(root) / "images" / f"{e}.jpg") as im:
            imgs.append(np.asarray(tfm(im.convert("RGB"))).copy())
    res, cams = {"images": pick, "classes": [split["classes"][c] for c in CAM_CLASSES], "models": {}}, {}
    for name, rid in CAM_RUNS.items():
        cfg, model, _ = load_model(rid, a.results, a.artifacts, device)
        model = model.float()                                  # Grad-CAM in fp32, no autocast
        if name == "ResNet-50":
            cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        else:
            cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=vit_reshape)
        maps, preds = [], []
        for arr in imgs:
            x = gpu_normalize(torch.from_numpy(arr).permute(2, 0, 1)[None], device)
            with torch.no_grad():
                preds.append(int(model(x).argmax(1)))
            maps.append(cam(input_tensor=x, targets=None)[0])
        cams[name] = maps
        res["models"][name] = {"run": rid, "pred": [split["classes"][p] for p in preds], "correct": [p == c for p, c in zip(preds, CAM_CLASSES)]}
        log(name, "correct:", sum(res["models"][name]["correct"]), "/", len(imgs))
    fig, axs = plt.subplots(6, 6, figsize=(9.0, 9.4))
    for k, arr in enumerate(imgs):
        r, c0 = k % 6, 3 * (k // 6)
        axs[r, c0].imshow(arr)
        axs[r, c0].set_title(res["classes"][k].replace("_", " "), fontsize=6.5)
        for j, name in enumerate(CAM_RUNS):
            axs[r, c0 + 1 + j].imshow(show_cam_on_image(arr.astype(np.float32) / 255.0, cams[name][k], use_rgb=True))
            ok = res["models"][name]["correct"][k]
            axs[r, c0 + 1 + j].set_title(f"{name}: {'đúng' if ok else 'sai: ' + res['models'][name]['pred'][k].replace('_', ' ')}", fontsize=6,
                                         color="#0b0b0b" if ok else "#c3361f")
    for ax in axs.ravel():
        ax.axis("off")
    fig.tight_layout()
    figdir = A1 / "figures" if not a.out else Path(a.out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "gradcam.png", dpi=150)
    fig.savefig(figdir / "gradcam.pdf")
    (Path(a.results) / "posthoc_gradcam.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    log("wrote posthoc_gradcam.json and figures/gradcam.*")


# ------------------------------------------------------------------------------------------------------------ top-loss training images
def toploss(a, k=200, show=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    root, split, device = setup(a)
    cidx = {c: i for i, c in enumerate(split["classes"])}
    entries = sorted(split["train"])
    labels = np.array([cidx[e.split("/")[0]] for e in entries])
    res = {"n_train": len(entries), "models": {}}
    top = {}
    for name, rid in TOPLOSS_RUNS.items():
        cfg, model, cl = load_model(rid, a.results, a.artifacts, device)
        ds = FoodDataset(Path(root) / "images", entries, cidx, eval_transform(cfg.augmentation))
        z, y = predict(model, make_loader(ds, cfg, shuffle=False, batch_size=128), device, cl)
        z = z.numpy().astype(np.float64)
        logp = z - z.max(1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
        loss = -logp[np.arange(len(labels)), labels]
        pred = z.argmax(1)
        order = np.argsort(-loss)[:k]
        top[name] = {"idx": order, "pred": pred, "loss": loss}
        res["models"][name] = {"run": rid, "train_acc": float((pred == labels).mean()), "mean_loss": float(loss.mean()),
                               "top": [{"entry": entries[i], "label": split["classes"][labels[i]], "pred": split["classes"][pred[i]],
                                        "loss": float(loss[i])} for i in order[:50]]}
        log(name, f"train accuracy {100 * (pred == labels).mean():.2f}%  mean loss {loss.mean():.3f}  top-1 loss {loss[order[0]]:.2f}")
    both = [i for i in top["ResNet-50"]["idx"] if i in set(top["DeiT-S"]["idx"].tolist())]
    both.sort(key=lambda i: -(top["ResNet-50"]["loss"][i] + top["DeiT-S"]["loss"][i]))
    agree = [i for i in both if top["ResNet-50"]["pred"][i] == top["DeiT-S"]["pred"][i] != labels[i]]
    res["both_in_top"] = len(both)
    res["both_predict_same_other_class"] = len(agree)
    res["shown"] = [{"entry": entries[i], "label": split["classes"][labels[i]], "pred_resnet": split["classes"][top["ResNet-50"]["pred"][i]],
                     "pred_deit": split["classes"][top["DeiT-S"]["pred"][i]]} for i in both[:show]]
    fig, axs = plt.subplots(3, 4, figsize=(8.0, 6.4))
    for ax, i in zip(axs.ravel(), both[:show]):
        with Image.open(Path(root) / "images" / f"{entries[i]}.jpg") as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"nhãn: {split['classes'][labels[i]].replace('_', ' ')}\ndự đoán: {split['classes'][top['ResNet-50']['pred'][i]].replace('_', ' ')} / "
                     f"{split['classes'][top['DeiT-S']['pred'][i]].replace('_', ' ')}", fontsize=6.5)
    for ax in axs.ravel():
        ax.axis("off")
    fig.tight_layout()
    figdir = A1 / "figures" if not a.out else Path(a.out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "toploss.png", dpi=150)
    fig.savefig(figdir / "toploss.pdf")
    (Path(a.results) / "posthoc_toploss.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    log(f"{len(both)} images are in the top-{k} loss of both models; {len(agree)} of them get the same wrong prediction from both")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["robustness", "gradcam", "toploss", "all"])
    ap.add_argument("--runs", default="", help="robustness: comma list (default: the 8 core runs)")
    ap.add_argument("--results", default=str(A1 / "results"))
    ap.add_argument("--artifacts", default=str(A1 / "artifacts"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None, help="figures go to OUT/figures instead of A1/figures")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    for c in (["robustness", "gradcam", "toploss"] if a.cmd == "all" else [a.cmd]):
        {"robustness": robustness, "gradcam": gradcam, "toploss": toploss}[c](a)


if __name__ == "__main__":
    main()
