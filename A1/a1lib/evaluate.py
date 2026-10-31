"""Re-evaluate the RELEASED weights on the test split and compare with what the report says.

    python -m a1lib.evaluate --runs m_resnet50_full_s0,m_deit_s_full_s0 [--all-core] [--data-root DIR] [--out DIR]

For every run: load A1/artifacts/<run>/model_final.pt into the architecture of the run's config, predict the 25,250 test images (the same
eval transform as in training), and compare with A1/results/<run>.json (top-1/top-5/macro-F1) and with the published logits.
The data comes from the pre-resized cache (`python -m a1lib.data build-cache`, see A1/README.md) or the Kaggle dataset.

Expected: on the GPU type of the original runs (Tesla T4) the top-1 is identical; on other GPUs (fp16 kernels differ) it agrees to a few
test images. The check passes when |delta top-1| <= --tol-points (default 0.1 pt)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from a1lib import metrics as M
from a1lib.analyze import load_logits
from a1lib.augment import eval_transform
from a1lib.config import from_dict
from a1lib.data import FoodDataset, N_CLASSES, ensure_data, load_split
from a1lib.models import FAMILY, make_model
from a1lib.paths import default_local
from a1lib.train import make_loader, predict

A1 = Path(__file__).resolve().parents[1]
CORE = [f"m_{a}_{m}_s0" for a in ("resnet50", "deit_s") for m in ("head", "partial", "full", "scratch")]


def evaluate_run(rid, results, artifacts, root, split, device):
    pub = json.loads((Path(results) / f"{rid}.json").read_text(encoding="utf8"))
    cfg = from_dict(pub["config"])
    wp = Path(artifacts) / rid / "model_final.pt"
    if not wp.exists():
        raise FileNotFoundError(f"{wp} not found: download the weights (see the notebook / A1/README.md)")
    cidx = {c: i for i, c in enumerate(split["classes"])}
    ds = FoodDataset(Path(root) / "images", split["test"], cidx, eval_transform(cfg.augmentation))
    model = make_model(cfg.model, False, cfg.drop_rate, cfg.drop_path_rate)
    model.load_state_dict(torch.load(wp, map_location="cpu"))
    channels_last = device.type == "cuda" and FAMILY.get(cfg.model) == "cnn"
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    t0 = time.time()
    z, y = predict(model, make_loader(ds, cfg, shuffle=False, batch_size=128), device, channels_last)
    z, y = z.numpy(), y.numpy()
    got = M.summarize(z, y, N_CLASSES)
    ref = pub["final"]["test"]
    row = {"run_id": rid, "published": {k: ref[k] for k in ("top1", "top5", "macro_f1")}, "recomputed": {k: got[k] for k in ("top1", "top5", "macro_f1")},
           "delta_top1_pts": 100 * (got["top1"] - ref["top1"]), "seconds": round(time.time() - t0, 1)}
    pl = load_logits(artifacts, rid, "test")
    if pl is not None:
        row["pred_agreement"] = float((pl[0].argmax(1) == z.argmax(1)).mean())
        row["max_abs_logit_diff"] = float(np.abs(pl[0] - z).max())
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="")
    ap.add_argument("--all-core", action="store_true", help="the 8 core runs (seed 0)")
    ap.add_argument("--results", default=str(A1 / "results"))
    ap.add_argument("--artifacts", default=str(A1 / "artifacts"))
    ap.add_argument("--data-root", default=None, help="folder containing images/ and split.json (default: ensure_data under A1_LOCAL)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol-points", type=float, default=0.1)
    a = ap.parse_args(argv)
    runs = CORE if a.all_core else [r for r in a.runs.split(",") if r]
    if not runs:
        ap.error("give --runs or --all-core")
    root = Path(a.data_root) if a.data_root else ensure_data(default_local() / "data")
    split = load_split(root / "split.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}; {len(runs)} run(s)")
    rows, bad = [], []
    for rid in runs:
        r = evaluate_run(rid, a.results, a.artifacts, root, split, device)
        ok = abs(r["delta_top1_pts"]) <= a.tol_points
        r["ok"] = ok
        rows.append(r)
        if not ok:
            bad.append(rid)
        print(f"{'OK  ' if ok else 'DIFF'} {rid:26s} published {100 * r['published']['top1']:.3f}  recomputed {100 * r['recomputed']['top1']:.3f}  "
              f"delta {r['delta_top1_pts']:+.3f} pt" + (f"  agreement {100 * r['pred_agreement']:.2f}%" if "pred_agreement" in r else ""))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "evaluate.json").write_text(json.dumps(rows, indent=1), encoding="utf8")
    if bad:
        raise SystemExit(f"re-evaluation differs from the published results for: {bad}")
    print("All re-evaluated runs reproduce the published test numbers.")


if __name__ == "__main__":
    main()
