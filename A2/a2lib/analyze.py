"""Tables, figures and summary.json for the A2 report from the committed results (A2/results/runs/<run_id>/result.json + confusion_*.npy).
No GPU, no torch, no data needed.

    python -m a2lib.analyze --results A2/results --out A2/results            # tables (tab_*.tex), figures/, summary.json
    python -m a2lib.analyze names --objectinfo <ADEChallengeData2016>/objectInfo150.csv --out A2/class_names.json

Statistics (protocol): one pretrained run per model per seed (0,1,2) on the test set (the official 2,000 validation images, never used for any
choice). Seeds give a mean +- sample std (ddof=1); the model comparison is a Welch t-test on the 3+3 test mIoUs (uncorrected, descriptive).
Everything marked 'seed 0 only' (scratch runs) has no variance estimate and is reported as a single number."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.repro import read_json, write_json

from a2lib import metrics as M
from a2lib.plan import LR_GRID, MODELS, SHORT

MODEL_NAME = {"segformer_b0": "SegFormer-B0", "deeplabv3p_mnv2": "DeepLabV3+ (MobileNetV2)"}
INIT_NAME = {"pretrained": "Pretrained ImageNet", "scratch": "Từ đầu (scratch)"}
COLOR = {"segformer_b0": "#0072B2", "deeplabv3p_mnv2": "#D55E00"}      # Okabe-Ito, colour-blind safe
N_CLASSES = 150


# ---------------------------------------------------------------------------------------------------------------- formatting
def num(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace(".", "{,}")


def pm(v, nd=2, scale=100.0):
    v = np.asarray(v, dtype=float) * scale
    return num(v[0], nd) if len(v) == 1 else f"{num(v.mean(), nd)} $\\pm$ {num(v.std(ddof=1), nd)}"


# ---------------------------------------------------------------------------------------------------------------- loading
def load_runs(results_dir):
    runs = {}
    for p in sorted((Path(results_dir) / "runs").glob("a2_*/result.json")):
        r = read_json(p)
        if not r or r.get("smoke") or r.get("fake"):
            continue
        d = p.parent
        r["_conf_test"] = np.load(d / "confusion_test.npy") if (d / "confusion_test.npy").exists() else None
        r["_conf_dev"] = np.load(d / "confusion_dev.npy") if (d / "confusion_dev.npy").exists() else None
        runs[r["run_id"]] = r
    return runs


def init_of(r):
    return "pretrained" if r["config"]["pretrained"] else "scratch"


def groups(runs):
    """{(model, init): [full-run results sorted by seed]} (kind main/seed only)."""
    out = {}
    for r in runs.values():
        if r["kind"] in ("main", "seed"):
            out.setdefault((r["config"]["model"], init_of(r)), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["config"]["seed"])
    return out


def class_names(path=None):
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "class_names.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf8"))
    return [f"class_{i:03d}" for i in range(N_CLASSES)]


def names_from_objectinfo(csv_path):
    """objectInfo150.csv (Idx, Ratio, Train, Val, Name) -> list of 150 first names (the part before the first comma)."""
    names = []
    with open(csv_path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            names.append(row["Name"].split(";")[0].split(",")[0].strip())
    assert len(names) == N_CLASSES, len(names)
    return names


# ---------------------------------------------------------------------------------------------------------------- tables
def tab_main(gs):
    rows = []
    for m in MODELS:
        for init in ("pretrained", "scratch"):
            rs = gs.get((m, init))
            if not rs:
                continue
            t = [r["final"]["test"] for r in rs]
            hours = np.mean([(r["timing"]["train_s_total"] + r["timing"]["val_s_total"]) / 3600 for r in rs])
            rows.append(" & ".join([MODEL_NAME[m], INIT_NAME[init], num(rs[0]["total_M"], 2), str(len(rs)),
                                    pm([x["mIoU"] for x in t]), pm([x["pixel_acc"] for x in t]), pm([x["mDice"] for x in t]),
                                    num(hours, 1)]) + " \\\\")
    return "\n".join(rows)


def tab_lr(runs, lr_choice):
    rows = []
    for m in MODELS:
        for init in ("pretrained", "scratch"):
            sw = sorted([r for r in runs.values() if r["kind"] == "sweep" and r["config"]["model"] == m and init_of(r) == init],
                        key=lambda r: r["config"]["lr_peak"])
            if not sw:
                continue
            chosen = (lr_choice.get(f"{m}|{init}") or {}).get("lr")
            cells = []
            for r in sw:
                lr, v = float(r["config"]["lr_peak"]), 100 * r["final"]["dev"]["mIoU"]
                c = f"{lr:.2g}: {num(v)}"
                cells.append(f"\\textbf{{{c}}}" if chosen is not None and abs(lr - chosen) < 1e-12 else c)
            rows.append(" & ".join([MODEL_NAME[m], INIT_NAME[init], " ; ".join(cells)]) + " \\\\")
    return "\n".join(rows)


def welch(a, b):
    from scipy import stats
    t = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1) / na, np.var(b, ddof=1) / nb
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return float(t.statistic), float(df), float(t.pvalue)


def seed_stats(gs):
    """Pretrained SegFormer vs DeepLab on the test set (needs >= 2 seeds each)."""
    a, b = gs.get(("segformer_b0", "pretrained")), gs.get(("deeplabv3p_mnv2", "pretrained"))
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    xa, xb = np.array([r["final"]["test"]["mIoU"] for r in a]) * 100, np.array([r["final"]["test"]["mIoU"] for r in b]) * 100
    t, df, p = welch(xa, xb)
    return {"segformer": xa.tolist(), "deeplab": xb.tolist(), "gap_pts": float(xa.mean() - xb.mean()), "welch_t": t, "df": df, "p": p,
            "n": [len(xa), len(xb)]}


def tab_seeds(gs):
    rows = []
    for m in MODELS:
        rs = gs.get((m, "pretrained"))
        if rs:
            rows.append(" & ".join([MODEL_NAME[m]] + [num(100 * r["final"]["test"]["mIoU"]) for r in rs]) + " \\\\")
    return "\n".join(rows)


def pretrain_gain(gs):
    out = {}
    for m in MODELS:
        p, s = gs.get((m, "pretrained")), gs.get((m, "scratch"))
        if p and s:
            out[m] = {"pretrained_seed0": 100 * p[0]["final"]["test"]["mIoU"], "scratch_seed0": 100 * s[0]["final"]["test"]["mIoU"],
                      "gain_pts": 100 * (p[0]["final"]["test"]["mIoU"] - s[0]["final"]["test"]["mIoU"])}
    return out


def per_class(gs):
    """seed-0 pretrained per-class IoU on the test set for both models + class pixel frequency (from the ground-truth row sums)."""
    a, b = gs.get(("segformer_b0", "pretrained")), gs.get(("deeplabv3p_mnv2", "pretrained"))
    if not a or not b or a[0]["_conf_test"] is None or b[0]["_conf_test"] is None:
        return None
    ca, cb = a[0]["_conf_test"], b[0]["_conf_test"]
    ia, ib = np.array(M.summarize(ca)["iou"], dtype=float), np.array(M.summarize(cb)["iou"], dtype=float)
    freq = ca.sum(1) / ca.sum()
    return {"iou_sf": ia, "iou_dl": ib, "freq": freq}


def freq_groups(pc):
    """mean IoU over the 50 most frequent / middle / 50 rarest classes (ranked by test pixel frequency)."""
    order = np.argsort(-pc["freq"])
    out = {}
    for name, idx in (("frequent", order[:50]), ("middle", order[50:100]), ("rare", order[100:])):
        out[name] = {"segformer": float(np.nanmean(pc["iou_sf"][idx]) * 100), "deeplab": float(np.nanmean(pc["iou_dl"][idx]) * 100),
                     "pixel_share": float(pc["freq"][idx].sum())}
    return out


def tab_perclass(pc, names, k=8):
    d = (pc["iou_sf"] - pc["iou_dl"]) * 100
    ok = np.where(~np.isnan(d))[0]
    top, bot = ok[np.argsort(-d[ok])][:k], ok[np.argsort(d[ok])][:k]
    def f(i):
        nm = names[i].replace("&", "\\&")
        return f"{nm} & {num(100 * pc['iou_sf'][i])} & {num(100 * pc['iou_dl'][i])} & {num(d[i], 1, True)} \\\\"
    return "\n".join(f(i) for i in top), "\n".join(f(i) for i in bot)


# ---------------------------------------------------------------------------------------------------------------- figures
def figures(runs, gs, pc, fg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    # 1) training curves (dev mIoU per epoch)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, init in zip(axs, ("pretrained", "scratch")):
        for m in MODELS:
            for r in gs.get((m, init), []):
                h = r["history"]
                ax.plot([x["epoch"] for x in h], [100 * x["dev_mIoU"] for x in h], color=COLOR[m], alpha=1.0 if r["config"]["seed"] == 0 else 0.45,
                        lw=1.8 if r["config"]["seed"] == 0 else 1.0, label=MODEL_NAME[m] if r["config"]["seed"] == 0 else None)
        ax.set_title(INIT_NAME[init].replace("Từ đầu", "From scratch").replace("Pretrained ImageNet", "ImageNet-pretrained"))
        ax.set_xlabel("epoch")
        ax.set_ylabel("dev mIoU (%)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)
    # 2) LR sweep
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, init in zip(axs, ("pretrained", "scratch")):
        for m in MODELS:
            sw = sorted([r for r in runs.values() if r["kind"] == "sweep" and r["config"]["model"] == m and init_of(r) == init],
                        key=lambda r: r["config"]["lr_peak"])
            if sw:
                ax.plot([r["config"]["lr_peak"] for r in sw], [100 * r["final"]["dev"]["mIoU"] for r in sw], "o-", color=COLOR[m], label=MODEL_NAME[m])
        ax.set_xscale("log")
        ax.set_xlabel("peak learning rate")
        ax.set_ylabel("dev mIoU after 3 epochs on 20% (%)")
        ax.set_title(init)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "lr_sweep.png", dpi=160)
    plt.close(fig)
    if not pc:
        return
    # 3) per-class IoU difference (SegFormer - DeepLab), classes sorted by pixel frequency
    order = np.argsort(-pc["freq"])
    d = (pc["iou_sf"] - pc["iou_dl"])[order] * 100
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(np.arange(N_CLASSES), np.nan_to_num(d), color=np.where(d >= 0, COLOR["segformer_b0"], COLOR["deeplabv3p_mnv2"]), width=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("class rank by pixel frequency (0 = most frequent)")
    ax.set_ylabel("IoU difference (points)")
    ax.set_title("SegFormer-B0 minus DeepLabV3+, pretrained, test set")
    fig.tight_layout()
    fig.savefig(out / "perclass_diff.png", dpi=160)
    plt.close(fig)
    # 4) frequency groups
    fig, ax = plt.subplots(figsize=(5, 3.2))
    w = 0.38
    for j, m in enumerate(("segformer", "deeplab")):
        ax.bar(np.arange(3) + (j - 0.5) * w, [fg[g][m] for g in ("frequent", "middle", "rare")], w, color=COLOR["segformer_b0" if j == 0 else "deeplabv3p_mnv2"],
               label=MODEL_NAME["segformer_b0" if j == 0 else "deeplabv3p_mnv2"])
    ax.set_xticks(range(3), ["50 most frequent", "middle 50", "50 rarest"])
    ax.set_ylabel("mean IoU (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "freq_groups.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- main
def run(results, out, names_path=None):
    results, out = Path(results), Path(out)
    runs = load_runs(results)
    gs = groups(runs)
    lr_choice = read_json(results / "lr_choice_a2.json") or {}
    names = class_names(names_path)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {"tab_main.tex": tab_main(gs), "tab_lr.tex": tab_lr(runs, lr_choice), "tab_seeds.tex": tab_seeds(gs)}
    ss, pg, pc = seed_stats(gs), pretrain_gain(gs), per_class(gs)
    summary = {"n_runs": len(runs), "n_full": sum(len(v) for v in gs.values()), "seed_stats": ss, "pretrain_gain": pg, "lr_choice": lr_choice}
    if pc:
        fg = freq_groups(pc)
        tabs["tab_classes_sf.tex"], tabs["tab_classes_dl.tex"] = tab_perclass(pc, names)
        summary["freq_groups"] = fg
        summary["n_classes_nan"] = {"segformer": int(np.isnan(pc["iou_sf"]).sum()), "deeplab": int(np.isnan(pc["iou_dl"]).sum())}
    else:
        fg = None
    for name, text in tabs.items():
        (out / name).write_text(text + "\n", encoding="utf8")
    figures(runs, gs, pc, fg, out / "figures")
    write_json(out / "summary.json", summary)
    print(f"{len(runs)} runs ({summary['n_full']} full); wrote {', '.join(tabs)} + summary.json; figures in {out / 'figures'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "names"])
    ap.add_argument("--results", default="A2/results")
    ap.add_argument("--out", default="A2/results")
    ap.add_argument("--names", default=None)
    ap.add_argument("--objectinfo", default=None)
    a = ap.parse_args()
    if a.cmd == "names":
        names = names_from_objectinfo(a.objectinfo)
        Path(a.out).write_text(json.dumps(names, indent=0, ensure_ascii=False), encoding="utf8")
        print(f"wrote {len(names)} class names to {a.out}")
    else:
        run(a.results, a.out, a.names)


if __name__ == "__main__":
    main()
