"""A1 tables, figures and post-hoc analyses from the committed results.   python -m a1lib.analyze [--results A1/results] [--out DIR]

Inputs  (all committed or downloadable; NO torch, NO GPU needed):
  A1/results/<run_id>.json                      one per finished run (written by `python -m a1lib.collect results`)
  A1/results/lr_choice.json                     LRs chosen from the sweep
  A1/artifacts/<run_id>/logits_{val,test}.npz   logits (fp16) + labels, from the GitHub Release `a1_logits.zip`  (optional: without
                                                them the tables that need predictions are skipped and the rest is still produced)
  A1/results/cost_gpu.json                      optional: GPU-only seconds/epoch per "<model-short>|<mode>" from the cost benchmark
Outputs: tab_*.tex (bodies for \\input; no trailing \\\\), summary.json, figures/*.png|pdf

Everything reported in the report is generated here from those files; nothing is typed by hand."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from a1lib import metrics as M  # noqa: E402

A1 = Path(__file__).resolve().parents[1]
N_CLASSES = 101
NAME = {"resnet50.a1_in1k": "ResNet-50", "deit_small_patch16_224.fb_in1k": "DeiT-S", "convnext_tiny.fb_in1k": "ConvNeXt-T",
        "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T"}
SHORT = {"resnet50.a1_in1k": "resnet50", "deit_small_patch16_224.fb_in1k": "deit_s", "convnext_tiny.fb_in1k": "convnext_t",
         "swin_tiny_patch4_window7_224.ms_in1k": "swin_t"}
FAMILY = {"ResNet-50": "CNN", "ConvNeXt-T": "CNN", "DeiT-S": "Transformer", "Swin-T": "Transformer"}
MODE = {"scratch": "From-scratch", "head": "Linear probing", "partial": "Freeze một phần", "full": "Full fine-tuning"}
PARTIAL = {"ResNet-50": "layer4 + đầu ra", "DeiT-S": "block 7--12 + đầu ra", "ConvNeXt-T": "stage 4 + đầu ra", "Swin-T": "stage 4 + đầu ra"}
ORDER_MODE = ["scratch", "head", "partial", "full"]
ARCH_COLOR = {"ResNet-50": "#2a78d6", "DeiT-S": "#eb6834", "ConvNeXt-T": "#1baf7a", "Swin-T": "#eda100"}
INK, GRID = "#0b0b0b", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})


# ---------------------------------------------------------------------------------------------------------------- loading
def load_runs(res):
    runs = {}
    for p in sorted(Path(res).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf8"))
        if isinstance(r, dict) and "config" in r and "final" in r and not r.get("smoke") and not r.get("fake"):
            runs[r["run_id"]] = r
    return runs


def load_logits(art, rid, split):
    p = Path(art) / rid / f"logits_{split}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["logits"].astype(np.float32), z["labels"].astype(np.int64)


def class_names(explicit=None):
    for c in ([Path(explicit)] if explicit else []) + [A1 / "split.json", A1 / "data" / "split.json", A1 / "results" / "split.json"]:
        if c.exists():
            return json.loads(c.read_text(encoding="utf8"))["classes"]
    return [f"class_{i:03d}" for i in range(N_CLASSES)]


# ---------------------------------------------------------------------------------------------------------------- formatting
def num(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace(".", "{,}")


def pm(v, nd=2, scale=100.0):
    v = np.asarray(v, dtype=float) * scale
    if len(v) == 1:
        return num(v[0], nd)
    return f"{num(v.mean(), nd)} $\\pm$ {num(v.std(ddof=1), nd)}"


def thou(n):
    return f"{int(n):,}".replace(",", "\\,")


def arch(r):
    return NAME[r["config"]["model"]]


def tex_rows(rows):
    return "\n".join(rows)


def save_fig(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- grouping
def main_groups(runs):
    """{(arch, mode): [runs over seeds, sorted by seed]} for the full-data protocol runs (kinds main and seed, plus pair 2)."""
    g = {}
    for r in runs.values():
        c = r["config"]
        std = c["kind"] in ("main", "seed") or (c["kind"] == "stretch" and c["epochs"] in (10, 40))   # pair 2 follows the same recipe
        if std and c["data_fraction"] == 1.0:
            g.setdefault((arch(r), c["mode"]), []).append(r)
    for v in g.values():
        v.sort(key=lambda r: r["config"]["seed"])
    return g


def stretch_runs(runs):
    return sorted((r for r in runs.values() if r["config"]["kind"] == "stretch"), key=lambda r: r["run_id"])


def ablation_runs(runs):
    return [r for r in runs.values() if r["config"]["kind"] == "ablation"]


def hours(r):
    t = r.get("timing", {})
    return (t.get("train_s_total", 0) + t.get("val_s_total", 0)) / 3600


def s_per_epoch(r):
    h = r["history"]
    return float(np.median([x["train_s"] + x["val_s"] for x in h])) if h else float("nan")


# ---------------------------------------------------------------------------------------------------------------- tables
def table_accuracy(g, archs=("ResNet-50", "DeiT-S")):
    rows = []
    for a in archs:
        for m in ORDER_MODE:
            rs = g.get((a, m))
            if not rs:
                continue
            tp = [r["final"]["test"] for r in rs]
            label = MODE[m] + (f" ({PARTIAL[a]})" if m == "partial" else "")
            rows.append(f"{FAMILY[a]} & {a} & {label} & {num(rs[0]['trainable_pct'], 1)}\\% & {pm([t['top1'] for t in tp])} & "
                        f"{pm([t['top5'] for t in tp])} & {pm([t['macro_f1'] for t in tp])} \\\\")
    return tex_rows(rows)


def table_seeds(g):
    rows = []
    for (a, m), rs in sorted(g.items(), key=lambda kv: (kv[0][0], ORDER_MODE.index(kv[0][1]))):
        if len(rs) < 2:
            continue
        v = [100 * r["final"]["test"]["top1"] for r in rs]
        rows.append(f"{a} & {MODE[m]} & " + " & ".join(num(x) for x in v) + f" & {num(np.mean(v))} & {num(np.std(v, ddof=1))} \\\\")
    return tex_rows(rows)


def table_cost(g, gpu):
    rows = []
    for a in ("ResNet-50", "DeiT-S"):
        for m in ORDER_MODE:
            rs = g.get((a, m))
            if not rs:
                continue
            r0 = rs[0]
            c = r0["config"]
            sg = gpu.get(f"{SHORT[c['model']]}|{m}")
            rows.append(f"{a} & {MODE[m]} & {num(r0['trainable_M'])} / {num(r0['trainable_pct'], 1)}\\% & {c['epochs']} & "
                        f"{num(sg, 1) if sg else '--'} & {num(s_per_epoch(r0) / 60, 1)} & {num(np.mean([hours(r) for r in rs]), 2)} & "
                        f"{thou(r0['peak_mem_mb_max'])} \\\\")
    return tex_rows(rows)


def table_freeze(g):
    """accuracy gained per freeze mode relative to linear probing, with cost: when does freezing pay off?"""
    rows = []
    for a in ("ResNet-50", "DeiT-S"):
        base = {m: np.mean([r["final"]["test"]["top1"] for r in g[(a, m)]]) * 100 for m in ("head", "partial", "full") if (a, m) in g}
        for m in ("head", "partial", "full"):
            if m not in base:
                continue
            r0 = g[(a, m)][0]
            rows.append(f"{a} & {MODE[m]} & {num(base[m])} & {num(base[m] - base.get('full', np.nan), 2, True)} & "
                        f"{num(r0['trainable_pct'], 1)}\\% & {num(s_per_epoch(r0) / 60, 1)} & {thou(r0['peak_mem_mb_max'])} \\\\")
    return tex_rows(rows)


def table_ablation(runs, g):
    rows, data = [], {}
    for a in ("ResNet-50", "DeiT-S"):
        for m in ("scratch", "full"):
            vals = {}
            for r in ablation_runs(runs):
                if arch(r) == a and r["config"]["mode"] == m:
                    vals[r["config"]["data_fraction"]] = r["final"]["test"]["top1"] * 100
            if (a, m) in g:
                vals[1.0] = g[(a, m)][0]["final"]["test"]["top1"] * 100       # seed 0 of the main run = the 100 % point
            data[(a, m)] = vals
            cells = " & ".join(num(vals[f]) if f in vals else "--" for f in (0.1, 0.25, 0.5, 1.0))
            gap = f"{num(vals[1.0] - vals[0.1], 1, True)}" if 1.0 in vals and 0.1 in vals else "--"
            rows.append(f"{a} & {MODE[m]} & {cells} & {gap} \\\\")
    return tex_rows(rows), data


def table_lr(runs, choice):
    rows = []
    by = {}
    for r in runs.values():
        c = r["config"]
        if c["kind"] == "sweep":
            by.setdefault((c["model"], c["mode"]), {})[float(c["lr_peak"])] = r["final"]["val"]["top1"] * 100
    group_of = {"head": "head", "full": "finetune", "scratch": "scratch"}
    for (model, mode), d in sorted(by.items(), key=lambda kv: (NAME[kv[0][0]], ORDER_MODE.index(kv[0][1]))):
        lrs = sorted(d)
        ch = (choice.get(f"{model}|{group_of[mode]}") or {}).get("lr")
        cells = " & ".join(f"{num(d[x])}" + ("$^{*}$" if ch is not None and abs(x - ch) < 1e-12 else "") for x in lrs)
        grid = ", ".join(f"{x:g}".replace("e-0", "e-") for x in lrs)
        rows.append(f"{NAME[model]} & {MODE[mode]} & {grid} & {cells} \\\\")
    return tex_rows(rows)


def table_stretch(runs):
    rows = []
    for r in stretch_runs(runs):
        c = r["config"]
        t = r["final"]["test"]
        rows.append(f"{arch(r)} & {MODE[c['mode']]} & {c['epochs']} & {num(100 * t['top1'])} & {num(100 * t['top5'])} & "
                    f"{num(100 * t['macro_f1'])} & {num(hours(r), 2)} \\\\")
    return tex_rows(rows)


# ---------------------------------------------------------------------------------------------------------------- logits based
def probs(z):
    return M.softmax(z)


def fit_temperature(z_val, y_val):
    """argmin_T NLL(logits / T) on the validation logits: coarse log grid, then three refinements."""
    lo, hi = np.log(0.05), np.log(10.0)
    for _ in range(4):
        grid = np.linspace(lo, hi, 41)
        nll = [M.nll(z_val / np.exp(t), y_val) for t in grid]
        i = int(np.argmin(nll))
        lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    return float(np.exp(grid[i]))


def calib_row(z, y):
    return {"ece": M.ece(z, y), "nll": M.nll(z, y), "brier": M.brier(z, y), "top1": float((z.argmax(1) == y).mean())}


def table_calibration(runs, art, summ):
    rows, rel = [], {}
    for rid, r in sorted(runs.items()):
        c = r["config"]
        if c["kind"] != "main" or c["data_fraction"] != 1.0:
            continue
        v, t = load_logits(art, rid, "val"), load_logits(art, rid, "test")
        if v is None or t is None:
            continue
        T = fit_temperature(*v)
        b, a = calib_row(*t), calib_row(t[0] / T, t[1])
        rel[rid] = {"T": T, "before": b, "after": a}
        rows.append(f"{arch(r)} & {MODE[c['mode']]} & {num(100 * b['ece'])} & {num(100 * a['ece'])} & {num(b['nll'], 3)} & "
                    f"{num(a['nll'], 3)} & {num(b['brier'], 3)} & {num(a['brier'], 3)} & {num(T, 2)} \\\\")
    summ["calibration"] = rel
    return tex_rows(rows)


def reliability(z, y, nb=15):
    p = probs(z)
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, nb + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()), ys.append(ok[m].mean()), ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig_reliability(runs, art, out):
    pick = [("m_resnet50_full_s0", "ResNet-50 full"), ("m_deit_s_full_s0", "DeiT-S full")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    done = False
    for ax, (rid, title) in zip(axs, pick):
        t, v = load_logits(art, rid, "test"), load_logits(art, rid, "val")
        if t is None or v is None:
            ax.set_visible(False)
            continue
        T = fit_temperature(*v)
        ax.plot([0, 1], [0, 1], color="#8a8984", lw=0.8, ls="--")
        for z, lab, col in ((t[0], "trước", "#eb6834"), (t[0] / T, f"sau (T={T:.2f})", "#2a78d6")):
            x, y, _ = reliability(z, t[1])
            ax.plot(x, y, marker="o", ms=3, lw=1.2, color=col, label=lab)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("độ tin cậy dự đoán")
        ax.legend(frameon=False, fontsize=8)
        done = True
    axs[0].set_ylabel("độ chính xác thực")
    if done:
        save_fig(fig, out, "reliability")
    else:
        plt.close(fig)
    return done


def per_class_acc(z, y):
    pred = z.argmax(1)
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(N_CLASSES)])


def top_pairs(z, y, k=10):
    pred = z.argmax(1)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    iu = np.triu_indices(N_CLASSES, 1)
    order = np.argsort(-sym[iu])[:k]
    return [(int(iu[0][i]), int(iu[1][i]), int(sym[iu][i])) for i in order]


def analysis_errors(runs, art, names, out, figdir, summ):
    """Per-class accuracy, confusion pairs, error overlap CNN vs Transformer (full fine-tuning, seed 0)."""
    a_id, b_id = "m_resnet50_full_s0", "m_deit_s_full_s0"
    A, B = load_logits(art, a_id, "test"), load_logits(art, b_id, "test")
    if A is None or B is None:
        return {}
    za, zb, y = A[0], B[0], A[1]
    pa, pb = za.argmax(1), zb.argmax(1)
    wa, wb = pa != y, pb != y
    n = len(y)
    ov = {"n": n, "both_right": int((~wa & ~wb).sum()), "only_cnn_wrong": int((wa & ~wb).sum()), "only_vit_wrong": int((~wa & wb).sum()),
          "both_wrong": int((wa & wb).sum())}
    ov["oracle_top1"] = (ov["both_right"] + ov["only_cnn_wrong"] + ov["only_vit_wrong"]) / n
    ov["jaccard_errors"] = ov["both_wrong"] / max(1, ov["both_wrong"] + ov["only_cnn_wrong"] + ov["only_vit_wrong"])
    ca, cb = per_class_acc(za, y), per_class_acc(zb, y)
    ok = ~np.isnan(ca) & ~np.isnan(cb)
    corr = float(np.corrcoef(ca[ok], cb[ok])[0, 1])
    pa_pairs, pb_pairs = top_pairs(za, y, 20), top_pairs(zb, y, 20)
    shared = [p for p in pa_pairs if (p[0], p[1]) in {(q[0], q[1]) for q in pb_pairs}]
    nm = lambda i: names[i].replace("_", " ")  # noqa: E731
    fmt = lambda p: f"{nm(p[0])} $\\leftrightarrow$ {nm(p[1])} ({p[2]})"  # noqa: E731
    rows = [f"{fmt(pa_pairs[i]) if i < 8 else ''} & {fmt(pb_pairs[i]) if i < 8 else ''} & "
            f"{fmt(shared[i]) if i < min(8, len(shared)) else ''} \\\\" for i in range(8)]
    (out / "tab_confpairs.tex").write_text(tex_rows(rows).removesuffix("\\\\").rstrip() + "\n", encoding="utf8")
    order = np.argsort(np.where(ok, ca + cb, np.inf))
    hard, easy = [int(i) for i in order[:6]], [int(i) for i in order[::-1] if ok[i]][:6]
    rows = [f"{nm(i)} & {num(100 * ca[i], 1)} & {num(100 * cb[i], 1)} \\\\" for i in hard]
    (out / "tab_hardclasses.tex").write_text(tex_rows(rows).removesuffix("\\\\").rstrip() + "\n", encoding="utf8")
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(100 * ca[ok], 100 * cb[ok], s=14, color="#2a78d6", alpha=0.8)
    ax.plot([40, 100], [40, 100], color="#8a8984", lw=0.8, ls="--")
    for i in hard[:4]:
        ax.annotate(nm(i), (100 * ca[i], 100 * cb[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("ResNet-50: độ chính xác theo lớp (%)")
    ax.set_ylabel("DeiT-S: độ chính xác theo lớp (%)")
    save_fig(fig, figdir, "perclass")
    return {"overlap": ov, "per_class_corr": corr, "hardest": [nm(i) for i in hard], "easiest": [nm(i) for i in easy],
            "pairs_cnn": [[nm(p[0]), nm(p[1]), p[2]] for p in pa_pairs[:8]], "pairs_vit": [[nm(p[0]), nm(p[1]), p[2]] for p in pb_pairs[:8]],
            "pairs_shared": [[nm(p[0]), nm(p[1]), p[2]] for p in shared[:8]]}


def ens_top1(zs, y):
    return float((np.mean([probs(z) for z in zs], axis=0).argmax(1) == y).mean())


def table_ensemble(runs, art, summ):
    """Mean-probability ensembles of the full-fine-tuned models: single, same-architecture (3 seeds), CNN+ViT (1+1 and 3+3)."""
    L = {}
    for a, s in (("resnet50", "ResNet-50"), ("deit_s", "DeiT-S")):
        for seed in (0, 1, 2):
            t = load_logits(art, f"m_{a}_full_s{seed}", "test")
            if t is not None:
                L[(s, seed)] = t
    if ("ResNet-50", 0) not in L or ("DeiT-S", 0) not in L:
        return ""
    y = L[("ResNet-50", 0)][1]
    z = lambda s, seeds: [L[(s, k)][0] for k in seeds if (s, k) in L]  # noqa: E731
    rows, res = [], {}

    def add(label, zs, n):
        if not zs:
            return
        v = 100 * ens_top1(zs, y)
        res[label] = v
        rows.append(f"{label} & {n} & {num(v)} \\\\")
    add("ResNet-50 (đơn, seed 0)", z("ResNet-50", [0]), 1)
    add("DeiT-S (đơn, seed 0)", z("DeiT-S", [0]), 1)
    add("ResNet-50 ensemble 3 seed", z("ResNet-50", [0, 1, 2]), len(z("ResNet-50", [0, 1, 2])))
    add("DeiT-S ensemble 3 seed", z("DeiT-S", [0, 1, 2]), len(z("DeiT-S", [0, 1, 2])))
    add("ResNet-50 + DeiT-S (seed 0)", z("ResNet-50", [0]) + z("DeiT-S", [0]), 2)
    add("ResNet-50 + DeiT-S (3 + 3 seed)", z("ResNet-50", [0, 1, 2]) + z("DeiT-S", [0, 1, 2]), len(z("ResNet-50", [0, 1, 2])) + len(z("DeiT-S", [0, 1, 2])))
    summ["ensemble_top1"] = res
    return tex_rows(rows)


# ---------------------------------------------------------------------------------------------------------------- figures
def fig_curves(g, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    style = {"head": ("#8a8984", "-"), "partial": ("#1baf7a", "-"), "full": ("#2a78d6", "-"), "scratch": ("#eb6834", "-")}
    for ax, a in zip(axs, ("ResNet-50", "DeiT-S")):
        for m in ORDER_MODE:
            rs = g.get((a, m))
            if not rs:
                continue
            h = rs[0]["history"]
            ax.plot([x["epoch"] for x in h], [100 * x["val_top1"] for x in h], color=style[m][0], lw=1.4, label=MODE[m])
        ax.set_title(a, fontsize=9, color=INK)
        ax.set_xlabel("epoch")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axs[0].set_ylabel("val top-1 (%)")
    save_fig(fig, out, "curves")


def fig_ablation(data, out):
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for (a, m), vals in data.items():
        xs = sorted(vals)
        ax.plot([100 * x for x in xs], [vals[x] for x in xs], marker="o", ms=3.5, lw=1.4, color=ARCH_COLOR[a], ls="-" if m == "full" else "--",
                label=f"{a} {MODE[m].lower()}")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10", "25", "50", "100"])
    ax.set_xlabel("tỷ lệ dữ liệu huấn luyện (%)")
    ax.set_ylabel("test top-1 (%)")
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, out, "ablation")


def fig_pareto(g, gpu, out):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for (a, m), rs in g.items():
        r = rs[0]
        x = np.mean([hours(r) for r in rs])
        ax.scatter(x, 100 * np.mean([r["final"]["test"]["top1"] for r in rs]), s=34, color=ARCH_COLOR[a], marker={"scratch": "s", "head": "^", "partial": "D", "full": "o"}[m])
        ax.annotate(MODE[m].split()[0], (x, 100 * np.mean([r["final"]["test"]["top1"] for r in rs])), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("thời gian huấn luyện thực (giờ, Kaggle)")
    ax.set_ylabel("test top-1 (%)")
    save_fig(fig, out, "pareto")


def fig_lr(runs, out):
    by = {}
    for r in runs.values():
        c = r["config"]
        if c["kind"] == "sweep":
            by.setdefault((arch(r), c["mode"]), {})[float(c["lr_peak"])] = 100 * r["final"]["val"]["top1"]
    if not by:
        return
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.8))
    for ax, mode in zip(axs, ("head", "full", "scratch")):
        for (a, m), d in by.items():
            if m == mode:
                xs = sorted(d)
                ax.plot(xs, [d[x] for x in xs], marker="o", ms=3.5, lw=1.4, color=ARCH_COLOR[a], label=a)
        ax.set_xscale("log")
        ax.set_title({"head": "linear probing", "full": "fine-tune", "scratch": "from-scratch"}[mode], fontsize=9, color=INK)
        ax.set_xlabel("peak LR")
    axs[0].set_ylabel("val top-1 sau 3 epoch (%)")
    axs[0].legend(frameon=False, fontsize=7)
    save_fig(fig, out, "lr_sweep")


# ---------------------------------------------------------------------------------------------------------------- main
def write_table(out, name, body):
    if body:
        (out / name).write_text(body.rstrip().removesuffix("\\\\").rstrip() + "\n", encoding="utf8")   # no trailing "\\" before \bottomrule


def main(argv=None):
    ap = argparse.ArgumentParser(description="A1 tables, figures and post-hoc analyses from results/*.json (+ logits)")
    ap.add_argument("--results", default=str(A1 / "results"))
    ap.add_argument("--artifacts", default=str(A1 / "artifacts"), help="folder with <run_id>/logits_{val,test}.npz (from the Release)")
    ap.add_argument("--out", default=None, help="write everything here instead of next to the results (replication notebook)")
    ap.add_argument("--split", default=None, help="split.json (class names)")
    a = ap.parse_args(argv)
    res, art = Path(a.results), Path(a.artifacts)
    out = Path(a.out) if a.out else res
    figdir = (out / "figures") if a.out else (A1 / "figures")    # default: where the report expects them (A1/figures), like E2
    figdir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(res)
    choice = json.loads((res / "lr_choice.json").read_text(encoding="utf8")) if (res / "lr_choice.json").exists() else {}
    gpu = json.loads((res / "cost_gpu.json").read_text(encoding="utf8")) if (res / "cost_gpu.json").exists() else {}
    names = class_names(a.split)
    g = main_groups(runs)
    print(f"{len(runs)} finished runs ({sum(len(v) for v in g.values())} in the main protocol); logits dir: {art} ({'found' if art.exists() else 'missing'})")
    summ = {"n_runs": len(runs), "kinds": {k: sum(1 for r in runs.values() if r["config"]["kind"] == k) for k in ("sweep", "main", "seed", "ablation", "stretch")}}
    summ["main"] = {f"{a_}|{m}": {"seeds": [r["config"]["seed"] for r in rs], "top1": [r["final"]["test"]["top1"] for r in rs],
                                  "top5": [r["final"]["test"]["top5"] for r in rs], "macro_f1": [r["final"]["test"]["macro_f1"] for r in rs],
                                  "val_top1": [r["final"]["val"]["top1"] for r in rs], "hours": [hours(r) for r in rs],
                                  "peak_mem_mb": rs[0]["peak_mem_mb_max"], "trainable_pct": rs[0]["trainable_pct"]} for (a_, m), rs in sorted(g.items())}
    write_table(out, "tab_acc.tex", table_accuracy(g))
    write_table(out, "tab_acc_ext.tex", table_accuracy(g, ("ConvNeXt-T", "Swin-T")))
    write_table(out, "tab_seeds.tex", table_seeds(g))
    write_table(out, "tab_traincost.tex", table_cost(g, gpu))
    write_table(out, "tab_freeze.tex", table_freeze(g))
    body, abl = table_ablation(runs, g)
    write_table(out, "tab_ablation.tex", body)
    write_table(out, "tab_lr.tex", table_lr(runs, choice))
    write_table(out, "tab_stretch.tex", table_stretch(runs))
    summ["ablation"] = {f"{a_}|{m}": {str(k): v for k, v in d.items()} for (a_, m), d in abl.items()}
    if g:
        fig_curves(g, figdir)
        fig_pareto(g, gpu, figdir)
    if abl:
        fig_ablation(abl, figdir)
    fig_lr(runs, figdir)
    if art.exists():
        write_table(out, "tab_calibration.tex", table_calibration(runs, art, summ))
        write_table(out, "tab_ensemble.tex", table_ensemble(runs, art, summ))
        fig_reliability(runs, art, figdir)
        summ["errors"] = analysis_errors(runs, art, names, out, figdir, summ)
    else:
        print("no logits found: calibration, ensemble and error analyses skipped (download a1_logits.zip from the Release)")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf8")
    print("wrote", ", ".join(sorted(p.name for p in out.glob("tab_*.tex"))), "+ summary.json; figures in", figdir)


if __name__ == "__main__":
    main()
