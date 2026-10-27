"""Aggregate results/*.json into LaTeX table bodies and figures.   python -m e2lib.analyze
Outputs: E2/results/summary.json, E2/results/tab_*.tex (\\input into the report), E2/figures/*.pdf|png"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

E2 = Path(__file__).resolve().parents[1]
RES, FIG = E2 / "results", E2 / "figures"
COLORS = {"patch4": "#2a78d6", "patch8": "#eb6834", "row": "#1baf7a", "cnn": "#eda100"}   # categorical slots 1-4, fixed order
NAMES = {"patch4": "Patch 4×4", "patch8": "Patch 8×8", "row": "Hàng (row)", "cnn": "CNN-stem"}
INK, GRID = "#0b0b0b", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})


def load():
    runs = []
    for p in sorted(RES.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf8"))
        if "config" in r and not r.get("smoke") and not r.get("fake"):
            runs.append(r)
    check_consistent(runs)
    return runs


def check_consistent(runs):
    """All runs of one study must share the split and the normalisation. Results written before the common/cifar.py change have no
    `channel_mean`: they used hard-coded constants, so mixing them with new runs would silently compare different preprocessing."""
    old = [r["run_id"] for r in runs if "channel_mean" not in r]
    if old:
        raise SystemExit(f"{len(old)} result file(s) were made before the shared-normalisation change (e.g. {old[0]}): "
                         "delete E2/results/*.json and E2/artifacts and rerun `python -m e2lib.plan`")
    variants = {(r["split_sha256_16"], tuple(round(m, 6) for m in r["channel_mean"]), tuple(round(v, 6) for v in r["channel_std"]))
                for r in runs}
    if len(variants) > 1:
        raise SystemExit(f"results use {len(variants)} different (split, mean, std) combinations: {sorted(variants)}")


def key(r):
    c = r["config"]
    return c["tokenizer"], c["attn"], c["use_pos"]


def tex_num(x, nd=2, sign=False):
    """LaTeX number with Vietnamese decimal comma and a proper minus sign."""
    t = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return t.replace(".", "{,}").replace("-", "$-$")


def thou(n):
    return f"{n:,}".replace(",", "\\,")


def ms(v, scale=1.0, nd=2):
    v = np.array(v) * scale
    return f"{tex_num(v.mean(), nd)} $\\pm$ {tex_num(v.std(ddof=1) if len(v) > 1 else 0, nd)}"


def group(runs):
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r)
    return {k: sorted(v, key=lambda r: r["config"]["seed"]) for k, v in g.items()}


def tables(g):
    out = {}
    # (i) scratch vs torch, same tokenizer, paired by seed
    rows = []
    for t in NAMES:
        a, b = g.get((t, "scratch", True)), g.get((t, "torch", True))
        if not a or not b:
            continue
        sa = {r["config"]["seed"]: r for r in a}
        sb = {r["config"]["seed"]: r for r in b}
        common = sorted(set(sa) & set(sb))
        diff = [100 * (sa[s]["final"]["test_acc"] - sb[s]["final"]["test_acc"]) for s in common]
        rows.append(f"{NAMES[t]} & {thou(a[0]['n_params'])} & {ms([r['final']['test_acc'] for r in a], 100)} & "
                    f"{ms([r['final']['test_acc'] for r in b], 100)} & {tex_num(np.mean(diff), 2, True)} & "
                    f"{tex_num(np.median([r['train_s_per_epoch_median'] for r in a]), 1)} & {tex_num(np.median([r['train_s_per_epoch_median'] for r in b]), 1)} \\\\")
    out["tab_impl.tex"] = "\n".join(rows)
    # (ii) tokenizers (scratch MSA)
    rows = []
    for t in NAMES:
        a = g.get((t, "scratch", True))
        if not a:
            continue
        rows.append(f"{NAMES[t]} & {a[0]['n_tokens']} & {thou(a[0]['n_params'])} & {ms([r['final']['val_acc'] for r in a], 100)} & "
                    f"{ms([r['final']['test_acc'] for r in a], 100)} & {tex_num(np.median([r['train_s_per_epoch_median'] for r in a]), 1)} & "
                    f"{np.max([r['peak_mem_mb'] for r in a]):.0f} \\\\")
    out["tab_tokenizers.tex"] = "\n".join(rows)
    # positional-encoding ablation
    rows = []
    for t in ("patch4", "row"):
        a, b = g.get((t, "scratch", True)), g.get((t, "scratch", False))
        if a and b:
            rows.append(f"{NAMES[t]} & {ms([r['final']['test_acc'] for r in a], 100)} & {ms([r['final']['test_acc'] for r in b], 100)} & "
                        f"{tex_num(100 * (np.mean([r['final']['test_acc'] for r in b]) - np.mean([r['final']['test_acc'] for r in a])), 2, True)} \\\\")
    out["tab_pos.tex"] = "\n".join(rows)
    return out


def mean_curve(rs, field):
    m = np.array([[h[field] for h in r["history"]] for r in rs])
    return m.mean(0), m.std(0)


def fig_curves(g):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for t in NAMES:
        rs = g.get((t, "scratch", True))
        if not rs:
            continue
        x = np.arange(1, len(rs[0]["history"]) + 1)
        for a, f, ttl in ((ax[0], "train_loss", "Loss huấn luyện"), (ax[1], "val_acc", "Độ chính xác validation")):
            m, s = mean_curve(rs, f)
            m = m * 100 if f == "val_acc" else m
            s = s * 100 if f == "val_acc" else s
            a.plot(x, m, color=COLORS[t], lw=1.6, label=NAMES[t])
            a.fill_between(x, m - s, m + s, color=COLORS[t], alpha=0.15, lw=0)
            a.set_title(ttl, loc="left", fontsize=9, color=INK)
            a.set_xlabel("Epoch")
    ax[1].set_ylabel("%")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "curves_tokenizers.pdf")
    fig.savefig(FIG / "curves_tokenizers.png")
    plt.close(fig)


def fig_impl(g):
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    for a, t in zip(ax, NAMES):
        for attn, ls, lab in (("scratch", "-", "tự viết"), ("torch", "--", "PyTorch")):
            rs = g.get((t, attn, True))
            if rs:
                m, _ = mean_curve(rs, "val_acc")
                a.plot(np.arange(1, len(m) + 1), m * 100, color=COLORS[t], ls=ls, lw=1.5, label=lab)
        a.set_title(NAMES[t], fontsize=8.5, loc="left", color=INK)
        a.set_xlabel("Epoch")
        a.legend(frameon=False, fontsize=7)
    ax[0].set_ylabel("Val acc (%)")
    fig.tight_layout()
    fig.savefig(FIG / "curves_impl.pdf")
    fig.savefig(FIG / "curves_impl.png")
    plt.close(fig)


def fig_confusion_and_errors(g, data_root=None):
    import torch
    best = max(((t, rs) for (t, a, p), rs in g.items() if a == "scratch" and p), key=lambda kv: np.mean([r["final"]["test_acc"] for r in kv[1]]),
               default=None)
    if best is None:
        return None
    t, rs = best
    r0 = rs[0]
    z = torch.load(E2 / "artifacts" / r0["run_id"] / "logits_test.pt", map_location="cpu")
    logits, y = z["logits"].float(), z["labels"]
    pred = logits.argmax(1)
    cm = np.zeros((10, 10), int)
    for a, b in zip(y.tolist(), pred.tolist()):
        cm[a, b] += 1
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(10), CLASSES, fontsize=7)
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    fig.colorbar(im, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.pdf")
    fig.savefig(FIG / "confusion.png")
    plt.close(fig)
    per_class = np.diag(cm) / cm.sum(1)
    pairs = sorted(((cm[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j), reverse=True)[:5]
    # most confident mistakes
    from .data import DATA_ROOT, load_cifar
    _, _, xte, _ = load_cifar(data_root or DATA_ROOT)
    prob = logits.softmax(1)
    conf = prob.max(1).values
    wrong = [i for i in torch.argsort(conf, descending=True).tolist() if pred[i] != y[i]][:12]
    fig, axs = plt.subplots(2, 6, figsize=(7.2, 2.9))
    for a, i in zip(axs.ravel(), wrong):
        a.imshow(xte[i])
        a.set_title(f"thật: {CLASSES[y[i]]}\nđoán: {CLASSES[pred[i]]} ({100 * conf[i]:.0f}%)", fontsize=6.5)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "errors.pdf")
    fig.savefig(FIG / "errors.png")
    plt.close(fig)
    return {"model": r0["run_id"], "per_class_acc": dict(zip(CLASSES, per_class.round(4).tolist())),
            "top_confusions": [[int(n), a, b] for n, a, b in pairs]}


def main():
    FIG.mkdir(exist_ok=True)
    runs = load()
    g = group(runs)
    print(f"{len(runs)} finished runs in {len(g)} groups")
    for name, body in tables(g).items():
        # no trailing "\\": the report writes "\input{..}\\" (a "\\" at the very end of an \input'ed file breaks \bottomrule)
        (RES / name).write_text(body.rstrip().removesuffix("\\\\").rstrip() + "\n", encoding="utf8")
        print(f"--- {name}\n{body}")
    if runs:
        fig_curves(g)
        fig_impl(g)
    summary = {"n_runs": len(runs), "groups": {"|".join(map(str, k)): {"seeds": [r["config"]["seed"] for r in v],
               "test_acc": [r["final"]["test_acc"] for r in v], "val_acc": [r["final"]["val_acc"] for r in v]} for k, v in g.items()}}
    try:
        summary["errors"] = fig_confusion_and_errors(g)
    except Exception as e:  # noqa: BLE001  (artifacts or dataset not present on this machine)
        print("confusion/error figures skipped:", e)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
