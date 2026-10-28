"""Aggregate E1/results/*.json over seeds:  python -m e1lib.summary
Writes E1/results/summary.json and E1/results/tab_e1_summary.tex (table body for the report: Vietnamese decimal comma)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

E1_DIR = Path(__file__).resolve().parents[1]
NAMES = {"softmax": "Softmax", "mlp": "MLP", "minivgg": "MiniVGG"}


def tex_num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", "{,}").replace("-", "$-$")


def thou(n):
    return f"{n:,}".replace(",", "\\,")


def mean_std(v):
    v = np.array(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pm(v, scale=1.0, nd=2):
    m, s = mean_std(np.array(v, dtype=float) * scale)
    return f"{tex_num(m, nd)} $\\pm$ {tex_num(s, nd)}"


def main(results_dir=None, out_dir=None):
    """Read the run JSONs of `results_dir`; write summary.json + tab_e1_summary.tex to `out_dir` (default: `results_dir`)."""
    results_dir = Path(results_dir or E1_DIR / "results")
    out_dir = Path(out_dir or results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for p in sorted(results_dir.glob("*_s[0-9]*.json")):
        r = json.loads(p.read_text(encoding="utf8"))
        if r.get("smoke") or r.get("fake"):
            continue
        groups[(r["config"]["model"], r["config"]["augment"])].append(r)
    if not groups:
        raise SystemExit(f"no finished runs in {results_dir}")
    summary, rows = {}, []
    for (model, aug), rs in sorted(groups.items()):
        s = {"model": model, "augment": aug, "n_seeds": len(rs), "seeds": sorted(r["config"]["seed"] for r in rs),
             "params": rs[0]["n_params"],
             "test_acc_final_mean_std": mean_std([r["final"]["test_acc"] for r in rs]),
             "test_acc_best_val_loss_mean_std": mean_std([r["best"]["test_acc"] for r in rs]),
             "train_s_per_epoch": float(np.mean([r["train_s_per_epoch_median"] for r in rs])),
             "inference_ms_per_image": float(np.mean([r["inference_ms_per_image"] for r in rs])),
             "split_hashes": sorted({r["split_sha256_16"] for r in rs})}
        summary[f"{model}{'' if aug else '_noaug'}"] = s
        print(f"{NAMES[model]:8s} aug={aug!s:5} seeds={s['n_seeds']}  final {100 * s['test_acc_final_mean_std'][0]:.2f} +- "
              f"{100 * s['test_acc_final_mean_std'][1]:.2f}  best-val-loss {100 * s['test_acc_best_val_loss_mean_std'][0]:.2f}  "
              f"params {s['params']:,}")
        rows.append(f"{NAMES[model]}{'' if aug else ' (không augment)'} & {thou(s['params'])} & "
                    f"{pm([r['final']['test_acc'] for r in rs], 100)} & {pm([r['best']['test_acc'] for r in rs], 100)} & "
                    f"{tex_num(s['train_s_per_epoch'], 1)} & {tex_num(s['inference_ms_per_image'], 3)} \\\\")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    (out_dir / "tab_e1_summary.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
    if len({h for s in summary.values() for h in s["split_hashes"]}) > 1:
        print("WARNING: runs were made with different splits")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E1 summary over seeds")
    ap.add_argument("--results", help="folder with the run JSONs (default E1/results)")
    ap.add_argument("--out", help="write summary.json and the table here instead of next to the JSONs")
    a = ap.parse_args()
    main(a.results, a.out)
