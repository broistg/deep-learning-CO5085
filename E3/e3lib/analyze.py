"""Summarize E3 LSTM and GRU runs.

    PYTHONPATH=.:E3 python -m e3lib.analyze
    PYTHONPATH=.:E3 python -m e3lib.analyze --out E3/regenerated
"""

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.repro import config_hash, write_json

from .plan import all_cfgs
from .train import E3_DIR, RESULTS_DIR

NAMES = {
    "lstm_row": "LSTM / row",
    "lstm_patch4": "LSTM / patch 4×4",
    "gru_row": "GRU / row",
    "gru_patch4": "GRU / patch 4×4",
}
COLORS = {
    "lstm_row": "#2a78d6", "lstm_patch4": "#6b55ba",
    "gru_row": "#1baf7a", "gru_patch4": "#eb6834",
}


def group_key(run):
    cfg = run["config"]
    return f"{cfg['model']}_{cfg['tokenizer']}"


def load_results(results_dir, allow_empty=False, hidden_size=128):
    """Load only the 12 planned recurrent run IDs and reject incompatible results."""
    results_dir = Path(results_dir)
    runs = []
    for cfg in all_cfgs(hidden_size=hidden_size):
        path = results_dir / f"{cfg.run_id}.json"
        if not path.exists():
            continue
        try:
            run = json.loads(path.read_text(encoding="utf8"))
            expected = dataclasses.asdict(cfg)
            if (run["run_id"] != cfg.run_id or run["config"] != expected
                    or run["config_hash"] != config_hash(expected)
                    or run.get("smoke") or run.get("fake")
                    or len(run["history"]) != cfg.epochs
                    or [row["epoch"] for row in run["history"]]
                       != list(range(1, cfg.epochs + 1))
                    or run["final"]["val_acc"] != run["history"][-1]["val_acc"]
                    or run["final"]["val_loss"] != run["history"][-1]["val_loss"]
                    or any(not math.isfinite(run["final"][key])
                           for key in ("val_acc", "val_loss", "test_acc", "test_loss"))
                    or run["n_params"] <= 0
                    or len(run["channel_mean"]) != 3
                    or len(run["channel_std"]) != 3):
                raise ValueError("result is incomplete or has a different config")
            runs.append(run)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Invalid E3 result {path}: {exc}") from exc
    if not runs and not allow_empty:
        raise ValueError(f"No completed E3 runs in {results_dir}")
    variants = {(r["split_sha256_16"], tuple(r["channel_mean"]),
                 tuple(r["channel_std"])) for r in runs}
    if len(variants) > 1:
        raise ValueError("E3 results use different CIFAR splits or normalization")
    return runs


def mean_std(values):
    x = np.asarray(values, dtype=float)
    return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0]


def summarize(runs):
    if not runs:
        raise ValueError("No E3 runs to summarize")
    variants = {(r["split_sha256_16"], tuple(r["channel_mean"]),
                 tuple(r["channel_std"])) for r in runs}
    if len(variants) != 1:
        raise ValueError("E3 runs use different CIFAR splits or normalization")
    grouped = defaultdict(list)
    for run in runs:
        grouped[group_key(run)].append(run)
    groups = {}
    for key, items in grouped.items():
        items.sort(key=lambda r: r["config"]["seed"])
        peaks = [r.get("peak_mem_mb") for r in items]
        peak_mean = (float(np.mean(peaks))
                     if [r["config"]["seed"] for r in items] == [0, 1, 2]
                     and all(isinstance(value, (int, float))
                             and math.isfinite(value) and value > 0
                             for value in peaks) else None)
        groups[key] = {
            "source": "E3",
            "seeds": [r["config"]["seed"] for r in items],
            "n_params": items[0]["n_params"],
            "n_tokens": items[0]["n_tokens"],
            "val_acc_mean_std": mean_std([r["final"]["val_acc"] for r in items]),
            "test_acc_mean_std": mean_std([r["final"]["test_acc"] for r in items]),
            "train_s_per_epoch_median_mean_std": mean_std(
                [r["train_s_per_epoch_median"] for r in items]),
            "peak_mem_mb_mean": peak_mean,
        }
    return {
        "n_runs": len(runs),
        "missing_run_ids": sorted({c.run_id for c in all_cfgs()}
                                  - {r["run_id"] for r in runs}),
        "split_sha256_16": runs[0]["split_sha256_16"],
        "channel_mean": runs[0]["channel_mean"],
        "channel_std": runs[0]["channel_std"],
        "groups": groups,
    }


def _tex_num(value, decimals=2):
    return f"{value:.{decimals}f}".replace(".", "{,}").replace("-", "$-$")


def _tex_pm(pair, scale=1, decimals=2):
    return (f"{_tex_num(pair[0] * scale, decimals)} $\\pm$ "
            f"{_tex_num(pair[1] * scale, decimals)}")


def tables(summary):
    rows = []
    for key, name in NAMES.items():
        if key not in summary["groups"]:
            continue
        group = summary["groups"][key]
        tokens = group["n_tokens"] if group["n_tokens"] is not None else "2D"
        params = f"{group['n_params']:,}".replace(",", "\\,")
        rows.append(
            f"{name} & {tokens} & {params} & "
            f"{_tex_pm(group['val_acc_mean_std'], 100)} & "
            f"{_tex_pm(group['test_acc_mean_std'], 100)} & "
            f"{_tex_num(group['train_s_per_epoch_median_mean_std'][0], 1)} & "
            f"{_tex_num(group['peak_mem_mb_mean'], 1) if group['peak_mem_mb_mean'] is not None else '--'} & "
            f"{len(group['seeds'])} " + r"\\")
    return {"tab_e3_summary.tex": "\n".join(rows) + "\n"}


def plot_curves(runs, figure_dir):
    """Mean training loss and final-epoch validation curves across available seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for run in runs:
        grouped[group_key(run)].append(run)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for key, name in NAMES.items():
        items = grouped.get(key)
        if not items:
            continue
        for ax, field, scale in ((axes[0], "train_loss", 1),
                                 (axes[1], "val_acc", 100)):
            values = np.asarray([[row[field] for row in r["history"]]
                                 for r in items], dtype=float) * scale
            epochs = np.arange(1, values.shape[1] + 1)
            mean = values.mean(0)
            std = values.std(0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
            ax.plot(epochs, mean, label=name, color=COLORS[key])
            ax.fill_between(epochs, mean - std, mean + std,
                            color=COLORS[key], alpha=0.12)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"curves_e3.{suffix}", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize E3 result JSON files")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="expected E3 hidden size; use 64 for the original E3 results")
    parser.add_argument("--out", type=Path,
                        help="write regenerated outputs here instead of beside results")
    parser.add_argument("--no-plots", action="store_true",
                        help="write JSON and LaTeX only")
    args = parser.parse_args(argv)
    runs = load_results(args.results, hidden_size=args.hidden_size)
    summary = summarize(runs)
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    for name, body in tables(summary).items():
        (out / name).write_text(body, encoding="utf8")
    (out / "tab_e3_pairs.tex").unlink(missing_ok=True)
    if not args.no_plots:
        plot_curves(runs, (out / "figures") if args.out else E3_DIR / "figures")
    print(f"E3: {len(runs)}/12 LSTM and GRU runs; wrote summary to {out}")


if __name__ == "__main__":
    main()
