"""Confusion matrix, per-class metrics and error grid from saved test logits."""

import argparse
from pathlib import Path

import numpy as np
import torch

from common.cifar import CLASSES, DATA_ROOT, load_cifar
from common.ckpt import verify_ckpt
from common.repro import write_json

from .analyze import load_results
from .train import ARTIFACTS_DIR, E3_DIR, RESULTS_DIR


def choose_run(runs):
    """Select by test accuracy, breaking ties by run ID."""
    return min(runs, key=lambda r: (-r["final"]["test_acc"], r["run_id"]))


def load_predictions(path):
    path = Path(path)
    if not path.is_file() or not verify_ckpt(path):
        raise ValueError(f"Missing or corrupt test logits: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    logits, labels = saved["logits"], saved["labels"]
    if (logits.shape != (10000, 10) or labels.shape != (10000,)
            or not torch.isfinite(logits).all()
            or not torch.all((labels >= 0) & (labels < 10))):
        raise ValueError(f"Invalid test predictions in {path}")
    return logits.float(), labels.long()


def diagnose(logits, labels, run_id):
    prediction = logits.argmax(1)
    cm = torch.bincount(labels * 10 + prediction, minlength=100).reshape(10, 10)
    counts = cm.sum(1)
    per_class = (cm.diag() / counts.clamp_min(1)).tolist()
    confusions = sorted(
        ((int(cm[i, j]), CLASSES[i], CLASSES[j])
         for i in range(10) for j in range(10) if i != j),
        reverse=True,
    )[:5]
    probabilities = logits.softmax(1)
    confidence = probabilities.max(1).values
    mistakes = torch.where(prediction != labels)[0]
    mistakes = mistakes[torch.argsort(confidence[mistakes], descending=True)][:12]
    return {
        "run_id": run_id,
        "test_acc": float((prediction == labels).float().mean()),
        "class_counts": dict(zip(CLASSES, counts.tolist())),
        "per_class_acc": dict(zip(CLASSES, per_class)),
        "confusion_matrix": cm.tolist(),
        "top_confusions": [[n, actual, predicted]
                           for n, actual, predicted in confusions],
        "most_confident_error_indices": mistakes.tolist(),
    }


def plot_confusion(matrix, figure_dir, config, test_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(matrix, dtype=float)
    normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    config_name = f"{config['model'].upper()}_{config['tokenizer'].capitalize()}"
    ax.set_title(f"E3 {config_name} (seed {config['seed']}) | test {test_acc:.2%}",
                 fontsize=10, pad=12)
    ax.set_xticks(range(10), CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(10), CLASSES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(10):
        for col in range(10):
            if normalized[row, col] >= 0.05:
                ax.text(col, row, f"{normalized[row, col] * 100:.0f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if normalized[row, col] > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"confusion_e3.{suffix}", dpi=160)
    plt.close(fig)


def plot_error_grid(images, logits, labels, indices, run_id, test_acc, figure_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = logits.argmax(1)
    confidence = logits.softmax(1).max(1).values
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle(f"E3 {run_id} | test accuracy {test_acc:.2%} | most confident errors",
                 fontsize=11)
    for ax, index in zip(axes.flat, indices):
        ax.imshow(images[index])
        actual = CLASSES[labels[index].item()]
        predicted = CLASSES[predictions[index].item()]
        ax.set_title(f"True: {actual}\nPred: {predicted} ({confidence[index]:.0%})",
                     fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(indices):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"error_grid_{run_id.rsplit('_s', 1)[0]}"
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="E3 test-set diagnostics")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="expected E3 hidden size; use 64 for the original E3 results")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="CIFAR-10 data directory for the error grid")
    parser.add_argument("--out", type=Path,
                        help="write diagnostics and figures here instead of beside results")
    parser.add_argument("--run-id", help="choose a completed run (default: best test accuracy)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    runs = load_results(args.results, hidden_size=args.hidden_size)
    if args.run_id:
        matches = [r for r in runs if r["run_id"] == args.run_id]
        if not matches:
            raise ValueError(f"No completed result for {args.run_id}")
        selected = matches[0]
    else:
        selected = choose_run(runs)
    path = args.artifacts / selected["run_id"] / "logits_test.pt"
    logits, labels = load_predictions(path)
    report = diagnose(logits, labels, selected["run_id"])
    if abs(report["test_acc"] - selected["final"]["test_acc"]) > 1e-6:
        raise ValueError(f"Test accuracy does not match result JSON for {selected['run_id']}")
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "diagnostics.json", report)
    if not args.no_plots:
        figure_dir = (out / "figures") if args.out else E3_DIR / "figures"
        _, _, images, image_labels = load_cifar(args.data_root)
        if not np.array_equal(image_labels, labels.numpy()):
            raise ValueError("Saved predictions do not match CIFAR-10 test image order")
        plot_confusion(report["confusion_matrix"], figure_dir,
                       selected["config"], report["test_acc"])
        plot_error_grid(images, logits, labels,
                        report["most_confident_error_indices"],
                        selected["run_id"], report["test_acc"], figure_dir)
    print(f"E3 diagnostics for {selected['run_id']}: test accuracy {report['test_acc']:.4f}")
    return report


if __name__ == "__main__":
    main()
