"""E1 figures from saved JSON histories and test predictions; no retraining or inference."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import torch

from .summary import NAMES

SEEDS = (0, 1, 2)

E1_DIR = Path(__file__).resolve().parents[1]
DISPLAY = {"softmax": "SoftmaxClassifier", "mlp": "MLP", "minivgg": "MiniVGG"}
_SAVE = {"path": None}
COLORS = {"softmax": "#2a78d6", "mlp": "#1baf7a", "minivgg": "#eb6834"}
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"



def _show():
    if _SAVE["path"]:
        path = Path(_SAVE["path"])
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        _SAVE["path"] = None
    else:
        plt.show()


# ---------------------------------------------------------------- notebook code (moved) ----------------------------------------

def plot_class_distribution(dataset, title="Training Set Class Distribution"):

    # Count the number of samples in each class
    class_counts = torch.bincount(dataset.targets)

    # Print class frequencies
    for class_idx, count in enumerate(class_counts):
        print(
            f"{dataset.classes[class_idx]:15s}: "
            f"{count.item()} samples"
        )

    # Plot class distribution
    plt.figure(figsize=(10, 5))

    # Generate colors using the 'rocket' colormap
    colors = sns.color_palette("rocket", n_colors=len(dataset.classes))

    plt.bar(
        dataset.classes,
        class_counts.numpy(),
        color=colors
    )

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")

    plt.xticks(rotation=45)

    plt.tight_layout()
    _show()


def plot_samples_per_class(dataset, samples_per_class=5):

    num_classes = len(dataset.classes)

    fig, axes = plt.subplots(
        num_classes,
        samples_per_class,
        figsize=(2 * samples_per_class, 2.8 * num_classes)
    )

    for class_idx in range(num_classes):

        # Find all samples belonging to the current class
        class_indices = torch.where(
            dataset.targets == class_idx
        )[0]

        # Randomly select samples from the current class
        selected_indices = class_indices[
            torch.randperm(len(class_indices))[:samples_per_class]
        ]

        for j, idx in enumerate(selected_indices):
            image = dataset.data[idx]

            axes[class_idx, j].imshow(
                image,
                cmap="gray"
            )

            axes[class_idx, j].axis("off")

            # Add class name below each image
            axes[class_idx, j].set_title(
                dataset.classes[class_idx],
                fontsize=12,
                y=-0.35
            )

            if j == 0:
                axes[class_idx, j].set_ylabel(
                    dataset.classes[class_idx],
                    rotation=0,
                    labelpad=50,
                    va="center"
                )

    plt.tight_layout()
    _show()


def plot_loss_accuracy_curves(train_losses, train_accuracies,
                              val_losses, val_accuracies,
                              title="Loss and Accuracy Curves",
                              axes=None,
):
    """
    Plot the training and validation loss and accuracy curves.
    """
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("rocket", n_colors=4)

    standalone = axes is None
    if standalone:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss Curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Training Loss", color=colors[1], linewidth=2)
    axes[0].plot(epochs, val_losses, label="Validation Loss", color=colors[3], linewidth=2, linestyle="--")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title} Loss".strip())
    axes[0].legend()

    # Plot Accuracy Curves
    axes[1].plot(epochs, train_accuracies, label="Training Accuracy", color=colors[1], linewidth=2)
    axes[1].plot(epochs, val_accuracies, label="Validation Accuracy", color=colors[3], linewidth=2, linestyle="--")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{title} Accuracy".strip())
    axes[1].legend()

    if standalone:
        plt.tight_layout()
        _show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,
                          title="Confusion matrix", cmap="rocket", ax=None):
  cm = metrics.confusion_matrix(y_true, y_pred,
                                labels=np.arange(len(classes)),
                                normalize="true" if normalize else None)
  standalone = ax is None
  if standalone:
    fig, ax = plt.subplots(figsize=(10, 10))
  sns.heatmap(cm,
              annot=True,
              fmt=".2f" if normalize else "d",
              cmap=cmap,
              cbar=False,
              xticklabels=classes,
              yticklabels=classes,
              ax=ax)
  ax.set_title(title)
  ax.set_xlabel("Predicted label")
  ax.set_ylabel("True label")
  ax.tick_params(axis="x", rotation=45)
  if standalone:
    plt.tight_layout()
    _show()

  return cm


def show_errors(dataset, y_true, y_pred, y_prob, classes,
                n=12, num_cols=4, title="", mean=None, std=None):
  wrong_idx = np.where(y_pred != y_true)[0]
  if len(wrong_idx) == 0:
    print("No wrong predictions")
    return

  # Get top n wrong predictions with highest confidence
  conf = y_prob[wrong_idx, y_pred[wrong_idx]]
  display_n = min(n, len(wrong_idx))
  desending_conf_idx = wrong_idx[np.argsort(-conf)][:display_n]

  num_rows = int(np.ceil(display_n / num_cols))
  fig, axes = plt.subplots(num_rows, num_cols,
                           figsize=(num_cols * 3, num_rows * 3))
  axes = np.array(axes).reshape(-1)

  for i, (ax, idx) in enumerate(zip(axes, desending_conf_idx)):
    img, label = dataset[idx]

    if mean is not None and std is not None:
      img = img.clone()
      for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    img = img.squeeze().cpu().numpy() \
     if img.shape[0] == 1 \
     else img.permute(1, 2, 0).cpu().numpy()

    # Clip pixel values between 0 and 1 for safe rendering
    img = np.clip(img, 0, 1)

    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(f"True: {classes[y_true[idx]]}\n"
                 f"Pred: {classes[y_pred[idx]]} ({y_prob[idx, y_pred[idx]]:.2f})",
                 fontsize=9, color="crimson")
    ax.axis("off")

  for ax in axes[display_n:]:
    ax.axis("off")

  fig.suptitle(f"{title} - {len(wrong_idx)} errors / {len(y_true)}", fontsize=12)
  plt.tight_layout()
  _show()


def plot_accuracy(results):
    model_names = list(results.keys())
    accuracies = [(results[m]["y_pred"] == results[m]["y_true"]).mean() * 100 for m in model_names]

    sns.set_theme(style="white")
    plt.figure(figsize=(6, 5))
    colors = sns.color_palette("rocket", n_colors=len(model_names))

    ax = sns.barplot(x=model_names, y=accuracies, palette=colors, hue=model_names, legend=False)
    plt.title("Test Accuracy (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(accuracies) * 1.15)

    for idx, val in enumerate(accuracies):
        ax.text(idx, val + (max(accuracies) * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    _show()

def plot_training_time(results):
    model_names = list(results.keys())
    training_times = [sum(results[m]["history"]["epoch_times"]) for m in model_names]

    sns.set_theme(style="white")
    plt.figure(figsize=(6, 5))
    colors = sns.color_palette("rocket", n_colors=len(model_names))

    ax = sns.barplot(x=model_names, y=training_times, palette=colors, hue=model_names, legend=False)
    plt.title("Training Time (seconds)", fontsize=12, fontweight="bold")
    plt.ylabel("Time (s)")
    plt.ylim(0, max(training_times) * 1.15)

    for idx, val in enumerate(training_times):
        ax.text(idx, val + (max(training_times) * 0.02), f"{val:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    _show()

def plot_inference_time(results, test_dataset):
    model_names = list(results.keys())
    # Calculate inference time per sample (in milliseconds for better readability)
    inf_times_per_sample = [(results[m]["inference_time"] / len(test_dataset)) * 1000 for m in model_names]

    sns.set_theme(style="white")
    plt.figure(figsize=(6, 5))
    colors = sns.color_palette("rocket", n_colors=len(model_names))

    ax = sns.barplot(x=model_names, y=inf_times_per_sample, palette=colors, hue=model_names, legend=False)
    plt.title("Inference Time per Sample (ms)", fontsize=12, fontweight="bold")
    plt.ylabel("Time (ms)")
    plt.ylim(0, max(inf_times_per_sample) * 1.15)

    for idx, val in enumerate(inf_times_per_sample):
        ax.text(idx, val + (max(inf_times_per_sample) * 0.02), f"{val:.4f} ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    _show()


def plot_number_of_parameters(results):
    import pandas as pd
    model_names = list(results.keys())
    trainable_params = [results[m]["history"]["params_trainable"] for m in model_names]

    # Create a DataFrame for easy plotting with seaborn
    data = []
    for name, trainable in zip(model_names, trainable_params):
        data.append({"Model": name, "Parameters": trainable, "Type": "Trainable"})
    df_plot = pd.DataFrame(data)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot grouped bar chart with a 3-color palette matching each model
    colors = sns.color_palette("rocket", n_colors=len(model_names))
    ax = sns.barplot(x="Model", y="Parameters", hue="Model", data=df_plot, palette=colors, legend=False)

    plt.title("Number of Parameters per Model", fontsize=14, fontweight="bold")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # Add values on top of the bars
    for p in ax.patches:
        val = int(p.get_height())
        if val > 0:
            ax.annotate(f"{val:,}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontweight='bold', fontsize=9)

    plt.tight_layout()
    _show()


# ---------------------------------------------------------------- new: rebuild `results`, write all figures ---------------------

def load_seed_groups(results_dir, suffix=""):
    """Read the three saved histories for every model; no predictions or training needed."""
    groups = {}
    for model in NAMES:
        runs = []
        for seed in SEEDS:
            path = Path(results_dir) / f"{model}_s{seed}{suffix}.json"
            run = json.loads(path.read_text(encoding="utf8"))
            if run.get("fake") or run.get("smoke"):
                raise ValueError(f"{path}: expected a completed real-data run")
            if run["config"]["seed"] != seed or run["config"]["model"] != model:
                raise ValueError(f"{path}: model or seed does not match its filename")
            runs.append(run)
        epochs = [[row["epoch"] for row in run["history"]] for run in runs]
        if not epochs[0] or any(values != epochs[0] for values in epochs[1:]):
            raise ValueError(f"{model}: the three seeds must have the same epoch history")
        groups[(model, suffix != "_noaug")] = runs
    return groups


def curve_stats(runs, field):
    """Per-epoch mean and sample standard deviation over the three saved seeds."""
    values = np.asarray([[row[field] for row in r["history"]] for r in runs], dtype=float)
    if values.shape[0] != len(SEEDS):
        raise ValueError(f"{field}: expected {len(SEEDS)} seeds, found {values.shape[0]}")
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def save_figure(fig, fig_dir, name):
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_curves(groups, fig_dir, augment=True):
    """Three models by two metrics, with mean and sample-std ribbons from seeds 0–2."""
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.0))
    for row_idx, model in enumerate(NAMES):
        runs = groups[(model, augment)]
        epochs = [row["epoch"] for row in runs[0]["history"]]
        for col_idx, fields in enumerate((("train_loss", "val_loss"), ("train_acc", "val_acc"))):
            ax = axes[row_idx, col_idx]
            for field, label, color, style in ((fields[0], "Training", TRAIN_COLOR, "-"),
                                               (fields[1], "Validation", VAL_COLOR, "--")):
                mean, std = curve_stats(runs, field)
                scale = 100 if col_idx == 1 else 1
                ax.plot(epochs, mean * scale, label=label, color=color, ls=style, lw=1.7)
                ax.fill_between(epochs, (mean - std) * scale, (mean + std) * scale,
                                color=color, alpha=0.16, linewidth=0)
            ax.set_title(f"{NAMES[model]} — {'Loss' if col_idx == 0 else 'Accuracy'}", loc="left")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss" if col_idx == 0 else "Accuracy (%)")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Mean ± sample std (3 seeds: 0, 1, 2)", fontsize=12, y=0.995)
    note = ("Training metrics use randomly augmented images; validation metrics use unaugmented images."
            if augment else "Training and validation metrics use unaugmented images.")
    fig.text(0.5, 0.006, note,
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    save_figure(fig, fig_dir, "curves")


def plot_aggregate_bars(groups, fig_dir, augment=True):
    """Cost and final-accuracy charts use all three seeds, matching the summary table."""
    specs = (
        ("accuracy", "Final-epoch test accuracy", "Accuracy (%)", lambda r: 100 * r["final"]["test_acc"]),
        ("training_time", "Training time", "Total training seconds", lambda r: sum(h["train_s"] for h in r["history"])),
        ("inference_time", "Test prediction pipeline", "Amortized ms/image at batch 64", lambda r: r["inference_ms_per_image"]),
        ("parameters", "Trainable parameters", "Parameters", lambda r: r["n_params"]),
    )
    models = list(NAMES)
    for filename, title, ylabel, get_value in specs:
        values = [np.asarray([get_value(r) for r in groups[(model, augment)]], dtype=float) for model in models]
        means = [v.mean() for v in values]
        stds = [v.std(ddof=1) for v in values]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds if filename != "parameters" else None, capsize=4,
               color=[COLORS[m] for m in models], alpha=0.9)
        ax.set_xticks(x, [NAMES[m] for m in models])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {'3-seed mean ± sample std' if filename != 'parameters' else 'model capacity'}", loc="left")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        save_figure(fig, fig_dir, filename)


def load_results(results_dir=None, seed=0, suffix=""):
    """-> {display name: {history, y_true, y_pred, y_prob, inference_time}}, the structure `run_experiment` returned in the notebook."""
    results_dir = Path(results_dir or E1_DIR / "results")
    out = {}
    for key, name in DISPLAY.items():
        jp = results_dir / f"{key}_s{seed}{suffix}.json"
        if not jp.exists():
            continue
        r = json.loads(jp.read_text(encoding="utf8"))
        p = np.load(results_dir / f"{key}_s{seed}{suffix}_preds.npz")
        h = r["history"]
        out[name] = {
            "history": {"train_losses": [x["train_loss"] for x in h], "train_accuracies": [x["train_acc"] for x in h],
                        "val_losses": [x["val_loss"] for x in h], "val_accuracies": [x["val_acc"] for x in h],
                        "epoch_times": [x["train_s"] for x in h], "best_epoch": r["best"]["epoch"],
                        "params_trainable": r["n_params"], "params_total": r["n_params"]},
            "y_true": p["y_true"], "y_pred": p["y_pred"], "y_prob": p["y_prob"], "inference_time": r["inference_s_test"]}
    return out


def make_figures(results_dir=None, fig_dir=None, seed=0, suffix=""):
    """Write three-seed comparison charts; EDA and seed-specific examples use the original pipeline."""
    from common.cifar import CLASSES, DATA_ROOT, SPLIT_PATH, channel_stats, get_splits
    from .data import CifarArrays
    results_dir = Path(results_dir or E1_DIR / "results")
    fig_dir = Path(fig_dir or E1_DIR / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    groups = load_seed_groups(results_dir, suffix)
    augment = suffix != "_noaug"
    plot_mean_curves(groups, fig_dir, augment)
    plot_aggregate_bars(groups, fig_dir, augment)

    results = load_results(results_dir, seed, suffix)
    splits, _ = get_splits(DATA_ROOT, SPLIT_PATH)
    mean, std = channel_stats(splits["train"][0])
    train_ds, test_ds = CifarArrays(*splits["train"], mean, std), CifarArrays(*splits["test"], mean, std)

    def save(name):
        _SAVE["path"] = fig_dir / f"{name}.png"

    save("eda_class_distribution")
    plot_class_distribution(train_ds, title="Training Class Distribution (45,000 images)")
    save("eda_samples")
    plot_samples_per_class(train_ds, samples_per_class=10)
    for name, res in results.items():
        save(f"confusion_{name}")
        plot_confusion_matrix(res["y_true"], res["y_pred"], CLASSES, normalize=False,
                              title=f"Confusion Matrix - {name} (seed {seed})")
        save(f"errors_{name}")
        show_errors(test_ds, res["y_true"], res["y_pred"], res["y_prob"], CLASSES, n=8, num_cols=4,
                    title=f"Top Prediction Errors - {name} (seed {seed})", mean=list(mean), std=list(std))
    print(f"figures -> {fig_dir}")

def main():
    ap = argparse.ArgumentParser(description="Write three-seed E1 comparison charts and seed-specific examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="'_noaug' for the no-augmentation runs")
    ap.add_argument("--results", help="folder with the run JSONs and *_preds.npz (default E1/results)")
    ap.add_argument("--out", help="figure folder (default E1/figures)")
    a = ap.parse_args()
    plt.switch_backend("Agg")
    make_figures(results_dir=a.results, fig_dir=a.out, seed=a.seed, suffix=a.suffix)


if __name__ == "__main__":
    main()
