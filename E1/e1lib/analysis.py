"""E1 plots. The functions below are Ly's notebook code, moved unchanged except that `plt.show()` became `_show()`:
it saves the figure when a path was set (scripts, `python -m e1lib.analysis`) and shows it otherwise (notebook).
`load_results` rebuilds the `results` dict of the notebook from E1/results/*.json and the saved predictions."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import torch

E1_DIR = Path(__file__).resolve().parents[1]
DISPLAY = {"softmax": "SoftmaxClassifier", "mlp": "MLP", "minivgg": "MiniVGG"}
_SAVE = {"path": None}


def _show():
    if _SAVE["path"]:
        plt.savefig(_SAVE["path"], dpi=150, bbox_inches="tight")
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
    """Write the notebook's charts for one seed to E1/figures/*.png (EDA charts need the data download)."""
    from common.cifar import CLASSES, DATA_ROOT, SPLIT_PATH, channel_stats, get_splits
    from .data import CifarArrays
    results_dir = Path(results_dir or E1_DIR / "results")
    fig_dir = Path(fig_dir or E1_DIR / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_dir, seed, suffix)
    if not results:
        raise SystemExit(f"no results for seed {seed} in {results_dir}: run `python -m e1lib.run` first")
    splits, _ = get_splits(DATA_ROOT, SPLIT_PATH)
    mean, std = channel_stats(splits["train"][0])
    train_ds, test_ds = CifarArrays(*splits["train"], mean, std), CifarArrays(*splits["test"], mean, std)

    def save(name):
        _SAVE["path"] = fig_dir / f"{name}.png"

    save("eda_class_distribution")
    plot_class_distribution(train_ds, title="Training Class Distribution (45,000 images)")
    save("eda_samples")
    plot_samples_per_class(train_ds, samples_per_class=10)

    fig, axes = plt.subplots(len(results), 2, figsize=(16, 5 * len(results)))
    axes = np.atleast_2d(axes)
    for idx, (name, res) in enumerate(results.items()):
        h = res["history"]
        plot_loss_accuracy_curves(h["train_losses"], h["train_accuracies"], h["val_losses"], h["val_accuracies"],
                                  title=name, axes=axes[idx])
    plt.tight_layout()
    fig.savefig(fig_dir / "curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for name, res in results.items():
        save(f"confusion_{name}")
        plot_confusion_matrix(res["y_true"], res["y_pred"], CLASSES, normalize=False, title=f"Confusion Matrix - {name}")
        save(f"errors_{name}")
        show_errors(test_ds, res["y_true"], res["y_pred"], res["y_prob"], CLASSES, n=8, num_cols=4,
                    title=f"Top Prediction Errors - {name}", mean=list(mean), std=list(std))
    for fn, name in ((plot_accuracy, "accuracy"), (plot_training_time, "training_time"), (plot_number_of_parameters, "parameters")):
        save(name)
        fn(results)
    save("inference_time")
    plot_inference_time(results, test_ds)
    print(f"figures -> {fig_dir}")


def main():
    ap = argparse.ArgumentParser(description="Write the E1 figures for one seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="'_noaug' for the no-augmentation runs")
    a = ap.parse_args()
    plt.switch_backend("Agg")
    make_figures(seed=a.seed, suffix=a.suffix)


if __name__ == "__main__":
    main()
