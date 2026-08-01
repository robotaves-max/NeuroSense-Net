from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import MELD_EMOTIONS

# Style configuration for presentation/publication grade plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def plot_training_history(history: List[Dict[str, Any]], save_path: Path) -> None:
    """Plot train/val loss, macro-F1, accuracy, and learning rate across training epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_loss = [h.get("val_loss", 0.0) for h in history]
    val_f1 = [h.get("val_macro_f1", 0.0) for h in history]
    val_acc = [h.get("val_accuracy", 0.0) for h in history]
    lrs = [h.get("lr", 0.0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Loss Curve
    axes[0, 0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.5)
    axes[0, 0].plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.5, linestyle="--")
    axes[0, 0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Validation Macro-F1
    axes[0, 1].plot(epochs, val_f1, label="Val Macro-F1", color="#2ca02c", linewidth=2.5, marker="o")
    axes[0, 1].set_title("Validation Macro-F1 (Primary Metric)", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Macro-F1")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Validation Accuracy
    axes[1, 0].plot(epochs, val_acc, label="Val Accuracy", color="#9467bd", linewidth=2.5, marker="s")
    axes[1, 0].set_title("Validation Accuracy", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, lrs, label="Learning Rate", color="#d62728", linewidth=2)
    axes[1, 1].set_title("Learning Rate Schedule", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_confusion_matrices(
    cm_counts: np.ndarray,
    cm_norm: np.ndarray,
    class_names: List[str],
    save_counts_path: Path,
    save_norm_path: Path,
) -> None:
    """Plot raw count and row-normalized confusion matrices."""
    save_counts_path.parent.mkdir(parents=True, exist_ok=True)

    # Counts Plot
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix (Counts)", fontweight="bold", fontsize=14)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.savefig(save_counts_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Normalized Plot
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("Normalized Confusion Matrix (Recall per Class)", fontweight="bold", fontsize=14)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.savefig(save_norm_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_per_class_f1(per_class_metrics: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Plot per-class F1 score bar chart with sample support annotated."""
    classes = list(per_class_metrics.keys())
    f1_scores = [per_class_metrics[c]["f1"] for c in classes]
    supports = [per_class_metrics[c]["support"] for c in classes]

    plt.figure(figsize=(10, 6), dpi=300)
    bars = plt.bar(classes, f1_scores, color="#3498db", edgecolor="#2980b9", linewidth=1.2)
    plt.title("Per-Class F1 Score (With Sample Support)", fontweight="bold", fontsize=14)
    plt.xlabel("Emotion Class", fontweight="bold")
    plt.ylabel("F1 Score", fontweight="bold")
    plt.ylim(0, 1.05)

    for bar, score, supp in zip(bars, f1_scores, supports):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.2f}\n(n={supp})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_gate_weights_overall(gate_weights: np.ndarray, save_path: Path) -> None:
    """Plot mean adaptive gate weights with confidence intervals per modality."""
    modalities = ["Text", "Audio", "Video"]
    means = gate_weights.mean(axis=0)
    stds = gate_weights.std(axis=0)

    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2ecc71", "#e74c3c", "#9b59b6"]
    bars = plt.bar(modalities, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.85)

    plt.title("Overall Modality Gate Weight Allocation (Mean ± Std)", fontweight="bold", fontsize=14)
    plt.ylabel("Average Gate Weight", fontweight="bold")
    plt.ylim(0, 1.0)

    for bar, m in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            m / 2.0,
            f"{m:.3f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=12,
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_gate_weights_by_class(
    gate_weights: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Plot average modality gate weights grouped by true emotion class."""
    modalities = ["Text", "Audio", "Video"]
    n_classes = len(class_names)
    class_means = np.zeros((n_classes, 3))

    for c in range(n_classes):
        idx = y_true == c
        if idx.any():
            class_means[c] = gate_weights[idx].mean(axis=0)

    x = np.arange(n_classes)
    width = 0.25

    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width, class_means[:, 0], width, label="Text", color="#2ecc71")
    plt.bar(x, class_means[:, 1], width, label="Audio", color="#e74c3c")
    plt.bar(x + width, class_means[:, 2], width, label="Video", color="#9b59b6")

    plt.title("Modality Gate Weights Breakdown by True Emotion Class", fontweight="bold", fontsize=14)
    plt.xlabel("Emotion Class", fontweight="bold")
    plt.ylabel("Average Gate Weight", fontweight="bold")
    plt.xticks(x, class_names, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_class_distribution(
    train_dist: Dict[str, int],
    val_dist: Dict[str, int],
    test_dist: Dict[str, int],
    save_path: Path,
) -> None:
    """Plot sample counts across emotion classes for train, val, and test splits."""
    classes = MELD_EMOTIONS
    x = np.arange(len(classes))
    width = 0.25

    tr_counts = [train_dist.get(c, 0) for c in classes]
    va_counts = [val_dist.get(c, 0) for c in classes]
    te_counts = [test_dist.get(c, 0) for c in classes]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width, tr_counts, width, label="Train", color="#34495e")
    plt.bar(x, va_counts, width, label="Validation", color="#3498db")
    plt.bar(x + width, te_counts, width, label="Test", color="#e67e22")

    plt.title("Sample Distribution Across Splits (MELD Class Imbalance)", fontweight="bold", fontsize=14)
    plt.xlabel("Emotion Class", fontweight="bold")
    plt.ylabel("Number of Samples", fontweight="bold")
    plt.xticks(x, classes, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_modality_ablation(ablation_results: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Plot Macro-F1 performance comparison across modality ablation conditions."""
    conditions = list(ablation_results.keys())
    f1_scores = [ablation_results[c]["macro_f1"] for c in conditions]

    plt.figure(figsize=(12, 6), dpi=300)
    colors = ["#27ae60" if "All" in c else "#2980b9" for c in conditions]
    bars = plt.bar(conditions, f1_scores, color=colors, edgecolor="black", linewidth=1.1)

    plt.title("Modality Ablation Protocol (Macro-F1 Robustness Analysis)", fontweight="bold", fontsize=14)
    plt.xlabel("Evaluated Modality Condition", fontweight="bold")
    plt.ylabel("Test Macro-F1", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")

    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
