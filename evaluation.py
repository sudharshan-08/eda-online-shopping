"""
Model Evaluation Module
Computes performance metrics (Accuracy, Precision, Recall, F1, Balanced Acc, ROC-AUC, PR-AUC),
and generates publication-quality evaluation visualizations:
1. Training and Validation Loss Curves (over epochs)
2. Training and Validation Accuracy Curves (over epochs)
3. Confusion Matrix Heatmap
4. Receiver Operating Characteristic (ROC) Curve with AUC
5. Precision-Recall Curve
6. Baseline vs Deep Learning MLP Comparative Benchmark
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)
from typing import Dict, Any, List, Optional


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes all standard classification metrics for given ground truth and prediction probabilities.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Specificity = TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "specificity": round(float(spec), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "matrix": cm.tolist()
        }
    }


def plot_loss_curves(history: Dict[str, List[float]], save_path: str = "outputs/figures/model_1_loss_curves.png") -> None:
    """
    Evaluation Visualization 1: Training Loss vs Validation Loss over Epochs
    Examines convergence, learning stability, and overfitting gap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_loss, "b-o", linewidth=2, label="Training Loss (Weighted BCE)", markersize=4)
    plt.plot(epochs, val_loss, "r--s", linewidth=2, label="Validation Loss", markersize=4)

    min_val_idx = np.argmin(val_loss)
    best_epoch = epochs[min_val_idx]
    best_loss = val_loss[min_val_idx]
    plt.axvline(best_epoch, color="green", linestyle=":", label=f"Best Checkpoint (Epoch {best_epoch}, Loss {best_loss:.3f})")

    plt.title("PyTorch Deep Learning MLP: Training Loss vs Validation Loss Across Epochs", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Epoch", fontsize=11)
    plt.ylabel("Loss (Binary Cross-Entropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curves(history: Dict[str, List[float]], save_path: str = "outputs/figures/model_2_accuracy_curves.png") -> None:
    """
    Evaluation Visualization 2: Training Accuracy vs Validation Accuracy over Epochs
    Demonstrates model generalization capacity across iterations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = history["epoch"]
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc = [a * 100 for a in history["val_acc"]]

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_acc, "g-o", linewidth=2, label="Training Accuracy (%)", markersize=4)
    plt.plot(epochs, val_acc, "m--^", linewidth=2, label="Validation Accuracy (%)", markersize=4)

    plt.title("PyTorch Deep Learning MLP: Training vs Validation Accuracy Across Epochs", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Epoch", fontsize=11)
    plt.ylabel("Accuracy Percentage (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    save_path: str = "outputs/figures/model_3_confusion_matrix.png"
) -> None:
    """
    Evaluation Visualization 3: Confusion Matrix Heatmap
    Shows True Negatives, False Positives, False Negatives, and True Positives with counts and percentages.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = np.array([
        [f"TN: {cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"FP: {cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
        [f"FN: {cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"TP: {cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=True,
        xticklabels=["Predicted: No Purchase", "Predicted: Purchase"],
        yticklabels=["Actual: No Purchase", "Actual: Purchase"],
        annot_kws={"fontsize": 12, "fontweight": "bold"}
    )
    plt.title(f"PyTorch ShoppersMLP Test Confusion Matrix (Threshold = {threshold})", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Ground Truth (Revenue)", fontsize=11)
    plt.xlabel("Model Predicted Outcome", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve_viz(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = "outputs/figures/model_4_roc_auc_curve.png") -> None:
    """
    Evaluation Visualization 4: Receiver Operating Characteristic (ROC) Curve with AUC
    Evaluates True Positive Rate vs False Positive Rate across all discrimination thresholds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"PyTorch ShoppersMLP (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Receiver Operating Characteristic (ROC) Curve on Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_viz(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = "outputs/figures/model_5_precision_recall_curve.png") -> None:
    """
    Evaluation Visualization 5: Precision-Recall Curve with Average Precision (PR-AUC)
    Crucial evaluation for imbalanced classification tasks.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline_pr = (y_true == 1).mean()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#27ae60", lw=2.5, label=f"PyTorch ShoppersMLP (PR-AUC = {pr_auc:.4f})")
    plt.axhline(baseline_pr, color="red", linestyle="--", lw=1.5, label=f"Random Prevalence Baseline ({baseline_pr:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Precision-Recall Curve on Test Set (Imbalanced Class Evaluation)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison_benchmark(
    benchmark_dict: Dict[str, Dict[str, float]],
    save_path: str = "outputs/figures/model_6_baseline_vs_mlp_comparison.png"
) -> None:
    """
    Evaluation Visualization 6: Comparative Benchmark Bar Chart
    Directly compares Deep Learning BankMLP vs Logistic Regression and Random Forest Baselines
    across Accuracy, Recall, F1-Score, and ROC-AUC.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_to_plot = ["accuracy", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"]

    models = list(benchmark_dict.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#95a5a6", "#3498db", "#2ecc71"]

    for i, (model_name, c) in enumerate(zip(models, colors)):
        values = [benchmark_dict[model_name][m] for m in metrics_to_plot]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=c, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("Comprehensive Model Benchmark: Deep Learning MLP vs Baseline Models", fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold")
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim([0.0, 1.1])
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
