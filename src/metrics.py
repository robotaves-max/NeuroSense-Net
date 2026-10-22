from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils import MELD_EMOTIONS


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = MELD_EMOTIONS,
    y_prob: np.ndarray = None,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for 7-class emotion recognition."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_p, per_r, per_f1, per_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    per_class_metrics = {}
    for idx, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(per_p[idx]),
            "recall": float(per_r[idx]),
            "f1": float(per_f1[idx]),
            "support": int(per_supp[idx]),
        }

    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "confusion_matrix_counts": cm_counts.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "total_samples": len(y_true),
    }
    if y_prob is not None:
        metrics["ece"] = expected_calibration_error(np.asarray(y_prob), y_true)
    return metrics


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> float:
    """Compute equal-width Expected Calibration Error from class probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probabilities must have shape [N, C] matching targets")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        in_bin = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fit one positive logit temperature by minimizing validation NLL."""
    logits = torch.as_tensor(logits, dtype=torch.float64).detach()
    targets = torch.as_tensor(targets, dtype=torch.long).detach()
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must be [N, C] and targets must be [N]")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
