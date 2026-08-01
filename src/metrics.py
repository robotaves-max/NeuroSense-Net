from typing import Any, Dict, List

import numpy as np
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

    return {
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
