from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Numerically stable multiclass Focal Loss.

    $FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: [B, C] unnormalized logit tensor
        targets: [B] target class index tensor (long)
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits tensor [B, C], got shape {logits.shape}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(f"Expected 1D target tensor of length B, got shape {targets.shape}")

        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()

        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt

        if self.weight is not None:
            alpha_t = self.weight.to(logits.device).gather(dim=0, index=targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            if self.weight is not None:
                alpha_t = self.weight.to(logits.device).gather(dim=0, index=targets)
                return loss.sum() / alpha_t.sum().clamp(min=1e-7)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_class_weights(
    train_labels: torch.Tensor,
    num_classes: int = 7,
    strategy: str = "effective_number",
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute per-class weights from training split labels only.

    Strategies:
      - 'none': Equal weight 1.0 for all classes
      - 'inverse_frequency': N / (C * count_c)
      - 'effective_number': (1 - beta) / (1 - beta^count_c)
    """
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()

    counts = np.bincount(train_labels, minlength=num_classes)
    counts = np.maximum(counts, 1)  # avoid division by zero

    if strategy == "none" or strategy is None:
        weights = np.ones(num_classes, dtype=np.float32)

    elif strategy == "inverse_frequency":
        total_samples = len(train_labels)
        weights = total_samples / (num_classes * counts.astype(np.float32))

    elif strategy == "effective_number":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-7)

    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")

    # Normalize weights so mean weight == 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
