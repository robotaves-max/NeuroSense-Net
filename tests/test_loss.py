import pytest
import torch
import torch.nn as nn

from src.losses import FocalLoss, compute_class_weights


def test_focal_loss_ce_equivalence():
    """Verify FocalLoss with gamma=0.0 and no weight is mathematically identical to CrossEntropyLoss."""
    torch.manual_seed(42)
    logits = torch.randn(8, 7)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])

    ce_loss = nn.CrossEntropyLoss(reduction="mean")(logits, targets)
    focal_loss = FocalLoss(gamma=0.0, weight=None, reduction="mean")(logits, targets)

    assert torch.allclose(ce_loss, focal_loss, atol=1e-5)


def test_focal_loss_reductions():
    logits = torch.randn(4, 7)
    targets = torch.tensor([0, 2, 4, 6])

    fl_none = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    fl_mean = FocalLoss(gamma=2.0, reduction="mean")(logits, targets)
    fl_sum = FocalLoss(gamma=2.0, reduction="sum")(logits, targets)

    assert fl_none.shape == (4,)
    assert torch.allclose(fl_none.mean(), fl_mean)
    assert torch.allclose(fl_none.sum(), fl_sum)


def test_class_weights_computation():
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3, 4, 5, 6])
    weights = compute_class_weights(targets, num_classes=7, strategy="effective_number")

    assert weights.shape == (7,)
    assert torch.is_floating_point(weights)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-3)
