"""Evaluation utilities for controlled modality corruption experiments."""

from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from src.metrics import compute_metrics
from src.utils import MELD_EMOTIONS


def evaluate_gate_robustness(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    modality: str,
    strengths: Iterable[float],
    corruption: str = "mask",
) -> List[Dict[str, Any]]:
    """Measure F1 and gate allocation while progressively corrupting one input."""
    if modality not in {"text", "audio", "video"}:
        raise ValueError(f"Unknown modality: {modality}")
    if corruption not in {"mask", "noise"}:
        raise ValueError(f"Unknown corruption type: {corruption}")
    results = []
    model.eval()
    for strength in strengths:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("corruption strengths must be in [0, 1]")
        logits_all, targets_all, gates_all = [], [], []
        with torch.inference_mode():
            for original in dataloader:
                batch = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in original.items()
                }
                feature = batch[modality]
                if corruption == "mask":
                    keep = torch.rand_like(feature) >= strength
                    feature.mul_(keep)
                elif strength:
                    noise = torch.randn_like(feature) * feature.std().clamp(min=1e-8) * strength
                    feature.add_(noise)
                outputs = model(batch)
                logits_all.append(outputs["logits"].cpu())
                targets_all.append(batch["label"].cpu())
                gates_all.append(outputs["gate_weights"].cpu())
        logits = torch.cat(logits_all)
        targets = torch.cat(targets_all).numpy()
        predictions = logits.argmax(dim=1).numpy()
        metrics = compute_metrics(targets, predictions, class_names=MELD_EMOTIONS)
        gate_mean = torch.cat(gates_all).mean(dim=0).numpy()
        results.append(
            {
                "modality": modality,
                "corruption": corruption,
                "strength": float(strength),
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mean_gate_weights": gate_mean.tolist(),
            }
        )
    return results
