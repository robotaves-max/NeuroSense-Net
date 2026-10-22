"""Run controlled feature corruption experiments on a trained checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collate import MultimodalCollate
from src.data.dataset import MELDDataset, SyntheticMELDDataset
from src.models.neurosense_net import NeuroSenseNet
from src.robustness import evaluate_gate_robustness
from src.utils import get_device


def main() -> None:
    """Load a checkpoint, run a corruption sweep, and save results and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate_robustness"))
    parser.add_argument("--modality", choices=["text", "audio", "video"], default="audio")
    parser.add_argument("--corruption", choices=["mask", "noise"], default="mask")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, dict):
        from src.config import dict_to_config
        config = dict_to_config(config)
    device = get_device()
    model = NeuroSenseNet(
        feature_dims=checkpoint["feature_dims"],
        num_classes=7,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_fusion_layers=config.model.num_fusion_layers,
        dropout=config.model.dropout,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
        sequence_encoder_type=config.model.sequence_encoder,
        baseline=config.model.baseline,
        use_gate=config.model.use_gate,
        use_cross_modal_attention=config.model.use_cross_modal_attention,
        use_speaker_embedding=config.context.speaker_identity or config.model.use_speaker_embedding,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if config.data.use_synthetic:
        dataset = SyntheticMELDDataset(num_samples=128, seed=config.seed + 2)
    else:
        if args.data_root is None:
            raise ValueError("--data-root is required for a real-data checkpoint")
        data_path = args.data_root / "test.pt"
        dataset = MELDDataset(data_path, context_window=config.context.window)
    loader = DataLoader(dataset, batch_size=config.data.batch_size, collate_fn=MultimodalCollate())
    results = evaluate_gate_robustness(
        model, loader, device, args.modality, args.strengths, args.corruption
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_dir / "gate_robustness.csv", index=False)
    gate_index = {"text": 0, "audio": 1, "video": 2}[args.modality]
    plt.plot(frame["strength"], frame["macro_f1"], marker="o", label="Macro-F1")
    plt.plot(
        frame["strength"],
        frame["mean_gate_weights"].map(lambda weights: weights[gate_index]),
        marker="s",
        label=f"{args.modality.title()} mean gate weight",
    )
    plt.xlabel(f"{args.corruption.title()} strength")
    plt.ylabel("Score / gate weight")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "gate_robustness.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
