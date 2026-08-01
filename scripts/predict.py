import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collate import MultimodalCollate
from src.data.dataset import MELDDataset, SyntheticMELDDataset
from src.engine import Evaluator
from src.models.neurosense_net import NeuroSenseNet
from src.utils import MELD_EMOTIONS, get_device, setup_logger

logger = setup_logger("neurosense.predict")


def main():
    parser = argparse.ArgumentParser(description="Run NeuroSense-Net inference on input data.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--input", type=str, default=None, help="Input data file (.pt, .pth, .npz).")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output predictions CSV path.")
    parser.add_argument("--use-synthetic", action="store_true", help="Generate synthetic input for quick test.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    logger.info(f"Loading model checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    feature_dims = ckpt["feature_dims"]

    model = NeuroSenseNet(
        feature_dims=feature_dims,
        num_classes=7,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_fusion_layers=config.model.num_fusion_layers,
        dropout=config.model.dropout,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
        sequence_encoder_type=config.model.sequence_encoder,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if args.use_synthetic or args.input is None:
        logger.info("Running inference on synthetic test samples...")
        dataset = SyntheticMELDDataset(num_samples=16, seed=42)
    else:
        logger.info(f"Loading input file: {args.input}")
        dataset = MELDDataset(args.input)

    collate_fn = MultimodalCollate()
    loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=collate_fn)

    evaluator = Evaluator(model, loader, device)
    res = evaluator.evaluate()

    df = pd.DataFrame(
        {
            "sample_id": res["ids"],
            "true_label": [MELD_EMOTIONS[i] for i in res["y_true"]],
            "predicted_label": [MELD_EMOTIONS[i] for i in res["y_pred"]],
            "confidence": res["probs"].max(axis=1),
        }
    )
    for idx, em in enumerate(MELD_EMOTIONS):
        df[f"prob_{em}"] = res["probs"][:, idx]
    df["gate_text"] = res["gate_weights"][:, 0]
    df["gate_audio"] = res["gate_weights"][:, 1]
    df["gate_video"] = res["gate_weights"][:, 2]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Inference complete! Saved {len(df)} predictions to: {out_path}")
    print("\nSample Predictions:")
    print(df[["sample_id", "true_label", "predicted_label", "confidence", "gate_text", "gate_audio", "gate_video"]].head())


if __name__ == "__main__":
    main()
