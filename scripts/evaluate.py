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
from src.data.inspection import DatasetInspector
from src.engine import Evaluator, run_modality_ablation_study
from src.models.neurosense_net import NeuroSenseNet
from src.utils import (
    MELD_EMOTIONS,
    get_device,
    save_json,
    setup_logger,
)
from src.visualization import (
    plot_confusion_matrices,
    plot_gate_weights_by_class,
    plot_gate_weights_overall,
    plot_modality_ablation,
    plot_per_class_f1,
)

logger = setup_logger("neurosense.evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NeuroSense-Net checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to MELD dataset root.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for eval reports.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic dataset for evaluation.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = get_device()
    logger.info(f"Loading checkpoint from: {checkpoint_path} on {device}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    feature_dims = ckpt["feature_dims"]

    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Reconstruct Model
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

    # Load Data
    if args.use_synthetic or config.data.use_synthetic:
        logger.info("Evaluating on Synthetic dataset...")
        eval_dataset = SyntheticMELDDataset(num_samples=128, seed=999)
    else:
        inspector = DatasetInspector(data_root=args.data_root or config.paths.data_root)
        report = inspector.inspect()
        eval_dataset = MELDDataset(inspector.splits[args.split], keys_map=report["inferred_keys"])

    collate_fn = MultimodalCollate()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    evaluator = Evaluator(model, eval_loader, device)
    res = evaluator.evaluate()
    metrics = res["metrics"]

    logger.info("=" * 60)
    logger.info(f"EVALUATION METRICS [{args.split.upper()} SPLIT]:")
    logger.info(f"  Loss:              {metrics['loss']:.4f}")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro-F1:          {metrics['macro_f1']:.4f} (PRIMARY)")
    logger.info(f"  Weighted-F1:       {metrics['weighted_f1']:.4f}")
    logger.info("=" * 60)

    save_json(metrics, out_dir / f"metrics_{args.split}.json")

    # Modality Ablation Study
    if config.evaluation.run_modality_ablation:
        ablation = run_modality_ablation_study(model, eval_loader, device)
        save_json(ablation, out_dir / "modality_ablation.json")
        plot_modality_ablation(ablation, plots_dir / "modality_ablation_macro_f1.png")

    # Visualizations
    plot_confusion_matrices(
        pd.DataFrame(metrics["confusion_matrix_counts"]).values,
        pd.DataFrame(metrics["confusion_matrix_normalized"]).values,
        MELD_EMOTIONS,
        plots_dir / "confusion_matrix_counts.png",
        plots_dir / "confusion_matrix_normalized.png",
    )
    plot_per_class_f1(metrics["per_class"], plots_dir / "per_class_f1.png")
    plot_gate_weights_overall(res["gate_weights"], plots_dir / "gate_weights_overall.png")
    plot_gate_weights_by_class(res["gate_weights"], res["y_true"], MELD_EMOTIONS, plots_dir / "gate_weights_by_true_class.png")

    logger.info(f"Evaluation complete! Reports saved to {out_dir}")


if __name__ == "__main__":
    main()
