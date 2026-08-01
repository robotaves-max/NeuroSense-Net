import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config_to_dict, load_config, save_config
from src.data.collate import MultimodalCollate
from src.data.dataset import MELDDataset, SyntheticMELDDataset
from src.data.inspection import DatasetInspector
from src.engine import Evaluator, Trainer, run_modality_ablation_study
from src.losses import compute_class_weights
from src.models.neurosense_net import NeuroSenseNet
from src.utils import (
    MELD_EMOTIONS,
    get_device,
    get_environment_info,
    save_json,
    seed_everything,
    setup_logger,
)
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_gate_weights_by_class,
    plot_gate_weights_overall,
    plot_modality_ablation,
    plot_per_class_f1,
    plot_training_history,
)


def main():
    parser = argparse.ArgumentParser(description="Train NeuroSense-Net Multimodal Emotion Model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root path.")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic dataset mode.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--output-root", type=str, default=None, help="Override output directory root.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    # Load base config
    config = load_config(Path(args.config))

    # Apply CLI overrides
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.use_synthetic:
        config.data.use_synthetic = True
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.output_root:
        config.paths.output_root = args.output_root
    if args.seed:
        config.seed = args.seed

    config.validate()

    # Set up reproducible run output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project_name}_{timestamp}"
    run_dir = Path(config.paths.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    preds_dir = run_dir / "predictions"
    plots_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)

    # Logger
    logger = setup_logger("neurosense.train", log_file=run_dir / "train.log")
    logger.info(f"Initialized NeuroSense-Net Training Run: {run_name}")
    logger.info(f"Run directory: {run_dir}")

    # Seed
    seed_everything(config.seed, config.deterministic)

    # Save initial metadata
    save_config(config, run_dir / "config_resolved.yaml")
    save_json(get_environment_info(), run_dir / "environment.json")
    label_map = {name: i for i, name in enumerate(MELD_EMOTIONS)}
    save_json(label_map, run_dir / "label_mapping.json")

    # Data Loading
    if config.data.use_synthetic:
        logger.info("Using Synthetic MELD Dataset for training and verification...")
        train_dataset = SyntheticMELDDataset(
            num_samples=config.data.synthetic_train_size, seed=config.seed
        )
        val_dataset = SyntheticMELDDataset(
            num_samples=config.data.synthetic_val_size, seed=config.seed + 1
        )
        test_dataset = SyntheticMELDDataset(
            num_samples=config.data.synthetic_test_size, seed=config.seed + 2
        )
        feature_dims = train_dataset.get_feature_dims()
        dataset_report = {
            "mode": "synthetic",
            "feature_dims": feature_dims,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        }
        save_json(dataset_report, run_dir / "dataset_report.json")
    else:
        logger.info("Inspecting real dataset paths...")
        inspector = DatasetInspector(
            data_root=config.paths.data_root,
            key_overrides=config.data.modality_key_overrides,
        )
        report = inspector.inspect()
        inspector.generate_reports(run_dir)

        feature_dims = report["inferred_dims"]
        if None in feature_dims.values():
            raise ValueError(f"Failed to infer all feature dimensions: {feature_dims}")

        splits = inspector.splits
        train_dataset = MELDDataset(splits["train"], keys_map=report["inferred_keys"])
        val_dataset = MELDDataset(splits["val"], keys_map=report["inferred_keys"])
        test_dataset = MELDDataset(splits["test"], keys_map=report["inferred_keys"])

        if config.data.normalize:
            norm_stats = train_dataset.compute_normalization_stats()
            val_dataset.normalization_stats = norm_stats
            test_dataset.normalization_stats = norm_stats

    logger.info(f"Inferred Feature Dimensions: {feature_dims}")

    collate_fn = MultimodalCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )

    # Class Weights (derived from TRAIN split only)
    train_targets = torch.tensor([s["label"] for s in train_dataset], dtype=torch.long)
    class_weights = compute_class_weights(
        train_targets,
        num_classes=7,
        strategy=config.loss.class_weight_strategy,
        beta=config.loss.effective_num_beta,
    )
    logger.info(f"Calculated training class weights: {class_weights.tolist()}")

    # Model Construction
    model = NeuroSenseNet(
        feature_dims=feature_dims,
        num_classes=7,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_fusion_layers=config.model.num_fusion_layers,
        dropout=config.model.dropout,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
        sequence_encoder_type=config.model.sequence_encoder,
        modality_dropout_prob=config.training.modality_dropout_probability,
    )

    # Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        run_dir=run_dir,
        class_weights=class_weights,
    )
    train_results = trainer.train()

    # Final Test Evaluation with Best Model
    device = get_device()
    test_evaluator = Evaluator(model, test_loader, device)
    test_res = test_evaluator.evaluate()

    metrics = test_res["metrics"]
    logger.info("=" * 60)
    logger.info("FINAL TEST EVALUATION RESULTS:")
    logger.info(f"  Test Loss:              {metrics['loss']:.4f}")
    logger.info(f"  Test Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Test Macro-F1:          {metrics['macro_f1']:.4f} (PRIMARY)")
    logger.info(f"  Test Weighted-F1:       {metrics['weighted_f1']:.4f}")
    logger.info("=" * 60)

    save_json(metrics, run_dir / "metrics.json")

    # Modality Ablation Study
    if config.evaluation.run_modality_ablation:
        ablation_results = run_modality_ablation_study(model, test_loader, device)
        save_json(ablation_results, run_dir / "modality_ablation.json")
        pd.DataFrame(ablation_results).T.to_csv(run_dir / "modality_ablation.csv")
        plot_modality_ablation(ablation_results, plots_dir / "modality_ablation_macro_f1.png")

    # Visualizations
    logger.info("Generating publication-grade visualization artifacts...")
    plot_training_history(train_results["history"], plots_dir / "training_curves.png")

    plot_confusion_matrices(
        np.array(metrics["confusion_matrix_counts"]),
        np.array(metrics["confusion_matrix_normalized"]),
        MELD_EMOTIONS,
        plots_dir / "confusion_matrix_counts.png",
        plots_dir / "confusion_matrix_normalized.png",
    )

    plot_per_class_f1(metrics["per_class"], plots_dir / "per_class_f1.png")

    plot_gate_weights_overall(test_res["gate_weights"], plots_dir / "gate_weights_overall.png")

    plot_gate_weights_by_class(
        test_res["gate_weights"],
        test_res["y_true"],
        MELD_EMOTIONS,
        plots_dir / "gate_weights_by_true_class.png",
    )

    # Class distributions
    def get_dist(dataset):
        c = np.bincount([s["label"] for s in dataset], minlength=7)
        return {name: int(c[i]) for i, name in enumerate(MELD_EMOTIONS)}

    plot_class_distribution(
        get_dist(train_dataset),
        get_dist(val_dataset),
        get_dist(test_dataset),
        plots_dir / "class_distribution.png",
    )

    # Predictions CSV
    if config.evaluation.save_predictions:
        preds_df = pd.DataFrame(
            {
                "sample_id": test_res["ids"],
                "split": "test",
                "true_label": [MELD_EMOTIONS[i] for i in test_res["y_true"]],
                "predicted_label": [MELD_EMOTIONS[i] for i in test_res["y_pred"]],
                "confidence": test_res["probs"].max(axis=1),
            }
        )
        for idx, em in enumerate(MELD_EMOTIONS):
            preds_df[f"prob_{em}"] = test_res["probs"][:, idx]
        preds_df["gate_text"] = test_res["gate_weights"][:, 0]
        preds_df["gate_audio"] = test_res["gate_weights"][:, 1]
        preds_df["gate_video"] = test_res["gate_weights"][:, 2]

        preds_df.to_csv(preds_dir / "predictions.csv", index=False)

        # Error Analysis CSV (highest confidence errors)
        incorrect_df = preds_df[preds_df["true_label"] != preds_df["predicted_label"]]
        error_df = incorrect_df.sort_values(by="confidence", ascending=False).head(50)
        error_df.to_csv(preds_dir / "error_analysis.csv", index=False)

    logger.info(f"Training run completed successfully! All artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
