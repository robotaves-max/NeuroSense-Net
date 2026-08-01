import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train as train_script
from src.utils import setup_logger

logger = setup_logger("smoke_test")


def verify_file_exists(path: Path, min_bytes: int = 10) -> None:
    """Assert file exists and is not empty."""
    if not path.exists():
        raise FileNotFoundError(f"[FAIL] Missing expected artifact: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(f"[FAIL] Artifact {path} is empty or corrupted (size: {size} bytes)")
    logger.info(f"  [OK] Verified {path.name} ({size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test for NeuroSense-Net pipeline.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of smoke test training epochs.")
    parser.add_argument("--use-synthetic", action="store_true", default=True, help="Use synthetic dataset.")
    args = parser.parse_args()

    output_root = "outputs/smoke_test"
    config_path = "configs/smoke_test.yaml"

    logger.info("=" * 70)
    logger.info("NEUROSENSE-NET AUTOMATED END-TO-END SMOKE TEST")
    logger.info("=" * 70)

    # Override sys.argv to simulate CLI train command
    sys.argv = [
        "train.py",
        "--config",
        config_path,
        "--use-synthetic",
        "--epochs",
        str(args.epochs),
        "--output-root",
        output_root,
    ]

    try:
        train_script.main()
    except Exception as e:
        logger.error(f"[FAIL] Smoke test training pipeline crashed with error: {e}")
        sys.exit(1)

    # Discover generated output directory
    smoke_out_dir = Path(output_root)
    run_dirs = list(smoke_out_dir.glob("NeuroSense-Net-SmokeTest_*"))
    if not run_dirs:
        logger.error("[FAIL] Could not find smoke test run output directory!")
        sys.exit(1)

    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Verifying generated artifacts in: {latest_run}")

    expected_files = [
        latest_run / "config_resolved.yaml",
        latest_run / "environment.json",
        latest_run / "dataset_report.json",
        latest_run / "train.log",
        latest_run / "history.csv",
        latest_run / "metrics.json",
        latest_run / "label_mapping.json",
        latest_run / "modality_ablation.json",
        latest_run / "modality_ablation.csv",
        latest_run / "checkpoints" / "best.pt",
        latest_run / "checkpoints" / "last.pt",
        latest_run / "predictions" / "predictions.csv",
        latest_run / "plots" / "training_curves.png",
        latest_run / "plots" / "confusion_matrix_counts.png",
        latest_run / "plots" / "confusion_matrix_normalized.png",
        latest_run / "plots" / "per_class_f1.png",
        latest_run / "plots" / "gate_weights_overall.png",
        latest_run / "plots" / "gate_weights_by_true_class.png",
        latest_run / "plots" / "class_distribution.png",
        latest_run / "plots" / "modality_ablation_macro_f1.png",
    ]

    logger.info("Checking file presence and non-zero size:")
    for fpath in expected_files:
        try:
            verify_file_exists(fpath)
        except Exception as err:
            logger.error(str(err))
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("SUCCESS: ALL SMOKE TEST VERIFICATION CHECKS PASSED PERFECTLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
