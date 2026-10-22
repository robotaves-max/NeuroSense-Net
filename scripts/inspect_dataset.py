import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.inspection import DatasetInspector
from src.utils import setup_logger

logger = setup_logger("inspect_dataset")


def main():
    parser = argparse.ArgumentParser(description="Discover and inspect MELD preprocessed dataset layout.")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing MELD dataset files.")
    parser.add_argument("--output-dir", type=str, default="outputs/dataset_inspection", help="Directory to write reports.")
    parser.add_argument("--key-text", type=str, default=None, help="Override key for text feature.")
    parser.add_argument("--key-audio", type=str, default=None, help="Override key for audio feature.")
    parser.add_argument("--key-video", type=str, default=None, help="Override key for video feature.")
    parser.add_argument("--key-label", type=str, default=None, help="Override key for label.")
    args = parser.parse_args()

    overrides = {}
    if args.key_text:
        overrides["text"] = args.key_text
    if args.key_audio:
        overrides["audio"] = args.key_audio
    if args.key_video:
        overrides["video"] = args.key_video
    if args.key_label:
        overrides["label"] = args.key_label

    logger.info("Initializing DatasetInspector...")
    inspector = DatasetInspector(data_root=args.data_root, key_overrides=overrides)

    try:
        inspector.inspect()
        out_json, out_txt = inspector.generate_reports(Path(args.output_dir))
        logger.info(f"Successfully generated dataset reports in {args.output_dir}")
        print("\n" + Path(out_txt).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        logger.error(f"Dataset inspection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
