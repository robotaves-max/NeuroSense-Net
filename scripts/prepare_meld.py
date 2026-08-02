"""Prepare official MELD text, audio and video features for training.

Example:
    python scripts/prepare_meld.py --data-root data/MELD.Raw --output-dir data/processed_meld
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.meld import MELDFeatureExtractor, discover_meld_csvs, read_meld_split


def _read_clip(path: Path):
    import cv2
    import librosa

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    waveform, rate = librosa.load(str(path), sr=16000, mono=True)
    return frames, waveform, rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real MELD text/audio/video tensors.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit for a quick validation run.")
    args = parser.parse_args()

    extractor = MELDFeatureExtractor(device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, csv_path in discover_meld_csvs(args.data_root).items():
        rows = read_meld_split(csv_path, split, args.data_root)
        if args.limit:
            rows = rows[:args.limit]
        samples: List[Dict] = []
        for row in rows:
            sample = {"id": row["id"], "label": row["label"], "text": extractor.text_features(row["text"]), "text_avail": True}
            path = Path(row["clip_path"]) if row["clip_path"] else None
            if path and path.exists():
                try:
                    frames, wave, rate = _read_clip(path)
                    sample.update(audio=extractor.audio_features(wave, rate), video=extractor.video_features(frames), audio_avail=True, video_avail=True)
                except Exception as exc:
                    print(f"[WARN] {row['id']}: {exc}; keeping text only")
                    sample.update(audio_avail=False, video_avail=False)
            else:
                sample.update(audio_avail=False, video_avail=False)
            samples.append(sample)
        torch.save(samples, args.output_dir / f"{split}.pt")
        print(f"Saved {len(samples)} {split} samples to {args.output_dir / f'{split}.pt'}")


if __name__ == "__main__":
    main()
