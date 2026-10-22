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
    parser.add_argument("--text-encoder", choices=["minilm", "roberta-base"], default="minilm")
    parser.add_argument("--audio-encoder", choices=["wav2vec2", "wavlm", "hubert"], default="wav2vec2")
    args = parser.parse_args()

    extractor = MELDFeatureExtractor(
        device=args.device,
        text_encoder=args.text_encoder,
        audio_encoder=args.audio_encoder,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / ".feature_cache"
    cache_dir.mkdir(exist_ok=True)
    for split, csv_path in discover_meld_csvs(args.data_root).items():
        rows = read_meld_split(csv_path, split, args.data_root)
        if args.limit:
            rows = rows[:args.limit]
        samples: List[Dict] = []
        for row in rows:
            cache_key = f"{row['id']}_{args.text_encoder}_{args.audio_encoder}".replace("/", "_")
            cache_path = cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                samples.append(torch.load(cache_path, map_location="cpu", weights_only=False))
                continue
            sample = {
                "id": row["id"],
                "label": row["label"],
                "text": extractor.text_features(row["text"]),
                "text_avail": True,
                "speaker": row.get("speaker", ""),
            }
            path = Path(row["clip_path"]) if row["clip_path"] else None
            if path and path.exists():
                try:
                    frames, wave, rate = _read_clip(path)
                    sample.update(
                        audio=extractor.audio_features(wave, rate),
                        video=extractor.video_features(frames),
                        audio_avail=True,
                        video_avail=True,
                    )
                except Exception as exc:
                    print(f"[WARN] {row['id']}: {exc}; keeping text only")
                    sample.update(audio_avail=False, video_avail=False)
            else:
                sample.update(audio_avail=False, video_avail=False)
            samples.append(sample)
            torch.save(sample, cache_path)
        torch.save(samples, args.output_dir / f"{split}.pt")
        print(f"Saved {len(samples)} {split} samples to {args.output_dir / f'{split}.pt'}")


if __name__ == "__main__":
    main()
