"""Convert the official MELD feature release into NeuroSense-Net split tensors.

The official archive supplies dialogue-indexed text (600D) and audio (300D)
features plus per-utterance labels. It does not include visual features, so
this importer records video as unavailable while retaining a one-dimensional
placeholder required by the generic three-branch fusion architecture.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official MELD text/audio features.")
    parser.add_argument("--features-dir", required=True, type=Path, help=".../MELD.Features.Models/features")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = load_pickle(args.features_dir / "data_emotion.p")[0]
    text_by_split = load_pickle(args.features_dir / "text_emotion.pkl")
    audio_by_split = load_pickle(args.features_dir / "audio_emotion.pkl")
    split_names = ["train", "val", "test"]
    source_names = ["train", "val", "test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for output_name, source_name, text_dialogues, audio_dialogues in zip(split_names, source_names, text_by_split, audio_by_split):
        samples = []
        split_records = [item for item in records if item["split"] == source_name]
        for item in split_records:
            dialogue, utterance = str(item["dialog"]), int(item["utterance"])
            if dialogue not in text_dialogues or dialogue not in audio_dialogues:
                continue
            text = np.asarray(text_dialogues[dialogue][utterance], dtype=np.float32)
            audio = np.asarray(audio_dialogues[dialogue][utterance], dtype=np.float32)
            samples.append({
                "id": f"{output_name}_d{dialogue}_u{utterance}",
                "label": item["y"],
                "dialogue_id": dialogue,
                "utterance_id": utterance,
                "text": torch.from_numpy(text), "text_avail": True,
                "audio": torch.from_numpy(audio), "audio_avail": True,
                "video": torch.zeros(1, dtype=torch.float32), "video_avail": False,
            })
        torch.save(samples, args.output_dir / f"{output_name}.pt")
        print(f"{output_name}: saved {len(samples)} samples")


if __name__ == "__main__":
    main()
