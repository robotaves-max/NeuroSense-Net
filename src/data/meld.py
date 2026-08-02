"""MELD manifest handling and pretrained feature extraction.

The raw MELD release stores utterance metadata in CSV files and optional MP4
clips.  This module turns the official split CSVs into a stable manifest and
extracts fixed-size representations suitable for ``NeuroSenseNet``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from src.utils import MELD_EMOTIONS

logger = logging.getLogger("neurosense.meld")


def _column(row: Dict[str, str], *names: str) -> str:
    """Fetch a CSV field case-insensitively, accepting MELD naming variants."""
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return str(value).strip()
    return ""


def read_meld_split(csv_path: Path, split: str, clips_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read an official MELD CSV into manifest rows.

    MELD has appeared with both ``Dialogue_ID``/``Utterance_ID`` and lower-case
    headers.  Clip discovery is deliberately tolerant: it records a missing
    path rather than silently inventing visual or audio data.
    """
    rows: List[Dict[str, Any]] = []
    # Index once. Re-running a recursive search for every MELD utterance turns
    # a normal preparation job into thousands of needless directory walks.
    clip_index = {}
    if clips_root and clips_root.exists():
        clip_index = {path.name.lower(): path for path in clips_root.rglob("*.mp4")}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            dialogue_id = _column(row, "Dialogue_ID", "dialogue_id")
            utterance_id = _column(row, "Utterance_ID", "utterance_id")
            utterance = _column(row, "Utterance", "text")
            emotion = _column(row, "Emotion", "emotion").lower()
            if emotion not in MELD_EMOTIONS:
                logger.warning("Skipping row %s in %s: unknown emotion %r", index, csv_path, emotion)
                continue
            sample_id = f"{split}_d{dialogue_id}_u{utterance_id}"
            clip_path = None
            if clips_root:
                candidates = [
                    clips_root / f"dia{dialogue_id}_utt{utterance_id}.mp4",
                    clips_root / split / f"dia{dialogue_id}_utt{utterance_id}.mp4",
                    clips_root / f"{dialogue_id}_{utterance_id}.mp4",
                ]
                clip_path = next((path for path in candidates if path.exists()), None)
                if clip_path is None:
                    # Official archives vary in their split folder names.
                    clip_path = clip_index.get(f"dia{dialogue_id}_utt{utterance_id}.mp4".lower())
            rows.append({
                "id": sample_id,
                "split": split,
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "text": utterance,
                "label": MELD_EMOTIONS.index(emotion),
                "emotion": emotion,
                "clip_path": str(clip_path) if clip_path else None,
            })
    return rows


def discover_meld_csvs(data_root: Path) -> Dict[str, Path]:
    """Locate train/dev/test label CSVs in a MELD download."""
    found: Dict[str, Path] = {}
    aliases = {"train": ("train",), "val": ("dev", "valid", "val"), "test": ("test",)}
    for path in data_root.rglob("*.csv"):
        name = path.name.lower()
        for split, tokens in aliases.items():
            if split not in found and any(token in name for token in tokens) and ("sent" in name or "label" in name or "meld" in name):
                found[split] = path
    missing = {"train", "val", "test"} - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not find MELD CSVs for {sorted(missing)} below {data_root}. "
            "Expected files similar to train_sent_emo.csv, dev_sent_emo.csv and test_sent_emo.csv."
        )
    return found


def _optional_imports() -> None:
    try:
        import cv2  # noqa: F401
        import librosa  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Real MELD feature extraction needs optional packages. Run: "
            "pip install -r requirements-optional.txt"
        ) from exc


class MELDFeatureExtractor:
    """Pretrained text, audio and visual encoders used for reproducible MELD preparation.

    Text uses a sentence-transformer-compatible BERT encoder. Audio uses
    wav2vec 2.0 representations and video uses ImageNet-pretrained ResNet-18
    frame features.  All encoders are frozen; the fusion classifier is trained
    on MELD labels, which is practical on a student GPU and avoids label leakage.
    """

    def __init__(self, device: Optional[str] = None, max_frames: int = 12):
        _optional_imports()
        import torchvision.models as tv_models
        from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_frames = max_frames
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device).eval()
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device).eval()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        backbone = tv_models.resnet18(weights=weights)
        self.video_model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device).eval()
        self.video_transform = weights.transforms()

    @torch.inference_mode()
    def text_features(self, text: str) -> torch.Tensor:
        encoded = self.text_tokenizer(text or "", truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.text_model(**encoded).last_hidden_state[0]
        return output.cpu().float()

    @torch.inference_mode()
    def audio_features(self, waveform: np.ndarray, sampling_rate: int) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        encoded = self.audio_processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        values = encoded.input_values.to(self.device)
        # Long clips can exceed memory.  A 20-second evenly sampled window is stable for MELD utterances.
        values = values[:, : sampling_rate * 20]
        return self.audio_model(values).last_hidden_state[0].cpu().float()

    @torch.inference_mode()
    def video_features(self, frames: Iterable[np.ndarray]) -> torch.Tensor:
        frame_list = list(frames)
        if not frame_list:
            raise ValueError("No frames supplied for video feature extraction.")
        positions = np.linspace(0, len(frame_list) - 1, min(self.max_frames, len(frame_list)), dtype=int)
        batch = torch.stack([self.video_transform(torch.from_numpy(frame_list[i]).permute(2, 0, 1)) for i in positions]).to(self.device)
        return self.video_model(batch).flatten(1).cpu().float()
