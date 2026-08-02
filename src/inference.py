"""Pretrained, no-checkpoint inference for the interactive application.

This module deliberately keeps the UI immediately usable: all three modality
models are public pretrained models and their probability vectors are mapped to
the seven MELD emotion classes before late fusion. A separately trained MELD
fusion checkpoint can still be evaluated with the research scripts.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.utils import MELD_EMOTIONS


_ALIASES = {
    "ang": "anger", "angry": "anger", "anger": "anger",
    "dis": "disgust", "disgust": "disgust",
    "fea": "fear", "fear": "fear",
    "hap": "joy", "happy": "joy", "happiness": "joy", "joy": "joy",
    "neu": "neutral", "neutral": "neutral",
    "sad": "sadness", "sadness": "sadness",
    "sur": "surprise", "surprised": "surprise", "surprise": "surprise",
}


def to_meld_probabilities(predictions: Iterable[Dict[str, float]]) -> np.ndarray:
    """Map arbitrary classifier labels to the canonical MELD label order."""
    scores = np.zeros(len(MELD_EMOTIONS), dtype=np.float32)
    for item in predictions:
        raw = str(item["label"]).lower().replace("label_", "").strip()
        label = _ALIASES.get(raw, raw)
        if label in MELD_EMOTIONS:
            scores[MELD_EMOTIONS.index(label)] += float(item["score"])
    total = scores.sum()
    # Some speech models only expose a subset of emotions. Retain the model's
    # confidence across the known classes and use uniform probabilities only if
    # its labels cannot be interpreted at all.
    return scores / total if total else np.full(len(MELD_EMOTIONS), 1 / len(MELD_EMOTIONS), dtype=np.float32)


class PretrainedEmotionSuite:
    """Lazy public-model wrapper used by Streamlit's selectable-input demo."""

    def __init__(self) -> None:
        # Lazy loading prevents a text-only user from waiting for audio and
        # vision weights to download on the first click.
        self.text = None
        self.audio = None
        self.face = None

    def text_probabilities(self, text: str) -> np.ndarray:
        if self.text is None:
            from transformers import pipeline
            self.text = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        return to_meld_probabilities(self.text(text, truncation=True)[0])

    def audio_probabilities(self, waveform: np.ndarray, sampling_rate: int) -> np.ndarray:
        if self.audio is None:
            from transformers import pipeline
            self.audio = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", top_k=None)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        return to_meld_probabilities(self.audio({"array": waveform, "sampling_rate": sampling_rate})[0])

    def video_probabilities(self, frames: List[np.ndarray]) -> np.ndarray:
        if self.face is None:
            from transformers import pipeline
            self.face = pipeline("image-classification", model="trpakov/vit-face-expression", top_k=None)
        if not frames:
            raise ValueError("No readable frames were found in the video.")
        positions = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
        probabilities = [to_meld_probabilities(self.face(frame)[0]) for frame in (frames[i] for i in positions)]
        return np.mean(probabilities, axis=0)


def fuse_probabilities(modalities: Dict[str, np.ndarray]) -> np.ndarray:
    """Confidence-weighted late fusion for whichever modalities are supplied."""
    if not modalities:
        raise ValueError("Select and provide at least one modality.")
    values = list(modalities.values())
    weights = np.asarray([float(prob.max()) for prob in values], dtype=np.float32)
    weights = weights / weights.sum() if weights.sum() else np.full(len(values), 1 / len(values))
    return sum(weight * probability for weight, probability in zip(weights, values))
