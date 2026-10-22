import logging
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import MELD_EMOTIONS

logger = logging.getLogger("neurosense.dataset")


class SyntheticMELDDataset(Dataset):
    """Deterministic synthetic MELD dataset for smoke testing and offline debugging.

    Produces realistic text, audio, and video features with realistic class imbalance
    and embedded class signal for smoke test convergence verification.
    """

    CLASS_PROBABILITIES = [0.47, 0.14, 0.12, 0.11, 0.09, 0.04, 0.03]  # Imbalanced like MELD

    def __init__(
        self,
        num_samples: int = 256,
        d_text: int = 768,
        d_audio: int = 128,
        d_video: int = 512,
        seed: int = 42,
        missing_rate: float = 0.05,
        sequence_data: bool = True,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.d_text = d_text
        self.d_audio = d_audio
        self.d_video = d_video
        self.seed = seed
        self.missing_rate = missing_rate
        self.sequence_data = sequence_data
        self.max_seq_len = max_seq_len

        rng = np.random.RandomState(seed)

        # Generate imbalanced labels
        self.labels = rng.choice(7, size=num_samples, p=self.CLASS_PROBABILITIES)

        # Generate modality availability flags (text is always present; audio/video may be missing)
        self.text_avail = np.ones(num_samples, dtype=bool)
        self.audio_avail = rng.rand(num_samples) >= missing_rate
        self.video_avail = rng.rand(num_samples) >= missing_rate

        # Ensure at least one modality is present for every sample
        for i in range(num_samples):
            if not self.audio_avail[i] and not self.video_avail[i]:
                self.audio_avail[i] = True

        self.samples = []
        for i in range(num_samples):
            label = self.labels[i]

            # Signal vector for class discriminability
            signal = np.sin(np.arange(7) * (label + 1))

            if sequence_data:
                seq_len = rng.randint(4, max_seq_len + 1)
                text_feat = rng.randn(seq_len, d_text).astype(np.float32)
                audio_feat = rng.randn(seq_len, d_audio).astype(np.float32)
                video_feat = rng.randn(seq_len, d_video).astype(np.float32)

                # Add class signal to features
                text_feat += signal[: d_text % 7 + 1].mean() * 0.5
                audio_feat += signal[: d_audio % 7 + 1].mean() * 0.5
                video_feat += signal[: d_video % 7 + 1].mean() * 0.5
            else:
                text_feat = rng.randn(d_text).astype(np.float32)
                audio_feat = rng.randn(d_audio).astype(np.float32)
                video_feat = rng.randn(d_video).astype(np.float32)

                text_feat[:7] += signal * 0.5
                audio_feat[:7] += signal * 0.5
                video_feat[:7] += signal * 0.5

            self.samples.append(
                {
                    "text": torch.tensor(text_feat, dtype=torch.float32),
                    "audio": torch.tensor(audio_feat, dtype=torch.float32),
                    "video": torch.tensor(video_feat, dtype=torch.float32),
                    "text_avail": torch.tensor(bool(self.text_avail[i]), dtype=torch.bool),
                    "audio_avail": torch.tensor(bool(self.audio_avail[i]), dtype=torch.bool),
                    "video_avail": torch.tensor(bool(self.video_avail[i]), dtype=torch.bool),
                    "label": torch.tensor(int(label), dtype=torch.long),
                    "id": f"synthetic_sample_{i:04d}",
                }
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_feature_dims(self) -> Dict[str, int]:
        return {"text": self.d_text, "audio": self.d_audio, "video": self.d_video}


class MELDDataset(Dataset):
    """Adapter dataset for real MELD preprocessed files supporting flexible keys and shapes."""

    def __init__(
        self,
        data_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        keys_map: Optional[Dict[str, str]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        normalization_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        context_window: int = 0,
    ):
        super().__init__()
        self.keys_map = keys_map or {
            "text": "text",
            "audio": "audio",
            "video": "video",
            "label": "label",
            "id": "id",
        }
        self.label_mapping = label_mapping or {name: i for i, name in enumerate(MELD_EMOTIONS)}
        self.normalization_stats = normalization_stats
        if context_window < 0:
            raise ValueError("context_window must be non-negative")
        self.context_window = context_window

        self.samples = self._load_data(data_path)
        for sample in self.samples:
            speaker_name = sample.pop("speaker_name", "")
            if speaker_name:
                speaker_hash = hashlib.sha256(speaker_name.encode("utf-8")).digest()
                sample["speaker_id"] = int.from_bytes(speaker_hash[:4], "big") % 511 + 1
        self._attach_dialogue_context()
        logger.info(f"Loaded {len(self.samples)} samples for dataset split.")

    def _load_data(self, data_path: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Load and parse arbitrary dict/list data into standard sample dictionaries."""
        raw_items = []
        if isinstance(data_path, (str, Path)):
            loaded = torch.load(data_path, map_location="cpu", weights_only=False)
        else:
            loaded = data_path

        if isinstance(loaded, dict):
            # Check if dict of lists or dict of samples
            first_val = next(iter(loaded.values()))
            if isinstance(first_val, (list, torch.Tensor, np.ndarray)):
                n_samples = len(first_val)
                for i in range(n_samples):
                    item = {}
                    for k, v in loaded.items():
                        item[k] = v[i]
                    raw_items.append(item)
            else:
                raw_items.append(loaded)
        elif isinstance(loaded, (list, tuple)):
            raw_items = list(loaded)
        else:
            raise ValueError(f"Unsupported data layout type: {type(loaded)}")

        parsed_samples = []
        for idx, raw in enumerate(raw_items):
            sample = self._parse_single_sample(raw, idx)
            parsed_samples.append(sample)

        return parsed_samples

    def _parse_single_sample(self, raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Normalize single sample dict into standard tensor format."""
        sample = {}

        # Extract ID
        id_key = self.keys_map.get("id", "id")
        sample["id"] = str(raw.get(id_key, f"sample_{idx:05d}"))

        # Extract Label
        label_key = self.keys_map.get("label", "label")
        raw_label = raw.get(label_key, 0)
        if isinstance(raw_label, str):
            label_int = self.label_mapping.get(raw_label.lower(), 0)
        elif isinstance(raw_label, (int, np.integer, torch.Tensor)):
            label_int = int(raw_label)
        else:
            label_int = 0
        sample["label"] = torch.tensor(label_int, dtype=torch.long)
        speaker = raw.get("speaker_id", raw.get("speaker", ""))
        sample["speaker_name"] = str(speaker) if speaker else ""
        sample["speaker_id"] = int(speaker) if isinstance(speaker, (int, np.integer)) else 0

        # Extract modalities and availability
        for mod in ["text", "audio", "video"]:
            mod_key = self.keys_map.get(mod, mod)
            feat = raw.get(mod_key, None)
            declared_available = raw.get(f"{mod}_avail", feat is not None)

            if feat is not None and bool(declared_available):
                if isinstance(feat, np.ndarray):
                    feat_tensor = torch.tensor(feat, dtype=torch.float32)
                elif isinstance(feat, torch.Tensor):
                    feat_tensor = feat.float()
                else:
                    feat_tensor = torch.zeros(1, dtype=torch.float32)
                sample[mod] = feat_tensor
                sample[f"{mod}_avail"] = torch.tensor(True, dtype=torch.bool)
            else:
                sample[mod] = torch.zeros(1, dtype=torch.float32)
                sample[f"{mod}_avail"] = torch.tensor(False, dtype=torch.bool)

        return sample

    def _attach_dialogue_context(self) -> None:
        """Prepend up to N preceding utterance embeddings from the same dialogue."""
        groups: Dict[str, List[Tuple[int, int]]] = {}
        for index, sample in enumerate(self.samples):
            match = re.search(r"(?:^|_)d(\d+)_u(\d+)$", sample["id"])
            if match:
                groups.setdefault(match.group(1), []).append((int(match.group(2)), index))

        for utterances in groups.values():
            utterances.sort()
            for position, (_, sample_index) in enumerate(utterances):
                sample = self.samples[sample_index]
                current = sample["text"]
                if self.context_window == 0:
                    continue
                previous = utterances[max(0, position - self.context_window):position]
                context = [
                    self.samples[index]["text"]
                    .reshape(-1, self.samples[index]["text"].shape[-1])
                    .mean(dim=0, keepdim=True)
                    for _, index in previous
                ]
                target = current.reshape(-1, current.shape[-1])
                sample["text"] = torch.cat([*context, target], dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if not self.normalization_stats:
            return sample

        normalized = dict(sample)
        for mod in ["text", "audio", "video"]:
            if sample[f"{mod}_avail"] and mod in self.normalization_stats:
                mean, std = self.normalization_stats[mod]
                feat = sample[mod]
                normalized[mod] = (feat - mean) / (std + 1e-7)

        return normalized

    def compute_normalization_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute training split mean and std per modality."""
        stats = {}
        for mod in ["text", "audio", "video"]:
            valid_feats = [s[mod] for s in self.samples if s[f"{mod}_avail"]]
            if not valid_feats:
                continue
            cat_feats = torch.cat([f.view(-1, f.shape[-1]) for f in valid_feats], dim=0)
            mean = cat_feats.mean(dim=0)
            std = cat_feats.std(dim=0)
            stats[mod] = (mean, std)
        self.normalization_stats = stats
        return stats
