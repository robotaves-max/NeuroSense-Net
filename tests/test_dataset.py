import pytest
import torch
from torch.utils.data import DataLoader

from src.data.collate import MultimodalCollate
from src.data.dataset import SyntheticMELDDataset


def test_synthetic_dataset_shapes():
    ds = SyntheticMELDDataset(num_samples=32, d_text=768, d_audio=128, d_video=512, seed=42)
    assert len(ds) == 32
    sample = ds[0]
    assert "text" in sample and "audio" in sample and "video" in sample
    assert "text_avail" in sample and "audio_avail" in sample and "video_avail" in sample
    assert "label" in sample and "id" in sample
    assert 0 <= sample["label"].item() < 7


def test_multimodal_collate():
    ds = SyntheticMELDDataset(num_samples=16, seed=42, sequence_data=True, max_seq_len=12)
    collate = MultimodalCollate()
    loader = DataLoader(ds, batch_size=4, collate_fn=collate)

    batch = next(iter(loader))
    assert batch["text"].ndim == 3  # [B, T, D]
    assert batch["audio"].ndim == 3
    assert batch["video"].ndim == 3
    assert batch["text_mask"].shape[0] == 4
    assert batch["label"].shape[0] == 4
    assert len(batch["ids"]) == 4
