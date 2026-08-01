import pytest
import torch

from src.models.neurosense_net import NeuroSenseNet


@pytest.fixture
def dummy_feature_dims():
    return {"text": 768, "audio": 128, "video": 512}


@pytest.fixture
def dummy_batch():
    B = 4
    return {
        "text": torch.randn(B, 10, 768),
        "audio": torch.randn(B, 10, 128),
        "video": torch.randn(B, 10, 512),
        "text_mask": torch.ones(B, 10, dtype=torch.bool),
        "audio_mask": torch.ones(B, 10, dtype=torch.bool),
        "video_mask": torch.ones(B, 10, dtype=torch.bool),
        "text_avail": torch.tensor([True, True, True, True]),
        "audio_avail": torch.tensor([True, False, True, False]),
        "video_avail": torch.tensor([True, True, False, False]),
        "label": torch.tensor([0, 1, 2, 3]),
        "ids": ["s0", "s1", "s2", "s3"],
    }


def test_model_forward_shapes(dummy_feature_dims, dummy_batch):
    model = NeuroSenseNet(feature_dims=dummy_feature_dims, hidden_dim=128, num_heads=4, num_fusion_layers=1)
    outputs = model(dummy_batch)

    assert "logits" in outputs and "gate_weights" in outputs and "fused_embedding" in outputs
    assert outputs["logits"].shape == (4, 7)
    assert outputs["gate_weights"].shape == (4, 3)
    assert outputs["fused_embedding"].shape == (4, 128)


def test_gate_weights_properties(dummy_feature_dims, dummy_batch):
    model = NeuroSenseNet(feature_dims=dummy_feature_dims, hidden_dim=128, num_heads=4, num_fusion_layers=1)
    outputs = model(dummy_batch)
    gate_weights = outputs["gate_weights"]

    # 1. Non-negativity
    assert (gate_weights >= 0.0).all()

    # 2. Sum to 1.0
    sums = gate_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # 3. Missing modality gate weights must be 0.0
    # Sample 1: audio is missing (index 1)
    assert gate_weights[1, 1].item() == 0.0
    # Sample 2: video is missing (index 2)
    assert gate_weights[2, 2].item() == 0.0


def test_gradient_flow(dummy_feature_dims, dummy_batch):
    model = NeuroSenseNet(feature_dims=dummy_feature_dims, hidden_dim=128, num_heads=4)
    outputs = model(dummy_batch)
    loss = outputs["logits"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter {name}"
