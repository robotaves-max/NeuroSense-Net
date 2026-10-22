import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import NeuroSenseConfig, dict_to_config
from src.data.collate import MultimodalCollate
from src.data.dataset import MELDDataset, SyntheticMELDDataset
from src.metrics import expected_calibration_error, fit_temperature
from src.models.neurosense_net import NeuroSenseNet
from src.robustness import evaluate_gate_robustness


def _sample(identifier, feature, speaker=""):
    return {
        "id": identifier,
        "label": "neutral",
        "speaker": speaker,
        "text": torch.full((4,), feature),
        "audio": torch.ones(3),
        "video": torch.ones(2),
        "video_avail": False,
    }


def test_context_window_uses_only_previous_utterances_in_dialogue():
    dataset = MELDDataset(
        [
            _sample("train_d1_u0", 1, "Alex"),
            _sample("train_d1_u1", 2, "Sam"),
            _sample("train_d2_u0", 9, "Alex"),
        ],
        context_window=2,
    )
    first, second, other_dialogue = dataset[0], dataset[1], dataset[2]
    assert first["text"].shape == (1, 4)
    assert torch.equal(second["text"][:, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(other_dialogue["text"][:, 0], torch.tensor([9.0]))
    assert first["speaker_id"] > 0 and second["speaker_id"] > 0


def test_context_disabled_preserves_existing_sequence_shape():
    dataset = MELDDataset([_sample("train_d1_u0", 1)], context_window=0)
    assert dataset[0]["text"].shape == (4,)


def test_ece_and_temperature_scaling():
    probabilities = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    targets = np.array([0, 1, 1])
    assert expected_calibration_error(probabilities, targets, n_bins=2) > 0
    logits = torch.tensor([[8.0, 0.0], [2.0, 0.0], [0.0, 7.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    assert torch.nn.functional.cross_entropy(logits / temperature, labels) < torch.nn.functional.cross_entropy(
        logits, labels
    )


def test_model_baseline_modes_and_config_validation():
    for baseline in ("full", "text_only", "audio_only", "video_only", "concat", "late"):
        config = dict_to_config({"model": {"baseline": baseline}})
        assert config.model.baseline == baseline
        model = NeuroSenseNet(
            {"text": 6, "audio": 4, "video": 3},
            hidden_dim=8,
            num_heads=2,
            num_fusion_layers=1,
            baseline=baseline,
        )
        batch = next(
            iter(
                DataLoader(
                    SyntheticMELDDataset(num_samples=2, d_text=6, d_audio=4, d_video=3),
                    batch_size=2,
                    collate_fn=MultimodalCollate(),
                )
            )
        )
        assert model(batch)["logits"].shape == (2, 7)

    try:
        NeuroSenseConfig(context=type("Context", (), {"window": -1, "speaker_identity": False})()).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("negative context windows must be rejected")


def test_gate_robustness_sweep_returns_metrics():
    dataset = SyntheticMELDDataset(num_samples=4, d_text=6, d_audio=4, d_video=3)
    loader = DataLoader(dataset, batch_size=2, collate_fn=MultimodalCollate())
    model = NeuroSenseNet(
        dataset.get_feature_dims(), hidden_dim=8, num_heads=2, num_fusion_layers=1
    )
    result = evaluate_gate_robustness(
        model, loader, torch.device("cpu"), "audio", [0.0, 0.5], "mask"
    )
    assert len(result) == 2
    assert all(0 <= row["macro_f1"] <= 1 for row in result)
    assert all(len(row["mean_gate_weights"]) == 3 for row in result)
