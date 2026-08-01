from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, EvaluationConfig, LossConfig, ModelConfig, NeuroSenseConfig, PathsConfig, TrainingConfig
from src.data.collate import MultimodalCollate
from src.data.dataset import SyntheticMELDDataset
from src.engine import Trainer, Evaluator
from src.models.neurosense_net import NeuroSenseNet


def test_short_training_and_checkpoint_loop():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        config = NeuroSenseConfig(
            paths=PathsConfig(output_root=tmpdir),
            data=DataConfig(use_synthetic=True, batch_size=8),
            training=TrainingConfig(epochs=2, early_stopping_patience=2, mixed_precision=False),
        )

        train_ds = SyntheticMELDDataset(num_samples=16, seed=42)
        val_ds = SyntheticMELDDataset(num_samples=8, seed=43)

        collate_fn = MultimodalCollate()
        train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

        model = NeuroSenseNet(
            feature_dims=train_ds.get_feature_dims(),
            hidden_dim=64,
            num_heads=2,
            num_fusion_layers=1,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            run_dir=run_dir,
        )

        res = trainer.train()
        assert res["best_macro_f1"] >= 0.0
        assert (run_dir / "checkpoints" / "best.pt").exists()
        assert (run_dir / "checkpoints" / "last.pt").exists()
