from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import load_yaml, save_yaml


@dataclass
class PathsConfig:
    data_root: Optional[str] = None
    output_root: str = "outputs"
    checkpoint_root: str = "checkpoints"


@dataclass
class DataConfig:
    use_synthetic: bool = False
    synthetic_train_size: int = 256
    synthetic_val_size: int = 64
    synthetic_test_size: int = 64
    num_workers: int = 0
    batch_size: int = 32
    normalize: bool = True
    modality_key_overrides: Dict[str, str] = field(default_factory=dict)
    label_mapping: Optional[Dict[str, int]] = None


@dataclass
class ContextConfig:
    window: int = 0
    speaker_identity: bool = False


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    num_fusion_layers: int = 2
    dropout: float = 0.20
    classifier_hidden_dim: int = 256
    sequence_encoder: str = "attention_pooling"
    baseline: str = "full"
    use_gate: bool = True
    use_cross_modal_attention: bool = True
    use_speaker_embedding: bool = False


@dataclass
class LossConfig:
    name: str = "focal"
    gamma: float = 2.0
    class_weight_strategy: str = "effective_number"
    effective_num_beta: float = 0.999


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    modality_dropout_probability: float = 0.10
    early_stopping_patience: int = 8
    mixed_precision: bool = True


@dataclass
class EvaluationConfig:
    primary_metric: str = "macro_f1"
    save_predictions: bool = True
    run_modality_ablation: bool = True


@dataclass
class NeuroSenseConfig:
    project_name: str = "NeuroSense-Net"
    seed: int = 42
    deterministic: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Validate configuration hyperparameter consistency."""
        if self.model.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.model.hidden_dim}")
        if self.model.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.model.num_heads}")
        if self.model.hidden_dim % self.model.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.model.hidden_dim}) must be divisible by num_heads ({self.model.num_heads})"
            )
        if not (0.0 <= self.model.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.model.dropout}")
        if not (0.0 <= self.training.modality_dropout_probability <= 1.0):
            raise ValueError(
                f"modality_dropout_probability must be in [0, 1], got {self.training.modality_dropout_probability}"
            )
        if self.data.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.data.batch_size}")
        if self.context.window < 0:
            raise ValueError(f"context.window must be non-negative, got {self.context.window}")
        if self.model.baseline not in {"full", "text_only", "audio_only", "video_only", "concat", "late"}:
            raise ValueError(f"Unknown model baseline: {self.model.baseline}")


def dict_to_config(d: Dict[str, Any]) -> NeuroSenseConfig:
    """Convert raw nested dictionary to NeuroSenseConfig dataclass."""
    paths_dict = d.get("paths", {})
    data_dict = d.get("data", {})
    model_dict = d.get("model", {})
    context_dict = d.get("context", {})
    loss_dict = d.get("loss", {})
    training_dict = d.get("training", {})
    eval_dict = d.get("evaluation", {})

    config = NeuroSenseConfig(
        project_name=d.get("project_name", "NeuroSense-Net"),
        seed=d.get("seed", 42),
        deterministic=d.get("deterministic", True),
        paths=PathsConfig(**paths_dict),
        data=DataConfig(**data_dict),
        context=ContextConfig(**context_dict),
        model=ModelConfig(**model_dict),
        loss=LossConfig(**loss_dict),
        training=TrainingConfig(**training_dict),
        evaluation=EvaluationConfig(**eval_dict),
    )
    config.validate()
    return config


def config_to_dict(config: NeuroSenseConfig) -> Dict[str, Any]:
    """Convert NeuroSenseConfig to standard python nested dictionary."""
    return {
        "project_name": config.project_name,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "paths": {
            "data_root": config.paths.data_root,
            "output_root": config.paths.output_root,
            "checkpoint_root": config.paths.checkpoint_root,
        },
        "data": {
            "use_synthetic": config.data.use_synthetic,
            "synthetic_train_size": config.data.synthetic_train_size,
            "synthetic_val_size": config.data.synthetic_val_size,
            "synthetic_test_size": config.data.synthetic_test_size,
            "num_workers": config.data.num_workers,
            "batch_size": config.data.batch_size,
            "normalize": config.data.normalize,
            "modality_key_overrides": config.data.modality_key_overrides,
            "label_mapping": config.data.label_mapping,
        },
        "context": {
            "window": config.context.window,
            "speaker_identity": config.context.speaker_identity,
        },
        "model": {
            "hidden_dim": config.model.hidden_dim,
            "num_heads": config.model.num_heads,
            "num_fusion_layers": config.model.num_fusion_layers,
            "dropout": config.model.dropout,
            "classifier_hidden_dim": config.model.classifier_hidden_dim,
            "sequence_encoder": config.model.sequence_encoder,
            "baseline": config.model.baseline,
            "use_gate": config.model.use_gate,
            "use_cross_modal_attention": config.model.use_cross_modal_attention,
            "use_speaker_embedding": config.model.use_speaker_embedding,
        },
        "loss": {
            "name": config.loss.name,
            "gamma": config.loss.gamma,
            "class_weight_strategy": config.loss.class_weight_strategy,
            "effective_num_beta": config.loss.effective_num_beta,
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_ratio": config.training.warmup_ratio,
            "gradient_clip_norm": config.training.gradient_clip_norm,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "modality_dropout_probability": config.training.modality_dropout_probability,
            "early_stopping_patience": config.training.early_stopping_patience,
            "mixed_precision": config.training.mixed_precision,
        },
        "evaluation": {
            "primary_metric": config.evaluation.primary_metric,
            "save_predictions": config.evaluation.save_predictions,
            "run_modality_ablation": config.evaluation.run_modality_ablation,
        },
    }


def load_config(config_path: Path) -> NeuroSenseConfig:
    """Load config from YAML file path."""
    raw = load_yaml(config_path)
    return dict_to_config(raw)


def save_config(config: NeuroSenseConfig, save_path: Path) -> None:
    """Save config object to YAML file path."""
    d = config_to_dict(config)
    save_yaml(d, save_path)
