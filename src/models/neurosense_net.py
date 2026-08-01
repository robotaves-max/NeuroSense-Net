import random
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.fusion import CrossModalAttention, ReliabilityAdaptiveGate
from src.models.layers import ModalityProjection, SequenceEncoder


class NeuroSenseNet(nn.Module):
    """NeuroSense-Net: Multimodal Emotion Recognition Architecture.

    Incorporates dynamic modality projections, sequence encoders, cross-modal
    transformer self-attention, reliability-aware adaptive gating, modality dropout,
    and missing-modality handling.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.20,
        classifier_hidden_dim: int = 256,
        sequence_encoder_type: str = "attention_pooling",
        modality_dropout_prob: float = 0.10,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.modality_dropout_prob = modality_dropout_prob

        # Projection heads per modality
        self.text_proj = ModalityProjection(feature_dims["text"], hidden_dim, dropout)
        self.audio_proj = ModalityProjection(feature_dims["audio"], hidden_dim, dropout)
        self.video_proj = ModalityProjection(feature_dims["video"], hidden_dim, dropout)

        # Sequence encoders per modality
        self.text_seq = SequenceEncoder(hidden_dim, sequence_encoder_type, dropout)
        self.audio_seq = SequenceEncoder(hidden_dim, sequence_encoder_type, dropout)
        self.video_seq = SequenceEncoder(hidden_dim, sequence_encoder_type, dropout)

        # Cross-modal Transformer attention fusion
        self.cross_modal_fusion = CrossModalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
        )

        # Reliability adaptive gating
        self.adaptive_gate = ReliabilityAdaptiveGate(hidden_dim=hidden_dim, dropout=dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def apply_modality_dropout(
        self, availability_mask: torch.Tensor
    ) -> torch.Tensor:
        """Randomly drop available modalities during training with probability p,

        ensuring at least one modality remains active per sample.
        """
        if not self.training or self.modality_dropout_prob <= 0.0:
            return availability_mask

        batch_size = availability_mask.shape[0]
        mask = availability_mask.clone()

        for b in range(batch_size):
            available_indices = [i for i in range(3) if mask[b, i]]
            if len(available_indices) > 1:
                for idx in available_indices:
                    if random.random() < self.modality_dropout_prob:
                        # Drop this modality if another active modality remains
                        if (mask[b].sum() > 1):
                            mask[b, idx] = False

        return mask

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        override_availability: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dictionary containing text, audio, video tensors, masks, and availabilities.
            override_availability: Optional dict to force disable modalities during ablation (e.g. {'text': False})

        Returns:
            Dictionary containing 'logits', 'gate_weights', 'fused_embedding'.
        """
        device = next(self.parameters()).device

        # Extract features and masks
        text_feat = batch["text"].to(device)
        audio_feat = batch["audio"].to(device)
        video_feat = batch["video"].to(device)

        text_seq_mask = batch.get("text_mask", None)
        audio_seq_mask = batch.get("audio_mask", None)
        video_seq_mask = batch.get("video_mask", None)

        if text_seq_mask is not None:
            text_seq_mask = text_seq_mask.to(device)
        if audio_seq_mask is not None:
            audio_seq_mask = audio_seq_mask.to(device)
        if video_seq_mask is not None:
            video_seq_mask = video_seq_mask.to(device)

        text_avail = batch["text_avail"].to(device)
        audio_avail = batch["audio_avail"].to(device)
        video_avail = batch["video_avail"].to(device)

        # Apply evaluation ablation overrides if provided
        if override_availability:
            if "text" in override_availability and not override_availability["text"]:
                text_avail = torch.zeros_like(text_avail)
            if "audio" in override_availability and not override_availability["audio"]:
                audio_avail = torch.zeros_like(audio_avail)
            if "video" in override_availability and not override_availability["video"]:
                video_avail = torch.zeros_like(video_avail)

        # Build initial availability mask: [B, 3] (0: Text, 1: Audio, 2: Video)
        availability_mask = torch.stack([text_avail, audio_avail, video_avail], dim=1)

        # Apply modality dropout during training
        availability_mask = self.apply_modality_dropout(availability_mask)

        # 1. Project raw features to hidden_dim
        t_proj = self.text_proj(text_feat)
        a_proj = self.audio_proj(audio_feat)
        v_proj = self.video_proj(video_feat)

        # 2. Encode sequence dimensions to single latent token per modality [B, hidden_dim]
        t_token = self.text_seq(t_proj, text_seq_mask)
        a_token = self.audio_seq(a_proj, audio_seq_mask)
        v_token = self.video_seq(v_proj, video_seq_mask)

        # Stack modality tokens: [B, 3, hidden_dim]
        modality_tokens = torch.stack([t_token, a_token, v_token], dim=1)

        # 3. Cross-modal Transformer Self-Attention
        contextualized_tokens = self.cross_modal_fusion(modality_tokens, availability_mask)

        # 4. Reliability Adaptive Gating Fusion
        fused_embedding, gate_weights = self.adaptive_gate(contextualized_tokens, availability_mask)

        # 5. 7-Class Emotion Classification Head
        logits = self.classifier(fused_embedding)

        return {
            "logits": logits,
            "gate_weights": gate_weights,
            "fused_embedding": fused_embedding,
        }
