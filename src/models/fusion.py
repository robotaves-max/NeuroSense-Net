from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Transformer self-attention block over text, audio, and video modality tokens."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Modality type embeddings: 0=Text, 1=Audio, 2=Video
        self.modality_type_embeddings = nn.Parameter(torch.randn(3, hidden_dim))
        nn.init.normal_(self.modality_type_embeddings, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, modality_tokens: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
        """Args:

        modality_tokens: [B, 3, hidden_dim]
        availability_mask: [B, 3] (True = available, False = missing)

        Returns:
            contextualized_tokens: [B, 3, hidden_dim]
        """
        # Add modality type embeddings
        tokens = modality_tokens + self.modality_type_embeddings.unsqueeze(0)  # [B, 3, hidden_dim]

        # PyTorch Transformer key_padding_mask: True means element SHOULD be ignored (missing)
        key_padding_mask = ~availability_mask  # [B, 3]

        # Handle edge case: if all 3 modalities are missing (should not happen), unmask the first token to prevent NaN
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing, 0] = False

        contextualized = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        # Replace outputs of missing modalities with zeros to avoid propagating garbage
        mask_expanded = availability_mask.unsqueeze(-1).float()
        return contextualized * mask_expanded


class ReliabilityAdaptiveGate(nn.Module):
    """Computes sample-wise modality confidence weights summing to 1 over available modalities."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, contextualized_tokens: torch.Tensor, availability_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        contextualized_tokens: [B, 3, hidden_dim]
        availability_mask: [B, 3] (True = available, False = missing)

        Returns:
            fused_embedding: [B, hidden_dim]
            gate_weights: [B, 3] (non-negative, sums to 1.0 over available modalities)
        """
        batch_size, num_mods, hidden_dim = contextualized_tokens.shape

        # Compute unnormalized logit score per modality: [B, 3, 1] -> [B, 3]
        raw_logits = self.gate_mlp(contextualized_tokens).squeeze(-1)

        # Mask out unavailable modalities with negative infinity before softmax
        masked_logits = raw_logits.masked_fill(~availability_mask, -1e9)

        # Numerically stable softmax
        gate_weights = F.softmax(masked_logits, dim=-1)

        # Zero out weights for missing modalities explicitly
        gate_weights = gate_weights * availability_mask.float()

        # Re-normalize to guarantee sum == 1.0 across available modalities
        weight_sum = gate_weights.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        gate_weights = gate_weights / weight_sum

        # Calculate fused representation: weighted sum across available modalities
        # [B, 3, 1] * [B, 3, hidden_dim] -> sum over dim 1 -> [B, hidden_dim]
        fused_embedding = torch.sum(gate_weights.unsqueeze(-1) * contextualized_tokens, dim=1)

        return fused_embedding, gate_weights
