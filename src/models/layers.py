import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjection(nn.Module):
    """Projects dynamic raw modality feature vectors into a common hidden dimension."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceEncoder(nn.Module):
    """Encodes variable sequence lengths [B, T, D] into a fixed vector token [B, D]."""

    def __init__(self, hidden_dim: int = 256, encoder_type: str = "attention_pooling", dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        if encoder_type == "attention_pooling":
            self.query = nn.Parameter(torch.randn(hidden_dim, 1))
            nn.init.xavier_uniform_(self.query)
        elif encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.query = nn.Parameter(torch.randn(hidden_dim, 1))
            nn.init.xavier_uniform_(self.query)
        else:
            # Mean pooling fallback
            pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Args:

        x: [B, T, hidden_dim] or [B, hidden_dim]
        mask: [B, T] (True = valid, False = padded)
        """
        if x.ndim == 2:
            return x

        batch_size, seq_len, _ = x.shape

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)

        if self.encoder_type == "transformer":
            # Transformer expects key_padding_mask where True indicates padded tokens
            key_padding_mask = ~mask
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if hasattr(self, "query"):
            # Attention pooling: score = softmax(X * query)
            # shape: [B, T, D] x [D, 1] -> [B, T, 1]
            attn_logits = torch.matmul(x, self.query).squeeze(-1) / math.sqrt(self.hidden_dim)
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, T, 1]
            pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
            return pooled

        # Masked mean pooling default
        mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
        sum_masked = torch.sum(x * mask_expanded, dim=1)
        cnt_masked = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        return sum_masked / cnt_masked
