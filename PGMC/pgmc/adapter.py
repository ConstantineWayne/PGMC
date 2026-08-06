"""Feature reconstruction adapter used by PGMC."""

import torch
from torch import nn


class PGMCFeatureAdapter(nn.Module):
    """Residual Transformer with an explicit one-token sequence dimension."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(1)
        attended, _ = self.attention(features, features, features, need_weights=False)
        features = self.norm1(features + attended)
        features = self.norm2(features + self.feed_forward(features))
        return features.squeeze(1) if squeeze else features

