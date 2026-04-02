import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class ResidualTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=0.2):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x,mode=None):
        x2, _ = self.attn(x, x, x)
        x = x + x2
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + x2
        x = self.norm2(x)
        return x


