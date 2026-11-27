"""Student 2: TODO implement the position-wise feed-forward module."""

from torch import nn

class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), # Layer normalization for stable training
            nn.Linear(dim, dim * expansion), # (B, T, D) -> (B, T, D * expansion)
            nn.SiLU(), # Swish activation function
            nn.Dropout(dropout), # for regularization
            nn.Linear(dim * expansion, dim), # (B, T, D * expansion) -> (B, T, D)
            nn.Dropout(dropout), # for regularization
        )

    def forward(self, x):
        return self.net(x)