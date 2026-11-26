"""Student 2: TODO implement relative-position multi-head self-attention and padding masks."""

from __future__ import annotations
import math
import torch
from torch import nn

def _sinusoidal_relative_positions(length: int, dim: int, device, dtype):
    """Generate sinusoidal embeddings for all possible relative positions."""

    # Relative positions range from -(T-1) to +(T-1)
    # Shape: (2T-1,)
    positions = torch.arange(
        -(length - 1), length,
        device=device,
        dtype=torch.float32
    )

    # Calculate sinusoidal frequency scaling factors
    # Shape: (ceil(D/2),)
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dim)
    )

    # Initialize the relative positional embedding matrix
    # Shape: (2T-1, D)
    result = torch.zeros(
        2 * length - 1, dim,
        device=device,
        dtype=torch.float32
    )

    # Apply sine to even dimensions
    result[:, 0::2] = torch.sin(positions[:, None] * scale)

    # Apply cosine to odd dimensions
    result[:, 1::2] = torch.cos(
        positions[:, None] * scale[:result[:, 1::2].shape[1]]
    )

    return result.to(dtype=dtype)


class RelativeMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Ensure that all attention heads have equal dimensions
        if dim % num_heads:
            raise ValueError(
                "encoder dimension must be divisible by attention heads"
            )

        # Attention dimensions
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 1. Pre-normalization
        # Use layer normalization before attention to stabilize training
        self.norm = nn.LayerNorm(dim)

        # 2. Query, key, and value projections
        # Usually we use 3 separate linear layers for Q, K, V, but we can combine them into one for efficiency
        self.qkv = nn.Linear(dim, 3 * dim)

        # 3. Relative positional embedding projection
        self.position = nn.Linear(dim, dim, bias=False)

        # 4. Final output projection
        self.out = nn.Linear(dim, dim)

        # 5. Trainable relative attention biases
        # Shape: (H, Dh)
        # Global content bias (u): Which key features are generally important, regardless of the query?
        self.pos_bias_u = nn.Parameter(
            torch.empty(num_heads, self.head_dim)
        )
        # Global positional bias (v): Which relative distances are generally important, regardless of the query?
        self.pos_bias_v = nn.Parameter(
            torch.empty(num_heads, self.head_dim)
        )

        # 6. Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Initialize trainable attention biases
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, T, D) - input acoustic representations.

            key_padding_mask: (B, T) - boolean padding mask.
                True  = padded frame.
                False = valid frame.

        Returns:
            output: (B, T, D) - attention output.

        B  = batch size
        T  = number of subsampled time frames
        D  = encoder dimension (144)
        H  = number of attention heads (4)
        Dh = dimension per head (36)
        """

        batch, length, _ = x.shape

        # 1. Layer normalization
        # (B, T, D) -> (B, T, D)
        normalized = self.norm(x)

        # 2. Generate queries, keys, and values
        # Linear projection: (B, T, D) -> (B, T, 3D)
        # Split into Q, K, V, each with shape (B, T, D)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)

        # Split the feature dimension into multiple heads
        # (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh)
        reshape = lambda z: z.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q, k, v = map(reshape, (q, k, v))

        # 3. Calculate content-based attention scores
        # Calculate (Q + u) @ K^T
        # (B, H, T, Dh) @ (B, H, Dh, T)
        # Output: (B, H, T, T)
        content = torch.matmul(
            q + self.pos_bias_u[None, :, None],
            k.transpose(-2, -1)
        )

        # 4. Generate relative positional embeddings
        # Generate embeddings for all possible relative distances
        # Shape: (2T-1, D)
        positions = _sinusoidal_relative_positions(
            length,
            self.dim,
            x.device,
            normalized.dtype
        )

        # Project positional embeddings and split into heads
        # (2T-1, D) -> (2T-1, H, Dh) -> (H, 2T-1, Dh)
        positions = self.position(positions).view(
            2 * length - 1,
            self.num_heads,
            self.head_dim
        ).permute(1, 0, 2)

        # 5. Calculate relative position attention scores
        # Compare every query against all relative positions
        # Calculate (Q + v_bias) @ R^T
        # Output: (B, H, T, 2T-1)
        relative_all = torch.einsum(
            "bhtd,hrd->bhtr",
            q + self.pos_bias_v[None, :, None],
            positions
        )

        # Generate query and key position indices
        query = torch.arange(length, device=x.device)[:, None]
        key = torch.arange(length, device=x.device)[None, :]

        # Calculate the relative distance between each query-key pair
        # Add T-1 to convert negative distances into valid indices
        # (T, T) -> (B, H, T, T)
        relative_index = (
            key - query + length - 1
        )[None, None].expand(
            batch, self.num_heads, -1, -1
        )

        # Select the correct relative position score for each pair
        # (B, H, T, 2T-1) -> (B, H, T, T)
        relative = relative_all.gather(-1, relative_index)

        # 6. Combine and scale attention scores
        # Combine content-based and position-based attention
        # Scale by sqrt(Dh) for numerical stability
        # Shape: (B, H, T, T)
        scores = (content + relative) / math.sqrt(self.head_dim)

        # 7. Apply padding mask
        # Prevent valid queries from attending to padded keys
        # Mask: (B, T) -> (B, 1, 1, T)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(scores.dtype).min
            )

        # 8. Calculate attention weights
        # Apply softmax across the key dimension
        # Use float32 for numerical stability
        # Shape: (B, H, T, T)
        weights = torch.softmax(
            scores.float(), dim=-1
        ).to(dtype=scores.dtype)

        # Regularize attention weights
        weights = self.attention_dropout(weights)

        # 9. Apply attention weights to values
        # (B, H, T, T) @ (B, H, T, Dh)
        # Output: (B, H, T, Dh)
        output = torch.matmul(weights, v)

        # Concatenate all attention heads
        # (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, D)
        output = output.transpose(1, 2).contiguous().view(
            batch, length, self.dim
        )

        # 10. Output projection and dropout
        # (B, T, D) -> (B, T, D)
        output = self.output_dropout(self.out(output))

        # Set outputs corresponding to padded queries to zero
        if key_padding_mask is not None:
            output = output.masked_fill(
                key_padding_mask.unsqueeze(-1), 0
            )

        return output