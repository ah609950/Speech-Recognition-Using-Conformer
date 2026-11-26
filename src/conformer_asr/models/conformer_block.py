"""Student 2: TODO assemble the two half-step FFNs, attention, convolution and normalization."""

from torch import nn
from .attention import RelativeMultiHeadSelfAttention
from .convolution import ConvolutionModule
from .feed_forward import FeedForwardModule

class ConformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1. First feed-forward module
        # Expand and project back: D -> 4D -> D
        # Conformer-Small: 144 -> 576 -> 144
        self.ffn1 = FeedForwardModule(
            dim, ffn_expansion, dropout
        )

        # 2. Relative-position multi-head self-attention
        # Capture global relationships between time frames
        self.attention = RelativeMultiHeadSelfAttention(
            dim, num_heads, dropout
        )

        # 3. Convolution module
        # Capture local temporal relationships
        self.convolution = ConvolutionModule(
            dim, conv_kernel_size, dropout
        )

        # 4. Second feed-forward module
        # Same architecture as FFN1
        # D -> 4D -> D
        self.ffn2 = FeedForwardModule(
            dim, ffn_expansion, dropout
        )

        # 5. Final layer normalization
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T, D) - input acoustic representations.

            padding_mask: (B, T) - boolean padding mask.
                True  = padded frame.
                False = valid frame.

        Returns:
            x: (B, T, D) - Conformer block output.

        B = batch size
        T = number of subsampled time frames
        D = encoder dimension (144)
        """

        # 1. First feed-forward module
        # Half-step residual connection
        # Scale only the FFN output by 0.5
        # (B, T, D) -> (B, T, D)
        x = x + 0.5 * self.ffn1(x)

        # 2. Multi-head self-attention module
        # Capture global temporal dependencies
        # Full-step residual connection
        # Padding mask prevents attention to padded keys
        # (B, T, D) -> (B, T, D)
        x = x + self.attention(x, padding_mask)

        # 3. Convolution module
        # Capture local temporal dependencies
        # Full-step residual connection
        # (B, T, D) -> (B, T, D)
        x = x + self.convolution(x, padding_mask)

        # 4. Second feed-forward module
        # Refine features after attention and convolution
        # Half-step residual connection
        # (B, T, D) -> (B, T, D)
        x = x + 0.5 * self.ffn2(x)

        # 5. Final layer normalization
        # Normalize the final feature representation
        # (B, T, D) -> (B, T, D)
        x = self.final_norm(x)

        # 6. Apply padding mask
        # Set all features of padded time frames to zero
        # Mask: (B, T) -> (B, T, 1)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        return x