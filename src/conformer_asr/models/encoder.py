"""Student 2: TODO stack the subsampling front end and 16 Conformer blocks."""

import torch
from torch import nn
from .conformer_block import ConformerBlock
from .subsampling import Conv2dSubsampling

class ConformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim=80,
        dim=144,
        num_blocks=16,
        num_heads=4,
        ffn_expansion=4,
        conv_kernel_size=32,
        dropout=0.1
    ):
        super().__init__()

        # 1. Convolutional subsampling
        # Reduce the time dimension by approximately 4x
        # Project acoustic features from input_dim to dim (80 -> 144)
        self.subsampling = Conv2dSubsampling(
            input_dim, dim, dropout
        )

        # 2. Stack Conformer blocks
        # Conformer-Small uses 16 blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim,
                num_heads,
                ffn_expansion,
                conv_kernel_size,
                dropout
            )
            for _ in range(num_blocks)
        ])

    def forward(self, features, lengths):
        """
        Args:
            features: (B, T, F) - input acoustic features.
            lengths:  (B,)      - actual audio frame lengths.

        Returns:
            x:       (B, T', D) - encoded acoustic representations.
            lengths: (B,)       - updated audio frame lengths.

        B  = batch size
        T  = original number of time frames
        T' = number of subsampled time frames
        F  = input feature dimension (80)
        D  = encoder dimension (144)
        """

        # 1. Convolutional subsampling
        # Reduce the time dimension and project the features
        # (B, T, F) -> (B, T', D)
        x, lengths = self.subsampling(features, lengths)

        # 2. Generate padding mask
        # Generate all time frame indices: [0, 1, ..., T'-1]
        # Time indices: (1, T')
        # Actual lengths: (B, 1)
        # True  = frame index >= actual sequence length (padding)
        # False = frame index < actual sequence length (valid)
        # Output mask shape: (B, T')
        padding_mask = (
            torch.arange(x.shape[1], device=x.device)[None]
            >= lengths[:, None]
        )

        # 3. Zero out padded input features
        # Expand mask to broadcast across the feature dimension
        # Mask: (B, T') -> (B, T', 1)
        # Set all features of padded time frames to zero
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 4. Pass through all Conformer blocks
        # Apply the 16 Conformer blocks sequentially
        # Each block receives the output of the previous block
        # (B, T', D) -> (B, T', D)
        for block in self.blocks:
            x = block(x, padding_mask)

        # 5. Return encoder output and valid sequence lengths
        
        # Encoder representations are passed to the RNN-T joint network
        # Sequence lengths are required for RNN-T loss and decoding
        return x, lengths