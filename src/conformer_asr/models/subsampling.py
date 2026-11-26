"""Student 2: TODO implement acoustic subsampling and exact encoder lengths."""

import torch
from torch import nn


def conv_output_length(lengths, kernel_size=3, stride=2):
    """Calculate output lengths after a convolution with no padding."""
    return torch.div(
        lengths - kernel_size,
        stride,
        rounding_mode="floor"
    ) + 1


class Conv2dSubsampling(nn.Module):

    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 144,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1. Convolutional subsampling (approximately 4x reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=3, stride=2), # (Batch, 1, T, Feature_dim) -> (Batch, output_dim, T/2, Feature_dim/2)
            nn.ReLU(),

            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2), # (Batch, output_dim, T/2, Feature_dim/2) -> (Batch, output_dim, T/4, Feature_dim/4)
            nn.ReLU(),
        )

        # 2. Calculate feature dimension after both convolutions
        feature_dim = (input_dim - 3) // 2 + 1  # 80 -> 39
        feature_dim = (feature_dim - 3) // 2 + 1  # 39 -> 19

        # 3. Project flattened features to the encoder dimension
        self.projection = nn.Sequential(
            nn.Linear(output_dim * feature_dim, output_dim), # (Batch, T/4, output_dim * feature_dim) -> (Batch, T/4, output_dim)
            nn.Dropout(dropout), # for regularization
        )

    def forward(self, features, lengths):
        """
        Args:
            features: (B, T, Feature_dim) - input acoustic features.
            lengths:  (B,)              - original audio frame lengths.

        Returns:
            x:       (B, T/4, D) - subsampled encoder features.
            lengths: (B,)       - updated audio frame lengths.
        """

        # 1. Add CNN channel dimension
        # (B, T, Feature_dim) -> (B, 1, T, Feature_dim)
        x = features.unsqueeze(1)

        # 2. Apply two convolutional layers
        # (B, 1, T, Feature_dim) -> (B, D, T/4, Feature_dim/4)
        x = self.conv(x)

        # 3. Extract output dimensions
        batch, channels, time, feature_dim = x.shape

        # 4. Move the time dimension before the channel dimension
        # (B, D, T/4, Feature_dim/4) -> (B, T/4, D, Feature_dim/4)
        x = x.transpose(1, 2).contiguous()

        # 5. Flatten the channel and frequency dimensions
        # (B, T/4, D, Feature_dim/4) -> (B, T/4, D * Feature_dim/4)
        x = x.view(batch, time, channels * feature_dim)

        # 6. Project to the encoder dimension and apply dropout
        # (B, T/4, D * Feature_dim/4) -> (B, T/4, D)
        x = self.projection(x)

        # 7. Calculate exact sequence lengths after both convolutions
        length_after_conv1 = conv_output_length(lengths)
        length_after_conv2 = conv_output_length(length_after_conv1)
        
        # 8. Clamp lengths to ensure they are non-negative
        lengths = length_after_conv2.clamp_min(0)

        return x, lengths