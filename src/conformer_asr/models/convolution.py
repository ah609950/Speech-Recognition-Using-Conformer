"""Student 2: TODO implement gated depthwise temporal convolution with length preservation."""

import torch.nn.functional as F
from torch import nn

class ConvolutionModule(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.kernel_size = kernel_size

        # 1. Pre-normalization across the feature dimension
        self.norm = nn.LayerNorm(dim)

        # 2. First pointwise convolution
        # Capture channel-wise dependencies independently at each time frame
        # No temporal mixing
        # D -> 2D (144 -> 288)
        self.pointwise_in = nn.Conv1d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=1
        )

        # 3. Depthwise convolution
        # Capture temporal relationships across 32 frames for each channel independently
        # groups=dim prevents mixing different channels
        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim
        )

        # 4. Batch normalization after depthwise convolution
        self.batch_norm = nn.BatchNorm1d(dim)

        # 5. Swish activation (SiLU is equivalent to Swish with beta=1)
        self.activation = nn.SiLU()

        # 6. Second pointwise convolution
        # Mix channels after extracting temporal features
        # D -> D (144 -> 144)
        self.pointwise_out = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )

        # 7. Output dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T, D) - input acoustic representations.

            padding_mask: (B, T) - boolean padding mask.
                True  = padded frame.
                False = valid frame.

        Returns:
            y: (B, T, D) - convolution module output.

        B = batch size
        T = number of subsampled time frames
        D = encoder dimension (144)
        """

        # 1. Layer normalization and transpose
        # Normalize the feature dimension of each time frame
        # (B, T, D) -> (B, T, D)
        y = self.norm(x)

        # Conv1d expects channels before the time dimension
        # (B, T, D) -> (B, D, T)
        y = y.transpose(1, 2)

        # 2. First pointwise convolution
        # Mix channels independently at each time frame
        # Double the number of channels
        # (B, D, T) -> (B, 2D, T)
        y = self.pointwise_in(y)

        # 3. Gated Linear Unit (GLU)
        # Split the channels into two equal parts: A and B
        # GLU(A, B) = A * sigmoid(B)
        # B controls how much information from A passes through
        # (B, 2D, T) -> (B, D, T)
        y = F.glu(y, dim=1)

        # 4. Calculate padding for depthwise convolution
        # Preserve the original number of time frames
        # Total required padding = kernel_size - 1
        # Asymmetric padding is needed because the kernel is even
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left

        # Pad along the time dimension
        # (B, D, T) -> (B, D, T + kernel_size - 1)
        y = F.pad(y, (left, right))

        # 5. Depthwise convolution
        # Capture local temporal relationships across 32 frames
        # Each channel has its own independent convolution filter
        # (B, D, T + kernel_size - 1) -> (B, D, T)
        y = self.depthwise(y)

        # 6. Batch normalization
        # Normalize each channel across batch and time dimensions
        # during training
        # (B, D, T) -> (B, D, T)
        y = self.batch_norm(y)

        # 7. Swish activation
        # Swish(x) = x * sigmoid(x)
        # (B, D, T) -> (B, D, T)
        y = self.activation(y)

        # 8. Second pointwise convolution
        # Mix information across channels after temporal processing
        # (B, D, T) -> (B, D, T)
        y = self.pointwise_out(y)

        # Restore the original tensor layout
        # (B, D, T) -> (B, T, D)
        y = y.transpose(1, 2)

        # 9. Output dropout
        y = self.dropout(y)

        # 10. Apply padding mask
        # Set the output features of padded time frames to zero
        # Mask: (B, T) -> (B, T, 1)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0)

        return y