"""Student 2: TODO implement the single-layer LSTM with empty-prefix and cached step interfaces."""

import torch
from torch import nn

class PredictionNetwork(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        dim: int = 320,
        num_layers: int = 1,
        blank_id: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Blank token ID, also used as the initial start symbol
        self.blank_id = blank_id

        # 1. Token embedding
        # Convert each discrete token ID into a learnable vector
        # Token ID -> D-dimensional embedding (320)
        self.embedding = nn.Embedding(vocab_size, dim)

        # 2. Unidirectional LSTM
        # Capture dependencies between previously generated tokens
        # Conformer-Small uses one LSTM layer with hidden size 320
        # Input dimension:  320
        # Hidden dimension: 320
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, targets):
        """
        Args:
            targets: (B, U) - ground-truth token IDs.

        Returns:
            outputs: (B, U+1, D) - prediction representations.

        B = batch size
        U = number of target tokens (including batch padding)
        D = prediction network dimension (320)
        """

        # 1. Create the initial start token for each recording
        # Shape: (B, 1)
        start = torch.full(
            (targets.shape[0], 1),
            self.blank_id,
            dtype=targets.dtype,
            device=targets.device
        )

        # 2. Prepend the start token to the target sequence
        # (B, U) -> (B, U+1)
        inputs = torch.cat((start, targets), dim=1)

        # 3. Convert token IDs into embedding vectors
        # (B, U+1) -> (B, U+1, D)
        embedded = self.embedding(inputs)

        # 4. Process the complete sequence using the LSTM
        # Each output represents the token history up to that point
        # (B, U+1, D) -> (B, U+1, D)
        outputs, _ = self.lstm(embedded)

        return outputs

    # Inference: Initialize the empty token history
    def initial(self, batch_size: int, device):
        """
        Returns:
            output: (B, D) - empty-history representation.
            state:  (h, c) - initial cached LSTM states.
        """

        # 1. Create the initial blank/start token
        # Shape: (B, 1)
        token = torch.full(
            (batch_size, 1),
            self.blank_id,
            dtype=torch.long,
            device=device
        )

        # 2. Convert the initial token into an embedding
        # (B, 1) -> (B, 1, D)
        embedded = self.embedding(token)

        # 3. Process the initial token using the LSTM
        # Output: (B, 1, D)
        # Hidden state h: (num_layers, B, D)
        # Cell state c:   (num_layers, B, D)
        output, state = self.lstm(embedded)

        # 4. Remove the single-token sequence dimension
        # (B, 1, D) -> (B, D)
        return output[:, 0], state

    # Inference step: Update the token history with a new token
    def step(self, token, state):
        """
        Args:
            token: (B,) or (B, 1) - newly generated token IDs.
            state: (h, c) - previously cached LSTM states.

        Returns:
            output:    (B, D) - updated token-history representation.
            new_state: (h, c) - updated cached LSTM states.
        """

        # 1. Ensure the token has a sequence dimension
        # (B,) -> (B, 1)
        if token.ndim == 1:
            token = token[:, None]

        # 2. Convert the new token into an embedding
        # (B, 1) -> (B, 1, D)
        embedded = self.embedding(token)

        # 3. Process only the new token using the previous LSTM state
        # Output: (B, 1, D)
        # New state: updated hidden and cell states
        output, new_state = self.lstm(embedded, state)

        # 4. Remove the single-token sequence dimension
        # (B, 1, D) -> (B, D)
        return output[:, 0], new_state
