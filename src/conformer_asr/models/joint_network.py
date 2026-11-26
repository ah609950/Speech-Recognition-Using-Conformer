"""Student 2: TODO implement training-grid and single-step vocabulary logits."""

"""RNN-T joint network for combining acoustic and linguistic representations."""

from torch import nn

class JointNetwork(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        prediction_dim: int,
        vocab_size: int,
        joint_dim: int = 320
    ):
        super().__init__()

        # 1. Encoder projection
        # Project acoustic representations to the joint dimension
        # Conformer-Small: 144 -> 320
        self.encoder_projection = nn.Linear(
            encoder_dim, joint_dim
        )

        # 2. Prediction network projection
        # Project linguistic representations to the joint dimension
        # Conformer-Small: 320 -> 320
        self.prediction_projection = nn.Linear(
            prediction_dim, joint_dim
        )

        # 3. Output projection
        # Convert joint representations into vocabulary logits
        # 320 -> vocab_size (including the blank token)
        self.output = nn.Linear(
            joint_dim, vocab_size
        )

    # Training
    def forward(self, encoder, prediction):
        """
        Args:
            encoder:    (B, T, De) - acoustic representations.
            prediction: (B, U+1, Dp) - linguistic representations.

        Returns:
            logits: (B, T, U+1, V) - vocabulary logits for
                    every audio-frame and text-prefix combination.

        B  = batch size
        T  = number of subsampled audio frames
        U  = number of target tokens
        De = encoder dimension (144)
        Dp = prediction network dimension (320)
        Dj = joint network dimension (320)
        V  = vocabulary size, including blank
        """

        # 1. Project acoustic representations
        # (B, T, De) -> (B, T, Dj)
        acoustic = self.encoder_projection(encoder)

        # Add a text-prefix dimension for broadcasting
        # (B, T, Dj) -> (B, T, 1, Dj)
        acoustic = acoustic[:, :, None, :]

        # 2. Project linguistic representations
        # (B, U+1, Dp) -> (B, U+1, Dj)
        linguistic = self.prediction_projection(prediction)

        # Add an audio-time dimension for broadcasting
        # (B, U+1, Dj) -> (B, 1, U+1, Dj)
        linguistic = linguistic[:, None, :, :]

        # 3. Combine acoustic and linguistic representations
        # Broadcasting combines every audio frame with
        # every text-prefix representation
        # (B, T,   1, Dj) + (B, 1, U+1, Dj) = (B, T, U+1, Dj)
        joint = acoustic + linguistic

        # 4. Apply nonlinear activation
        joint = joint.tanh()

        # 5. Generate vocabulary logits
        # including the blank token
        # (B, T, U+1, Dj) -> (B, T, U+1, V)
        logits = self.output(joint)

        # Return raw logits for the RNN-T loss
        return logits

    # Inference: Process one audio frame and one text prefix
    def step(self, encoder, prediction):
        """
        Args:
            encoder:    (B, De) - current audio frame.
            prediction: (B, Dp) - current token history.

        Returns:
            logits: (B, V) - next-token vocabulary logits.
        """

        # 1. Project the current acoustic representation
        # (B, De) -> (B, Dj)
        acoustic = self.encoder_projection(encoder)

        # 2. Project the current linguistic representation
        # (B, Dp) -> (B, Dj)
        linguistic = self.prediction_projection(prediction)

        # 3. Combine both representations
        # (B, Dj) + (B, Dj) -> (B, Dj)
        joint = acoustic + linguistic

        # 4. Apply nonlinear activation
        joint = joint.tanh()

        # 5. Generate next-token logits
        # (B, Dj) -> (B, V)
        logits = self.output(joint)

        return logits