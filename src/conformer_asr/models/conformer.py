"""Student 2: TODO assemble encoder, predictor and joint network"""

"""Complete Conformer Small RNN-Transducer."""

from torch import nn
from .encoder import ConformerEncoder
from .joint_network import JointNetwork
from .prediction_network import PredictionNetwork

class ConformerTransducer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        *,
        input_dim=80,
        num_blocks=16,
        encoder_dim=144,
        attention_heads=4,
        conv_kernel_size=32,
        ffn_expansion=4,
        prediction_layers=1,
        prediction_dim=320,
        joint_dim=320,
        residual_dropout=0.1,
        blank_id=0
    ):
        super().__init__()

        # Store the RNN-T blank token ID
        self.blank_id = blank_id

        # 1. Conformer encoder
        # Extract acoustic representations from audio features
        # Includes convolutional subsampling and 16 Conformer blocks
        # (B, T, 80) -> (B, T', 144)
        self.encoder = ConformerEncoder(
            input_dim,
            encoder_dim,
            num_blocks,
            attention_heads,
            ffn_expansion,
            conv_kernel_size,
            residual_dropout
        )

        # 2. Prediction network
        # Extract linguistic representations from previous tokens
        # Uses a single-layer unidirectional LSTM
        # (B, U) -> (B, U+1, 320)
        self.prediction = PredictionNetwork(
            vocab_size,
            prediction_dim,
            prediction_layers,
            blank_id
        )

        # 3. Joint network
        # Combine acoustic and linguistic representations
        # Project both representations to the joint dimension
        # Produce vocabulary logits for each audio-frame/text-prefix pair
        # (B, T', U+1, V)
        self.joint = JointNetwork(
            encoder_dim,
            prediction_dim,
            vocab_size,
            joint_dim
        )

    def encode(self, features, feature_lengths):
        """
        Args:
            features:        (B, T, F) - input acoustic features.
            feature_lengths: (B,)      - actual audio frame lengths.

        Returns:
            encoded:         (B, T', De) - acoustic representations.
            encoded_lengths: (B,)        - subsampled sequence lengths.
        """

        # Pass audio features through the Conformer encoder
        # Also return the updated valid sequence lengths
        return self.encoder(features, feature_lengths)

    def forward(
        self,
        features,
        feature_lengths,
        tokens,
        token_lengths=None
    ):
        """
        Args:
            features:        (B, T, F) - input acoustic features.
            feature_lengths: (B,)      - actual audio frame lengths.
            tokens:          (B, U)    - ground-truth token IDs.
            token_lengths:   (B,)      - actual target sequence lengths.
                                        Accepted but not used here.

        Returns:
            logits:          (B, T', U+1, V) - raw vocabulary logits.
            encoded_lengths: (B,)            - updated audio lengths.

        B  = batch size
        T  = original number of audio frames
        T' = subsampled number of audio frames
        F  = input acoustic feature dimension (80)
        U  = padded number of target tokens
        De = encoder dimension (144)
        Dp = prediction network dimension (320)
        V  = vocabulary size, including blank
        """

        # 1. Acoustic encoding
        # Extract acoustic representations from input audio
        # (B, T, F) -> (B, T', De)
        #
        # Update audio lengths after convolutional subsampling
        encoded, encoded_lengths = self.encode(
            features, feature_lengths
        )

        # 2. Linguistic prediction
        # Process the ground-truth tokens using teacher forcing
        # The prediction network prepends the initial blank token
        #
        # (B, U) -> (B, U+1, Dp)
        predicted = self.prediction(tokens)

        # 3. Joint network
        # Combine every encoded audio frame with every prediction network representation
        # Encoder:   (B, T', De)
        # Predictor: (B, U+1, Dp)
        # Logits:    (B, T', U+1, V)
        logits = self.joint(encoded, predicted)

        # 4. Return outputs for RNN-T loss
        return logits, encoded_lengths

    @classmethod
    def from_config(cls, config, vocab_size, blank_id=0):
        """
        Create a ConformerTransducer model from a configuration file.
        """
        return cls(
            vocab_size=vocab_size,
            blank_id=blank_id,
            **config
        )
