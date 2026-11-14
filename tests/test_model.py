import unittest
from conformer_asr.config import load_config


class TestModelConfig(unittest.TestCase):
    def test_small_dimensions(self):
        cfg = load_config("configs/model.yaml")
        self.assertEqual((cfg["num_blocks"], cfg["encoder_dim"],
                          cfg["attention_heads"], cfg["conv_kernel_size"],
                          cfg["prediction_layers"], cfg["prediction_dim"]),
                         (16, 144, 4, 32, 1, 320))

    # TODO: add a variable-length forward pass and verify joiner shape.
