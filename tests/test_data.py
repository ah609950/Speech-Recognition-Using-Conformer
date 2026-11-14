import unittest
from conformer_asr.config import load_config


class TestDataConfig(unittest.TestCase):
    def test_feature_contract(self):
        config = load_config("configs/data.yaml")
        self.assertEqual(config["sample_rate"], 16000)
        self.assertEqual(config["num_mel_bins"], 80)
        self.assertFalse(set(config["train_splits"]) & set(config["test_splits"]))

    # TODO: add a variable-length collation and tokenizer round-trip test.
