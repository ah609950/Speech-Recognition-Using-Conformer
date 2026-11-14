import unittest
from conformer_asr.config import load_config


class TestTrainingConfig(unittest.TestCase):
    def test_paper_adam_settings(self):
        cfg = load_config("configs/training.yaml")
        self.assertEqual(cfg["adam_betas"], [0.9, 0.98])
        self.assertEqual(cfg["adam_eps"], 1e-9)
        self.assertEqual(cfg["warmup_steps"], 10000)

    # TODO: check a known loss example, finite gradients and checkpoint resume.
