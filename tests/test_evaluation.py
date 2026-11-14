import unittest
from conformer_asr.evaluation.metrics import corpus_error_rates, edit_distance


class TestMetrics(unittest.TestCase):
    def test_edits(self):
        self.assertEqual(edit_distance("CAT", "BAT"), 1)
        self.assertEqual(edit_distance([], ["A", "B"]), 2)

    def test_corpus_weighting(self):
        result = corpus_error_rates([("a", ""), ("b c d", "b c d")])
        self.assertEqual(result["wer"], 0.25)
        self.assertEqual(result["cer"], 0.25)

    def test_empty_reference_and_normalization(self):
        self.assertIsNone(corpus_error_rates([("", "A")])["wer"])
        self.assertEqual(corpus_error_rates([("a   b", "A B")])["wer"], 0)

    # TODO: test blank transitions and predictor state in greedy decoding.
