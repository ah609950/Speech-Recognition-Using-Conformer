import unittest

from conformer_asr.evaluation.metrics import (corpus_error_rates, edit_counts,
                                              edit_distance, normalize)

class TestMetrics(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("THE CAT SAT".split(), "THE CAT SIT".split()), 1)
        self.assertEqual(edit_distance("KITTEN", "SITTING"), 3)
        self.assertEqual(edit_distance([], ["A", "B"]), 2)

    def test_error_breakdown(self):
        self.assertEqual(edit_counts("A B C".split(), "A C".split()),
                        {"substitutions": 0, "deletions": 1, "insertions": 0})
        self.assertEqual(edit_counts("A C".split(), "A B C".split()),
                        {"substitutions": 0, "deletions": 0, "insertions": 1})
        self.assertEqual(edit_counts("A B C D".split(), "A X C D E".split()),
                        {"substitutions": 1, "deletions": 0, "insertions": 1})

    def test_guide_example(self):
        self.assertAlmostEqual(corpus_error_rates([("THE CAT SAT", "THE CAT SIT")])["wer"], 1 / 3)

    def test_corpus_wer_is_not_mean_of_utterance(self):
        self.assertEqual(corpus_error_rates([("THE CAT SAT", "THE CAT SIT")])["wer"], 1 / 3)

    def test_wer_can_exceed_one(self):
        self.assertEqual(corpus_error_rates([("A", "B C D")])["wer"], 3.0)

    def test_normalize(self):
        self.assertEqual(normalize("  it's over. "), "IT'S OVER")
        self.assertEqual(normalize("IT\u2019S OVER"), "IT'S OVER")
        self.assertEqual(normalize("well-known"), "WELL KNOWN")
        self.assertEqual(corpus_error_rates([("IT'S OVER", "it's over.")])["wer"], 0.0)

    def test_cer_spaces(self):
        self.assertEqual(corpus_error_rates([("AB CD", "ABCD")])["cer"], 0.0)
        self.assertEqual(corpus_error_rates([("AB CD", "ABCD")], include_spaces=True)["cer"], 0.2)

    def test_empty_reference(self):
        self.assertIsNone(corpus_error_rates([("", "A")])["wer"])