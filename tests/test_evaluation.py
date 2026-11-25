import unittest

from types import SimpleNamespace

import torch

from conformer_asr.evaluation.decoder import GreedyRNNTDecoder

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

class FakePredictor:
    def __init__(self):
        self.steps = [] # every token the LSTM was fed

    def initial(self, batch_size, device):
        return torch.zeros(1, 1), 0 # prediction 0 = nothing emitted yet

    def step(self, token, state):
        self.steps.append(int(token))
        return token.float().view(1, 1), state + 1 # prediction = the last token

class FakeJoint:
    def __init__(self, table):
        self.table = table # {(frame, last_token): token_to_emit}

    def step(self, frame, prediction):
        key = (int(frame.item()), int(prediction.item()))
        logits = torch.full((1, 6), -10.0) # 6-token vocabulary, all low scores
        logits[0, self.table.get(key, 0)] = 10.0 # chosen token (blank = 0 if not listed)
        return logits

def make_decoder(table, max_symbols=4):
    predictor = FakePredictor()
    model = SimpleNamespace(
        prediction=predictor,
        joint=FakeJoint(table),
        encode=lambda features, lengths: (features, lengths), # fake encoder: pass-through
    )
    tokenizer = SimpleNamespace(blank_id=0, decode=lambda ids: " ".join(map(str, ids)))
    return GreedyRNNTDecoder(model, tokenizer, max_symbols), predictor

class TestGreedyDecoder(unittest.TestCase):
    def test_blank_does_not_update_lstm(self):
        decoder, predictor = make_decoder({(0, 0): 2, (1, 2): 3})
        frames = torch.tensor([[[0.0], [1.0]]]) # shape [1 utterance, 2 frames, 1 number]
        result = decoder.decode_features(frames, torch.tensor([2]))[0]
        self.assertEqual(result["tokens"], [2, 3])
        self.assertEqual(predictor.steps, [2, 3]) # LSTM fed only real tokens, never blanks