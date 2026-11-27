"""The step-by-step decoding path must reproduce the full training path exactly"""

import unittest

import torch

from conformer_asr.evaluation.decoder import GreedyRNNTDecoder
from conformer_asr.models import ConformerTransducer

def small_model(vocab_size=32):
    torch.manual_seed(0) # same random weights every run
    model = ConformerTransducer(vocab_size, num_blocks=2, blank_id=0) # 2 blocks: fast, same code paths
    model.eval() # dropout off, or the two paths differ randomly

    return model

class TestDecoderMatchesModel(unittest.TestCase):
    def test_stepwise_scores_equal_training_scores(self):
        model = small_model()
        features = torch.randn(1, 120, 80) # 1 utterance, 120 frames, 80 features
        lengths = torch.tensor([120])
        tokens = torch.tensor([[5, 9, 3, 17]]) # a made-up 4-token transcript

        with torch.inference_mode():
            # training path: everything at once
            full_logits, _ = model(features, lengths, tokens) # [1, T', 5, 32]
            encoded, _ = model.encode(features, lengths) # [1, T', 144]

            # decoding path: build the history one token at a time
            prediction, state = model.prediction.initial(1, features.device)
            histories = [prediction] # u = 0: empty history
            for token in tokens[0]:
                prediction, state = model.prediction.step(token.view(1), state)
                histories.append(prediction) # u = 1, 2, 3, 4

            # compare at the first, middle and last frame, for every history length
            for t in (0, encoded.shape[1] // 2, encoded.shape[1] - 1):
                for u, pred in enumerate(histories):
                    step_logits = model.joint.step(encoded[0, t][None], pred) # [1, 32]
                    torch.testing.assert_close(step_logits[0], full_logits[0, t, u],
                                               rtol=1e-4, atol=1e-4,
                                               msg=f"mismatch at frame t={t}, history u={u}")

    def test_greedy_decoder_runs_on_real_model(self):
        model = small_model()
        tokenizer = type("Tok", (), {"blank_id": 0, "decode": staticmethod(lambda ids: ids)})()
        decoder = GreedyRNNTDecoder(model, tokenizer, max_symbols_per_frame=8)
        out = decoder.decode_features(torch.randn(2, 120, 80), torch.tensor([120, 80]))
        self.assertEqual(len(out), 2) # one result per utterance
        for result in out:
            self.assertTrue(all(0 < t < 32 for t in result["tokens"])) # real tokens only, never blank