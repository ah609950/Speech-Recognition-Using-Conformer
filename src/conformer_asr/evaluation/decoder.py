"""Greedy RNN-Transducer decoding"""
from __future__ import annotations

import torch

class GreedyRNNTDecoder:
    def __init__(self, model, tokenizer, max_symbols_per_frame=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_symbols_per_frame = max_symbols_per_frame

    @torch.inference_mode()
    def decode_features(self, features, feature_lengths):
        """features: [B, T, 80] padded batch
        feature_lengths: [B] true lengths"""
        encoded, encoded_lengths = self.model.encode(features, feature_lengths)
        results = []
        for i in range(encoded.shape[0]): # each utterance in the batch
            real_frames = encoded[i, :int(encoded_lengths[i])] # drop the padding
            results.append(self._decode_one(real_frames))

        return results

    def _decode_one(self, encoded):
        """encoded: [T', D], the frames of ONE utterance"""
        prediction, state = self.model.prediction.initial(1, encoded.device)
        tokens = []
        for frame in encoded: # frame: [D]
            for _ in range(self.max_symbols_per_frame):
                logits = self.model.joint.step(frame[None], prediction) # [1, V]
                token = int(logits.reshape(-1).argmax())
                if token == self.tokenizer.blank_id:
                    break # rule 1
                tokens.append(token)
                token_tensor = torch.tensor([token], device=encoded.device)
                prediction, state = self.model.prediction.step(token_tensor, state) # rule 2

        return {"tokens": tokens, "text": self.tokenizer.decode(tokens)}