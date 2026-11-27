"""Decodes a dataset split and save predictions + corpus metrics"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .metrics import corpus_error_rates

@torch.inference_mode()
def evaluate(model, decoder, dataloader, device, output_dir, split, tag=""):
    model.eval() # dropout/SpecAugment off
    out = Path(output_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    name = f"{split}_{tag}" if tag else split

    records = []
    with open(out / "predictions" / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for batch in dataloader:
            for key in ("utterance_ids", "transcripts"):
                if key not in batch:
                    raise KeyError(f"batch has no '{key}'; the collate function must return it")
            hyps = decoder.decode_features(batch["features"].to(device),
                                           batch["feature_lengths"].to(device))
            for utt_id, ref, hyp in zip(batch["utterance_ids"], batch["transcripts"], hyps):
                record = {"utterance_id": utt_id, "reference": ref, "prediction": hyp["text"]}
                records.append(record)
                f.write(json.dumps(record) + "\n") # one JSON object per line

    result = corpus_error_rates((r["reference"], r["prediction"]) for r in records)
    result.update({"split": split, "tag": tag, "utterances":len(records)})
    with open(out / "metrics" / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    return result