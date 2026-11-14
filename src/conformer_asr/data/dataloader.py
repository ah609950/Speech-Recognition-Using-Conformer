"""Student 1: combine variable-length samples into one batch."""


def collate_fn(samples, *, pad_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    features = [item["features"] for item in samples]
    tokens = [item["tokens"] for item in samples]
    return {
        "features": pad_sequence(features, batch_first=True),
        "feature_lengths": torch.tensor([len(x) for x in features]),
        "tokens": pad_sequence(tokens, batch_first=True, padding_value=pad_id),
        "token_lengths": torch.tensor([len(y) for y in tokens]),
        "transcripts": [item["transcript"] for item in samples],
        "utterance_ids": [item["utterance_id"] for item in samples],
    }


# TODO: expose create_dataloader() with split-specific shuffling.
