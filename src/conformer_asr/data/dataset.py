"""Student 1: use the library loader for paired recordings and transcripts."""
from pathlib import Path


def load_librispeech(root, split, *, download=False):
    from torchaudio.datasets import LIBRISPEECH

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return LIBRISPEECH(str(root), url=split, download=download)


# TODO: wrap the library Dataset and return model-ready example dictionaries.
