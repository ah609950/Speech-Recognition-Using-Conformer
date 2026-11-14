"""Corpus edit error rates for normalized text."""
from collections.abc import Iterable, Sequence


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    row = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        new = [i]
        for j, hyp in enumerate(prediction, 1):
            new.append(min(row[j] + 1, new[-1] + 1,
                           row[j - 1] + (ref != hyp)))
        row = new
    return row[-1]


def normalize(text: str) -> str:
    """Normalize case and whitespace, preserving punctuation."""
    return " ".join(text.upper().split())


def corpus_error_rates(pairs: Iterable[tuple[str, str]]) -> dict:
    """Return ratios. CER excludes spaces. A zero denominator gives None."""
    word_edits = words = char_edits = chars = 0
    for reference, prediction in pairs:
        ref, hyp = normalize(reference), normalize(prediction)
        ref_words, hyp_words = ref.split(), hyp.split()
        ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
        word_edits += edit_distance(ref_words, hyp_words)
        char_edits += edit_distance(ref_chars, hyp_chars)
        words += len(ref_words)
        chars += len(ref_chars)
    return {
        "wer": word_edits / words if words else None,
        "cer": char_edits / chars if chars else None,
        "word_edits": word_edits,
        "reference_words": words,
        "char_edits": char_edits,
        "reference_characters": chars,
    }
