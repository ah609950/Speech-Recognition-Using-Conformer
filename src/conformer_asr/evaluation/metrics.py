"""Corpus-level WER/CER"""
from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "`": "'"})
_DISALLOWED = re.compile(r"[^A-Z0-9']+")

def edit_distance(reference, prediction):
    """Fewest substitutions + deletions + insertions turning prediction into reference
    Works on any sequences: a list of words (WER) or a string (CER)"""
    return sum(edit_counts(reference, prediction).values())

def edit_counts(reference, prediction):
    n, m = len(reference), len(prediction)
    d = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0, i, 0) # first column: i deletions
    for j in range(m + 1):
        d[0][j] = (j, 0, 0, j) # first row: i insertions
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == prediction[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                continue
            t, s, de, ins = d[i - 1][j - 1]
            substitution = (t + 1, s + 1, de, ins)
            t, s, de, ins = d[i - 1][j]
            deletion = (t + 1, s, de + 1, ins)
            t, s, de, ins = d[i][j - 1]
            insertion = (t + 1, s, de, ins + 1)
            d[i][j] = min(substitution, deletion, insertion, key=lambda c: c[0])
    _, s, de, ins = d[n][m]

    return {"substitutions": s, "deletions": de, "insertions": ins}

def normalize(text: str) -> str:
    """Uppercase; keep only A-Z, 0-9, apostrophes; single spaces"""
    text = unicodedata.normalize("NFKC", text) # 1. unify odd Unicode forms
    text = text.translate(_APOSTROPHES).upper() # 2. curly/backtick -> ', then uppercase
    text = " ".join(text.split()) # 3. tabs/newlines/runs of spaces -> one space
    text = _DISALLOWED.sub(" ", text) # 4. every other character -> space
    return " ".join(text.split()) # 5. collapse the spaces step 4 created

def corpus_error_rates(pairs, include_spaces=False):
    """pairs: iterable of (reference_text, predicted_text)
    Sums all edits and all reference lengths, then divides once"""
    word = {"substitutions": 0, "deletions": 0, "insertions": 0}
    char = {"substitutions": 0, "deletions": 0, "insertions": 0}
    n_words = n_chars = 0

    for reference, prediction in pairs:
        ref, hyp = normalize(reference), normalize(prediction)

        #word level: compare lists of words
        for k, v in edit_counts(ref.split(), hyp.split()).items():
            word[k] += v
        n_words += len(ref.split())

        #character level: compare strings, spaces removed unless asked
        ref_c = ref if include_spaces else ref.replace(" ", "")
        hyp_c = hyp if include_spaces else hyp.replace(" ", "")
        for k, v in edit_counts(ref_c, hyp_c).items():
            char[k] += v
        n_chars += len(ref_c)

    return {
        "wer": sum(word.values()) / n_words if n_words else None,
        "cer": sum(char.values()) / n_chars if n_chars else None,
        "word_error_counts": word,
        "character_error_counts": char,
        "reference_words": n_words,
        "reference_characters": n_chars,
    }