"""Word segmentation for text PDF extraction welded together.

`split_run_together` in ingest.py handles welds that kept their capitals
("MitsubishiCorporation"). It cannot touch the larger problem: 92% of chunks in this
corpus contain an all-lowercase run of 16+ characters —
"investmentsinequitysecurities", "significantaccountingpoliciesandpractices" — and
those account for ~10% of all alphabetic tokens. FTS5 makes each run a single token,
so BM25 cannot match any word inside it.

The vocabulary is derived from the corpus itself rather than shipped as a wordlist:
only ~10% of tokens are welded, so the other 90% already spell out the domain
vocabulary ("insurance", "underwriting", "Berkshire") far better than a generic
English list would.

Segmentation is maximum-likelihood via dynamic programming — the standard approach —
and deliberately conservative: a split is kept only if EVERY piece is a known word,
so an unrecognised run is left exactly as it was rather than shredded into noise.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

# Runs at least this long are candidates. Real English words this long are rare, and
# the shorter a run is the likelier a split is coincidence rather than a repair.
MIN_SEGMENT_LEN = 16
# Longest single word a split piece may be — bounds the inner DP loop.
MAX_WORD_LEN = 18
# Pieces shorter than this are rejected: one- and two-letter fragments make almost
# any string "segmentable" and would turn real words into confetti.
MIN_PIECE_LEN = 3
# Only learn words up to this length. Welds SHORTER than MIN_SEGMENT_LEN would
# otherwise be learned as if they were words ("unpaidlossesand"), and the DP then
# happily "explains" a run using another weld — poisoning the vocabulary with the
# very defect this module exists to repair.
VOCAB_MAX_LEN = 12
# A word must appear at least this often to be trusted; drops OCR debris.
VOCAB_MIN_COUNT = 2
# The connectives that glue welds together are nearly all two letters, so the
# MIN_PIECE_LEN floor would block exactly the splits that matter. Allow these by
# name rather than lowering the floor for everything.
_SHORT_WORDS = frozenset(
    "of to in is it be as at by on or an we do if no so up he us my me"
    .split()
)

_WORD = re.compile(r"[A-Za-z]+")


def build_vocabulary(texts: Iterable[str]) -> dict[str, float]:
    """Word -> log-probability, learned from correctly-spaced tokens in the corpus.

    Tokens long enough to be welds are excluded, so the vocabulary is built only from
    text that PDF extraction handled correctly.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        for word in _WORD.findall(text):
            lowered = word.lower()
            if len(lowered) <= VOCAB_MAX_LEN and (
                len(lowered) >= MIN_PIECE_LEN or lowered in _SHORT_WORDS
            ):
                counts[lowered] += 1
    total = sum(counts.values())
    if not total:
        return {}
    return {
        w: math.log(c / total)
        for w, c in counts.items()
        if c >= VOCAB_MIN_COUNT or w in _SHORT_WORDS
    }


def _segment_token(token: str, vocab: dict[str, float]) -> list[str] | None:
    """Best full segmentation of `token`, or None if it can't be fully explained.

    Maximum-likelihood DP: best[i] is the score of the best segmentation of the first
    i characters. Requiring a complete parse is what keeps this safe — a run with any
    unknown fragment is rejected whole rather than partially mangled.
    """
    n = len(token)
    best: list[float] = [-math.inf] * (n + 1)
    back: list[int] = [-1] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        start = max(0, i - MAX_WORD_LEN)
        for j in range(start, i):
            if best[j] == -math.inf:
                continue
            piece = token[j:i]
            if len(piece) < MIN_PIECE_LEN and piece not in _SHORT_WORDS:
                continue
            score = vocab.get(piece)
            if score is None:
                continue
            cand = best[j] + score
            if cand > best[i]:
                best[i] = cand
                back[i] = j
    if best[n] == -math.inf:
        return None
    pieces: list[str] = []
    i = n
    while i > 0:
        j = back[i]
        pieces.append(token[j:i])
        i = j
    return pieces[::-1]


# A weld candidate: an optional leading capital followed by lowercase. Matching the
# capital is essential — restricting this to all-lowercase runs skipped every weld at
# a sentence start or in a heading ("Investmentsinequitysecurities"), because the
# capital orphaned the first letter and left an unparseable remainder. Those are
# exactly the positions entity words occupy.
_WELD = re.compile(rf"[A-Za-z][a-z]{{{MIN_SEGMENT_LEN - 1},}}")


def segment_text(text: str, vocab: dict[str, float]) -> str:
    """Re-space long welded runs using the learned vocabulary.

    Lookup is case-insensitive; if the run was capitalised, the capital is restored on
    the first piece so headings and sentence starts read correctly afterwards.
    """
    if not vocab:
        return text

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        pieces = _segment_token(token.lower(), vocab)
        if not pieces:
            return token
        if token[0].isupper():
            pieces[0] = pieces[0].capitalize()
        return " ".join(pieces)

    return _WELD.sub(repl, text)
