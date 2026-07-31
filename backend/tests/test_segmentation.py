"""Unit tests for dictionary word segmentation (no PDF / no DB)."""
from __future__ import annotations

from app.segmentation import build_vocabulary, segment_text

# A miniature corpus standing in for the correctly-spaced 90% of the real one.
CORPUS = [
    "investments in equity securities are carried at fair value",
    "equity securities and investments in the insurance business",
    "the insurance business generated float in 2021 and 2022",
    "significant accounting policies and practices are described",
    "accounting policies and practices of the insurance group",
    "we used cash to repurchase shares of its common stock",
    "repurchase of shares reduced the count of common stock",
]


def _vocab():
    return build_vocabulary(CORPUS)


def test_segments_a_welded_phrase_into_searchable_words():
    assert segment_text("investmentsinequitysecurities", _vocab()) == (
        "investments in equity securities"
    )


def test_segments_using_two_letter_connectives():
    # "in"/"of"/"to" are the glue in most welds; blocking them would block the split.
    assert segment_text("torepurchaseshares", _vocab()) == "to repurchase shares"


def test_leaves_unknown_runs_completely_alone():
    # All-or-nothing: one unexplainable fragment means the run is returned intact
    # rather than shredded into plausible-looking noise.
    welded = "zzzqqqxxxwwwvvvuuu"
    assert segment_text(welded, _vocab()) == welded


def test_leaves_short_tokens_and_ordinary_prose_alone():
    prose = "the insurance business generated float"
    assert segment_text(prose, _vocab()) == prose
    # Below the length threshold, so never a candidate.
    assert segment_text("equityvalue", _vocab()) == "equityvalue"


def test_vocabulary_excludes_long_tokens_so_welds_cannot_poison_it():
    # A weld appearing in the source must not be learned as if it were a word,
    # or the segmenter would "explain" one weld with another.
    vocab = build_vocabulary(["investmentsinequitysecurities appears here"])
    assert "investmentsinequitysecurities" not in vocab


def test_empty_vocabulary_is_a_no_op():
    assert segment_text("investmentsinequitysecurities", {}) == (
        "investmentsinequitysecurities"
    )


def test_segments_a_capitalised_weld_and_restores_the_capital():
    # Sentence starts and headings are exactly where entity words live, and an
    # all-lowercase-only pattern skipped every one of them.
    assert segment_text("Investmentsinequitysecurities", _vocab()) == (
        "Investments in equity securities"
    )


def test_lowercase_welds_still_segment_unchanged():
    assert segment_text("investmentsinequitysecurities", _vocab()) == (
        "investments in equity securities"
    )
