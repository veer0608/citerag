"""Unit tests for golden-set answer matching (no DB).

Ground-truth integrity: a question must not score as a hit on a passage that never
contained the answer, and must still hit passages where PDF extraction mangled the
spacing. Those two requirements pull in opposite directions.
"""
from __future__ import annotations

from app.eval.run_eval import _contains


def test_matches_despite_welded_pdf_text():
    # pdfplumber welds words; the figure is only findable with whitespace stripped.
    content = "Berkshirepaid$27.1billionin2021torepurchasesharesofitsClassAand"
    assert _contains(content, "27.1 billion")


def test_matches_a_table_cell_next_to_another_number():
    # Stripping whitespace would jam "232" and "7,693" together; the space-collapsed
    # form keeps them distinct so the real BYD market value still matches.
    assert _contains("BYDCo.Ltd. 7.7 232 7,693", "7,693")


def test_rejects_a_number_inside_a_longer_number():
    # "67,693" is a totals row, not the BYD holding — this must NOT count as a hit.
    assert not _contains("Totalinsurance 75,140 69,361 67,693 6,585", "7,693")


def test_rejects_a_figure_that_only_appears_as_a_suffix():
    # "3,225,000,000 shares authorized" is not the 225,000,000-share BYD holding.
    assert not _contains("ClassB 3,225,000,000 sharesauthorized", "225,000,000")


def test_accepts_the_same_figure_standing_alone():
    assert _contains("225,000,000 BYDCo.Ltd. 7.7 232", "225,000,000")


def test_non_numeric_needles_are_unaffected():
    assert _contains("its runner-up Giant by year-end value", "runner-up")
    assert not _contains("no such phrase here", "runner-up")


def test_empty_needle_never_matches():
    assert not _contains("anything at all", "")


def test_a_sentence_comma_does_not_extend_the_number():
    # "$99,497, a meaningful gain" — the trailing comma is punctuation, not a
    # thousands separator, so the figure must still match.
    assert _contains("goingfrom$79,387per A shareto$99,497,a meaningfulgain", "99,497")
    assert _contains("Charlie Munger died on November 28, just 33 days before", "November 28")


def test_a_thousands_separator_still_blocks_the_match():
    # Distinguish the above from a comma that genuinely continues the number.
    assert not _contains("authorized 3,225,000,000 shares", "225,000,000")
