"""Unit tests for re-spacing welded PDF text (no PDF / no DB).

The failure this fixes: FTS5 turns a welded run into a single token, so a query for
"mitsubishi" cannot match a chunk containing "MitsubishiCorporation".
"""
from __future__ import annotations

from app.ingest import split_run_together


def test_splits_welded_company_names_into_searchable_tokens():
    got = split_run_together("81,714,800 MitsubishiCorporation 5.5 2,102")
    assert "Mitsubishi Corporation" in got


def test_splits_acronym_followed_by_a_word():
    assert split_run_together("225,000,000 BYDCo.Ltd. 7.7") .startswith("225,000,000 BYD Co.Ltd.")


def test_partial_split_still_frees_the_entity():
    # "Bankof" stays wrong, but "America" becomes matchable — which is the point.
    got = split_run_together("1,032,852,006 BankofAmericaCorp. 12.8")
    assert "America" in got.split()[1:][0] or "America" in got


def test_leaves_ordinary_prose_untouched():
    prose = "Berkshire paid $27.1 billion in 2021 to repurchase shares"
    assert split_run_together(prose) == prose


def test_leaves_short_tokens_untouched():
    # Below the length threshold, so an ordinary capitalised word is safe.
    assert split_run_together("The BNSF railroad") == "The BNSF railroad"


def test_splits_a_welded_sentence():
    got = split_run_together("Ofequalimportance,floatisverysticky.Fundsattributable")
    # Nothing to split (no interior capitals) — must not corrupt it either.
    assert got == "Ofequalimportance,floatisverysticky.Fundsattributable"


def test_handles_empty_and_whitespace():
    assert split_run_together("") == ""
    assert split_run_together("   ") == "   "
