"""Unit tests for the hybrid-retrieval building blocks (no DB / no models).

Covers the two pieces most likely to break silently: turning free-text into a safe
FTS5 MATCH expression, and reciprocal rank fusion of two ranked lists.
"""
from __future__ import annotations

import pytest

from app.retrieval import RetrievedChunk, _rrf_fuse
from app.vectorstore import _fts_match_query


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="d", page_number=1, content=cid, score=0.0
    )


def test_fts_match_query_quotes_and_ors_tokens():
    # Punctuation that FTS5 would treat as operators must not leak through raw.
    q = _fts_match_query('What were "total revenues" in 2021 (Q4)?')
    assert q == '"what" OR "were" OR "total" OR "revenues" OR "in" OR "2021" OR "q4"'


def test_fts_match_query_empty_for_no_tokens():
    assert _fts_match_query("   ?!  ") == ""


def test_rrf_fuse_orders_by_summed_reciprocal_rank():
    # b is rank 1 in both lists -> should win over a, which tops neither.
    dense = [_chunk("b"), _chunk("a"), _chunk("c")]
    lexical = [_chunk("b"), _chunk("d"), _chunk("a")]
    fused = _rrf_fuse([dense, lexical], rrf_k=60, limit=10)
    ids = [c.chunk_id for c in fused]

    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}  # union, de-duplicated
    # Fused score is the sum of 1/(k+rank) across both lists (b: rank 1 + rank 1).
    b = next(c for c in fused if c.chunk_id == "b")
    assert b.score == pytest.approx((1 / 61) + (1 / 61))


def test_rrf_fuse_respects_limit():
    dense = [_chunk(x) for x in "abcde"]
    assert len(_rrf_fuse([dense], rrf_k=60, limit=3)) == 3
