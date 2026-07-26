"""Regression gate on retrieval quality.

Asserts recall@5 stays at or above the committed baseline. This fails loudly if a
later change makes retrieval worse — the same "assert a real number, not just that
it ran" pattern used for the AgencyDesk isolation tests.

Skips (rather than failing) when there's nothing to measure: no golden questions,
or a corpus that hasn't been ingested (e.g. CI before the Phase 4 seed step).
"""
from __future__ import annotations

import pytest

from app.eval.run_eval import load_golden_set, run_eval
from app.models import Chunk
from tests.conftest import requires_db

# Committed recall@5 floor over the 30-question Berkshire golden set. This gate
# tests the VECTOR-ONLY path (rerank=False) on purpose: it stays fast and needs no
# 1.1GB cross-encoder in CI. The re-ranker lifts recall to 0.500 on top of this.
#   Phase 2 baseline (fixed-size chunking):        0.367
#   Phase 3 exp1 (structure-aware chunking, kept): 0.467  <- measured floor
#   Phase 3 exp2 (+ re-ranker, default on):        0.500  (measured, not CI-gated)
# Set one question (1/30 ~ 0.033) below the measured 0.467 so cross-platform float
# jitter (CI is Linux, dev is Windows) can't flake the build, while a real
# regression — anything that drops 2+ questions — still fails loudly.
RECALL_AT_5_THRESHOLD = 0.43


@requires_db
def test_recall_at_5_meets_threshold(session):
    if not load_golden_set():
        pytest.skip("golden_set.json has no questions yet (Phase 2 not started)")
    if session.query(Chunk).count() == 0:
        pytest.skip("corpus not ingested (run scripts/seed_corpus.py)")
    result = run_eval(session, top_k=5, rerank=False, persist=False)
    recall = result["metrics"]["recall_at_k"]
    assert recall is not None
    assert recall >= RECALL_AT_5_THRESHOLD, (
        f"recall@5 {recall} fell below committed threshold {RECALL_AT_5_THRESHOLD}"
    )
