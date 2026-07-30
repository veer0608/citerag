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

# Committed answer-bearing recall@5 floor over the 30-question Berkshire golden set.
# This gate tests the NO-RERANK path on purpose: it stays fast and needs no 1.1GB
# cross-encoder in CI. Hybrid (dense + BM25) IS exercised — FTS5 is built into
# SQLite, so it's free — because it's the default retrieval path. The re-ranker
# lifts recall further (to 0.733) on top of this.
#   Phase 2 baseline (fixed-size chunking):             0.367
#   Phase 3 exp1 (structure-aware chunking, kept):      0.467
#   Phase 3 exp2 (+ re-ranker, default on):             0.500
#   Phase 5 exp3 (+ hybrid dense+BM25, no rerank):      0.633  <- measured floor
#   Phase 5 exp3 (+ hybrid + re-ranker, default):       0.733  (measured, not CI-gated)
# Set ~2 questions (2/30 ~ 0.067) below the measured 0.633 so cross-platform float
# jitter (CI is Linux, dev is Windows) can't flake the build, while a real
# regression — anything that drops 3+ questions — still fails loudly.
RECALL_AT_5_THRESHOLD = 0.57


@requires_db
def test_recall_at_5_meets_threshold(session):
    if not load_golden_set():
        pytest.skip("golden_set.json has no questions yet (Phase 2 not started)")
    if session.query(Chunk).count() == 0:
        pytest.skip("corpus not ingested (run scripts/seed_corpus.py)")
    # Explicit hybrid=True/rerank=False so the gate is deterministic regardless of
    # env-var config: it measures the default hybrid path minus the heavy reranker.
    result = run_eval(session, top_k=5, rerank=False, hybrid=True, persist=False)
    recall = result["metrics"]["recall_at_k"]
    assert recall is not None
    assert recall >= RECALL_AT_5_THRESHOLD, (
        f"recall@5 {recall} fell below committed threshold {RECALL_AT_5_THRESHOLD}"
    )
