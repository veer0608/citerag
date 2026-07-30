"""End-to-end check of POST /query against a live DB with an ingested corpus."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Chunk
from tests.conftest import requires_db

client = TestClient(app)


@requires_db
def test_health_reports_config():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding_dim"] > 0


@requires_db
def test_query_returns_citations(session):
    if session.query(Chunk).count() == 0:
        pytest.skip("no chunks ingested yet (run scripts/seed_corpus.py)")
    r = client.post("/query", json={"question": "What were total revenues?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["chunks"]) > 0
    # Citations are the passages the ANSWER cited, so they're a subset of what was
    # retrieved — never more, and each must point at a real retrieved chunk.
    retrieved_ids = {c["chunk_id"] for c in body["chunks"]}
    assert len(body["citations"]) <= len(body["chunks"])
    assert all(c["chunk_id"] in retrieved_ids for c in body["citations"])
    assert all(1 <= c["marker"] <= len(body["chunks"]) for c in body["citations"])
    # An answer either cites something or is flagged unverified — never both/neither.
    assert body["uncited"] == (len(body["citations"]) == 0)
    # Scores are only meaningful alongside the scale they're on, and every chunk in
    # one response comes from the same stage.
    assert body["score_type"] in {"cosine", "rrf", "cross-encoder"}
    assert all(c["score_type"] == body["score_type"] for c in body["chunks"])
