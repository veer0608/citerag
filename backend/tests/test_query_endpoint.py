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
    # Every retrieved chunk has a citation with a stable id and (usually) a page.
    assert len(body["citations"]) == len(body["chunks"])
    assert all("chunk_id" in c for c in body["citations"])
