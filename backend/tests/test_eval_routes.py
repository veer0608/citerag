"""The /eval/run contract: GET measures, POST records.

Asserted against the OpenAPI schema rather than by running the harness, which would
mean 30 retrievals per test. The point being locked down is that no GET can write an
eval_runs row — a refresh or prefetch must not mutate the run history.
"""
from __future__ import annotations

from app.main import app


def _params(method: str) -> set[str]:
    op = app.openapi()["paths"]["/eval/run"][method]
    return {p["name"] for p in op.get("parameters", [])}


def test_get_cannot_request_persistence():
    # No `persist` knob at all, so there is no way to make a GET write.
    assert "persist" not in _params("get")
    assert {"top_k", "rerank", "hybrid"} <= _params("get")


def test_post_is_the_recording_route():
    assert "persist" in _params("post")
