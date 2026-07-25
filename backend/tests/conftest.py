"""Shared fixtures. DB-backed tests skip cleanly when the schema isn't migrated,
so the pure-unit suite (chunking) still runs anywhere."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db import SessionLocal, engine


def _schema_ready() -> bool:
    # SQLite always "connects" (it just makes a file), so gate DB tests on the
    # schema actually existing — i.e. migrations have been run.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM chunks LIMIT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


DB_AVAILABLE = _schema_ready()
requires_db = pytest.mark.skipif(
    not DB_AVAILABLE, reason="schema not migrated (run: alembic upgrade head)"
)


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
