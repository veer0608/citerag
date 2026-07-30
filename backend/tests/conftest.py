"""Shared fixtures. DB-backed tests skip cleanly when the schema isn't migrated,
so the pure-unit suite (chunking) still runs anywhere."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db import SessionLocal, engine


# Everything the DB-backed tests touch that a migration introduced. Probing only
# `chunks` was not enough: a database left at an older revision passed the check and
# then failed hard mid-test on a missing column, which is exactly what this gate
# exists to turn into a clean skip. Add a probe here with each migration that tests
# depend on.
_SCHEMA_PROBES = (
    "SELECT page_label FROM chunks LIMIT 1",  # 0003
    "SELECT 1 FROM fts_chunks LIMIT 1",  # 0002 (FTS5 virtual table)
    "SELECT 1 FROM vec_chunks LIMIT 1",  # 0001 (sqlite-vec virtual table)
    "SELECT 1 FROM eval_runs LIMIT 1",  # 0001
)


def _schema_ready() -> bool:
    # SQLite always "connects" (it just makes a file), so gate DB tests on the
    # schema actually being at a revision the tests can run against.
    try:
        with engine.connect() as conn:
            for probe in _SCHEMA_PROBES:
                conn.execute(text(probe))
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
