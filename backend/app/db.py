"""SQLAlchemy engine / session wiring for SQLite + sqlite-vec.

The sqlite-vec extension is loaded on EVERY sqlite connection via a listener on
the base Engine class, so both the app engine and Alembic's migration engine can
use the vec0 virtual table. Foreign keys are enabled per-connection too (SQLite
requires this) so ON DELETE CASCADE works.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import sqlite_vec
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


@event.listens_for(Engine, "connect")
def _configure_sqlite_connection(dbapi_conn, _record) -> None:
    # Guard so this is a no-op if the URL ever points at a non-sqlite backend.
    if not isinstance(dbapi_conn, sqlite3.Connection):
        return
    dbapi_conn.enable_load_extension(True)
    sqlite_vec.load(dbapi_conn)
    dbapi_conn.enable_load_extension(False)
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


# check_same_thread=False lets the connection be shared across FastAPI threads.
_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args=_connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yields a session, always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
