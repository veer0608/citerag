"""ORM models. Mirrors the schema created by alembic/versions/0001_schema.py.

Storage is SQLite. Chunk *embeddings* do not live on this table — they live in the
`vec_chunks` sqlite-vec virtual table, joined back by SQLite's implicit rowid (see
app/vectorstore.py). Everything else is plain relational data.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")


class EvalQuestion(Base):
    __tablename__ = "eval_questions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The chunk id(s) that SHOULD be retrieved for this question (JSON array).
    expected_chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Exact retrieval config used, so runs are comparable across history.
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    # recall@k, precision@k, mrr, and per-question detail.
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
