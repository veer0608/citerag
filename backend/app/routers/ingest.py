"""POST /ingest — ingest a PDF already present on the server, by path."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session
from app.ingest import ingest_pdf

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    # Path to a PDF inside the corpus directory — either relative to it
    # ("berkshire_2023.pdf") or the absolute path of a file within it. Uploads via
    # multipart could be added later; for a corpus-seeding project, path is simpler.
    path: str
    title: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    n_pages: int
    n_chunks: int
    n_empty_pages: int


def resolve_corpus_path(raw: str) -> Path:
    """Resolve a request path to a real PDF inside the corpus directory.

    Confinement matters because this endpoint is unauthenticated: without it, any
    caller could name an arbitrary server file and have its text extracted into the
    response. Resolving BOTH sides before comparing is what defeats `..` traversal
    and symlinks — a lexical prefix check on the raw string would not.

    Raises HTTPException(400) for anything outside the corpus or not a .pdf, and
    404 when the file simply isn't there.
    """
    root = Path(settings.corpus_dir).resolve()
    candidate = Path(raw)
    # A relative path is interpreted against the corpus, never the process cwd.
    path = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    if path != root and root not in path.parents:
        raise HTTPException(
            status_code=400,
            detail=f"path must be inside the corpus directory ({root})",
        )
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="only .pdf files can be ingested")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {raw}")
    return path


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, session: Session = Depends(get_session)) -> IngestResponse:
    path = resolve_corpus_path(req.path)
    try:
        result = ingest_pdf(session, path, title=req.title)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return IngestResponse(**result.__dict__)
