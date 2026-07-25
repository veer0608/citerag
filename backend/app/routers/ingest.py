"""POST /ingest — ingest a PDF already present on the server, by path."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_session
from app.ingest import ingest_pdf

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    # Path on the server (e.g. under backend/data/corpus). Uploads via multipart
    # could be added later; for a corpus-seeding project, path-based is simpler.
    path: str
    title: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    n_pages: int
    n_chunks: int
    n_empty_pages: int


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, session: Session = Depends(get_session)) -> IngestResponse:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    try:
        result = ingest_pdf(session, path, title=req.title)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return IngestResponse(**result.__dict__)
