"""POST /query — retrieve chunks and answer with citations."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_session
from app.llm import answer
from app.retrieval import retrieve

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    rerank: bool | None = None
    hybrid: bool | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    page_number: int | None  # physical 1-based index in the PDF
    page_label: str | None  # number printed on the page ("7", "K-83")
    page_citation: str  # human-facing form, e.g. "page K-83 (PDF page 98)"
    # False = the model declared this citation via an [n] marker. True = the model
    # cited nothing and this was attributed afterwards by matching the answer's
    # figures against the passage. Weaker evidence — don't present them as equal.
    inferred: bool
    # For an inferred citation, the answer figures found in this passage.
    matched_figures: list[str]


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int | None
    page_label: str | None
    score: float
    # What `score` is on: "cosine" (0..1), "rrf" (~0..0.05) or "cross-encoder"
    # (unbounded). Scores are only comparable within one score_type, so a client
    # must not render a bare number or scale it to a bar without checking this.
    score_type: str
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    # The score_type shared by every chunk below (null when nothing was retrieved).
    score_type: str | None
    # The passages backing the answer — a subset of `chunks`, which is the full
    # retrieved pool. Either the model's own [n] markers, or (when it cited nothing)
    # passages attributed by figure matching and flagged `inferred`.
    citations: list[Citation]
    # True when the MODEL cited nothing — still true when citations were inferred, so
    # a caller can tell declared provenance from reconstructed provenance.
    uncited: bool
    chunks: list[RetrievedChunkOut]


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, session: Session = Depends(get_session)) -> QueryResponse:
    chunks = retrieve(
        session, req.question, top_k=req.top_k, rerank=req.rerank, hybrid=req.hybrid
    )
    result = answer(req.question, chunks)
    return QueryResponse(
        question=req.question,
        answer=result.text,
        model=result.model,
        score_type=chunks[0].score_type if chunks else None,
        citations=[Citation(**c) for c in result.citations],
        uncited=result.uncited,
        chunks=[
            RetrievedChunkOut(
                chunk_id=str(c.chunk_id),
                document_id=str(c.document_id),
                page_number=c.page_number,
                page_label=c.page_label,
                score=round(c.score, 4),
                score_type=c.score_type,
                content=c.content,
            )
            for c in chunks
        ],
    )
