"""Query -> embed -> sqlite-vec cosine search -> (optional) cross-encoder re-rank.

The re-ranker is off by default in Phase 1. Phase 3 turns it on and the eval
harness measures whether recall@k actually improves.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.orm import Session

from app import vectorstore
from app.config import settings
from app.embeddings import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    page_number: int | None
    content: str
    score: float  # cosine similarity (1 = identical); re-rank overwrites this


def _vector_search(session: Session, query_embedding: list[float], limit: int) -> list[RetrievedChunk]:
    hits = vectorstore.knn(session, query_embedding, limit)
    return [
        RetrievedChunk(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            page_number=h.page_number,
            content=h.content,
            # cosine distance -> similarity
            score=1.0 - h.distance,
        )
        for h in hits
    ]


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def _rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    model = _reranker()
    scores = model.predict([(query, c.content) for c in candidates])
    for cand, score in zip(candidates, scores, strict=True):
        cand.score = float(score)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def retrieve(
    session: Session,
    query: str,
    *,
    top_k: int | None = None,
    rerank: bool | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query.

    When re-ranking is enabled, pull `rerank_candidates` via vector search first,
    then re-rank down to `top_k` with the cross-encoder.
    """
    top_k = settings.retrieval_top_k if top_k is None else top_k
    rerank = settings.rerank_enabled if rerank is None else rerank

    query_embedding = embed_query(query)

    if rerank:
        candidates = _vector_search(session, query_embedding, settings.rerank_candidates)
        return _rerank(query, candidates, top_k)

    return _vector_search(session, query_embedding, top_k)
