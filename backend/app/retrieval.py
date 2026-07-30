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


def _to_chunks(hits, *, score) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            page_number=h.page_number,
            content=h.content,
            score=score(h),
        )
        for h in hits
    ]


def _vector_search(session: Session, query_embedding: list[float], limit: int) -> list[RetrievedChunk]:
    hits = vectorstore.knn(session, query_embedding, limit)
    # cosine distance -> similarity
    return _to_chunks(hits, score=lambda h: 1.0 - h.distance)


def _keyword_search(session: Session, query: str, limit: int) -> list[RetrievedChunk]:
    hits = vectorstore.keyword_search(session, query, limit)
    # BM25 score is kept only for reference; hybrid fusion uses rank position.
    return _to_chunks(hits, score=lambda h: h.distance)


def _rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]], *, rrf_k: int, limit: int
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: a chunk's fused score is sum(1 / (rrf_k + rank)) over
    every list it appears in (rank is 1-based). Rank-based, so the dense and lexical
    scores never have to be on the same scale. The RRF score overwrites `score`."""
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = str(chunk.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            fused.setdefault(key, chunk)
    for key, chunk in fused.items():
        chunk.score = scores[key]
    ordered = sorted(fused.values(), key=lambda c: c.score, reverse=True)
    return ordered[:limit]


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


def _candidates(
    session: Session, query: str, *, hybrid: bool, pool: int
) -> list[RetrievedChunk]:
    """Build the candidate pool: dense-only, or dense + lexical fused by RRF."""
    query_embedding = embed_query(query)
    vec = _vector_search(session, query_embedding, pool)
    if not hybrid:
        return vec
    kw = _keyword_search(session, query, pool)
    return _rrf_fuse([vec, kw], rrf_k=settings.rrf_k, limit=pool)


def retrieve(
    session: Session,
    query: str,
    *,
    top_k: int | None = None,
    rerank: bool | None = None,
    hybrid: bool | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query.

    Candidate generation is dense (sqlite-vec) or hybrid (dense + BM25 fused by RRF).
    When re-ranking is enabled the candidate pool is `rerank_candidates` deep and the
    cross-encoder picks the final `top_k`; otherwise the pool is `top_k` deep and
    returned as-is.
    """
    top_k = settings.retrieval_top_k if top_k is None else top_k
    rerank = settings.rerank_enabled if rerank is None else rerank
    hybrid = settings.hybrid_enabled if hybrid is None else hybrid

    # Deep enough for whichever stage consumes the pool.
    pool = settings.rerank_candidates if rerank else top_k
    if hybrid:
        pool = max(pool, settings.hybrid_candidates)

    candidates = _candidates(session, query, hybrid=hybrid, pool=pool)

    if rerank:
        return _rerank(query, candidates, top_k)
    return candidates[:top_k]
