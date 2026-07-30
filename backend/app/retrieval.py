"""Query -> candidate generation -> (optional) cross-encoder re-rank.

Candidates come from dense vector search (sqlite-vec), optionally fused with
lexical BM25 search (SQLite FTS5) by reciprocal rank fusion. Hybrid fusion and
re-ranking are both ON by default — each was kept only because the eval harness
showed recall@k improve.

Every stage that writes a chunk's `score` also sets its `score_type`, because the
three stages produce numbers on completely different scales (see SCORE_* below).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.orm import Session

from app import vectorstore
from app.config import settings
from app.embeddings import embed_query


# What a chunk's `score` actually is. The stages are on incompatible scales, so a
# bare number is meaningless (and misleading) without this tag: cosine sits near
# 0.7, an RRF score near 0.03, a cross-encoder score anywhere.
SCORE_COSINE = "cosine"  # 0..1, higher = closer
SCORE_BM25 = "bm25"  # SQLite bm25(), NEGATIVE, lower = better
SCORE_RRF = "rrf"  # sum of 1/(k+rank), ~0..0.05, higher = better
SCORE_CROSS_ENCODER = "cross-encoder"  # unbounded, higher = better


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    page_number: int | None  # physical 1-based index in the PDF
    content: str
    score: float  # meaning depends on score_type — never compare across types
    score_type: str = SCORE_COSINE
    page_label: str | None = None  # number printed on the page ("7", "K-83")


def _to_chunks(hits, *, score, score_type: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            page_number=h.page_number,
            page_label=h.page_label,
            content=h.content,
            score=score(h),
            score_type=score_type,
        )
        for h in hits
    ]


def _vector_search(session: Session, query_embedding: list[float], limit: int) -> list[RetrievedChunk]:
    hits = vectorstore.knn(session, query_embedding, limit)
    # cosine distance -> similarity
    return _to_chunks(hits, score=lambda h: 1.0 - h.distance, score_type=SCORE_COSINE)


def _keyword_search(session: Session, query: str, limit: int) -> list[RetrievedChunk]:
    hits = vectorstore.keyword_search(session, query, limit)
    # BM25 score is kept only for reference; hybrid fusion uses rank position.
    return _to_chunks(hits, score=lambda h: h.distance, score_type=SCORE_BM25)


def _rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]], *, rrf_k: int, limit: int
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: a chunk's fused score is sum(1 / (rrf_k + rank)) over
    every list it appears in (rank is 1-based). Rank-based, so the dense and lexical
    scores never have to be on the same scale. The RRF score replaces `score`, and
    `score_type` is retagged to match."""
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = str(chunk.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            fused.setdefault(key, chunk)
    for key, chunk in fused.items():
        chunk.score = scores[key]
        chunk.score_type = SCORE_RRF
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
        cand.score_type = SCORE_CROSS_ENCODER
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
