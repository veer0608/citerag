"""Compute retrieval metrics over the golden set and persist one eval_runs row.

Metrics per run: recall@k, precision@k, MRR, plus per-question detail so a failing
question can be eyeballed. Every run records the exact retrieval config, so any
future change is compared against history rather than just the last run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import settings
from app.models import EvalRun
from app.retrieval import RetrievedChunk, retrieve

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def _squash(text: str) -> str:
    """Lowercase and strip all whitespace, for whitespace-insensitive matching."""
    return re.sub(r"\s+", "", text).lower()


@dataclass
class GoldenQuestion:
    question: str
    expected_answer: str | None = None
    expected_page_numbers: list[int] = field(default_factory=list)
    expected_answer_substring: str | None = None
    expected_chunk_ids: list[str] = field(default_factory=list)

    def matches(self, chunk: RetrievedChunk) -> bool:
        """Does this retrieved chunk satisfy the question's ground truth?"""
        if self.expected_chunk_ids and str(chunk.chunk_id) in self.expected_chunk_ids:
            return True
        if (
            self.expected_page_numbers
            and chunk.page_number in self.expected_page_numbers
        ):
            return True
        if self.expected_answer_substring:
            # Whitespace-insensitive: PDF extraction spacing is noise (pdfplumber
            # yields "164 billion" on one page and "164billion" on another), and a
            # re-chunk in Phase 3 shifts whitespace again. Comparing with all
            # whitespace stripped keeps the golden set robust to both.
            needle = _squash(self.expected_answer_substring)
            if needle and needle in _squash(chunk.content):
                return True
        return False


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenQuestion(**q) for q in data.get("questions", [])]


@dataclass
class QuestionResult:
    question: str
    hit: bool
    rank: int | None  # 1-based rank of the first matching chunk, else None
    n_relevant_in_topk: int
    retrieved_chunk_ids: list[str]


def _evaluate_question(
    session: Session, gq: GoldenQuestion, top_k: int, rerank: bool
) -> QuestionResult:
    chunks = retrieve(session, gq.question, top_k=top_k, rerank=rerank)
    rank: int | None = None
    n_relevant = 0
    for i, chunk in enumerate(chunks, start=1):
        if gq.matches(chunk):
            n_relevant += 1
            if rank is None:
                rank = i
    return QuestionResult(
        question=gq.question,
        hit=rank is not None,
        rank=rank,
        n_relevant_in_topk=n_relevant,
        retrieved_chunk_ids=[str(c.chunk_id) for c in chunks],
    )


def run_eval(
    session: Session,
    *,
    top_k: int | None = None,
    rerank: bool | None = None,
    persist: bool = True,
) -> dict:
    top_k = settings.retrieval_top_k if top_k is None else top_k
    rerank = settings.rerank_enabled if rerank is None else rerank

    golden = load_golden_set()
    config = {
        "top_k": top_k,
        "rerank_enabled": rerank,
        "rerank_candidates": settings.rerank_candidates if rerank else None,
        "reranker_model": settings.reranker_model if rerank else None,
        "embedding_model": settings.embedding_model,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "structure_max_tokens": settings.structure_max_tokens,
        "structure_overlap_lines": settings.structure_overlap_lines,
    }

    if not golden:
        metrics = {
            "n_questions": 0,
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
            "note": "golden_set.json has no questions yet (Phase 2 not started).",
            "per_question": [],
        }
        if persist:
            _persist(session, config, metrics)
        return {"config": config, "metrics": metrics}

    results = [_evaluate_question(session, gq, top_k, rerank) for gq in golden]
    n = len(results)
    recall = sum(1 for r in results if r.hit) / n
    precision = sum(r.n_relevant_in_topk for r in results) / (n * top_k)
    mrr = sum((1.0 / r.rank) if r.rank else 0.0 for r in results) / n

    metrics = {
        "n_questions": n,
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
        "per_question": [
            {
                "question": r.question,
                "hit": r.hit,
                "rank": r.rank,
                "n_relevant_in_topk": r.n_relevant_in_topk,
            }
            for r in results
        ],
    }

    if persist:
        _persist(session, config, metrics)
    return {"config": config, "metrics": metrics}


def _persist(session: Session, config: dict, metrics: dict) -> None:
    session.add(EvalRun(config=config, metrics=metrics))
    session.commit()


if __name__ == "__main__":
    from app.db import SessionLocal

    with SessionLocal() as s:
        out = run_eval(s)
    print(json.dumps(out, indent=2))
