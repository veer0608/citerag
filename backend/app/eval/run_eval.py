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


def _collapse(text: str) -> str:
    """Lowercase, collapsing runs of whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _extends_number_left(haystack: str, i: int) -> bool:
    """Do the characters ending at index i-1 continue a number leftwards?

    A digit always does. A comma does only when a digit precedes it — that's a
    thousands separator ("3,225,000,000"). A trailing sentence comma does not.
    """
    if i <= 0:
        return False
    prev = haystack[i - 1]
    if prev.isdigit():
        return True
    return prev == "," and i - 2 >= 0 and haystack[i - 2].isdigit()


def _extends_number_right(haystack: str, j: int) -> bool:
    """Mirror of the above for the characters starting at index j."""
    if j >= len(haystack):
        return False
    nxt = haystack[j]
    if nxt.isdigit():
        return True
    return nxt == "," and j + 1 < len(haystack) and haystack[j + 1].isdigit()


def _bounded_in(haystack: str, needle: str) -> bool:
    """`needle` occurs in `haystack` without being embedded in a longer number."""
    if not needle:
        return False
    starts_num, ends_num = needle[0].isdigit(), needle[-1].isdigit()
    if not (starts_num or ends_num):
        return needle in haystack
    for m in re.finditer(re.escape(needle), haystack):
        if (not starts_num or not _extends_number_left(haystack, m.start())) and (
            not ends_num or not _extends_number_right(haystack, m.end())
        ):
            return True
    return False


def _contains(content: str, needle_raw: str) -> bool:
    """Does this passage contain the expected answer?

    Two PDF artefacts pull in opposite directions, so both forms are tried:

    * pdfplumber welds words together, so "27.1 billion" only matches once ALL
      whitespace is stripped ("berkshirepaid$27.1billionin2021");
    * stripping whitespace also jams neighbouring TABLE CELLS together, so
      "... 232 7,693 ..." becomes "...2327,693..." and a naive test scores "7,693"
      as present inside "67,693" in an unrelated totals row — a question then
      "passes" on a chunk that never contained the answer.

    Matching against the space-collapsed form as well recovers the table case, and a
    digit/comma boundary check on both forms rejects number-inside-number hits.
    """
    if not needle_raw:
        return False
    return _bounded_in(_collapse(content), _collapse(needle_raw)) or _bounded_in(
        _squash(content), _squash(needle_raw)
    )


@dataclass
class GoldenQuestion:
    question: str
    expected_answer: str | None = None
    expected_page_numbers: list[int] = field(default_factory=list)
    expected_answer_substring: str | None = None
    expected_chunk_ids: list[str] = field(default_factory=list)

    def matches_strict(self, chunk: RetrievedChunk) -> bool:
        """Answer-bearing relevance: the chunk actually contains the answer.

        This is the honest signal — "did we retrieve the chunk that answers the
        question", not merely "a chunk from the right page". An explicit
        expected_chunk_id counts too. Only when a question has NO answer-level
        ground truth (no substring, no chunk ids) do we fall back to the page.
        """
        if self.expected_chunk_ids and str(chunk.chunk_id) in self.expected_chunk_ids:
            return True
        if self.expected_answer_substring:
            # Whitespace-insensitive: PDF extraction spacing is noise (pdfplumber
            # yields "164 billion" on one page and "164billion" on another), and a
            # re-chunk shifts whitespace again. Comparing with all whitespace
            # stripped keeps the golden set robust to both.
            return _contains(chunk.content, self.expected_answer_substring)
        if self.expected_page_numbers:
            return self._page_matches(chunk)
        return False

    def _page_matches(self, chunk: RetrievedChunk) -> bool:
        """Compare against the PHYSICAL PDF page index.

        Measured against the corpus, the golden set's expected_page_numbers track
        the physical index (agreeing on 14/30 questions) rather than the printed
        label (6/30) — they were recorded by eye from a PDF viewer. They're also
        often off by one (an answer on physical 99 recorded as 98), which is why
        page_recall is only an indicative signal and recall@k is answer-bearing
        instead. Do NOT switch this to page_label: that was tried and is wrong.
        """
        return chunk.page_number in set(self.expected_page_numbers)

    def matches_page(self, chunk: RetrievedChunk) -> bool:
        """Page-level relevance: chunk came from an expected page.

        Looser than matches_strict — any chunk on the page counts, even one that
        doesn't contain the answer. Kept only so we can report the historical
        page-level number alongside the strict one. Falls back to strict when a
        question has no page ground truth.
        """
        if self.expected_page_numbers:
            return self._page_matches(chunk)
        return self.matches_strict(chunk)


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenQuestion(**q) for q in data.get("questions", [])]


@dataclass
class QuestionResult:
    question: str
    hit: bool  # strict / answer-bearing: an answer-containing chunk was in top-k
    rank: int | None  # 1-based rank of the first answer-bearing chunk, else None
    n_relevant_in_topk: int  # count of answer-bearing chunks in top-k
    page_hit: bool  # looser: any chunk from an expected page was in top-k
    retrieved_chunk_ids: list[str]


def _evaluate_question(
    session: Session, gq: GoldenQuestion, top_k: int, rerank: bool, hybrid: bool
) -> QuestionResult:
    chunks = retrieve(session, gq.question, top_k=top_k, rerank=rerank, hybrid=hybrid)
    rank: int | None = None
    n_relevant = 0
    page_hit = False
    for i, chunk in enumerate(chunks, start=1):
        if gq.matches_strict(chunk):
            n_relevant += 1
            if rank is None:
                rank = i
        if gq.matches_page(chunk):
            page_hit = True
    return QuestionResult(
        question=gq.question,
        hit=rank is not None,
        rank=rank,
        n_relevant_in_topk=n_relevant,
        page_hit=page_hit,
        retrieved_chunk_ids=[str(c.chunk_id) for c in chunks],
    )


def run_eval(
    session: Session,
    *,
    top_k: int | None = None,
    rerank: bool | None = None,
    hybrid: bool | None = None,
    persist: bool = True,
) -> dict:
    top_k = settings.retrieval_top_k if top_k is None else top_k
    rerank = settings.rerank_enabled if rerank is None else rerank
    hybrid = settings.hybrid_enabled if hybrid is None else hybrid

    golden = load_golden_set()
    config = {
        "top_k": top_k,
        "rerank_enabled": rerank,
        "rerank_candidates": settings.rerank_candidates if rerank else None,
        "reranker_model": settings.reranker_model if rerank else None,
        "hybrid_enabled": hybrid,
        "hybrid_candidates": settings.hybrid_candidates if hybrid else None,
        "rrf_k": settings.rrf_k if hybrid else None,
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
            "page_recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
            "note": "golden_set.json has no questions yet (Phase 2 not started).",
            "per_question": [],
        }
        if persist:
            _persist(session, config, metrics)
        return {"config": config, "metrics": metrics}

    results = [_evaluate_question(session, gq, top_k, rerank, hybrid) for gq in golden]
    n = len(results)
    # recall_at_k / precision_at_k / mrr are all ANSWER-BEARING (strict): they count
    # only chunks that actually contain the answer. page_recall_at_k is the looser,
    # historical page-level number, reported alongside for continuity.
    recall = sum(1 for r in results if r.hit) / n
    page_recall = sum(1 for r in results if r.page_hit) / n
    precision = sum(r.n_relevant_in_topk for r in results) / (n * top_k)
    mrr = sum((1.0 / r.rank) if r.rank else 0.0 for r in results) / n

    metrics = {
        "n_questions": n,
        "recall_at_k": round(recall, 4),
        "page_recall_at_k": round(page_recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
        "per_question": [
            {
                "question": r.question,
                "hit": r.hit,
                "page_hit": r.page_hit,
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
