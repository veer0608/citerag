"""Run the eval harness on demand.

Two routes on purpose: GET measures and never writes, POST measures and records an
`eval_runs` row. Persisting used to be the GET default, which made a refresh, a
prefetch or a link scanner silently append rows to the run history the project's
whole before/after story is read from.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_session
from app.eval.run_eval import run_eval

router = APIRouter(tags=["eval"])


@router.get("/eval/run")
def eval_run(
    top_k: int | None = None,
    rerank: bool | None = None,
    hybrid: bool | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Measure and return metrics. Side-effect free — nothing is written."""
    return run_eval(
        session, top_k=top_k, rerank=rerank, hybrid=hybrid, persist=False
    )


@router.post("/eval/run")
def eval_run_recorded(
    top_k: int | None = None,
    rerank: bool | None = None,
    hybrid: bool | None = None,
    persist: bool = True,
    session: Session = Depends(get_session),
) -> dict:
    """Measure and record the run, so the result becomes part of eval history."""
    return run_eval(
        session, top_k=top_k, rerank=rerank, hybrid=hybrid, persist=persist
    )
