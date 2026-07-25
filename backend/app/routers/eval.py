"""GET /eval/run — run the eval harness on demand and return live metrics."""
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
    persist: bool = True,
    session: Session = Depends(get_session),
) -> dict:
    return run_eval(session, top_k=top_k, rerank=rerank, persist=persist)
