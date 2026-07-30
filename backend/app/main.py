"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import settings
from app.routers import eval as eval_router
from app.routers import ingest, query

logger = logging.getLogger("citerag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the models at startup, in the main thread. Loading heavy torch models
    lazily inside a request (which FastAPI runs in a worker thread) can trip a
    'meta tensor' error; doing it here once avoids that and makes the first query
    fast instead of cold."""
    from app import embeddings, llm, retrieval

    try:
        embeddings._local_model()
        if settings.rerank_enabled:
            retrieval._reranker()
        if settings.local_llm_enabled:
            llm._local_model()
        logger.info("model warmup complete")
    except Exception:
        logger.exception("model warmup failed (requests may still lazy-load)")
    yield


app = FastAPI(title="CiteRAG", version="0.1.0", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_router.router)

_STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Minimal one-page query UI (the plan's sanctioned single static page)."""
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "hybrid_enabled": settings.hybrid_enabled,
        "rerank_enabled": settings.rerank_enabled,
        "llm": (
            "openai"
            if settings.openai_api_key
            else "anthropic"
            if settings.anthropic_api_key
            else f"local:{settings.local_llm_model}"
            if settings.local_llm_enabled
            else "extractive-fallback"
        ),
    }
