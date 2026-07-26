"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.routers import eval as eval_router
from app.routers import ingest, query

app = FastAPI(title="CiteRAG", version="0.1.0")

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_router.router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
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
