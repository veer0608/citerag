"""Embedding backends behind one interface.

Default is the local bge-small model (free, no API key). If EMBEDDING_MODEL is set
to an OpenAI model and a key is present, that path is used instead. Whatever the
backend, embeddings are L2-normalised so cosine distance in sqlite-vec is meaningful.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import HF_EMBEDDING_IDS, settings

# bge models recommend prefixing the query (not the passages) with this instruction.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@lru_cache
def _local_model():
    from sentence_transformers import SentenceTransformer

    hf_id = HF_EMBEDDING_IDS.get(settings.embedding_model, settings.embedding_model)
    return SentenceTransformer(hf_id)


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _local_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return _normalise(np.asarray(vecs, dtype=np.float32)).tolist()


@lru_cache
def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)


def _embed_openai(texts: list[str]) -> list[list[float]]:
    resp = _openai_client().embeddings.create(
        model=settings.embedding_model, input=texts
    )
    vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    return _normalise(vecs).tolist()


def _is_openai_model() -> bool:
    return settings.embedding_model.startswith("text-embedding-")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for storage."""
    if not texts:
        return []
    if _is_openai_model():
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query. bge wants an instruction prefix on the query side."""
    if _is_openai_model():
        return _embed_openai([text])[0]
    prefixed = _BGE_QUERY_PREFIX + text
    return _embed_local([prefixed])[0]
