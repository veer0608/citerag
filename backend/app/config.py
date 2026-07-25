"""Central configuration.

Everything that a later phase might want to A/B (chunk size, top_k, reranker on/off,
embedding model) lives here so an eval run can record the exact config it used.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default SQLite DB lives at backend/data/citerag.db, resolved absolutely so the
# path is stable regardless of the process working directory.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DB_PATH = _BACKEND_DIR / "data" / "citerag.db"
_DEFAULT_DATABASE_URL = f"sqlite:///{_DEFAULT_DB_PATH.as_posix()}"

# Known embedding models -> their vector dimension. The DB column dimension must
# match, which is why the Alembic migration reads EMBEDDING_DIM from here.
EMBEDDING_DIMS: dict[str, int] = {
    "bge-small-en-v1.5": 384,
    "text-embedding-3-small": 1536,
}

# Map our short name to the actual HuggingFace model id.
HF_EMBEDDING_IDS: dict[str, str] = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = _DEFAULT_DATABASE_URL

    # Embeddings
    embedding_model: str = "bge-small-en-v1.5"

    # Chunking (Phase 1 defaults — deliberately naive, improved in Phase 3)
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Retrieval
    retrieval_top_k: int = 5
    rerank_enabled: bool = False
    rerank_candidates: int = 20
    reranker_model: str = "BAAI/bge-reranker-base"

    # LLM answer step (optional — retrieval + eval work without any key)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    @property
    def embedding_dim(self) -> int:
        if self.embedding_model not in EMBEDDING_DIMS:
            raise ValueError(
                f"Unknown embedding model {self.embedding_model!r}; "
                f"add it to EMBEDDING_DIMS in config.py"
            )
        return EMBEDDING_DIMS[self.embedding_model]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
