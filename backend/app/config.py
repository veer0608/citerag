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
    # Absolute .env path so the file is found no matter which directory the
    # process was launched from (same reasoning as the absolute DB path above).
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"), extra="ignore"
    )

    database_url: str = _DEFAULT_DATABASE_URL

    # Embeddings
    embedding_model: str = "bge-small-en-v1.5"

    # Chunking. Strategy "fixed" = Phase 1 naive token windows; "structure" =
    # Phase 3 page-bounded, line-preserving chunks (default — it lifted recall@5
    # from 0.367 to 0.467 on the golden set).
    chunk_strategy: str = "structure"
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50
    # Used only by the "structure" strategy.
    structure_max_tokens: int = 220
    structure_overlap_lines: int = 1

    # Retrieval. Re-ranking is on by default (Phase 3 exp2): embed the top
    # `rerank_candidates`, then cross-encode down to `retrieval_top_k`. It lifted
    # recall@5 0.467 -> 0.500 and MRR 0.302 -> 0.365 on the golden set.
    retrieval_top_k: int = 5
    rerank_enabled: bool = True
    rerank_candidates: int = 20
    reranker_model: str = "BAAI/bge-reranker-base"

    # LLM answer step. Priority: OpenAI key -> Anthropic key -> local model ->
    # extractive fallback. Retrieval + eval never depend on any of this.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Local instruct model — free, no API key, runs on CPU. Off by default so CI
    # and tests don't pull a model; the local server turns it on via LOCAL_LLM_ENABLED.
    local_llm_enabled: bool = False
    local_llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

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
