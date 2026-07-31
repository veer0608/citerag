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
# The only directory /ingest will read PDFs from. Absolute, so the confinement check
# doesn't depend on the process working directory.
_DEFAULT_CORPUS_DIR = _BACKEND_DIR / "data" / "corpus"

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

    # Where the corpus PDFs live, and the boundary /ingest refuses to read outside of.
    corpus_dir: Path = _DEFAULT_CORPUS_DIR

    # Embeddings
    embedding_model: str = "bge-small-en-v1.5"

    # Chunking. Strategy "fixed" = Phase 1 naive token windows; "structure" =
    # Phase 3 page-bounded, line-preserving chunks (default — it lifted recall@5
    # from 0.367 to 0.467 on the golden set).
    chunk_strategy: str = "structure"
    # Table-aware ingestion: pull tables with pdfplumber.extract_tables() and append
    # each row as a compact "label | v1 | v2 ..." fact. Intended to help the
    # equity-holdings fair-value questions. OFF by default: measured on the golden
    # set it HURT the default config (recall@5 0.733 -> 0.667, MRR 0.532 -> 0.506) —
    # appending rows duplicates numbers the text extractor already caught, and the
    # near-duplicate chunks crowd the re-ranker. Kept behind the flag as a recorded
    # experiment; see the README results table.
    table_extraction_enabled: bool = False
    # Re-space words that PDF extraction welded together ("MitsubishiCorporation").
    # FTS5 makes a welded run into one token, so BM25 cannot match the entity at all;
    # this restores it. On by default — see the results table for the measured effect.
    split_run_together_enabled: bool = True
    # Dictionary segmentation for all-lowercase welds ("investmentsinequitysecurities").
    # Case-boundary splitting can't touch these, and they are ~10% of all alphabetic
    # tokens across 92% of chunks. The vocabulary is learned from the document itself.
    word_segmentation_enabled: bool = True
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

    # Hybrid retrieval (Phase 5 exp3): fuse dense (sqlite-vec) with lexical
    # (SQLite FTS5 / BM25) via reciprocal rank fusion, then hand the fused pool to
    # the re-ranker. Dense search blurs exact tokens ("$3.3 billion", "BNSF",
    # "2021") that a keyword index nails, so the two are complementary. Each
    # retriever contributes its top `hybrid_candidates`; rrf_k is the standard RRF
    # damping constant.
    hybrid_enabled: bool = True
    hybrid_candidates: int = 20
    rrf_k: int = 60

    # LLM answer step. Priority: OpenAI key -> Anthropic key -> local model ->
    # extractive fallback. Retrieval + eval never depend on any of this.
    openai_api_key: str | None = None
    # Any OpenAI-compatible endpoint works here — Google AI Studio (Gemini), Groq,
    # OpenRouter and Mistral all speak this protocol, which is how a free tier can be
    # used without a second client implementation. Leave unset for OpenAI itself.
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
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
