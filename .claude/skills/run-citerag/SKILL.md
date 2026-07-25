---
name: run-citerag
description: Bring up the CiteRAG stack (local SQLite + sqlite-vec + FastAPI), seed the corpus, and verify it end to end with a real query. Use when asked to run, start, boot, or smoke-test CiteRAG, or to confirm ingestion/retrieval works.
---

# Run CiteRAG end to end

Goal: a working stack that answers a real corpus question with a citation.
No Docker, no database server — everything runs locally against a SQLite file.

## Prerequisites
- Python 3.11. From `backend/`: `pip install -r requirements.txt`.
- First run needs outbound network: it downloads the `bge-small` embedding model
  (~130MB) and the corpus PDFs.

## Steps (run from `backend/`)
1. Create the schema (relational tables + the `vec_chunks` sqlite-vec table):
   ```bash
   alembic upgrade head
   ```
2. Seed the corpus (idempotent — safe to re-run):
   ```bash
   python scripts/seed_corpus.py
   ```
   Watch for `empty pages (scanned? OCR needed)` warnings — do not ignore them,
   they mean text extraction failed for those pages.
3. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Confirm health, then ask a real question:
   ```bash
   curl -s localhost:8000/health
   curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question":"What was Berkshire’s insurance float at year-end 2022?"}'
   ```

## Done when
`/query` returns retrieved chunks plus a `citations` array with `chunk_id` and
`page_number`. The answer text may be the extractive fallback if no LLM key is set —
that is expected and fine; retrieval is what matters here.

## Notes
- The SQLite DB lives at `backend/data/citerag.db` (git-ignored). Delete it and
  re-run `alembic upgrade head` for a clean slate.
- The vector store is isolated in `app/vectorstore.py`; swapping to Postgres+pgvector
  later touches only that module and the migration, not the app.
- To run pytest: `cd backend && pytest` (DB-backed tests skip until migrated/seeded).
