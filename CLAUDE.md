# CiteRAG

Cite-everything RAG over messy 10-K PDFs, built against a golden-set eval harness.
README.md covers the design and the numbers; this file is how to work in it.

## Layout

The repo root holds the docs (`README.md`, `PLAN.md`).
**The Python project is `backend/`** — everything below runs from there.

- `backend/app/` — ingestion, chunking, retrieval, re-ranking, `llm.py`, the API
- `backend/tests/` — 77 tests
- `backend/alembic/` — migrations
- `backend/data/`, `backend/scripts/` — corpus and helpers

## Commands

Run from `backend/`. Python 3.11.

```bash
pip install -r requirements.txt
python -m pytest -q          # 77 tests
```

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`, so pytest must be invoked
from `backend/` or collection fails on `app` imports.

## Skills

This repo already has four, in `.claude/skills/` — `run-citerag`, `run-eval`,
`add-golden-questions`, `retrieval-experiment`. **Read the relevant one before
starting**, especially before touching retrieval or the golden set. They encode
decisions that were expensive to reach.

## The numbers, and how not to break them

- **recall@5 went 0.37 → 0.767 against a measured ceiling of 0.850** (bge-reranker-large).
  Those are the figures on the portfolio site and in the profile README. If a change
  moves them, the published numbers must move too — do not leave three places
  disagreeing.
- **The largest gains came from repairing PDF text extraction**, not from retrieval
  design: run-together words, capital-aware segmentation. When retrieval looks bad,
  suspect the text before the ranker.
- **Every change is scored before it is kept.** That is the whole point of the repo.
  Do not tune by eye, and do not report a retrieval change without running the eval.
- **Embeddings and re-ranking run locally, no API key.** This is deliberate: the
  retrieval numbers hold independently of whichever model writes the answer. Keep it
  that way.
- CI ingests the corpus and asserts a recall gate. If it fails, the retrieval got
  worse — that is the gate working.

## Environment

- Shell is PowerShell; POSIX flags like `ls -la` fail there.
- Docker cannot run on this machine (Win11 Home / VBS). Do not propose it.
- The only live key is Groq free tier; its real limit is tokens-per-day and it
  appears in no response header.
