# CiteRAG

Cite-everything RAG over messy 10-K PDFs, built against a golden-set eval harness.
README.md covers the design and the numbers; this file is how to work in it.

## Layout

The repo root holds the docs (`README.md`, `PLAN.md`).
**The Python project is `backend/`** — everything below runs from there.

- `backend/app/` — ingestion, chunking, retrieval, re-ranking, `llm.py`, the API
- `backend/tests/` — 84 tests
- `backend/alembic/` — migrations
- `backend/data/`, `backend/scripts/` — corpus and helpers

## Commands

Run from `backend/`. Python 3.11.

```bash
pip install -r requirements.txt
python -m pytest -q          # 84 tests
```

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`, so pytest must be invoked
from `backend/` or collection fails on `app` imports.

## Skills

This repo already has four, in `.claude/skills/` — `run-citerag`, `run-eval`,
`add-golden-questions`, `retrieval-experiment`. **Read the relevant one before
starting**, especially before touching retrieval or the golden set. They encode
decisions that were expensive to reach.

## The numbers, and how not to break them

- **recall@5 is 0.767 on the 60-question set, against a measured ceiling of 0.850**
  (bge-reranker-large). The 0.367 baseline was on the earlier 30-question set, where
  recall@5 went 0.367 → 0.733. Never put 0.37 and 0.77 in one sentence: they are two sets.
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

## A second corpus exists — do not mix it with the Berkshire one

`POST /ingest/web` (Firecrawl) ingests a second, unrelated corpus (`docs.firecrawl.dev`)
into its own database, `backend/data/citerag_web.db`, with its own golden set,
`golden_set_web.json`. **Never point that ingestion route, or any web-corpus experiment,
at the default `citerag.db`** — doing so once already moved the published Berkshire
recall@5 from 0.767 to 0.75 with zero code changes, just from more chunks sharing the
vector/FTS index. See the README's "second corpus" section for its own numbers.

## Environment

Machine-wide constraints (PowerShell, no Docker, Groq's invisible daily cap) live in
`~/.claude/CLAUDE.md` and are not repeated here.
