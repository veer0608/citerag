# CiteRAG

A retrieval-augmented generation system that answers questions over a **real, messy
document set**, cites the exact source passage (with page number) for every answer,
and — the part most tutorial RAG projects skip — ships a **golden-question eval
harness that proves retrieval quality with numbers, not vibes**.

> The one sentence this repo is built to support:
> *"My naive retrieval got the right chunk about half the time. I built an eval set of
> real questions, found the tables were getting split mid-row, fixed the chunking, then
> added a re-ranker — recall went from X to Y. Here's the table."*

## Corpus

**Berkshire Hathaway annual reports, 2021–2023** (public, free, from
`berkshirehathaway.com`). One company across three years is a deliberately hard
retrieval target: dense financial tables, footnotes, boilerplate repeated across
years, and exact numbers that must be pulled from the *right year's* table. That
makes *"did it retrieve the right number from the right year"* an objective,
writable eval question — the reason this beats clean Wikipedia/markdown corpora.

Swap in your own PDFs by dropping them in `backend/data/corpus/` and re-running the
seed script.

## Stack

| Layer | Choice |
|---|---|
| Vector store | **SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec)** (`vec0` virtual table, cosine). Zero-infra and fully local. The vector search is isolated in `app/vectorstore.py` — the one seam to swap for Postgres+pgvector later without touching the rest of the app. |
| Backend | FastAPI |
| ORM / migrations | SQLAlchemy 2 + Alembic |
| Embeddings | **`bge-small-en-v1.5`** (local, 384-dim, no API key). Swappable to OpenAI `text-embedding-3-small` via `EMBEDDING_MODEL`. |
| Re-ranker | `bge-reranker-base` cross-encoder (off by default; A/B'd in Phase 3) |
| LLM answer step | Priority: OpenAI → Anthropic (if a key is set) → **local `Qwen2.5-0.5B-Instruct`** (free, no key, runs on CPU via `LOCAL_LLM_ENABLED=true`) → labelled extractive fallback. **Retrieval and the entire eval harness need no LLM at all.** |
| Testing / CI | pytest / GitHub Actions |

> Why SQLite+sqlite-vec instead of the originally-planned Postgres+pgvector: the
> target machine (Windows 11 Home + Ryzen) couldn't run Docker Desktop — Memory
> Integrity/VBS blocks its WSL2 engine. Rather than burn the build on infra, the
> vector store was swapped for a zero-install local equivalent behind
> `vectorstore.py`. The schema, migrations, citations, and the entire eval harness
> are unchanged.

## Quickstart

No Docker, no database server — it's all local.

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 1. Create the SQLite schema + the vec_chunks vector table.
alembic upgrade head

# 2. Seed the corpus (downloads the PDFs, then ingests them). First run also
#    downloads the ~130MB bge-small embedding model.
python scripts/seed_corpus.py

# 3. Start the API.
uvicorn app.main:app --reload

# 4. Ask a question — every answer comes back with citations + page numbers.
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What was Berkshire’s insurance float at year-end 2022?"}'

# 5. Run the eval harness on demand.
curl -s 'localhost:8000/eval/run'
```

Health/config check: `curl -s localhost:8000/health`.

## API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/ingest` | Ingest one PDF by server path (`{"path": "...", "title": "..."}`) |
| `POST` | `/query` | Retrieve top-k chunks + answer with citations |
| `GET`  | `/eval/run` | Run recall@k / precision@k / MRR over the golden set, store an `eval_runs` row |
| `GET`  | `/health` | Effective config (embedding model/dim, reranker, LLM backend) |

## How it works

```
PDF ──pdfplumber──▶ pages ──chunking.py──▶ chunks ──embeddings.py──▶ sqlite-vec
                                                                        │
question ──embed_query──▶ cosine top-k ──(optional) cross-encoder re-rank──▶ chunks
                                                                        │
                                                            llm.py: answer + citations
```

- `chunking.py` is isolated so the strategy can be swapped and unit-tested. Phase 1
  is naive fixed-size token windows (500 tokens, 50 overlap).
- Ingestion is **idempotent** (deterministic ids; re-ingesting replaces prior rows)
  and surfaces empty/scanned pages instead of silently dropping them.
- Every `eval_runs` row stores the exact retrieval config, so any future change is
  compared against history, not just the last run.

## Build phases

- **Phase 1 — naive pipeline, end to end.** ✅ *(this scaffold)* Ingest → chunk →
  embed → sqlite-vec search → answer with citations. Deliberately un-tuned, to create a
  baseline to measure against.
- **Phase 2 — measure before you fix.** ✅ 30-question hand-written golden set;
  baseline recall@5 = 0.367 recorded and committed as a regression gate.
- **Phase 3 — fix what the eval found broken.** ✅ structure-aware chunking
  (0.367 → 0.467) then re-ranker (0.467 → 0.500), each measured and kept only
  because the number moved. Query rewriting is the next lever but needs an LLM key.
- **Phase 4 — the defensible layer.** ✅ citations with page numbers, live
  `/eval/run`, and CI that ingests the corpus and asserts the recall gate.
- **Phase 5 — hybrid retrieval.** ✅ dense (sqlite-vec) + lexical (SQLite FTS5/BM25)
  fused with reciprocal rank fusion, then re-ranked. recall@5 0.500 → 0.733 — the
  biggest single lever, and free of any new model or service. Query rewriting is the
  next lever but needs an LLM key.

## Results (before/after)

Measured over a **30-question hand-written golden set** (`backend/app/eval/golden_set.json`)
spanning the 2021–2023 reports, top_k=5. This table is the point of the project.

| Change | recall@5 | precision@5 | MRR | notes |
|---|---|---|---|---|
| Naive fixed-size chunking (500/50) | 0.367 | 0.10 | 0.188 | baseline — 11/30. |
| + structure-aware chunking (page-bounded, 220-tok, line-preserving) | 0.467 | 0.147 | 0.302 | +0.100. 14/30. Diagnosis: naive windows spanned pages and diluted specific facts; smaller page-bounded chunks concentrate them. |
| + re-ranker (bge-reranker-base, top-20 → top-5) | 0.500 | 0.173 | 0.365 | +0.033 recall, but MRR 0.302 → 0.365 — the right chunk, when found, ranks higher. 15/30. |
| + hybrid retrieval (dense + BM25/FTS5, RRF-fused) | 0.633 | 0.193 | 0.458 | +0.133 (measured with rerank off, to isolate the fusion). 19/30. Diagnosis: the corpus is full of exact tokens — dollar amounts, tickers, years — that dense embeddings blur; a keyword index nails them. |
| + hybrid **and** re-ranker (default config) | **0.733** | **0.220** | **0.532** | +0.100 on top of hybrid. 22/30. Re-ranker orders the richer fused pool better than it did the dense-only one. |

**Net: recall@5 0.367 → 0.733 (2×), MRR 0.188 → 0.532 (2.8×).** The two biggest levers
were structure-aware chunking (+0.100) and hybrid retrieval (+0.133).

**What "recall@5" means here (verified, not assumed):** a question counts as hit only
when a retrieved chunk *actually contains the answer* (whitespace-insensitive substring
of the expected answer), not merely a chunk from the expected page. The eval also reports
a looser `page_recall@5` for continuity. Tightening this in `run_eval.py` left the strict
number unchanged (0.467 / 0.500) — confirming the headline was already answer-bearing —
but the low `page_recall@5` (0.167) surfaced a separate bug: the golden set's
`expected_page_numbers` are the page labels a human reads, while ingest stores the 1-based
*physical* PDF index, and the two are offset by the reports' front matter. See the gap note
below.

**How each change was chosen (not guessed):** the misses were diagnosed by checking
whether the correct chunk was even in the candidate pool, and if so, where it ranked:
- *Not retrieved at all* (most baseline misses) — narrative facts diluted inside big
  cross-page windows → **structure-aware chunking** (exp1: +0.100).
- *Answer is an exact token dense search blurred* (dollar figures, tickers, years) →
  **hybrid dense + BM25 fusion** (exp3: +0.133, the biggest single win).
- *In the pool but out-ranked* (e.g. Apple's 2023 fair value sat at rank 6) →
  **re-ranker** (exp2/exp3: reorders the fused pool, +0.100 on top of hybrid).

Cost note: the re-ranker adds a ~1.1GB cross-encoder and per-query latency. It's on
by default because it's the best-scoring config; set `RERANK_ENABLED=false` to skip it
(hybrid alone still scores 0.633). Hybrid itself is nearly free — FTS5 is built into
SQLite, so there's no extra model or service. **CI** ingests the whole corpus and
asserts the no-rerank recall floor (0.57, ~2 questions below the measured 0.633 to
absorb cross-platform float jitter) — the "assert a real number, not just that it
built" gate — without needing the reranker.

**Still on the table (honest remaining gap — 8/30 still miss):** the equity-holdings
fair-value tables (rows get flattened into ragged text at ingest, so the row structure
is lost) and a few narrative facts that phrase the answer very differently from the
question. Next levers: table-aware ingestion (`extract_tables()` → one fact per row)
and query rewriting (restate the question in the document's vocabulary before
embedding) — neither done yet.

**Known bug — citation page numbers are physical, not printed.** Ingest stores each
chunk's 1-based *physical* PDF page index; the number printed on the page (and used by
the golden set) is offset by the reports' cover/front matter, and the offset isn't even
constant across the three years. So a returned citation of "page 6" may not match the "6"
a reader sees on the page. This is why `page_recall@5` sits at 0.167 while answer-bearing
recall is 0.467 — retrieval finds the right content, but the page label it cites is off.
Fix options: label citations explicitly as "PDF page N", or extract printed labels at
ingest and store both.

## Claude Code skills

Repeatable workflows are packaged as project skills under `.claude/skills/`, so a
fresh Claude Code session can operate the repo per the plan's rules:

| Skill | Purpose |
|---|---|
| `run-citerag` | Bring up the stack, seed the corpus, smoke-test a real query |
| `run-eval` | Run the eval harness, record the `eval_runs` row, update the before/after table |
| `add-golden-questions` | Hand-write Phase 2 golden questions with robust ground-truth matching |
| `retrieval-experiment` | Disciplined Phase 3 loop: one change → re-eval → keep only if the number improved |

## Running tests

```bash
# Pure unit tests (chunking) run anywhere. DB-backed tests skip until the schema
# is migrated (alembic upgrade head) and, for the query test, the corpus is seeded.
cd backend && pytest
```

## Environment / config

See `backend/.env.example`. Notable knobs (all recorded per eval run):
`EMBEDDING_MODEL`, `RETRIEVAL_TOP_K`, `RERANK_ENABLED`, `RERANK_CANDIDATES`,
`HYBRID_ENABLED`, `HYBRID_CANDIDATES`, `RRF_K`, `CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`.
