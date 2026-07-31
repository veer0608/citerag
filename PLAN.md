# CiteRAG — Build Plan

Hand this file to a fresh Claude Code session with: **"Read PLAN.md and start Phase 1."**
Everything below is written so it can execute without needing to ask you clarifying
questions about scope — the decisions are already made.

## What this is

A RAG (retrieval-augmented generation) system that answers questions over a real,
messy document set, cites the exact source passage for every answer, and — the part
almost nobody builds — has a golden-question eval harness that proves retrieval
quality improved, with numbers, not vibes.

The interview pitch this is built to support: *"My naive version retrieved the right
chunk 60% of the time. Here's the eval set I built, here's what I found was actually
broken, and here's what fixed it and by how much."*

## Non-negotiable constraints

- **Measure before you fix.** Phase 2 (the eval harness) happens before any retrieval
  improvement in Phase 3. Improving something you haven't measured is not allowed —
  the whole point of this project is being able to show a before/after number.
- **No agentic multi-step reasoning bolted on.** This is a retrieval project, not an
  agent project. Do not add tool use, planning loops, or multi-hop reasoning — that
  dilutes the story this project is meant to tell.
- **No UI polish.** A bare FastAPI + curl (or one static HTML page) is enough. Time
  spent on frontend is time not spent on retrieval quality or the eval harness.

## Corpus — pick one, do not skip this step

The corpus is the single most important decision. It must have **real** retrieval
problems: tables, multi-column layout, inconsistent headers, numbers that need exact
retrieval rather than paraphrase. Do not use clean Wikipedia articles or markdown —
that produces a tutorial-clone project that proves nothing.

**Default (use this unless told otherwise): a company's 10-K / annual report PDFs**
(2-3 years of one company, e.g. from SEC EDGAR — public, free, genuinely messy: dense
tables, footnotes, repeated boilerplate across years, numbers that must be retrieved
exactly). Good because "did it retrieve the right number from the right year's table"
is an easy, objective eval question to write.

Two other acceptable options, pick whichever the user prefers if asked:
- Your own college course PDFs (MITS syllabus/slides) — scanned pages, tables, mixed layouts.
- NPTEL course transcripts + slides for a course you've taken — mixed formats, and you
  already know the right answers, which makes writing the golden eval set fast.

Once picked, note it at the top of the README and do not revisit the choice mid-build.

## Stack

Reuse what is already known from the AgencyDesk project rather than learning new
tools for their own sake.

> **What shipped differs from this table in two places.** The vector store is
> **SQLite + sqlite-vec**, not Postgres + pgvector (Docker wouldn't run on the build
> machine — see the checklist below), and lexical **BM25 via SQLite FTS5** was added
> alongside dense search, which the plan didn't anticipate at all and which turned out
> to be one of the two biggest wins. The rows below are the original reasoning, kept
> as written.

| Layer | Choice | Why |
|---|---|---|
| Vector store | **Postgres + pgvector** | Already comfortable operating Postgres with real schemas and migrations |
| Backend | **FastAPI** | Same framework as AgencyDesk |
| ORM / migrations | **SQLAlchemy + Alembic** | Same as AgencyDesk |
| Embeddings | `text-embedding-3-small` (OpenAI) *or* `bge-small-en-v1.5` (local, free) | Start with whichever has an available API key; note the choice in the README |
| Re-ranker | `bge-reranker-base` (via `sentence-transformers` or a small hosted call) | This is the piece most tutorial RAG projects skip — it is a deliberate differentiator here |
| LLM for answering | Whatever API is available (OpenAI/Anthropic) | Swappable behind one function, `llm.py` |
| Testing | `pytest` | Same as AgencyDesk |
| CI | GitHub Actions | Same pattern as AgencyDesk: build, then assert real numbers, not just "it ran" |

## Repo layout

```
citerag/
  README.md                 corpus choice, setup, the before/after eval table
  PLAN.md                   this file
  docker-compose.yml        postgres (with pgvector) + api   [not built — see above]
  backend/
    alembic/versions/
      0001_schema.py         documents, chunks, eval_questions, eval_runs
    app/
      config.py
      db.py
      ingest.py              PDF -> parsed pages -> chunks -> embeddings -> DB
      chunking.py            chunking strategy, isolated so it can be swapped/tested
      retrieval.py           embed query -> vector search -> (optional) re-rank
      llm.py                 one function: (question, chunks) -> answer + citations
      routers/
        ingest.py             POST /ingest
        query.py              POST /query
        eval.py                GET  /eval/run
      eval/
        golden_set.json       25-40 real questions with expected chunk ids / answers
        run_eval.py            computes recall@k, precision@k, writes eval_runs row
    tests/
      test_chunking.py
      test_retrieval_recall.py   asserts recall@5 stays above a committed threshold
      test_query_endpoint.py
    scripts/
      seed_corpus.py           idempotent: ingest the chosen documents
  .github/workflows/ci.yml    build, ingest, run eval, assert recall@5 >= threshold
```

## Database schema (sketch — refine during Phase 1)

```sql
documents (
  id uuid primary key,
  title text not null,
  source_path text not null,
  ingested_at timestamptz not null default now()
)

chunks (
  id uuid primary key,
  document_id uuid not null references documents(id),
  chunk_index int not null,
  page_number int,
  content text not null,
  token_count int not null,
  embedding vector(1536)        -- dimension matches the chosen embedding model
)
-- ivfflat or hnsw index on embedding, chosen once corpus size is known

eval_questions (
  id uuid primary key,
  question text not null,
  expected_answer text,
  expected_chunk_ids uuid[] not null   -- the chunk(s) that should be retrieved
)

eval_runs (
  id uuid primary key,
  run_at timestamptz not null default now(),
  config jsonb not null,        -- chunking params, k, reranker on/off, etc.
  metrics jsonb not null        -- recall@k, precision@k, mrr, per-question detail
)
```

## Phase 1 — naive pipeline, working end to end

Goal: a bad-but-complete pipeline, so there is a baseline to measure against.

1. `ingest.py`: parse PDFs (start with `pypdf` or `pdfplumber` for text; note if any
   pages need OCR and handle that explicitly rather than silently dropping them).
2. `chunking.py`: fixed-size chunking (e.g. 500 tokens, 50 token overlap) to start —
   deliberately naive, to be improved with evidence in Phase 3.
3. Embed each chunk, store in `chunks.embedding`.
4. `retrieval.py`: embed the query, cosine-similarity top-k via pgvector *(shipped:
   sqlite-vec, later fused with BM25 — see the note above)*.
5. `llm.py`: stuff top-k chunks into a prompt, ask the LLM to answer **and cite which
   chunk(s) it used**.
6. `POST /query` wires it together end to end.

Done when: you can ask a real question from the corpus and get an answer with a
citation, even if the answer is sometimes wrong.

## Phase 2 — measure before you fix

Goal: know exactly what is broken, with numbers.

1. Write 25-40 real questions into `eval/golden_set.json`, each with the actual
   expected chunk id(s) and (where objective) the expected answer. Write these by
   hand, from the real corpus — this is the part that cannot be automated or faked.
2. `run_eval.py`: for each question, run retrieval, compute:
   - **recall@k** — was the correct chunk in the top-k?
   - **precision@k**
   - **MRR** (mean reciprocal rank)
   - Store the full result as one row in `eval_runs`, so every future change is
     compared against history, not just the last run.
3. `test_retrieval_recall.py`: assert recall@5 stays at or above whatever the current
   number is — this becomes a **regression test**, exactly like the isolation tests
   in AgencyDesk. It should fail loudly if a later change makes retrieval worse.

Done when: you have a real number (e.g. "56% recall@5") and know, question by
question, which ones are failing and can eyeball *why* (bad chunk boundary? wrong
phrasing in the query? number split across a table row?).

## Phase 3 — fix specifically what Phase 2 found broken

Do not apply "best practices" speculatively. Only fix what the eval showed is
actually failing, in priority order of biggest recall drop:

- Tables split mid-row across chunks → structure-aware chunking (keep table rows
  intact; chunk by section/heading boundary instead of raw token count).
- Right document, wrong specific chunk → try smaller chunks with more overlap, or
  add the re-ranker (embed top-20, re-rank down to top-5 with the cross-encoder).
- Query phrasing doesn't match document phrasing → query rewriting (ask the LLM to
  restate the question in the document's likely vocabulary before embedding it).

After each individual change: re-run `run_eval.py`, compare the new `eval_runs` row
to the previous one, and only keep the change if the number actually improved.
Record every attempt (including ones that made it worse) in the README — that list
is itself interview material.

## Phase 4 — the defensible layer

1. Every answer from `/query` returns citations with page numbers, not just chunk ids.
2. `GET /eval/run` is callable on demand and returns the metrics live.
3. CI (`ci.yml`) runs on every push: build the container, ingest the corpus, run
   `pytest`, and **assert recall@5 is at or above the committed threshold** — same
   pattern as the AgencyDesk CI asserting `rolbypassrls = f`, not just "the app booted".
4. README contains, prominently, a **before/after table**. What it actually holds
   (the figures below were illustrative when this plan was written; these are
   measured):

   | Change | recall@5 | notes |
   |---|---|---|
   | Naive fixed-size chunking | 0.367 | baseline, 30-question set |
   | + structure-aware chunking | 0.467 | facts no longer diluted across page-spanning windows |
   | + re-ranker | 0.500 | |
   | + hybrid dense + BM25 (RRF) | 0.733 | biggest retrieval-design gain |
   | *(golden set doubled to 60 — harder, so not comparable to the rows above)* | 0.650 | |
   | + re-spacing welded PDF text | 0.683 | |
   | + dictionary word segmentation | **0.767** | biggest single gain overall |

   This table is worth more in an interview than any other part of the repo — and
   the rejected rows are worth as much as the kept ones.

## Deliverables checklist

- [x] Corpus chosen and documented, with a note on why it's a good stress test
      — Berkshire annual reports 2021–2023: one company across three years, so an
      answer must come from the *right* year's table.
- [x] ~~`docker compose up` runs the whole thing end to end~~ **— deliberately
      dropped.** Docker Desktop cannot run on this machine (Windows 11 Home / VBS),
      so Postgres + pgvector was replaced by **SQLite + sqlite-vec**, which needs no
      daemon at all. `app/vectorstore.py` is the single seam that would be swapped
      back. The constraint turned out to be a net win: the whole stack now runs from
      `alembic upgrade head` + `uvicorn`, and CI needs no services.
- [x] `eval/golden_set.json` — real, hand-written questions
      — 60 questions, every answer verified to occur in a real chunk; audited to 0
      unanswerable and 0 over-broad.
- [x] `run_eval.py` producing recall@k / precision@k / MRR, stored per-run
- [x] At least one committed regression test on recall@5
      — CI-gated floor, raised with each kept improvement (currently 0.66).
- [x] Re-ranker implemented and A/B'd against no-re-ranker, with the number
      — and three further experiments **rejected** on the number: table-append
      ingestion, a deeper re-rank pool, and letter↔digit splitting.
- [x] Citations in every answer, with page numbers
      — the *printed* page label (`page K-83 (PDF page 98)`), parsed from the
      answer's own `[n]` markers, with inferred fallback when the model cites nothing.
- [x] CI asserting the recall threshold, not just "it built"
- [x] README with the before/after table front and center

## Explicitly out of scope

- Multi-hop / agentic reasoning
- A polished frontend
- Support for more than one corpus/domain at once
- Fine-tuning anything (that is a different project — see the fine-tuning idea
  discussed separately)

## The one sentence to have ready

*"My naive retrieval got the right chunk a third of the time. I built an eval set of
real questions, and it kept telling me I was wrong about what was broken — the biggest
win wasn't the re-ranker, it was discovering that PDF extraction had welded words
together so the keyword index couldn't match them at all. Recall went from 0.37 to
0.77, and three of the changes I tried got rejected on the number. Here's the table."*

### What the plan got wrong (worth saying out loud)

This document was written before any of the work. Keeping its mistakes visible is more
useful than editing them away:

- **The expected culprit was chunking splitting tables mid-row.** It contributed, but
  the two largest wins came from *hybrid dense+BM25 retrieval* and from repairing
  **PDF text extraction** — `pdfplumber` welds words together
  (`MitsubishiCorporation`, `investmentsinequitysecurities`), and FTS5 turns each weld
  into a single token, so BM25 was blind to ~10% of all text.
- **Docker was assumed.** It wasn't available, and the constraint improved the design.
- **The eval set was assumed adequate at 30 questions.** It wasn't: a diagnosis run on
  it pointed at the wrong next lever, and *reversed itself* once the set was doubled.
  Sizing the measurement instrument mattered more than any single tuning pass.
