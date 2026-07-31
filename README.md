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
| `POST` | `/ingest` | Ingest one PDF from the corpus dir (`{"path": "berkshire_2023.pdf", "title": "..."}`) |
| `POST` | `/query` | Retrieve top-k chunks + answer with citations |
| `GET`  | `/eval/run` | Run recall@k / precision@k / MRR over the golden set — read-only, writes nothing |
| `POST` | `/eval/run` | Same, and record the result as an `eval_runs` row |
| `GET`  | `/health` | Effective config (embedding model/dim, reranker, LLM backend) |

`/ingest` only reads PDFs **inside the corpus directory** (`CORPUS_DIR`, default
`backend/data/corpus`). Both sides of the path are resolved before comparison, so `..`
traversal and symlinks can't escape it — the endpoint is unauthenticated, so without
that check any caller could name an arbitrary server file and get its text back.

`/query` responses carry a **`score_type`** (`cosine` ~0–1, `rrf` ~0–0.05, or
`cross-encoder`, unbounded) naming the scale each `score` is on — the three stages
produce numbers on incompatible ranges, so a bare score can't be compared across
configs or rendered as a bar without it.

**`citations` are the passages the answer actually cited**, parsed from the `[n]`
markers in the answer text — a *subset* of `chunks`, which is the full retrieved pool
the model was shown. Markers outside that range are dropped (a model inventing `[9]`
against 5 passages has cited nothing real). When an answer asserts something and cites
nothing, the response sets **`uncited: true`** and the UI labels it unverified rather
than attaching the retrieved pool as if it were support.

**When the model cites nothing, citations are reconstructed rather than abandoned.**
The default local model (`Qwen2.5-0.5B-Instruct`) does *not* follow the marker
instruction — it answers correctly but silently. Rather than leave those answers with no
provenance, the answer's distinctive figures are matched back against the retrieved
passages, and the passages containing them are returned with **`inferred: true`** plus
the `matched_figures` that justified each one. The UI renders these dashed and labelled,
because reconstructed provenance is weaker evidence than a citation the model declared.

Guards keep inference from manufacturing support: a figure must look like a claim
(decimal, `%`, or 3+ digits — so date components like "January **31**" are skipped),
bare years are ignored as boilerplate, and at most 3 passages are attributed so an
echoed figure can't quietly re-attach the whole retrieved pool.

Worked example — the local model answers *"an additional 41.4% interest in Pilot Travel
Centers on January 31, 2023"* with no markers, and inference attributes it to the three
passages that actually contain "41.4%" (verified: they read *"an agreement to acquire an
additional 41.4% of Pilot"*), on printed pages `K-58`, `K-85` and `K-112`.

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
- **Phase 2 — measure before you fix.** ✅ 30-question hand-written golden set (later
  doubled to 60); baseline recall@5 = 0.367 recorded and committed as a regression gate.
- **Phase 3 — fix what the eval found broken.** ✅ structure-aware chunking
  (0.367 → 0.467) then re-ranker (0.467 → 0.500), each measured and kept only
  because the number moved.
- **Phase 4 — the defensible layer.** ✅ citations with page numbers, live
  `/eval/run`, and CI that ingests the corpus and asserts the recall gate.
- **Phase 5 — hybrid retrieval.** ✅ dense (sqlite-vec) + lexical (SQLite FTS5/BM25)
  fused with reciprocal rank fusion, then re-ranked. recall@5 0.500 → 0.733 — the
  biggest single lever, and free of any new model or service.
- **Phase 6 — diagnose, then trust the diagnosis only as far as the data allows.** ✅
  Located the answer-bearing chunk for every failing question, which stopped a planned
  change (query rewriting looked worthless) and killed a tried one (a deeper re-rank
  pool measured worse). Then doubled the golden set to 60 questions — and the same
  diagnosis, rerun, **reversed itself**: query rewriting went from the least valuable
  lever to the most. See *Where the misses actually are* below.

## Results (before/after)

Measured over a hand-written golden set (`backend/app/eval/golden_set.json`) spanning
the 2021–2023 reports, top_k=5. This table is the point of the project.

> **The golden set was expanded from 30 to 60 questions**, so the table below and the
> current baseline are measured on **different question sets and are not directly
> comparable**. The improvement arc (0.367 → 0.733) is the real, sequential history on
> the original 30; the 60-question restatement follows it. Nothing about retrieval
> changed between them.

| Change | recall@5 | precision@5 | MRR | notes |
|---|---|---|---|---|
| Naive fixed-size chunking (500/50) | 0.367 | 0.10 | 0.188 | baseline — 11/30. |
| + structure-aware chunking (page-bounded, 220-tok, line-preserving) | 0.467 | 0.147 | 0.302 | +0.100. 14/30. Diagnosis: naive windows spanned pages and diluted specific facts; smaller page-bounded chunks concentrate them. |
| + re-ranker (bge-reranker-base, top-20 → top-5) | 0.500 | 0.173 | 0.365 | +0.033 recall, but MRR 0.302 → 0.365 — the right chunk, when found, ranks higher. 15/30. |
| + hybrid retrieval (dense + BM25/FTS5, RRF-fused) | 0.633 | 0.193 | 0.458 | +0.133 (measured with rerank off, to isolate the fusion). 19/30. Diagnosis: the corpus is full of exact tokens — dollar amounts, tickers, years — that dense embeddings blur; a keyword index nails them. |
| + hybrid **and** re-ranker (default config) | **0.733** | **0.220** | **0.532** | +0.100 on top of hybrid. 22/30. Re-ranker orders the richer fused pool better than it did the dense-only one. |
| + table-aware ingestion (serialize table rows) — **rejected** | 0.667 | 0.193 | 0.506 | −0.067. 20/30. Appending `extract_tables()` rows duplicates numbers the text extractor already caught; the near-duplicate chunks crowd the re-ranker and displace the answer-bearing narrative chunk. Kept behind `TABLE_EXTRACTION_ENABLED` (off) as a recorded experiment. |
| deeper re-rank pool (`RERANK_CANDIDATES` 20 → 50) — **rejected** | 0.700 | 0.213 | 0.502 | −0.033. 21/30. Chosen because three answers sat at ranks 21–29, outside the pool the cross-encoder ever sees. Widening the pool does surface them, but the extra distractors cost more than they recover — the re-ranker, not the pool depth, is the binding constraint. Reverted. |

**Net on that set: recall@5 0.367 → 0.733 (2×), MRR 0.188 → 0.532 (2.8×).** The two
biggest levers were structure-aware chunking (+0.100) and hybrid retrieval (+0.133). Not
every idea helped: table-aware ingestion and a deeper re-rank pool both measured *worse*
and were rejected rather than shipped on faith — the point of the harness.

### Restated on the expanded 60-question set

The original set was too small to justify further tuning: at n=30 a single question is
±0.033, so every remaining candidate change sat inside the noise. The set was doubled to
60 questions (**1 question = ±0.017**), deliberately weighted toward the failure modes
the diagnosis had exposed — equity-holdings table lookups, facts that differ across the
three years, and questions phrased away from the document's own wording.

| Change | recall@5 | precision@5 | MRR |
|---|---|---|---|
| hybrid, no re-ranker | 0.600 (36/60) | 0.157 | 0.474 |
| hybrid + re-ranker | 0.650 (39/60) | 0.183 | 0.519 |
| **+ re-spacing welded PDF text** (no re-ranker) | 0.667 (40/60) | 0.183 | 0.551 |
| **+ re-spacing welded PDF text** + re-ranker (**current default**) | **0.683 (41/60)** | **0.197** | **0.587** |

The first two rows are lower than 0.733 because the questions are harder, not because
retrieval regressed — the code was identical. That 0.650 was the baseline the next
change had to beat.

**Re-spacing welded text: +0.033 with the re-ranker, +0.067 without it** (and MRR
0.519 → 0.587). The larger gain on the no-re-ranker path is exactly what the mechanism
predicts: the defect was blinding BM25, so fixing it shows up most clearly where the
cross-encoder isn't papering over the ranking. Details below.

Every added question was written from the corpus text and its answer verified to occur
in a real chunk; the page numbers are derived from where the answer actually sits rather
than read off a viewer. Three drafts were thrown out because their answer substrings
(`8.3`, `1.3 billion`, `Alleghany`) appear on 18–35 pages — too common to be honest
ground truth, since any chunk containing them would score as a hit.

**What "recall@5" means here (verified, not assumed):** a question counts as hit only
when a retrieved chunk *actually contains the answer* (whitespace-insensitive substring
of the expected answer), not merely a chunk from the expected page. The eval also reports
a looser `page_recall@5` for continuity. Tightening this in `run_eval.py` left the strict
number unchanged (0.467 / 0.500) — confirming the headline was already answer-bearing —
but the low `page_recall@5` (0.167) prompted a look at page numbering, which turned up a
real citation defect: ingest stored only the 1-based *physical* PDF index, never the number
printed on the page. Printed labels are now extracted at ingest; see the citations note below.

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

### Where the misses actually are — and why the small set misled us

Measured, not guessed: for every golden question, the answer-bearing chunk is located in
a depth-50 hybrid pool to separate *"never retrieved"* from *"retrieved but out-ranked"*.
Those need different fixes, so the split decides which lever is worth building.

Running it on both sets is the most instructive result in this project:

| Failure mode | n=30 | n=60 | What would fix it |
|---|---|---|---|
| Answer never retrieved | 1 | **10** | query rewriting / HyDE — vocabulary mismatch |
| Retrieved at rank 7–20 (inside the re-rank pool) | 5 | 9 | a stronger re-ranker |
| Retrieved at rank 21+ (outside the pool) | 3 | 3 | a deeper pool — *tried, made things worse* |
| Already in top-5 | 21 | 38 | — |

**The 30-question set pointed at the wrong lever.** On it, query rewriting looked like
the least valuable change available — one question, +0.033, indistinguishable from
noise — and re-ranker quality looked like the binding constraint. On 60 questions the
never-retrieved bucket is the *largest* single failure mode: 10 questions, 45% of all
failures, a ceiling of +0.167.

Nothing about the system changed between those two measurements. The first diagnosis was
simply built on too few observations to be trustworthy — a one-question bucket is not a
signal. **Expanding the golden set before tuning was worth more than any tuning pass
would have been**, because it changed which experiment is worth running at all.

### The never-retrieved bucket had nothing to do with vocabulary

Reading those 10 failures instead of trusting the bucket label found the real cause.
`pdfplumber` frequently emits words **welded together** — the equity-holdings table
renders as `907,559,761 AppleInc. ... 1,032,852,006 BankofAmericaCorp.` FTS5 tokenizes
on non-alphanumerics, so each weld becomes **one token**. Checked directly against the
index:

| query term | before | after |
|---|---|---|
| `"mitsubishi"` | **MISS** | HIT |
| `"america"` | **MISS** | HIT |
| `"MitsubishiCorporation"` | HIT | — |

BM25 — the half of hybrid retrieval that produced the single biggest win in this
project — was **structurally blind** to holdings-table rows for any company-name query,
while the dense half saw only a wall of digits. That is why those questions were never
retrieved at any depth.

**Query rewriting could not have fixed this.** No rephrasing of "how many Mitsubishi
shares" yields a token matching `MitsubishiCorporation`. Had HyDE been built first — as
the n=60 diagnosis appeared to justify — it would have measured flat on exactly these
questions and the technique would have taken the blame.

The fix is at ingest: split tokens at case boundaries (`lower→Upper` and
`ACRONYM→Word`) when they're long enough to be suspicious, leaving ordinary prose
untouched. It's deterministic, costs nothing per query, and repairs the lexical and
dense paths at once. Toggle with `SPLIT_RUN_TOGETHER_ENABLED`.

Known limits, pinned by tests rather than left as surprises: splitting needs surviving
capitals, so an all-lowercase weld (`Ofequalimportance,floatisverysticky`) stays welded;
and `BankofAmericaCorp` becomes `Bankof America Corp` — "america" is searchable, "bank"
still isn't. Partial repair, but enough to reach the row.

**A note on measuring the mechanism, not a proxy.** The first attempt to test this used
an aggregate "share of very long words" metric, and it *contradicted* the hypothesis —
gold chunks scored 0.112 against a corpus average of 0.165. The aggregate was simply the
wrong instrument: it measured whole-chunk noise rather than whether the query's entity
was matchable. One targeted FTS5 lookup settled it.

Remaining candidates: a **stronger cross-encoder** (the rank 7–20 bucket), **query
rewriting / HyDE** for genuine vocabulary gaps, and a table strategy that *replaces*
garbled page text instead of appending it. A deeper re-rank pool is ruled out.

**Citations quote the page a reader actually sees.** Ingest reads each page's *printed*
label off the page and stores it next to the 1-based physical PDF index, so a citation
reads `page K-83 (PDF page 98)` — the first half matches the paper report, the second
half is what a PDF viewer's page box wants.

This mattered more than a fixed offset would suggest: these reports use **two different
numbering schemes** — plain integers in the shareholder letter, a `K-` prefix in the 10-K
— so the printed number can't be derived from the physical index by any arithmetic. It
has to be read off the page. 98% of chunks get a label; the rest (covers, section
dividers, back matter) legitimately print none and fall back to the physical index.

**What this did *not* fix:** the low `page_recall@5`. The obvious theory was that the
golden set's `expected_page_numbers` were printed labels — so I measured it, and they
aren't: they track the *physical* index (agreeing on 14/30 questions vs 6/30 for labels),
and they're often off by one (an answer on physical 99 recorded as 98), consistent with
being read by eye off a PDF viewer. So page matching in the eval stays on the physical
index, and `page_recall` remains only an indicative signal — which is exactly why the
headline metric is answer-bearing recall instead.

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
`HYBRID_ENABLED`, `HYBRID_CANDIDATES`, `RRF_K`, `CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`,
`TABLE_EXTRACTION_ENABLED` (off — a rejected experiment).
