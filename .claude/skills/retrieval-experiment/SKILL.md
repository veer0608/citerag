---
name: retrieval-experiment
description: Run a disciplined Phase 3 retrieval-improvement experiment for CiteRAG — change one thing, re-run the eval, keep it only if the number improved, and log the attempt. Use when improving chunking, adding the re-ranker, query rewriting, or otherwise tuning retrieval.
---

# Retrieval experiment (Phase 3)

Fix only what the eval showed is actually broken. Do not apply "best practices"
speculatively. Work in priority order of biggest recall drop.

## The loop (one change at a time)
1. **Pick the target from the eval.** Read the latest `per_question` detail. Identify
   the dominant failure mode among `hit: false` questions:
   - Table split mid-row across chunks → structure-aware chunking (keep table rows
     / section boundaries intact instead of raw token windows). Edit `chunking.py`.
   - Right document, wrong specific chunk → smaller chunks + more overlap, or turn on
     the re-ranker (`RERANK_ENABLED=true`, embed top-20 → re-rank to top-5).
   - Query phrasing doesn't match document phrasing → query rewriting before embed.
2. **Change exactly one thing.** Keep `chunking.py` isolated and unit-tested; if you
   change chunking, re-ingest (`seed_corpus.py`) so chunks reflect the new strategy.
3. **Re-run the eval** (`run-eval` skill) and compare the new `eval_runs` row to the
   previous one.
4. **Keep or revert by the number.** Keep the change only if recall@5 (or the metric
   you're targeting) actually improved. If it regressed or was flat, revert it.
5. **Log every attempt** — including failures — as a row in the README before/after
   table with a one-line note. That list is itself interview material.

## Guardrails
- Never tune against the eval so hard that you overfit it — the golden set is small.
  If you suspect overfitting, add a few more real questions first.
- After a genuine improvement, bump `RECALL_AT_5_THRESHOLD` in
  `backend/tests/test_retrieval_recall.py` to the new committed floor.
- Stay in scope: no multi-hop/agentic reasoning, no frontend polish, one corpus.
