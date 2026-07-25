---
name: run-eval
description: Run the CiteRAG retrieval eval harness (recall@k / precision@k / MRR over the golden set), record the eval_runs row, and update the before/after table in the README. Use when asked to measure retrieval quality, get a baseline number, or compare a change.
---

# Run the CiteRAG eval harness

The whole project exists to produce real before/after numbers. This skill runs the
harness and records the result so every change is comparable against history.

## Prerequisites
- Stack running and corpus ingested (see the `run-citerag` skill).
- `backend/app/eval/golden_set.json` has questions. If it is empty, the harness
  reports `n_questions: 0` — you must write the golden set first (see the
  `add-golden-questions` skill). Do NOT proceed to "fix" anything before this.

## Steps
1. Run the eval (persists one `eval_runs` row by default):
   ```bash
   curl -s 'localhost:8000/eval/run?top_k=5'
   ```
   Or against a specific config, e.g. with re-rank on:
   ```bash
   curl -s 'localhost:8000/eval/run?top_k=5&rerank=true'
   ```
   Or directly without the API (from `backend/`):
   ```bash
   python -m app.eval.run_eval
   ```
2. Read the returned `metrics`: `recall_at_k`, `precision_at_k`, `mrr`, and
   `per_question`. Inspect `per_question` for the failing questions (`hit: false`)
   and reason about *why* each failed (bad chunk boundary? number split across a
   table row? query phrasing mismatch?).
3. Record the number:
   - Add/refresh a row in the README **Results (before/after)** table: the change
     name, the recall@5, and a one-line note on what moved.
   - Every run is already stored in `eval_runs` with its exact config, so trust
     that table for history rather than re-deriving.

## Non-negotiable
- **Measure before you fix.** Never make a retrieval change without a recorded
  baseline eval to compare against.
- When you establish a trustworthy baseline, commit it as the threshold in
  `backend/tests/test_retrieval_recall.py` (`RECALL_AT_5_THRESHOLD`) so a future
  regression fails loudly.
