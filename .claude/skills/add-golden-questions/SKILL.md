---
name: add-golden-questions
description: Hand-write real golden-set questions for CiteRAG from the actual corpus (Phase 2), with correct ground-truth matching. Use when building or extending backend/app/eval/golden_set.json or when asked to create the eval question set.
---

# Add golden-set questions (Phase 2)

The golden set is the part of this project that **cannot be automated or faked**.
Each question must come from reading the real corpus and knowing the true answer.
Aim for **25–40 questions** total.

## How to write a good question
- Ask something with an objective, checkable answer that lives in a specific place
  in the corpus — a number in a table, a figure from a specific year, a named fact.
- Prefer questions that stress retrieval: exact numbers, values that differ across
  years (which forces retrieving from the *right* year's document), figures inside
  tables and footnotes.
- Phrase the question in your own words, NOT by copying the document sentence — the
  point is to test whether retrieval bridges the vocabulary gap.

## Ground-truth matching (how a hit is judged)
A question counts as retrieved-correctly if **any** top-k chunk satisfies the
criteria you provide. Provide at least one of:

- `expected_page_numbers`: a retrieved chunk from one of these pages counts as a hit.
- `expected_answer_substring`: a retrieved chunk containing this text counts as a hit.
- `expected_chunk_ids`: exact chunk-id match. **Fragile** — only stable while the
  chunking config is unchanged; it breaks when Phase 3 re-chunks. Avoid relying on
  it alone.

**Prefer page + substring matching** so the golden set survives re-chunking in
Phase 3. To find the right page/text, query the running system or read the PDF.

## Format
Edit `backend/app/eval/golden_set.json`, appending to `questions`:
```json
{
  "question": "By how much did Berkshire's insurance float grow from 2021 to 2022?",
  "expected_answer": "roughly $147B to $164B",
  "expected_page_numbers": [7],
  "expected_answer_substring": "float"
}
```
`expected_answer` is for human reference; matching uses the fields above.

## After writing
1. Run the eval (`run-eval` skill) to get the **baseline** recall@5.
2. Record that baseline in the README before/after table as the first row.
3. Only now is Phase 3 (fixing retrieval) allowed to begin.
