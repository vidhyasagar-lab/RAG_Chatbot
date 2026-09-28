# Answer Reasoning and Honest Rewrite Scores - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the three partial answers from the 2026-09-28 re-test (8.75/10): compute totals from listed parts, answer "why might" questions from the facts, and stop showing a rejected draft's score on the rewrite that replaced it.

**Architecture:** Two prompt edits in `app/core/rag_engine.py`; one extra faithfulness pass in the background gate on a rewrite, stored beside the draft's score in `gate_results`; the UI reads the draft's score for the rejection reason and the rewrite's for the badge.

**Tech Stack:** FastAPI, pytest, SQLite; Next.js, vitest.

**Spec:** the four-point fix approved in chat on 2026-09-28 (no spec file):
1. Totals from parts: a worked example that sums listed parts and gives percentages; if the parts may be incomplete, say the answer assumes they are, instead of refusing.
2. "Why might / how could" questions are answered from the facts that bear on them, like judgment questions; no opening "the context doesn't explain" when the facts support an answer.
3. Retry prompt: state supporting facts in plain words, no quoted "(stated ...)" labels after each claim.
4. After a rewrite, re-check the rewrite's faithfulness in the background and show that score; the rejected draft's score never appears on its replacement.

## Global Constraints

- Worked examples must not use terms from the review document (`REVIEW_TERMS` in tests/test_prompts.py).
- Nothing new blocks the stream: the extra scoring runs inside the background gate only.
- A rewrite whose re-check fails or times out shows "Not scored", never the draft's score.
- Never `git add -A`; no attribution lines in commits.

## Review Focus

1. Re-check times out -> rewrite faithfulness is None, draft score kept only as `draft_faithfulness`.
2. Old `gate_results` rows (no `draft_faithfulness` column) -> migration adds it; reads return None.
3. Chat reloaded after the gate finished -> stored meta carries the rewrite's score, not the draft's.
4. Gate finished before `_persist` (early path) -> same fields as the late path.
5. A passed answer -> `draft_faithfulness` is None and nothing is re-scored.

---

### Task 1: Answer prompt - totals from parts, and "why might" questions

**Files:** Modify `app/core/rag_engine.py` (SYSTEM_PROMPT); Test `tests/test_prompts.py`

- [ ] Step 1: failing tests `test_a_total_is_summed_from_its_listed_parts` (prompt contains "add them" and an example whose Good answer gives percentages that sum from listed parts), `test_an_explanation_question_is_answered_from_the_facts` ("why might" in the prompt; a Bad example opening "The context does not explain").
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: edit the prompt: rule + a totals example (regional sales) + a why-might example (two branches), unrelated to the review document.
- [ ] Step 4: run -> PASS; full suite green; commit.

### Task 2: Retry prompt - facts in plain words

**Files:** Modify `app/core/rag_engine.py` (REFINED_SYSTEM_PROMPT); Test `tests/test_prompts.py`

- [ ] Step 1: failing test `test_the_retry_prompt_forbids_quoted_evidence_tags` (prompt says "plain words" and forbids "(stated" tags).
- [ ] Step 2: FAIL. Step 3: edit. Step 4: PASS, suite, commit.

### Task 3: Backend - score the rewrite, keep the draft's score apart

**Files:** Modify `app/core/rag_engine.py` (`_run_gate`), `app/core/eval_store.py` (column + fields), `app/api/routes/chat.py` (`_persist` early path); Test `tests/test_streaming_gate.py`, `tests/test_gate_results.py`

**Produces:** gate result dict gains `draft_faithfulness: float | None`; `get_gate_result` returns it; stored message meta `eval` gains `draft_faithfulness`.

- [ ] Step 1: failing tests: a rejected-then-rewritten answer saves the rewrite's faithfulness as `faithfulness` and the draft's as `draft_faithfulness`; a rewrite whose re-check raises saves `faithfulness=None`; a passed answer has `draft_faithfulness=None`; the revision meta carries both; `save/get_gate_result` round-trips `draft_faithfulness`, including on a table created without the column.
- [ ] Step 2: FAIL. Step 3: implement (re-score with `_run_metric` under `_EVAL_SLOTS` and `eval_timeout_seconds`; `ALTER TABLE` migration; `_persist` passes the field). Replace `test_the_regenerated_answer_is_not_gated_again` with a test that the rewrite is scored for faithfulness only and never regenerated again.
- [ ] Step 4: PASS, suite, commit.

### Task 4: UI - rejection reason uses the draft's score

**Files:** Modify `Chatbot-ui/src/lib/gate.ts`; Test `Chatbot-ui/src/lib/gate.test.ts`

- [ ] Step 1: failing test: with `faithfulness: 0.86, draft_faithfulness: 0.43, revised_answer` set, the reason reads "faithfulness 0.43 < 0.50" and the badge scores carry 0.86; missing `draft_faithfulness` (older backend) falls back to `faithfulness`.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: vitest, tsc, eslint green; commit.

### Task 5: Deploy and validate

- [ ] Deploy backend to the VM (`down`/`up -d`), push UI.
- [ ] Re-run the Q2, Q7 and Q10 questions on the VM through `ask_stream` + gate; record answers, verdicts and scores.
