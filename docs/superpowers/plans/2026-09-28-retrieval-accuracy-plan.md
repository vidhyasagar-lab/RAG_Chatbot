# Retrieval Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tables and figures findable, let answers reason from named evidence, and add one bounded follow-up search. Target: Q2/Q3/Q6/Q10 on the 10-question review.

**Architecture:** Phase 1 changes ingestion (DOCX body-order walk, merged cells, heading-only filter, headers on parents) and retrieval size (top_k 12 under a token budget). Phase 2 rewrites both prompts. Phase 3 adds `app/core/followup.py`, used by `_build_context` and by `ask_stream`, which emits a new `searching` stage.

**Tech Stack:** Python 3.12, FastAPI, python-docx 1.2, LangChain, FAISS + BM25, Azure OpenAI; Next.js UI in `../Chatbot-ui`.

**Spec:** `docs/superpowers/specs/2026-09-28-retrieval-accuracy-design.md`

## Global Constraints

- Tests run with `uv run python -m pytest` (console-script shims are blocked on this machine).
- No test may call Azure. Embeddings are stubbed in `tests/conftest.py`; LLM and vision boundaries are stubbed per test.
- Stage every file by name. Never `git add -A` (it once swept up a `.env.bak` with a live key).
- No `Co-Authored-By` or "Generated with" lines in commits.
- The UI is a separate git repo (`../Chatbot-ui`).
- `eval_quality_threshold` stays 0.5.
- The prompts must not contain `attractiveness`, `interest rate`, `India`, `offshore` or `renewable`.

## Rulings made while planning

- **Parents carry the breadcrumb.** `chunk_documents` returns header-free parents, and `add_documents` stores those for parent expansion, so the model's context never showed the `[Document | Section]` line the spec's 1.1 example depends on. `chunk_documents` now returns the header-injected parents. Cost if wrong: about 20 extra tokens per retrieved parent.
- **Heading-only means no body at all, not <40 chars.** A 40-character floor would silently drop real short sections ("## Contact\nops@x.com"). Parents whose non-heading text is empty are dropped; nothing else is. `min_chunk_body_chars` is not added. Cost if wrong: a few near-empty decoys survive (for example "## X\nSee below.").
- **The coverage digest reads metadata, not the first line.** Source, section and content type come from `metadata`, so the digest does not depend on header formatting.

## Review Focus

1. A DOCX whose images are anchored (floating) rather than inline. They must still be emitted with a breadcrumb. `.//a:blip` covers both; the Task 1 test uses an inline picture, and the floating case is checked by review.
2. A picture inside a table cell. It must be emitted once, under the table's breadcrumb, not again as unreferenced. Covered by the Task 1 test `test_a_picture_inside_a_table_cell_is_emitted_once`.
3. The same picture referenced twice. It must be emitted once. Covered by the same test's dedupe on relationship id.
4. The follow-up check returning JSON with extra keys, non-string queries or more than 3 queries. At most 3 string queries are used. Covered in Task 6.
5. A `TOP_K_RESULTS` in the VM's `.env` overriding the new default of 12. Task 8 checks the effective value inside the container.

---

### Task 1: Walk the DOCX body in document order

**Files:**
- Modify: `app/core/preprocessing.py` (`preprocess_docx`; delete `_get_surrounding_text`)
- Create: `tests/docx_fixtures.py`, `tests/test_docx_body_walk.py`

**Interfaces:**
- Produces: table and visual `Document`s from `preprocess_docx` carry `metadata["section_header"]` (`" > "`-joined heading path) when under a heading.

- [ ] **Step 1: Write fixtures and failing tests.** `tests/docx_fixtures.py` builds a noise PNG with zlib (no Pillow dependency) and a DOCX: `# Overview`, paragraph, table 1, `## Composite ranking`, table 2, paragraph with picture A, `# Risks`, `## Supply`, table 3 containing picture B in a cell, paragraph referencing picture A again, and one unreferenced picture C via `part.get_or_add_image`. Tests stub `app.core.vision.describe_images`. Assert:
  - table section headers: `["Overview", "Overview > Composite ranking", "Risks > Supply"]`, labels `[Table 1..3]` in order
  - picture A: `Overview > Composite ranking`; picture B: `Risks > Supply`; C: no `section_header`
  - exactly 3 image docs (A deduped)
  - `text_docs[0]` is the text stream and contains `# Overview` and `## Composite ranking`
- [ ] **Step 2: Run.** `uv run python -m pytest tests/test_docx_body_walk.py -q` → Expected: FAIL (tables have no `section_header`).
- [ ] **Step 3: Implement.** Iterate `docx.iter_inner_content()`, keeping a heading stack of `(level, text)`. Before the walk, reserve `text_docs[0]` as the text placeholder. Tables are emitted via `_emit_docx_table(result, table, n, file_path, section)`. For each block, every `.//a:blip/@r:embed` not yet seen is emitted via `_emit_docx_image(...)` with the current breadcrumb and the text of up to 3 neighbouring paragraphs on each side as classification context. After the walk, image relationships the walk never saw are emitted with no section. Existing behaviour is preserved: the image counter counts every image, including ones under 5,000 bytes, and filenames stay `{stem}_img{n}.{ext}`. After `_resolve_pending_visuals`, drop `text_docs[0]` if the text stream is empty.
- [ ] **Step 4: Run.** Same command → Expected: PASS. Then run the full suite → Expected: green.
- [ ] **Step 5: Commit** `Give DOCX tables and figures the heading they sit under`.

### Task 2: Blank horizontally merged cells

**Files:** Modify `app/core/preprocessing.py` (`_docx_table_to_markdown`). Create `tests/test_docx_merged_cells.py`.

- [ ] **Step 1: Failing tests.** Build a 3-column table: row 0 merged across all columns ("Title"); row 1 "Low" spanning 2 columns and "High"; rows 2–3 with a vertical merge "V" in column 0; row 4 unmerged "I", "I", "x". Expected lines:
  - `| Title |  |  |`
  - `| Low |  | High |`
  - rows 2 and 3 both start `| V |`
  - `| I | I | x |`
- [ ] **Step 2: Run** → Expected: FAIL (`| Title | Title | Title |`).
- [ ] **Step 3: Implement.** In each row, a cell whose `_tc` is the previous cell's `_tc` renders as `""`.
- [ ] **Step 4: Run** → PASS, then run the full suite.
- [ ] **Step 5: Commit** `Render a merged table cell once instead of once per column`.

### Task 3: Drop heading-only parents; parents keep their header

**Files:** Modify `app/core/document_loader.py` (`chunk_documents`). Create `tests/test_chunk_filter.py`.

- [ ] **Step 1: Failing tests** (keep every section under 60 tokens so the semantic splitter, which would need real embeddings, is skipped):
  - `"# A\n\n## B\n\nA real sentence under B."` gives one parent, starting `[Document: r.docx | Section: A > B]`
  - a table doc `"[Table 1]\n| a |"` is kept
  - `"Revenue rose 12%."` with no heading is kept
- [ ] **Step 2: Run** → Expected: FAIL (the heading-only parent is present; parents have no header).
- [ ] **Step 3: Implement.** Add `_has_body(doc)`: true unless the doc's `content_type` is `text` and every non-blank line starts with `#`. Filter `parents` after step 4 of `chunk_documents`. Return `parents_with_headers` in place of `parents`.
- [ ] **Step 4: Run** → PASS, then run the full suite. Fix any test that asserted header-free parents, recording the reason in the ledger.
- [ ] **Step 5: Commit** `Drop chunks that are only a heading, and keep the breadcrumb on parents`.

### Task 4: Retrieve 12, within a token budget

**Files:** Modify `app/config.py` (`top_k_results = 12`, `max_context_tokens: int = 6000`) and `app/core/rag_engine.py` (extract `_format_context(docs)`, add `_apply_budget(docs, max_tokens)`, and have `_build_context` use both). Create `tests/test_context_budget.py`.

**Interfaces:** Produces `_apply_budget(docs: list[Document], max_tokens: int) -> list[Document]` and `_format_context(docs: list[Document]) -> tuple[str, list[dict], list[dict]]`.

- [ ] **Step 1: Failing tests:** `get_settings().top_k_results == 12`; `max_context_tokens == 6000`; the budget keeps a rank-order prefix; an oversized first doc is still kept; `_build_context` with `max_context_tokens` patched small returns only the first doc's text.
- [ ] **Step 2: Run** → FAIL (`5 != 12`; `_apply_budget` missing).
- [ ] **Step 3: Implement.** `_apply_budget` counts with `document_loader._token_len` and stops before the doc that would overflow, always keeping at least one.
- [ ] **Step 4: Run** → PASS, then run the full suite.
- [ ] **Step 5: Commit** `Retrieve twelve chunks under a context token budget`.

### Task 5: Three-way answer rule, in both prompts

**Files:** Modify `app/core/rag_engine.py` (`SYSTEM_PROMPT`, `REFINED_SYSTEM_PROMPT`). Create `tests/test_prompts.py`.

- [ ] **Step 1: Failing tests:** both prompts `.format(context="X")` without error; both contain `--- CONTEXT BEGINS ---`, `--- CONTEXT ENDS ---` and `untrusted`; both contain `name the facts`; neither contains `Nothing inferred`, `do not speculate` or any forbidden term from Global Constraints.
- [ ] **Step 2: Run** → FAIL (`renewable` is present; `name the facts` is absent).
- [ ] **Step 3: Implement.** Thinking step 3 becomes stated / entailed / unsupported. Answer bullets:
  - stated: answer it
  - entailed: answer it and name the facts it rests on
  - unsupported: say so in one sentence; never use outside knowledge

  Replace the Bad/Good example with a neutral one (product-line return rates), and add an entailed example (warehouse stock days). The refined prompt's audit keeps claims that are located in the context, or derived from located facts that the answer names.
- [ ] **Step 4: Run** → PASS, then run the full suite.
- [ ] **Step 5: Commit** `Let answers reason from named evidence, and keep the retry as permissive`.

### Task 6: One bounded follow-up search

**Files:** Create `app/core/followup.py` and `tests/test_followup.py`. Modify `app/config.py` (`followup_retrieval_enabled: bool = True`, `followup_timeout_seconds: float = 8.0`, `followup_max_queries: int = 3`), `app/core/rag_engine.py` (`_build_context`, `ask_stream`) and `tests/test_streaming_gate.py` (`_install` stubs `hybrid_search` and `plan_followups` instead of `_build_context`).

**Interfaces:** Produces:
- `coverage_digest(docs) -> str`
- `plan_followups(question: str, docs: list[Document]) -> list[str]` (never raises)
- `fuse_rounds(first: list[Document], extra: list[list[Document]], k: int) -> list[Document]`
- `retrieve_with_followup(question, k=None, user_id="", on_search=None) -> list[Document]`

- [ ] **Step 1: Failing tests** (Azure client stubbed via `followup._get_client`, search via `followup.hybrid_search`):
  - `missing` true → queries capped at 3; non-string queries dropped
  - `missing` false → `[]`
  - bad JSON, an exception, or the flag off → `[]`
  - the request passes `timeout=8.0`, `response_format` json_object and `max_completion_tokens=400`
  - the digest carries the section and cuts the body at 100 chars
  - `retrieve_with_followup` calls `on_search` once and includes a round-2 doc; with no queries, round 1 is returned unchanged; hybrid_search is called `1 + len(queries)` times
  - in `test_streaming_gate.py`: a `searching` stage precedes `meta` when queries exist, and is absent when there are none
- [ ] **Step 2: Run** → FAIL (no module `app.core.followup`).
- [ ] **Step 3: Implement** the module. `_build_context` becomes `retrieve_with_followup`, then `_apply_budget`, then `_format_context`. `ask_stream` runs round 1 and the check via `asyncio.to_thread`, yields `stage: searching` when there are queries, runs round 2, fuses, budgets and formats.
- [ ] **Step 4: Run** → PASS, then run the full suite.
- [ ] **Step 5: Commit** `Search once more when the first round misses part of the question`.

### Task 7: UI label for the new stage (separate repo)

**Files:** `../Chatbot-ui/src/lib/chat-types.ts`, `src/components/chat/assistant-message.tsx`, `src/lib/backend-stream.test.ts`.

- [ ] **Step 1: Failing test:** a `stage: "searching"` event maps to `data-status` `searching`, typed as `ChatStage`.
- [ ] **Step 2: Run** `npx tsc --noEmit` → FAIL (`"searching"` is not assignable to `ChatStage`).
- [ ] **Step 3: Implement.** Add `"searching"` to `ChatStage`, and `searching: "Looking in a few more places"` to `STAGE_COPY`.
- [ ] **Step 4: Run** `npm test` and `npx tsc --noEmit` → both pass.
- [ ] **Step 5: Commit** in `Chatbot-ui`: `Say so when the answer needs a second search`.

### Task 8: Deploy and validate (needs the user's go-ahead: push and production)

- [ ] Push `headless-api` and `Chatbot-ui`.
- [ ] Redeploy the VM (pull, `docker compose build`, `up -d`). Inside the container, print `top_k_results`, `max_context_tokens` and `followup_retrieval_enabled`. Expected: 12, 6000, True.
- [ ] The user deletes and re-uploads the review document. Re-run the breadcrumb diagnostic. Expected: table breadcrumbs 34/34 and no heading-only chunks.
- [ ] Measure time to first token in the container, with and without a follow-up. Expected: under 5s without one.
- [ ] The user re-runs the 10 questions.
