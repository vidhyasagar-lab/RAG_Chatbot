# Retrieval accuracy and answer relevance

**Status:** approved design, 2026-09-28
**Branch:** `headless-api`
**Scope:** backend (`app/`), plus one stage label in `Chatbot-ui`

## Problem

A 10-question manual review against `Global_Renewable_Energy_Outlook_2026.docx`
scored 5/10 strict, 6/10 lenient. Diagnosis on the live index (204 chunks)
found three retrieval defects and one prompt defect. Multi-hop reasoning was
suspected, but it was not the main cause.

| # | Defect | Evidence | Questions hit |
|---|--------|----------|---------------|
| D1 | Tables and figures carry no section breadcrumb | 0/34 table chunks, 6/29 chart chunks and 8/9 image chunks have `Section:` in their header | Q2, Q6, Q10 |
| D2 | Heading-only chunks outrank real content | 23/204 chunks have <40 chars of body; `## Composite ranking` (empty) ranks #1 for Q6 | Q6 |
| D3 | `top_k_results = 5` is too small | Table 17 (the answer to Q2/Q6/Q10) first appears at top_k=15 | Q2, Q6, Q10 |
| D4 | The prompt forbids inference | "Do not answer from your own knowledge and do not speculate" yields "insufficient information" for Q3, whose answer is entailed by retrieved Table 12 | Q3 |

Secondary finding: python-docx repeats a horizontally merged cell's text
across its span. Table 12's title row renders four times. Noisy, not lossy.

### Root cause of D1

`preprocess_docx` (`app/core/preprocessing.py:431`) makes three independent
passes: paragraphs (joined into one Markdown-headed text blob), `docx.tables`,
and `docx.part.rels` for images. Only the text blob goes through
`MarkdownHeaderTextSplitter`, which is what sets `metadata["section_header"]`.
Tables and images are emitted as separate documents, never learn which heading
they sat under, and so `_inject_context_headers` has no breadcrumb to add.

## Goals

- Q2, Q6, Q10 answered correctly after Phase 1.
- Q3 answered, with its evidence named, after Phase 2.
- No regression on the questions that currently pass.
- Time to first token for questions that do not trigger a follow-up stays
  within 5s (3.8s today).

## Non-goals

- **Docling.** Evaluated and rejected: it fixes the same questions Phase 1
  fixes, needs ~2.3 GB for its PDF pipeline on a 2 GB VM, replaces five
  type-specific vision prompts with one generic one, and `contextualize()` keeps
  only the nearest heading (docling#2055). That loses `9 Market attractiveness`,
  the level holding the word the Q6 query uses.
- PDF breadcrumbs. PDF keeps its current path; there is no measured failure.
- Rerankers or a different embedding model. Recall is not the bottleneck;
  structure is.
- An open-ended agentic retrieval loop.
- A golden-set evaluation harness. The 10 questions stay a manual re-test.

## Design

Three phases, shipped in order. Each is deployed and re-tested before the next,
so a score change can be attributed to the phase that caused it.

---

### Phase 1: Make the answers findable (D1, D2, D3)

#### 1.1 Walk the DOCX body in document order

Replace passes 1–3 of `preprocess_docx` with one walk over
`docx.element.body`'s children, tracking a heading stack.

- `w:p` with a `Heading N` style: pop the stack to depth N-1, push the text.
  Append `"#"*N + " " + text` to the text stream, exactly as today.
- Other `w:p`: append its text to the text stream. For each inline image in the
  paragraph (`a:blip/@r:embed` → `docx.part.related_parts[rId]`), emit a visual
  document as today (`_defer_description`), with `section_header` set to the
  current breadcrumb.
- `w:tbl`: render it with `_docx_table_to_markdown` and emit a table document
  and a `VisualElement` as today, with `section_header` set to the current
  breadcrumb.

The breadcrumb format matches the one `_structural_split` builds:
`" > ".join(stack)`, e.g. `9 Market attractiveness > Composite ranking`.

`_structural_split` already preserves pre-existing `section_header` metadata
on documents with no Markdown headers (`{**doc.metadata, **split.metadata}`),
so `_inject_context_headers` then produces:

```
[Document: Global_Renewable_Energy_Outlook_2026.docx | Section: 9 Market attractiveness > Composite ranking]
[Table 17]
| Rank | Market | Composite | Change vs 2025 | Band |
```

Preserved behaviour:

- Table numbering (`[Table N]`) is unchanged: `docx.tables` is already
  top-level tables in body order, and the walk visits the same tables in the
  same order.
- Image filenames keep the `{stem}_img{n}` scheme.
- Images under 5,000 bytes are still skipped.
- Image relationships the body never references are still emitted after the
  walk, with no breadcrumb, so no image extracted today is lost.
- `_classify_visual` now receives the real preceding and following paragraphs
  instead of `_get_surrounding_text`'s index approximation. That function is
  deleted.
- SmartArt extraction (`_extract_docx_smartart`) is untouched and still gets
  no breadcrumb. Out of scope.

#### 1.2 Blank horizontally merged cells

In `_docx_table_to_markdown`, when consecutive cells in a row share the same
underlying `cell._tc` element, emit the text once and blank the repeats:

```
| Capital allocation outlook - 2026 forecast band |  |  |  |
| Low case |  | High case |  |
```

- Uses element identity, not text equality. Text equality falsely flagged RACI
  tables whose adjacent cells legitimately read `I | I`.
- The column count is unchanged, so the Markdown stays well-formed.
- Vertical merges are not blanked. Repeating the value down a column keeps
  each row readable on its own, which is what a retrieved row needs.

#### 1.3 Drop heading-only parents

In `chunk_documents`, after step 4 (the token limit) and before parent IDs are
assigned, drop any parent whose `content_type` is `text` and whose body has
fewer than `min_chunk_body_chars` characters, where body means the text with
Markdown heading lines removed.

The design drops these parents rather than merging them forward. Merging would
attach one section's heading to the next section's text and produce a wrong
breadcrumb. No information is lost: after 1.1, a heading's text survives in
the breadcrumb of every table, figure and subsection beneath it.

New setting: `min_chunk_body_chars: int = 40`.

#### 1.4 Retrieve more, within a token budget

- `top_k_results`: 5 → 12. `fetch_k` follows as `k * 3` (36).
- New setting `max_context_tokens: int = 6000`. After fusion and parent
  expansion, `_build_context` keeps documents in rank order until the next one
  would exceed the budget. It always keeps at least one.

The budget exists because parent expansion lets 12 results reach ~6,000
tokens, against ~2,500 today, on every question.

Known limitation, unchanged by this design: `hybrid_search` filters by
`user_id` after fetching, so in a store shared by many users the effective
candidate pool is smaller than `fetch_k`. The larger `fetch_k` eases this but
does not fix it.

---

### Phase 2: Let the model reason from the evidence (D4)

#### 2.1 Three-way answer rule in `SYSTEM_PROMPT`

Replace the final two "How to answer" bullets and thinking step 3 with:

- **Stated:** the context states the answer → answer it.
- **Entailed:** the context does not state it, but it follows from what the
  context does state → answer it, and name the facts it rests on in the same
  sentence or the next.
- **Unsupported:** neither → say so in one sentence. Never draw on outside
  knowledge.

Naming the evidence does two jobs. It makes the inference checkable by the
reader, and it gives the faithfulness metric literal, supported claims to
score, which keeps a reasoned answer above the gate threshold.

Add one contrastive example for the entailed case. **It must not come from the
review document or resemble any of the 10 review questions.** An example built
from the test set would teach to the test and make the re-test meaningless.

The existing Bad/Good attribution example ("Middle East, 26.3% CAGR") was taken
from the review document, so it is rewritten on a neutral subject too. The
lesson it teaches (no source-naming preamble) stays the same.

Unchanged: the silent chain-of-thought, the ban on source attribution in prose,
the context fence, and the injection defence block and its restatement after
the fence.

#### 2.2 Same rule on regeneration

`REFINED_SYSTEM_PROMPT` currently says "Nothing inferred" and "Delete every
claim you cannot locate". A gate rejection of a correctly reasoned answer
therefore guarantees a worse retry. The refined prompt keeps its audit framing
(list claims, check each, delete what fails), but a claim passes if it is
either located in the context or derived from located facts that the answer
names.

`eval_quality_threshold` stays at 0.5.

---

### Phase 3: At most one follow-up search

For questions whose answer spans sections that one query does not reach.

#### 3.1 Flow

1. Round 1: `hybrid_search(question)`, as today.
2. Coverage check: one chat completion given the question and, for each
   retrieved document, its first line (the `[Document | Section]` header) plus
   the first 100 characters of body. Bodies are not sent in full. The check
   returns JSON `{"missing": bool, "queries": [string, ...]}` with at most 3
   queries.
3. If `missing` is true and `queries` is non-empty: emit
   `stage: "searching"`, run `hybrid_search` for each query, then fuse round 1
   and every follow-up list with the existing `_reciprocal_rank_fusion`, all
   weights equal. Fusion is keyed on content, so documents found in both
   rounds merge rather than duplicate.
4. Truncate to `top_k_results`, then apply the token budget (1.4).
5. Stop. There is never a third round.

The check fails open. On a timeout, an API error, malformed JSON or an empty
query list, round 1's results are used unchanged and no stage event is sent.
A broken check can never cost the reader an answer.

#### 3.2 Components

New module `app/core/followup.py`:

- `coverage_digest(docs) -> str`: builds the header-plus-snippet digest.
- `plan_followups(question, docs) -> list[str]`: runs the check and returns
  zero to three queries. It never raises.
- `retrieve_with_followup(question, k, user_id, on_search=None) -> list[Document]`:
  runs rounds 1–2, fusion and truncation, and calls `on_search()` before
  round 2 when there is one.

`_build_context` is split:

- `_format_context(docs) -> (text, sources, images)`: the existing
  formatting, unchanged.
- `_build_context` calls `retrieve_with_followup`, then the budget, then
  `_format_context`. The non-streaming `ask()` path gets the same retrieval.

In `ask_stream`, retrieval currently runs synchronously on the event loop.
It moves to `asyncio.to_thread`. The stage event must be emitted from the
generator, so `ask_stream` calls round 1, `plan_followups` and round 2 as
separate awaited steps rather than going through `retrieve_with_followup`.
All of this happens before `meta`, so the event contract gains one optional
event:

```
meta ← [stage:searching] ← (retrieval)   then as today:
meta → stage → token+ → [eval → [replace → stage → token+]] → done
```

Settings:

- `followup_retrieval_enabled: bool = True`
- `followup_timeout_seconds: float = 8.0`
- `followup_max_queries: int = 3`

The check uses the main Azure deployment with `max_completion_tokens=400` and
`response_format={"type": "json_object"}`.

#### 3.3 UI

`Chatbot-ui`:

- Add `"searching"` to `ChatStage` (`src/lib/chat-types.ts`).
- Add copy to `STAGE_COPY` (`src/components/chat/assistant-message.tsx`):
  "Looking in a few more places".
- `backend-stream.ts` already forwards `stage` events generically, so it does
  not change.

#### 3.4 Cost

About +3–5s, and only on questions where the check asks for more. Questions
that do not trigger it pay only the check itself, which is small because it
sees headers and snippets, not full chunks. Latency is measured during
implementation. If the check alone adds more than 1.5s to TTFT, that is
reported before the design proceeds.

---

## Testing

Every item is written failing-first.

| Test file | Covers |
|-----------|--------|
| `tests/test_docx_body_walk.py` | A table under `# A` / `## B` gets `section_header == "A > B"`; a table after a new `## C` gets `"A > C"`; an inline image gets the breadcrumb current at its paragraph; an unreferenced image is still emitted; table numbering matches `docx.tables` order. The fixture DOCX is built in the test with python-docx. |
| `tests/test_docx_merged_cells.py` | A horizontal merge renders once and then blanks; the column count is preserved; a vertical merge still repeats; identical adjacent text in *unmerged* cells is kept. |
| `tests/test_chunk_filter.py` | A heading-only text parent is dropped; a short table parent is kept; a parent with ≥40 chars of body is kept. |
| `tests/test_context_budget.py` | Documents are kept in rank order under the budget; an oversized first document is still kept; the default `top_k` is 12. |
| `tests/test_followup.py` | A missing piece produces queries and a fused result containing round 2's documents; `missing: false` means no round 2 and no `on_search` call; timeout, bad JSON and API errors all fall back to round 1; never more than one extra round; at most 3 queries used. The LLM is stubbed. |
| `tests/test_prompts.py` | Both prompts contain the three-way rule and the injection block, contain no "Nothing inferred", and neither prompt contains any of `attractiveness`, `interest rate`, `India`, `offshore`, `renewable` (terms from the review document and questions). |
| `Chatbot-ui/src/lib/backend-stream.test.ts` | A `searching` stage maps onto the status indicator. |

The existing suite (143 tests) must stay green at every phase.

## Rollout

Per phase:

1. Commit to `headless-api`, then push.
2. On the VM, run the existing redeploy (pull, then `docker compose build` and
   `up -d`). Verify inside the container that the new code is running.
3. **Phase 1 only:** delete the document and re-upload it through the UI. This
   re-runs vision over all 49 figures (a few minutes of Azure spend). Phases 2
   and 3 do not change the index.
4. **Phase 3 only:** push `Chatbot-ui` so the new stage label deploys.
5. You re-run the 10 questions by hand.

Rollback: each phase is its own commits. Reverting them and redeploying
restores the previous behaviour. Phase 1's rollback also needs a re-upload.

## Risks

| Risk | Mitigation |
|------|------------|
| Reasoned answers score below 0.5 faithfulness and get regenerated | 2.1 names evidence, so supported claims dominate; 2.2 stops regeneration from being stricter. Watched via the verdicts in Langfuse after Phase 2. |
| Larger context raises TTFT and cost | 6,000-token budget; TTFT is measured after Phase 1 against the 5s goal. |
| The coverage check asks for more on every question | Measured on the 10 review questions; if it fires on most of them, the prompt is tightened before rollout. |
| The body walk misses content the old passes caught | Unreferenced images are emitted after the walk; tests assert table-count parity with `docx.tables`. |
