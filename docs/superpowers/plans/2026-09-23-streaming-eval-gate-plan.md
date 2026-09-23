# Implementation plan — streaming answers with a non-blocking eval gate

Spec: `docs/superpowers/specs/2026-09-23-streaming-eval-gate-design.md`
Branch: `headless-api`

## Global constraints

- Tests run with `uv run python -m pytest` (never bare `pytest` — console-script
  shims are blocked on this machine).
- No test may call Azure. `tests/conftest.py` stubs embeddings; LLM boundaries
  are stubbed per test, matching the style of `tests/test_eval_gating_flag.py`.
- Single uvicorn worker assumption is unchanged.
- The UI lives in a **separate git repo** (`../Chatbot-ui`). Task 5 commits there.

## Review focus

Input classes the tests below do not exercise, to be checked deliberately:

- A client that disconnects mid-attempt-1 (Caddy logged `context canceled`).
- `eval_gating_enabled=False` must still stream and still fire background eval.
- An answer shorter than `eval_max_answer_chars` vs one longer (truncation edge).
- `context_precision` returning `None` (evaluator error) vs `0.0` (real zero) —
  these must not be conflated, `None` is not a retrieval failure.

---

## Task 1 — Eval model seam, telemetry off, timeout setting

**Produces:** `settings.eval_*`; `evaluator._get_eval_llm()`; `RAGAS_DO_NOT_TRACK` set at import.

1. Write `tests/test_eval_config.py` covering: defaults fall back to the main
   model; `eval_endpoint` set overrides only the endpoint; `RAGAS_DO_NOT_TRACK`
   is `"true"` after importing `app.core.evaluator`.
2. Run `uv run python -m pytest tests/test_eval_config.py -q`.
   **Expected:** fails — `eval_model` is not a Settings field.
3. Add the six `eval_*` fields to `app/config.py` with `effective_eval_*`
   properties mirroring the existing `effective_embedding_*` pattern.
4. In `app/core/evaluator.py`: set `os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")`
   at module top, and rename `_get_azure_llm` to `_get_eval_llm` reading the
   `effective_eval_*` settings.
5. Run the test. **Expected:** passes.
6. Commit: `Add eval model seam and disable ragas telemetry`.

## Task 2 — Faithfulness cache key includes the answer

**Consumes:** nothing. **Produces:** `_make_cache_key(question, contexts, answer)`.

1. Write `tests/test_eval_cache_key.py`: saving scores for answer A then reading
   with answer B returns `None`; reading with answer A returns the scores.
2. Run it. **Expected:** fails — same key for both answers, B returns A's score.
3. In `app/core/eval_store.py` add `answer: str = ""` to `_make_cache_key`,
   `get_eval_cache`, `save_eval_cache`.
4. In `app/core/evaluator.py`: `evaluate_faithfulness_sync` passes `answer`;
   `evaluate_context_precision_sync` drops its cache read (answer-independent,
   and it runs concurrently with generation so the cache saved nothing visible);
   `evaluate_query_async` passes `answer` when saving.
5. Run the test. **Expected:** passes.
6. Commit: `Key the faithfulness cache on the answer it scored`.

## Task 3 — Streaming gated pipeline

**Consumes:** Task 1 `_get_eval_llm`, `eval_timeout_seconds`, `eval_max_answer_chars`.
**Produces:** `ask_stream(..., gated: bool | None = None)` async generator emitting
the spec's contract. `ask_with_eval` deleted.

1. Write `tests/test_streaming_gate.py` with the LLM boundary stubbed
   (`_async_generate_answer`, the streaming client, and both evaluator
   functions). Cases, one test each:
   - first `token` precedes any `eval`
   - `rejected` → `replace` → attempt-2 tokens → `done` with `final_attempt: 2`
   - `context_precision == 0.0` → verdict `retrieval_failed`, no attempt 2
   - `context_precision is None` → still `rejected` (not conflated with 0.0)
   - faithfulness raising → verdict `unscored`, draft kept, no attempt 2
   - `gated=False` → `meta → stage → token+ → done`, no `eval`
2. Run. **Expected:** fails — `ask_stream` is a sync generator with no `gated` param.
3. Rewrite `ask_stream` in `app/core/rag_engine.py` per the spec's pipeline:
   async, `AsyncAzureOpenAI` streaming, `context_precision` started concurrently
   with generation, faithfulness under `asyncio.wait_for`, verdict table,
   regeneration only on `rejected`, background eval for attempt 2.
4. Delete `ask_with_eval`.
5. Run. **Expected:** passes.
6. Commit: `Stream tokens immediately and gate without blocking`.

## Task 4 — Attempt-aware persistence

**Consumes:** Task 3's `attempt` / `final_attempt` fields.

1. Extend `tests/test_streaming_gate.py`: after a rejected draft, session history
   contains only the attempt-2 text.
2. Run. **Expected:** fails — history holds draft+replacement concatenated.
3. In `app/api/routes/chat.py`: replace the single `full_answer` accumulator with
   `answers: dict[int, str]`; on `done`, save `answers[final_attempt]`. Drop the
   `ask_with_eval` branch and `iterate_in_threadpool`.
4. Update `tests/test_eval_gating_flag.py` for the new event shapes.
5. Run `uv run python -m pytest tests/ -q`. **Expected:** whole suite passes.
6. Commit: `Persist only the answer that survived the gate`.

## Task 5 — UI adapter (separate repo)

**Consumes:** Task 3's contract.

1. Extend `Chatbot-ui/src/lib/backend-stream.test.ts`: `attempt` routes tokens to
   per-attempt text ids; `replace` closes the draft and emits `data-rejected`;
   `stage` maps to `data-status`.
2. Run `npm test`. **Expected:** fails.
3. Update `backend-stream.ts`: `TEXT_ID` becomes `answer-${attempt}`; handle
   `stage` and `replace`; carry `verdict` and `attempt` on `data-eval`.
4. Add the `rejected` data type to `chat-types.ts`; render it in
   `assistant-message.tsx`.
5. Run `npm test` and `npx tsc --noEmit`. **Expected:** both pass.
6. Commit in the `Chatbot-ui` repo: `Render the eval gate rejecting a draft`.

## Task 6 — Deploy and validate end to end

1. Commit the two VM-only edits that were never in git (`docker-compose.yml`
   localhost bind, `Dockerfile` `--proxy-headers`).
2. Push `headless-api`; pull and rebuild on the VM.
3. Measure time-to-first-token against the live VM with `curl`.
   **Expected:** first `token` event under ~5s.
4. Drive the same question from the Next.js UI against the VM.
   **Expected:** text appears within seconds; eval badge lands later; a rejected
   draft is visibly replaced.
5. Ledger the measured numbers against the spec's Expected outcome table.
