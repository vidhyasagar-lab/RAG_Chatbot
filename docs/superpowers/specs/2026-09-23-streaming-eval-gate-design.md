# Streaming answers with a non-blocking eval gate

Date: 2026-09-23
Status: approved (conversational design approval, 2026-09-23)

## Problem

A single `/chat/stream` request measured **3m 41s** end to end. Langfuse trace
`30d37ec5cad2be782dba4fd649f6b82d`:

| step | time |
|---|---|
| retrieval | 1.1s |
| llm-completion-initial | 33.5s |
| faithfulness-gate | 1m 32s |
| llm-completion-regenerated | 20.0s |
| *(uninstrumented)* second faithfulness check | ~74s |

Faithfulness evaluation is ~166s of 221s — 75% of the request. The user sees
nothing at all until every one of those steps has finished.

Two root causes:

1. **`EVAL_GATING_ENABLED=true` disables streaming.** `chat.py` picks between
   `ask_stream` (real token streaming, no gate) and `ask_with_eval` (gate, but
   the "stream" is the finished string sliced into 4-character chunks after all
   the waiting is over). Real streaming already exists and is simply unreachable.
2. **The gate runs twice**, and the second run — on the regenerated answer — has
   no Langfuse span, which is why the trace appears to lose 74 seconds.

## What was measured, not assumed

Swapping the evaluator to a smaller model does **not** help. Four runs of the
real `Faithfulness` metric over the same sample:

```
gpt-5.2       decompose  7.8s / 7.7s   verify 14.2s / 17.5s   mean 24.0s
gpt-4.1-mini  decompose  5.0s / 4.9s   verify 34.9s / 41.9s   mean 43.8s
```

gpt-5.2 is **1.8x faster**. All responses were 200 — the mini resource is not
throttled; the model is genuinely slower at the verification call, where output
scales with claim count. An instrumented run also showed the metric makes
exactly **2 HTTP calls**, so the cost is payload size, not retry storms.

**Therefore: the gate cannot be made fast. It must stop blocking.**

## Goals

- Time to first token under ~5s, from 3m 41s.
- Keep eval gating visible and meaningful — it is the portfolio showcase, and a
  reviewer watching the gate reject an answer is a better demonstration than a
  blank screen followed by text.
- Remove work that cannot change the outcome.

## Non-goals

- Making the faithfulness metric itself faster. Measured; not available.
- Retrieval quality. `context_precision=0.0` on a whole-document question is a
  chunking problem and deserves its own round.
- CORS configuration, and the two uncommitted VM-only edits.

## Design

### SSE contract

```
{"type":"meta",  "sources":[…], "images":[…], "trace_id":…, "session_id":…}
{"type":"stage", "stage":"generating"|"scoring"|"regenerating", "attempt":1}
{"type":"token", "content":"…", "attempt":1}
{"type":"eval",  "attempt":1, "verdict":<verdict>, "scores":{…}}
{"type":"replace","reason":"faithfulness 0.28 < 0.50"}
{"type":"done",  "usage":{…}, "final_attempt":1}
```

Ordering guarantee:

```
meta → stage → token+ → eval → [replace → stage → token+] → done
```

There is no second `eval` after a replacement: the regenerated answer is
scored in the background, not before `done`. (An `eval` with
`verdict:"unscored"` does follow attempt 2 in one case — when the
regeneration came back empty and the draft was kept.)

`replace` means everything streamed so far is superseded. `attempt` tells the
client which answer a token belongs to; without it a client concatenates the
rejected draft onto its replacement.

`stage` and `replace` are additive — the existing adapter ignores unknown types
(asserted by `backend-stream.test.ts`). `attempt` is not backwards-compatible,
so both repos ship together.

### Verdicts

Evaluated in order; the first match wins:

| verdict | condition | regenerate? |
|---|---|---|
| `unscored` | faithfulness is None (error or timeout) | no |
| `passed` | faithfulness >= threshold | no |
| `retrieval_failed` | context_precision == 0.0 | no |
| `rejected` | otherwise | **yes** |

`retrieval_failed` is the significant one. When retrieval surfaced nothing
relevant, a stricter prompt against identical context cannot improve grounding —
regenerating cost ~94s on the measured trace and could not have helped. Saying
so is also a better showcase: the system diagnoses its own failure mode.

### Pipeline

`ask_with_eval` is deleted. `ask_stream` becomes async, always streams for real,
and takes the gate as a parameter:

```
retrieve → yield meta
         → stream tokens (attempt 1)
         → gated? no  → done
                   yes → faithfulness + context_precision, concurrently,
                         bounded by eval_timeout_seconds
                       → verdict
                       → rejected? no  → done
                                   yes → replace → stream tokens (attempt 2) → done
```

**Correction (found during implementation).** The original design had
`context_precision` starting before generation, on the belief that it needs
only the question and contexts. That is false:
`LLMContextPrecisionWithoutReference` declares `response` a required column and
judges each context against the answer that was produced — "without reference"
means without a *ground-truth* answer, not without the response. The existing
code passed the literal string `"placeholder"`, which pinned the score at 0.0
for every query ever run and, with the new verdict table, would have routed
every rejection to `retrieval_failed` and disabled regeneration entirely.

Both metrics therefore need the answer and run concurrently with each other
after streaming. The gate costs `max(the two)` rather than their sum.

The regenerated answer is **not** scored synchronously. It goes to
`evaluate_query_async`, the same background path `ask_stream` already used. That
is the ~74s.

### Persistence

The route accumulates `answers: dict[int, str]` keyed by attempt and, on `done`,
saves `answers[final_attempt]`. A rejected draft is never written to session
history — it was shown as a demonstration, not offered as an answer.

### Error handling

A gate that raises or exceeds `eval_timeout_seconds` yields `verdict:"unscored"`
and **keeps the draft**. An evaluation failure must never cost the user their
answer. Client disconnect abandons cleanly without starting a regeneration
nobody will read.

### Configuration

```python
eval_model: str = ""              # falls back to azure_openai_model
eval_endpoint: str = ""
eval_api_key: str = ""
eval_api_version: str = ""
eval_timeout_seconds: int = 120
eval_max_answer_chars: int = 6000
```

Shipped unset: gpt-5.2 remains the evaluator because it measured faster. The
seam exists so the choice can change by env var rather than code.

`RAGAS_DO_NOT_TRACK=true` is set before ragas is imported. Its telemetry thread
ships evaluation metadata to `t.explodinggradients.com` and retries on DNS
failure; neither is wanted.

### Correctness fix carried along

`_make_cache_key(question, contexts)` omits the answer, but faithfulness is a
property of the answer. On a repeat question with the same retrieved chunks the
gate reads back a *previous* answer's score and judges the new one by it. The
key gains the answer.

`context_precision` turned out to depend on the answer too (see the correction
above), so it shares the same answer-keyed cache.

One consequence worth naming: when an answer exceeds `eval_max_answer_chars`,
the gate looks up the *truncated* text while the background eval saves under
the *full* answer, so the two keys can never coincide and the cache is dead
weight for that request. That is a lost optimisation, not a wrong result.

## Expected outcome

| | before | after |
|---|---|---|
| first token | 3m 41s | ~3s |
| verdict | 3m 41s | ~25-90s, non-blocking |
| failure path | 3m 41s | ~60-120s, reading throughout |

The gate is not faster. It is no longer in the way.

## Acceptance

1. A `token` event precedes any `eval` event.
2. Gate failure produces `replace` then attempt-2 tokens.
3. `context_precision == 0.0` produces no attempt 2.
4. The regenerated answer is not scored before `done`.
5. Only the final attempt reaches session history.
6. A raising gate leaves the draft intact with `verdict:"unscored"`.
7. Verified end to end against the Oracle VM through the Next.js UI.
