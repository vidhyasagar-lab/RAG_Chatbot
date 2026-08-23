"""RAG engine – orchestrates multimodal retrieval and generation.

Supports text + image contexts. When retrieved chunks reference images
or tables, their descriptions are included in the LLM context and the
image paths are returned for display in the UI.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from openai import AsyncAzureOpenAI, AzureOpenAI

from app.config import get_settings
from app.core.evaluator import (
    evaluate_context_precision_sync,
    evaluate_faithfulness_sync,
    evaluate_query_async,
)
from app.core.logging import get_logger
from app.core.observability import create_trace
from app.core.vector_store import hybrid_search

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a helpful AI assistant. Answer the user's question using ONLY the \
context provided below. The context may include:
- Regular text content from documents
- Descriptions of images and photographs
- Extracted table data (formatted as Markdown tables)
- Flowchart / process diagram descriptions with steps and decision points
- Chart and graph descriptions with data points
- Architecture and system diagram descriptions

When referencing visual content, be explicit about the source type:
- "According to the table on page 3..."
- "The flowchart shows the process as..."
- "Based on the bar chart..."
- "The architecture diagram illustrates..."
When referencing text, cite the source document when available.

If the context does not contain enough information to answer, say so \
honestly — do not make things up.

Context:
{context}
"""


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class RAGResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    trace_id: str = ""


def _get_client() -> AzureOpenAI:
    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def _get_async_client() -> AsyncAzureOpenAI:
    settings = get_settings()
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def _build_context(query: str, top_k: int | None = None, user_id: str = "") -> tuple[str, list[dict], list[dict]]:
    """Retrieve relevant chunks via hybrid search and format into a context block.

    Returns ``(context_text, sources, images)`` where images contains
    paths to visual elements referenced by retrieved chunks.
    """
    docs = hybrid_search(query, k=top_k, user_id=user_id)
    sources: list[dict] = []
    images: list[dict] = []
    context_parts: list[str] = []
    seen_sources: set[tuple] = set()

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        content_type = doc.metadata.get("content_type", "text")
        image_path = doc.metadata.get("image_path", "")

        # Build label with content type indicator
        type_tags = {
            "image": " [IMAGE]",
            "table": " [TABLE]",
            "flowchart": " [FLOWCHART]",
            "chart": " [CHART]",
            "diagram": " [DIAGRAM]",
        }
        type_tag = type_tags.get(content_type, "")

        label = f"[{i}] {source}{type_tag}" + (f" (page {page})" if page else "")
        context_parts.append(f"{label}\n{doc.page_content}")

        # Deduplicate sources by (source, page, content_type)
        source_key = (source, str(page), content_type)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": source,
                "page": page,
                "chunk_index": i,
                "content_type": content_type,
            })

        # Collect image references for all visual content types
        if image_path and content_type in ("image", "table", "flowchart", "chart", "diagram"):
            images.append({
                "path": image_path,
                "page": page,
                "source": source,
                "content_type": content_type,
            })

    return "\n\n---\n\n".join(context_parts), sources, images


def ask(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    top_k: int | None = None,
    user_id: str = "",
) -> RAGResult:
    """Run the full multimodal RAG pipeline: retrieve → augment → generate."""
    settings = get_settings()

    # ── Langfuse trace for the full request ──────────────────────
    trace = create_trace(
        name="rag-chat",
        user_id=user_id,
        session_id=user_id,
        input={"question": question, "top_k": top_k},
        tags=["chat", "rag"],
        metadata={
            "chat_history_len": len(chat_history) if chat_history else 0,
            "model": settings.azure_openai_model,
        },
    )

    # ── Retrieval span ───────────────────────────────────────────
    retrieval_span = trace.span(
        name="retrieval",
        input={"query": question, "top_k": top_k},
    )
    context_text, sources, images = _build_context(question, top_k, user_id=user_id)
    retrieval_span.update(
        output={"sources_count": len(sources), "images_count": len(images)},
    )
    retrieval_span.end()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
    ]

    # Append prior conversation turns (if any)
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": question})

    # ── Generation span ──────────────────────────────────────────
    generation = trace.generation(
        name="llm-completion",
        model=settings.azure_openai_model,
        input=messages,
        model_parameters={
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        },
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.azure_openai_model,
        messages=messages,
        max_completion_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )

    answer = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }

    generation.update(
        output=answer,
        usage=usage,
    )
    generation.end()

    # ── Finalise trace ───────────────────────────────────────────
    trace.update(output={"answer": answer, "sources_count": len(sources)})
    trace_id = trace.id
    trace.end()

    logger.info(
        "rag_response_generated",
        question_len=len(question),
        context_chunks=len(sources),
        total_tokens=usage.get("total_tokens"),
        trace_id=trace_id,
    )

    return RAGResult(
        answer=answer, sources=sources, images=images,
        usage=usage, trace_id=trace_id,
    )


def ask_stream(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    top_k: int | None = None,
    user_id: str = "",
    session_id: str = "",
) -> Generator[str, None, None]:
    """Stream the RAG pipeline: retrieve → augment → generate token-by-token.

    Yields Server-Sent Event (SSE) formatted lines:
      - ``data: {"type":"meta", ...}``  — sources, images, trace_id
      - ``data: {"type":"token", "content":"..."}``  — each streamed token
      - ``data: {"type":"done", "usage":{...}}``  — final usage stats
    """
    settings = get_settings()

    trace = create_trace(
        name="rag-chat",
        user_id=user_id,
        session_id=user_id,
        input={"question": question, "top_k": top_k},
        tags=["chat", "rag", "stream"],
        metadata={
            "chat_history_len": len(chat_history) if chat_history else 0,
            "model": settings.azure_openai_model,
        },
    )

    # ── Retrieval ────────────────────────────────────────────────
    retrieval_span = trace.span(
        name="retrieval",
        input={"query": question, "top_k": top_k},
    )
    context_text, sources, images = _build_context(question, top_k, user_id=user_id)
    retrieval_span.update(
        output={"sources_count": len(sources), "images_count": len(images)},
    )
    retrieval_span.end()

    # Send metadata first so the UI can render sources while tokens stream
    meta = {
        "type": "meta",
        "sources": sources,
        "images": images,
        "trace_id": trace.id,
        "session_id": session_id,
    }
    yield f"data: {json.dumps(meta)}\n\n"

    # ── Build messages ───────────────────────────────────────────
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
    ]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": question})

    # ── Streaming generation ─────────────────────────────────────
    generation = trace.generation(
        name="llm-completion",
        model=settings.azure_openai_model,
        input=messages,
        model_parameters={
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        },
    )

    client = _get_client()
    stream = client.chat.completions.create(
        model=settings.azure_openai_model,
        messages=messages,
        max_completion_tokens=settings.max_tokens,
        temperature=settings.temperature,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # ── Finalise tracing ─────────────────────────────────────────
    usage = {}
    # The last chunk in Azure OpenAI streams may carry usage info
    if hasattr(chunk, "usage") and chunk.usage:
        usage = {
            "prompt_tokens": chunk.usage.prompt_tokens,
            "completion_tokens": chunk.usage.completion_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }

    generation.update(output=full_answer, usage=usage)
    generation.end()
    trace.update(output={"answer": full_answer, "sources_count": len(sources)})
    trace_id = trace.id
    trace.end()

    logger.info(
        "rag_stream_completed",
        question_len=len(question),
        context_chunks=len(sources),
        total_tokens=usage.get("total_tokens"),
        trace_id=trace_id,
    )

    # Fire async RAGAS evaluation (background, non-blocking)
    context_chunks = [part.split("\n", 1)[-1] for part in context_text.split("\n\n---\n\n") if part.strip()]
    evaluate_query_async(
        question=question,
        answer=full_answer,
        contexts=context_chunks,
        trace_id=trace_id,
        user_id=user_id,
    )

    yield f"data: {json.dumps({'type': 'done', 'usage': usage})}\n\n"


# ── Eval-Gated Pipeline ─────────────────────────────────────────────

REFINED_SYSTEM_PROMPT = """\
You are a helpful AI assistant. Your previous answer was flagged as \
potentially unfaithful to the source material. Answer the user's question \
using ONLY the context provided below. Be strictly factual — do not add \
information that is not explicitly stated in the context. If the context \
does not contain enough information, say so.

Context:
{context}
"""


async def _async_generate_answer(
    client: AsyncAzureOpenAI,
    messages: list[dict[str, str]],
    settings,
) -> tuple[str, dict]:
    """Generate a complete (non-streaming) answer using async client."""
    response = await client.chat.completions.create(
        model=settings.azure_openai_model,
        messages=messages,
        max_completion_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    answer = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return answer, usage


async def ask_with_eval(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    top_k: int | None = None,
    user_id: str = "",
    session_id: str = "",
) -> AsyncGenerator[str, None]:
    """Eval-gated async RAG pipeline with speculative parallelism.

    Fully async — uses AsyncAzureOpenAI for generation and
    asyncio.to_thread for CPU-bound RAGAS evaluations.

    Latency-optimised flow:
      1. Retrieve context
      2. IN PARALLEL: async generate answer + context precision (in thread)
      3. Run faithfulness check on the generated answer
      4. If faithfulness < threshold → regenerate with stricter prompt
      5. Stream the verified answer
      6. Full 3-metric eval runs in background

    SSE events:
      - ``data: {"type":"meta", ...}``          — sources, images, trace_id
      - ``data: {"type":"eval", "scores":{...}}``— quality gate result
      - ``data: {"type":"token", "content":".."}``— each token of the final answer
      - ``data: {"type":"done", "usage":{...}}`` — final usage stats
    """
    settings = get_settings()

    trace = create_trace(
        name="rag-chat-eval-gated",
        user_id=user_id,
        session_id=user_id,
        input={"question": question, "top_k": top_k},
        tags=["chat", "rag", "eval-gated"],
        metadata={
            "chat_history_len": len(chat_history) if chat_history else 0,
            "model": settings.azure_openai_model,
        },
    )

    # ── Retrieval ────────────────────────────────────────────────
    retrieval_span = trace.span(
        name="retrieval",
        input={"query": question, "top_k": top_k},
    )
    context_text, sources, images = _build_context(question, top_k, user_id=user_id)
    context_chunks = [
        part.split("\n", 1)[-1] for part in context_text.split("\n\n---\n\n") if part.strip()
    ]
    retrieval_span.update(
        output={"sources_count": len(sources), "images_count": len(images)},
    )
    retrieval_span.end()

    # Send metadata immediately
    meta = {
        "type": "meta",
        "sources": sources,
        "images": images,
        "trace_id": trace.id,
        "session_id": session_id,
    }
    yield f"data: {json.dumps(meta)}\n\n"

    # ── Build messages ───────────────────────────────────────────
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
    ]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": question})

    async_client = _get_async_client()
    threshold = settings.eval_quality_threshold

    # ── Step 1: PARALLEL — Async generate + context precision (thread) ──
    gen_span = trace.generation(
        name="llm-completion-initial",
        model=settings.azure_openai_model,
        input=messages,
        model_parameters={
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        },
    )

    # Run generation (async) and context precision (sync in thread) concurrently
    gen_task = asyncio.ensure_future(
        _async_generate_answer(async_client, messages, settings)
    )
    ctx_prec_task = asyncio.ensure_future(
        asyncio.to_thread(evaluate_context_precision_sync, question, context_chunks)
    )

    # Await both concurrently
    (answer, usage), context_precision = await asyncio.gather(gen_task, ctx_prec_task)
    gen_span.update(output=answer, usage=usage)
    gen_span.end()

    logger.info("context_precision_parallel_done", context_precision=context_precision)

    # ── Step 2: Faithfulness gate (needs the answer) ─────────────
    eval_span = trace.span(name="faithfulness-gate", input={"answer_len": len(answer)})
    faithfulness = await asyncio.to_thread(
        evaluate_faithfulness_sync, question, answer, context_chunks
    )
    passed = faithfulness is None or faithfulness >= threshold

    eval_result = {
        "type": "eval",
        "scores": {
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "threshold": threshold,
            "passed": passed,
        },
    }
    eval_span.update(output=eval_result["scores"])
    eval_span.end()

    yield f"data: {json.dumps(eval_result)}\n\n"

    final_answer = answer
    total_usage = dict(usage)

    # ── Step 3: Regenerate if faithfulness failed ────────────────
    if not passed and settings.eval_max_retries > 0:
        logger.warning(
            "faithfulness_gate_failed_regenerating",
            faithfulness=faithfulness,
            threshold=threshold,
            trace_id=trace.id,
        )

        refined_messages: list[dict[str, str]] = [
            {"role": "system", "content": REFINED_SYSTEM_PROMPT.format(context=context_text)},
        ]
        if chat_history:
            for msg in chat_history:
                refined_messages.append({"role": msg.role, "content": msg.content})
        refined_messages.append({"role": "user", "content": question})

        regen_span = trace.generation(
            name="llm-completion-regenerated",
            model=settings.azure_openai_model,
            input=refined_messages,
            model_parameters={
                "max_tokens": settings.max_tokens,
                "temperature": max(settings.temperature - 0.1, 0.0),
            },
        )

        regen_answer, regen_usage = await _async_generate_answer(
            async_client, refined_messages, settings
        )
        regen_span.update(output=regen_answer, usage=regen_usage)
        regen_span.end()

        # Check regenerated answer faithfulness
        regen_faith = await asyncio.to_thread(
            evaluate_faithfulness_sync, question, regen_answer, context_chunks
        )
        regen_passed = regen_faith is None or regen_faith >= threshold

        if regen_passed or (regen_faith is not None and faithfulness is not None and regen_faith > faithfulness):
            final_answer = regen_answer
            total_usage = {
                k: total_usage.get(k, 0) + regen_usage.get(k, 0)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            yield f"data: {json.dumps({'type': 'eval', 'scores': {'context_precision': context_precision, 'faithfulness': regen_faith, 'threshold': threshold, 'passed': regen_passed, 'regenerated': True}})}\n\n"

            logger.info(
                "faithfulness_gate_regenerated",
                original_score=faithfulness,
                new_score=regen_faith,
                trace_id=trace.id,
            )

    # ── Step 4: Stream the final answer token-by-token ───────────
    chunk_size = 4  # characters per token event
    for i in range(0, len(final_answer), chunk_size):
        token = final_answer[i:i + chunk_size]
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # ── Finalise tracing ─────────────────────────────────────────
    trace.update(output={"answer": final_answer, "sources_count": len(sources)})
    trace_id = trace.id
    trace.end()

    logger.info(
        "rag_eval_gated_completed",
        question_len=len(question),
        context_chunks=len(sources),
        total_tokens=total_usage.get("total_tokens"),
        context_precision=context_precision,
        faithfulness=faithfulness,
        trace_id=trace_id,
    )

    # Fire remaining metrics (answer_relevancy) in background
    # Context precision + faithfulness already computed above
    evaluate_query_async(
        question=question,
        answer=final_answer,
        contexts=context_chunks,
        trace_id=trace_id,
        user_id=user_id,
    )

    yield f"data: {json.dumps({'type': 'done', 'usage': total_usage})}\n\n"
