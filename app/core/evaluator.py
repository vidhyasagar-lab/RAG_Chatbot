"""RAG evaluator — uses RAGAS metrics with Azure OpenAI via LiteLLM.

Supports three flows:
1. **Auto golden gen on upload** — generates Q&A pairs from newly uploaded doc chunks
2. **Per-query async evaluation** — evaluates each chat response in background
3. **Batch evaluation** — runs golden dataset through the RAG pipeline then evaluates

Scores are pushed back to Langfuse via the existing observability module.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from typing import Any

# ragas starts an analytics thread on import that POSTs evaluation metadata
# (metric names, row counts, a generated user id) to t.explodinggradients.com,
# and retries on failure. Set before any ragas import — which happens lazily
# inside the functions below — so the thread never starts. setdefault, so an
# operator who has deliberately set it keeps their value.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from app.config import get_settings
from app.core.eval_store import (
    add_golden_samples_bulk,
    complete_eval_run,
    create_eval_run,
    fail_eval_run,
    get_eval_cache,
    get_golden_dataset,
    save_eval_cache,
    save_eval_result,
    save_query_scores,
)
from app.core.logging import get_logger
from app.core.observability import score_trace
from app.core.vector_store import hybrid_search

logger = get_logger(__name__)


def _get_eval_llm():
    """Create a RAGAS-compatible LLM from the eval_* settings.

    Defaults to the main chat deployment. A separate one can be configured,
    but measure before switching: on 2026-09-23 gpt-4.1-mini scored the
    faithfulness metric in 43.8s against gpt-5.2's 24.0s, because the metric's
    cost sits in the verification call, where a chattier model loses.
    """
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import AzureChatOpenAI

    settings = get_settings()
    lc_llm = AzureChatOpenAI(
        model=settings.effective_eval_model,
        azure_endpoint=settings.effective_eval_endpoint,
        api_key=settings.effective_eval_api_key,
        api_version=settings.effective_eval_api_version,
        max_tokens=8192,
    )
    return LangchainLLMWrapper(lc_llm)


def _get_azure_embeddings():
    """Create RAGAS-compatible embeddings via LangChain AzureOpenAIEmbeddings."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import AzureOpenAIEmbeddings

    settings = get_settings()
    lc_emb = AzureOpenAIEmbeddings(
        model=settings.azure_openai_embedding_model,
        azure_endpoint=settings.effective_embedding_endpoint,
        api_key=settings.effective_embedding_api_key,
        api_version=settings.effective_embedding_api_version,
    )
    return LangchainEmbeddingsWrapper(lc_emb)


def _get_openai_client():
    """Get a plain OpenAI client for Azure (used for synthetic generation)."""
    from openai import AzureOpenAI

    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


# ── Synthetic Golden Dataset Generation ──────────────────────────────

SYNTH_PROMPT = """\
You are a QA dataset generator for evaluating a document retrieval system.

Given the following document content, generate {count} question-answer pairs.
Each question should be answerable using ONLY the provided content.
Each answer should be a concise, factual response based on the document.

Document source: {source}
---
{content}
---

Return ONLY a JSON array of objects, each with "question" and "ground_truth" keys.
Example format:
[
  {{"question": "What is X?", "ground_truth": "X is..."}},
  {{"question": "How does Y work?", "ground_truth": "Y works by..."}}
]
"""


def generate_synthetic_dataset(user_id: str, count_per_doc: int = 3) -> int:
    """Generate Q&A pairs from user's documents using Azure OpenAI.

    Returns number of samples generated.
    """
    import json

    settings = get_settings()
    client = _get_openai_client()

    # Get unique document chunks (using a sample query to get diverse chunks)
    docs = hybrid_search("*", k=50, expand_parents=True, user_id=user_id)
    if not docs:
        return 0

    # Group chunks by source document
    doc_groups: dict[str, list[str]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in doc_groups:
            doc_groups[source] = []
        if len(doc_groups[source]) < 5:  # max 5 chunks per doc
            doc_groups[source].append(doc.page_content)

    all_samples: list[dict] = []
    for source, chunks in doc_groups.items():
        content = "\n\n---\n\n".join(chunks)
        try:
            response = client.chat.completions.create(
                model=settings.azure_openai_model,
                messages=[
                    {"role": "system", "content": "You generate evaluation datasets for RAG systems. Always respond with valid JSON only."},
                    {"role": "user", "content": SYNTH_PROMPT.format(
                        count=count_per_doc, source=source, content=content[:4000]
                    )},
                ],
                temperature=0.7,
                max_completion_tokens=2000,
            )
            raw = response.choices[0].message.content or "[]"
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            pairs = json.loads(raw)
            for p in pairs:
                if "question" in p and "ground_truth" in p:
                    all_samples.append({
                        "question": p["question"],
                        "ground_truth": p["ground_truth"],
                        "source_doc": source,
                        "source": "synthetic",
                    })
        except Exception:
            logger.exception("synth_generation_failed", source=source)
            continue

    if all_samples:
        add_golden_samples_bulk(all_samples)

    logger.info("synthetic_dataset_generated", count=len(all_samples))
    return len(all_samples)


# ── Auto Golden Generation on Document Upload ────────────────────────

def _generate_golden_for_document_sync(
    chunks: list[str], source: str, user_id: str, count_per_doc: int = 3,
) -> None:
    """Generate Q&A pairs from the given document chunks (runs in background)."""
    settings = get_settings()
    client = _get_openai_client()

    # Use up to 5 chunks to keep prompt reasonable
    content = "\n\n---\n\n".join(chunks[:5])
    if not content.strip():
        return

    try:
        response = client.chat.completions.create(
            model=settings.azure_openai_model,
            messages=[
                {"role": "system", "content": "You generate evaluation datasets for RAG systems. Always respond with valid JSON only."},
                {"role": "user", "content": SYNTH_PROMPT.format(
                    count=count_per_doc, source=source, content=content[:4000]
                )},
            ],
            temperature=0.7,
            max_completion_tokens=2000,
        )
        raw = response.choices[0].message.content or "[]"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        pairs = json.loads(raw)
        samples = []
        for p in pairs:
            if "question" in p and "ground_truth" in p:
                samples.append({
                    "question": p["question"],
                    "ground_truth": p["ground_truth"],
                    "source_doc": source,
                    "source": "synthetic",
                })
        if samples:
            add_golden_samples_bulk(samples)
            logger.info("golden_auto_generated", source=source, count=len(samples))
    except Exception:
        logger.exception("golden_auto_generation_failed", source=source)


def generate_golden_for_document(
    parent_chunks: list, source: str, user_id: str, count_per_doc: int = 3,
) -> None:
    """Fire-and-forget golden dataset generation for a newly uploaded document.

    Called from the document upload route after indexing completes.
    """
    # Extract text from Document objects or strings
    texts = []
    for chunk in parent_chunks:
        if hasattr(chunk, "page_content"):
            texts.append(chunk.page_content)
        elif isinstance(chunk, str):
            texts.append(chunk)
    if not texts:
        return

    thread = threading.Thread(
        target=_generate_golden_for_document_sync,
        args=(texts, source, user_id, count_per_doc),
        daemon=True,
    )
    thread.start()
    logger.info("golden_generation_queued", source=source)


# ── Per-Query Async Evaluation ───────────────────────────────────────

def _evaluate_single_query_sync(
    question: str,
    answer: str,
    contexts: list[str],
    trace_id: str,
    user_id: str,
) -> None:
    """Evaluate a single query/answer using RAGAS metrics, push scores to Langfuse."""
    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference

        eval_llm = _get_eval_llm()
        eval_embeddings = _get_azure_embeddings()

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]

        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=dataset, metrics=metrics,
            llm=eval_llm, embeddings=eval_embeddings,
            show_progress=False,
        )

        df = result.to_pandas()
        if len(df) > 0:
            row = df.iloc[0]
            scores = {
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("llm_context_precision_without_reference")),
            }

            # Push each score to Langfuse on the original trace
            for metric_name, value in scores.items():
                if value is not None:
                    score_trace(
                        trace_id,
                        name=metric_name,
                        value=value,
                        comment=f"RAGAS auto-eval: {metric_name}",
                    )

            # Store scores locally for UI badge polling, tagged with the owner
            # so /chat/scores and /feedback can be ownership checked.
            save_query_scores(trace_id, scores, user_id=user_id)

            # Cache under the answer that was actually scored, so a later
            # gate check on a different answer misses rather than inheriting
            # this verdict.
            save_eval_cache(question, contexts, scores, answer=answer)

            logger.info(
                "per_query_eval_completed",
                trace_id=trace_id,
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
                context_precision=scores.get("context_precision"),
            )

    except Exception:
        logger.exception("per_query_eval_failed", trace_id=trace_id)


def evaluate_context_precision_sync(
    question: str,
    contexts: list[str],
) -> float | None:
    """Run ONLY ContextPrecision synchronously.

    This metric does NOT need the answer — only the question and contexts.
    This allows it to run IN PARALLEL with answer generation, giving
    us a free quality check during generation time.

    Returns cached score if same question + contexts seen before.
    Returns the context precision score (0.0–1.0) or None on error.
    """
    if not contexts:
        return None

    # No cache read here. This metric runs concurrently with answer generation,
    # so its latency is already hidden behind the LLM call and a hit would save
    # nothing the reader can perceive.
    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference

        eval_llm = _get_eval_llm()

        sample = SingleTurnSample(
            user_input=question,
            response="placeholder",  # not used by this metric
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=dataset,
            metrics=[LLMContextPrecisionWithoutReference()],
            llm=eval_llm,
            show_progress=False,
        )
        df = result.to_pandas()
        if len(df) > 0:
            score = _safe_float(df.iloc[0].get("llm_context_precision_without_reference"))
            logger.info("context_precision_gate_checked", score=score)
            return score
        return None
    except Exception:
        logger.exception("context_precision_gate_failed")
        return None


def evaluate_faithfulness_sync(
    question: str,
    answer: str,
    contexts: list[str],
) -> float | None:
    """Run ONLY the Faithfulness metric synchronously (fast gate check).

    Returns cached score if same question + contexts seen before.
    Returns the faithfulness score (0.0–1.0) or None on error.
    This is much faster than the full 3-metric evaluation because it
    makes only a single LLM call to decompose + verify claims.
    """
    if not answer.strip() or not contexts:
        return None

    # Keyed on the answer too: this score describes THIS answer, and a
    # question asked twice produces different answers at temperature > 0.
    cached = get_eval_cache(question, contexts, answer)
    if cached and cached.get("faithfulness") is not None:
        logger.info("faithfulness_cache_hit")
        return cached["faithfulness"]

    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics._faithfulness import Faithfulness

        eval_llm = _get_eval_llm()

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness()],
            llm=eval_llm,
            show_progress=False,
        )
        df = result.to_pandas()
        if len(df) > 0:
            score = _safe_float(df.iloc[0].get("faithfulness"))
            logger.info("faithfulness_gate_checked", score=score)
            return score
        return None
    except Exception:
        logger.exception("faithfulness_gate_failed")
        return None


def evaluate_query_async(
    question: str,
    answer: str,
    contexts: list[str],
    trace_id: str,
    user_id: str = "",
) -> None:
    """Fire-and-forget evaluation of a single RAG query.

    Called from ask_stream after the answer is fully generated.
    Runs RAGAS metrics in a background thread and pushes scores to Langfuse.
    """
    if not answer.strip() or not contexts:
        return

    thread = threading.Thread(
        target=_evaluate_single_query_sync,
        args=(question, answer, contexts, trace_id, user_id),
        daemon=True,
    )
    thread.start()


# ── Batch Evaluation ─────────────────────────────────────────────────

def _run_evaluation_sync(run_id: str, user_id: str) -> None:
    """Run evaluation synchronously (called in background thread)."""
    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
        from ragas.metrics._context_recall import LLMContextRecall

        settings = get_settings()
        client = _get_openai_client()
        eval_llm = _get_eval_llm()
        eval_embeddings = _get_azure_embeddings()

        golden = get_golden_dataset()
        if not golden:
            fail_eval_run(run_id, "No golden dataset found")
            return

        # Build samples: run each question through the RAG pipeline
        samples = []
        sample_meta = []  # parallel list for metadata

        for item in golden:
            question = item["question"]
            ground_truth = item["ground_truth"]

            # Retrieve contexts
            docs = hybrid_search(question, user_id=user_id)
            contexts = [doc.page_content for doc in docs]

            # Generate answer using RAG
            context_text = "\n\n---\n\n".join(
                f"[{i+1}] {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
            try:
                response = client.chat.completions.create(
                    model=settings.azure_openai_model,
                    messages=[
                        {"role": "system", "content": f"Answer using ONLY this context:\n\n{context_text}"},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.3,
                    max_completion_tokens=1024,
                )
                answer = response.choices[0].message.content or ""
            except Exception:
                logger.exception("eval_answer_generation_failed", question=question)
                answer = "Error generating answer"

            samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            ))
            sample_meta.append({
                "golden_id": item["id"],
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })

        # Configure metrics
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithoutReference(),
            LLMContextRecall(),
        ]

        # Run RAGAS evaluation
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset, metrics=metrics,
            llm=eval_llm, embeddings=eval_embeddings,
        )

        # Save individual results
        df = result.to_pandas()
        for idx, row in df.iterrows():
            scores = {
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("llm_context_precision_without_reference")),
                "context_recall": _safe_float(row.get("context_recall")),
            }
            meta = sample_meta[idx]
            save_eval_result(
                run_id=run_id,
                question=meta["question"],
                answer=meta["answer"],
                contexts=meta["contexts"],
                ground_truth=meta["ground_truth"],
                scores=scores,
                golden_id=meta["golden_id"],
            )

        complete_eval_run(run_id)

        # Push aggregate scores to Langfuse
        _push_scores_to_langfuse(run_id)

        logger.info("evaluation_completed", run_id=run_id, samples=len(samples))

    except Exception as e:
        logger.exception("evaluation_failed", run_id=run_id)
        fail_eval_run(run_id, str(e))


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except (ValueError, TypeError):
        return None


def _push_scores_to_langfuse(run_id: str) -> None:
    """Push evaluation scores to Langfuse as a scored trace."""
    from app.core.eval_store import get_eval_run
    from app.core.observability import create_trace

    run = get_eval_run(run_id)
    if not run:
        return

    trace = create_trace(
        name="rag-evaluation",
        user_id="system",
        tags=["evaluation", "ragas"],
        input={"run_id": run_id, "total_samples": run["total_samples"]},
        metadata={
            "run_type": run["run_type"],
            "avg_faithfulness": run["avg_faithfulness"],
            "avg_relevancy": run["avg_relevancy"],
            "avg_context_precision": run["avg_context_precision"],
            "avg_context_recall": run["avg_context_recall"],
        },
    )
    trace.update(output={
        "avg_faithfulness": run["avg_faithfulness"],
        "avg_relevancy": run["avg_relevancy"],
        "avg_context_precision": run["avg_context_precision"],
        "avg_context_recall": run["avg_context_recall"],
    })
    trace.end()


def start_evaluation(user_id: str) -> str:
    """Start a batch evaluation in a background thread.

    Returns the run_id for tracking progress.
    """
    golden = get_golden_dataset()
    run_id = create_eval_run(run_type="batch", total_samples=len(golden))

    thread = threading.Thread(
        target=_run_evaluation_sync,
        args=(run_id, user_id),
        daemon=True,
    )
    thread.start()

    logger.info("evaluation_started", run_id=run_id, samples=len(golden))
    return run_id
