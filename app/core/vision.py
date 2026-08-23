"""Vision processing — describe images via Azure OpenAI GPT-5.2 vision.

Provides the core ``describe_image()`` function used by the preprocessing
pipeline to generate searchable text from images, tables, and diagrams.
"""

from __future__ import annotations

import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from openai import AzureOpenAI

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

_DEFAULT_DESCRIBE_PROMPT = """\
You are a document analysis assistant. Describe this image in detail for \
a searchable knowledge base. Include:
- All visible text (transcribe exactly)
- Data in tables (format as markdown tables)
- Chart/graph descriptions with key data points
- Diagram explanations with relationships
- Any other relevant visual information

Be thorough — your description will be used for search retrieval."""


@lru_cache(maxsize=1)
def _get_vision_client() -> AzureOpenAI:
    """Return a shared vision client.

    Cached deliberately: the client is thread-safe and owns an HTTP connection
    pool, so reusing one across concurrent describe calls avoids rebuilding a
    pool (and re-doing TLS handshakes) for every image.
    """
    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, mime_type).

    Validates that the path is within allowed directories to prevent
    arbitrary file read (SSRF / path traversal).
    """
    settings = get_settings()
    path = Path(image_path).resolve()

    # Only allow reading from the uploads directory
    allowed_dir = Path(settings.upload_dir).resolve()
    if not path.is_relative_to(allowed_dir):
        raise ValueError(f"Image path outside allowed directory: {image_path}")

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {path.suffix}")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type


def describe_image(image_path: str, prompt: str | None = None,
                   parent_trace=None) -> str:
    """Send an image to GPT-5.2 vision and get a text description.

    The description is suitable for embedding and retrieval.
    If ``parent_trace`` is provided, a generation span is attached to it.
    """
    settings = get_settings()
    b64_data, mime_type = _encode_image(image_path)
    client = _get_vision_client()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or _DEFAULT_DESCRIBE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_data}",
                        "detail": settings.validated_vision_detail,
                    },
                },
            ],
        }
    ]

    # Langfuse generation span (no-op if no parent trace)
    generation = None
    if parent_trace is not None:
        try:
            generation = parent_trace.generation(
                name="vision-describe",
                model=settings.azure_openai_model,
                input={"image_path": image_path, "prompt": (prompt or _DEFAULT_DESCRIBE_PROMPT)[:200]},
                model_parameters={
                    "max_tokens": settings.vision_max_tokens,
                    "temperature": 0.1,
                },
            )
        except Exception:
            generation = None

    response = client.chat.completions.create(
        model=settings.azure_openai_model,
        messages=messages,
        max_completion_tokens=settings.vision_max_tokens,
        temperature=0.1,
    )

    description = response.choices[0].message.content or ""
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }

    if generation is not None:
        try:
            generation.update(output=description, usage=token_usage)
            generation.end()
        except Exception:
            pass

    logger.info(
        "image_described",
        path=image_path,
        description_len=len(description),
        tokens=token_usage.get("total_tokens", 0),
    )
    return description


def describe_images(
    jobs: Sequence[tuple[str, str | None]],
    max_workers: int | None = None,
) -> list[tuple[str | None, Exception | None]]:
    """Describe several images concurrently.

    ``jobs`` is a sequence of ``(image_path, prompt)`` pairs.  Returns a list
    of ``(description, error)`` tuples in the **same order as the input** —
    exactly one element of each tuple is non-None.

    Vision calls are network-bound (a second or two each, almost all of it
    waiting on Azure), so a thread pool turns an N-image document from N
    round-trips into roughly N/``max_workers``.  A thread pool rather than
    asyncio because ``describe_image`` uses the synchronous client and is
    called from synchronous preprocessing code — this keeps it usable from
    both sync and async callers without an event loop.

    Failures are captured per job, never raised: one unreadable image must not
    abandon the rest of the document.
    """
    if not jobs:
        return []

    settings = get_settings()
    workers = max_workers if max_workers is not None else settings.vision_max_concurrency
    workers = max(1, min(workers, len(jobs)))

    results: list[tuple[str | None, Exception | None]] = [(None, None)] * len(jobs)

    if workers == 1:
        # Skip pool overhead when there is nothing to overlap.
        for i, (path, prompt) in enumerate(jobs):
            try:
                results[i] = (describe_image(path, prompt=prompt), None)
            except Exception as e:  # noqa: BLE001 - recorded per job
                results[i] = (None, e)
        return results

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vision") as pool:
        futures = {
            pool.submit(describe_image, path, prompt=prompt): i
            for i, (path, prompt) in enumerate(jobs)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = (future.result(), None)
            except Exception as e:  # noqa: BLE001 - recorded per job
                results[i] = (None, e)

    ok = sum(1 for d, _ in results if d is not None)
    logger.info(
        "images_described_parallel",
        total=len(jobs),
        succeeded=ok,
        failed=len(jobs) - ok,
        workers=workers,
    )
    return results
