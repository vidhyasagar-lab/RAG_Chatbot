"""Langfuse observability — centralised tracing for the RAG pipeline.

Provides a lazy singleton ``Langfuse`` client and thin helpers that
every module can import to create traces and spans.  When Langfuse is
disabled (``LANGFUSE_ENABLED=false``) every helper silently returns a
no-op object so instrumented code never needs to check a flag.

Targets the Langfuse **v4** SDK.  v4 replaced the v3 surface this module
was originally written against:

===========================  =============================================
v3 (removed)                 v4 (used here)
===========================  =============================================
``client.start_span()``      ``client.start_observation(as_type="span")``
``span.start_generation()``  ``span.start_observation(as_type="generation")``
``span.update_trace()``      trace-level OTEL attributes on the root span
``update(usage=...)``        ``update(usage_details=...)``
===========================  =============================================

The wrapper classes below keep the *caller-facing* API unchanged
(``trace.span()`` / ``trace.generation()`` / ``.update()`` / ``.end()``)
so instrumented modules did not need to change.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Lazy singleton ───────────────────────────────────────────────────────────

_langfuse_client = None
_initialised = False


def _get_langfuse():
    """Return the Langfuse client singleton (or None if disabled)."""
    global _langfuse_client, _initialised

    if _initialised:
        return _langfuse_client

    _initialised = True
    settings = get_settings()

    if not settings.langfuse_enabled:
        logger.info("langfuse_disabled")
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("langfuse_credentials_missing")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("langfuse_initialised", host=settings.langfuse_host)
    except Exception:
        logger.exception("langfuse_init_failed")
        _langfuse_client = None

    return _langfuse_client


# ── v3 → v4 translation helpers ──────────────────────────────────────────────

# Trace-level attribute keys.  Imported from the SDK when available so we track
# any rename; the literals are the on-the-wire names and are the fallback.
try:  # pragma: no cover - exercised implicitly via create_trace
    from langfuse._client.attributes import LangfuseOtelSpanAttributes as _Attr

    _TRACE_NAME = _Attr.TRACE_NAME
    _TRACE_USER_ID = _Attr.TRACE_USER_ID
    _TRACE_SESSION_ID = _Attr.TRACE_SESSION_ID
    _TRACE_TAGS = _Attr.TRACE_TAGS
    _TRACE_METADATA = _Attr.TRACE_METADATA
    _TRACE_INPUT = _Attr.TRACE_INPUT
    _TRACE_OUTPUT = _Attr.TRACE_OUTPUT
except Exception:  # pragma: no cover
    _TRACE_NAME = "langfuse.trace.name"
    _TRACE_USER_ID = "user.id"
    _TRACE_SESSION_ID = "session.id"
    _TRACE_TAGS = "langfuse.trace.tags"
    _TRACE_METADATA = "langfuse.trace.metadata"
    _TRACE_INPUT = "langfuse.trace.input"
    _TRACE_OUTPUT = "langfuse.trace.output"

# v4 reports token counts as ``usage_details`` with input/output/total keys.
_USAGE_KEY_MAP = {
    "prompt_tokens": "input",
    "completion_tokens": "output",
    "total_tokens": "total",
}


def _to_usage_details(usage: Any) -> dict[str, int] | None:
    """Translate an OpenAI-style usage dict into v4 ``usage_details``.

    v4's ``update()`` silently *ignores* unknown kwargs, so passing the old
    ``usage=`` through unchanged would quietly drop all token accounting.
    """
    if not isinstance(usage, dict):
        return None
    details: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        details[_USAGE_KEY_MAP.get(key, key)] = value
    return details or None


def _encode(value: Any) -> str:
    """JSON-encode a value for an OTEL attribute (which must be a scalar)."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _set_trace_attributes(span: Any, **fields: Any) -> None:
    """Set trace-level attributes on a root span (the v4 ``update_trace``).

    v4 dropped ``span.update_trace()``.  Its public replacement,
    ``langfuse.propagate_attributes()``, is a *context manager*, which does not
    fit this module's long-lived trace object (created in one call, ended in
    another, often across an ``await``).  Holding one open would risk leaking
    OTEL context between requests, so the attributes are written directly onto
    the root span instead — which is what v3's ``update_trace`` did internally.
    """
    otel_span = getattr(span, "_otel_span", None)
    if otel_span is None or not getattr(otel_span, "is_recording", lambda: False)():
        return

    attributes: dict[str, Any] = {}
    if fields.get("name"):
        attributes[_TRACE_NAME] = fields["name"]
    if fields.get("user_id"):
        attributes[_TRACE_USER_ID] = str(fields["user_id"])
    if fields.get("session_id"):
        attributes[_TRACE_SESSION_ID] = str(fields["session_id"])
    if fields.get("tags"):
        attributes[_TRACE_TAGS] = [str(t) for t in fields["tags"]]
    if fields.get("metadata"):
        attributes[_TRACE_METADATA] = _encode(fields["metadata"])
    if fields.get("input") is not None:
        attributes[_TRACE_INPUT] = _encode(fields["input"])
    if fields.get("output") is not None:
        attributes[_TRACE_OUTPUT] = _encode(fields["output"])

    if attributes:
        otel_span.set_attributes(attributes)


# ── Public helpers ───────────────────────────────────────────────────────────


class _NoOpSpan:
    """Quacks like a Langfuse span/generation but does nothing."""

    def update(self, **_: Any) -> "_NoOpSpan":
        return self

    def end(self, **_: Any) -> None:
        pass

    def generation(self, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()

    def span(self, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()

    @property
    def id(self) -> str:
        return ""


class _NoOpTrace(_NoOpSpan):
    """Quacks like a Langfuse trace but does nothing."""

    def score(self, **_: Any) -> None:
        pass


class _SpanWrapper:
    """Wraps a Langfuse v4 observation, preserving the v3-style call surface.

    Child observations are created through ``start_observation(as_type=...)``
    and ``usage=`` is translated to ``usage_details=``.  Every call is guarded:
    telemetry must never break the request it is measuring.
    """

    def __init__(self, obs: Any) -> None:
        self._obs = obs

    @property
    def id(self) -> str:
        return getattr(self._obs, "id", "") or ""

    def _child(self, as_type: str, **kwargs: Any) -> Any:
        try:
            return _SpanWrapper(self._obs.start_observation(as_type=as_type, **kwargs))
        except Exception:
            logger.exception("langfuse_start_observation_failed", as_type=as_type)
            return _NoOpSpan()

    def span(self, **kwargs: Any) -> Any:
        return self._child("span", **kwargs)

    def generation(self, **kwargs: Any) -> Any:
        return self._child("generation", **kwargs)

    def update(self, **kwargs: Any) -> "_SpanWrapper":
        usage = kwargs.pop("usage", None)
        if usage is not None and "usage_details" not in kwargs:
            details = _to_usage_details(usage)
            if details:
                kwargs["usage_details"] = details
        try:
            self._obs.update(**kwargs)
        except Exception:
            logger.exception("langfuse_update_failed")
        return self

    def end(self) -> None:
        try:
            self._obs.end()
        except Exception:
            pass


class _TraceWrapper(_SpanWrapper):
    """Wraps the root span of a trace.

    ``.id`` is the *trace* id (used to attach scores later), and ``.update()``
    also mirrors output onto the trace-level attributes.  The root span must be
    ended via ``.end()`` for the trace to be exported.
    """

    @property
    def id(self) -> str:
        return getattr(self._obs, "trace_id", "") or ""

    def update(self, **kwargs: Any) -> "_TraceWrapper":
        if "output" in kwargs:
            try:
                _set_trace_attributes(self._obs, output=kwargs["output"])
            except Exception:
                logger.exception("langfuse_set_trace_output_failed")
        super().update(**kwargs)
        return self


def create_trace(*, name: str, user_id: str = "", session_id: str = "",
                 metadata: dict | None = None, tags: list[str] | None = None,
                 input: Any = None, **kwargs: Any):
    """Create a Langfuse trace. Returns a trace object (or a no-op)."""
    lf = _get_langfuse()
    if lf is None:
        return _NoOpTrace()

    try:
        span_kwargs: dict[str, Any] = {"name": name, "as_type": "span"}
        if metadata:
            span_kwargs["metadata"] = metadata
        if input is not None:
            span_kwargs["input"] = input
        span_kwargs.update(kwargs)
        root_span = lf.start_observation(**span_kwargs)

        # Set trace-level attributes (user_id, session_id, tags, etc.)
        _set_trace_attributes(
            root_span,
            name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            input=input,
        )

        return _TraceWrapper(root_span)
    except Exception:
        logger.exception("langfuse_create_trace_failed")
        return _NoOpTrace()


def score_trace(trace_id: str, *, name: str, value: float,
                comment: str = "") -> None:
    """Attach a score (e.g. user feedback) to an existing trace."""
    lf = _get_langfuse()
    if lf is None or not trace_id:
        return

    try:
        score_kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
        }
        if comment:
            score_kwargs["comment"] = comment
        lf.create_score(**score_kwargs)
    except Exception:
        logger.exception("langfuse_score_failed", trace_id=trace_id)


def flush() -> None:
    """Flush pending Langfuse events (call on shutdown)."""
    lf = _get_langfuse()
    if lf is not None:
        try:
            lf.flush()
            logger.info("langfuse_flushed")
        except Exception:
            logger.exception("langfuse_flush_failed")


def shutdown() -> None:
    """Flush and shut down the Langfuse client."""
    global _langfuse_client, _initialised
    flush()
    if _langfuse_client is not None:
        try:
            _langfuse_client.shutdown()
        except Exception:
            pass
    _langfuse_client = None
    _initialised = False
