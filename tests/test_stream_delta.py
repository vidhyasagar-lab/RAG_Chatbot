"""Azure streams chunks whose choice carries no delta; the answer must survive them."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace


def _chunk(content=None, *, no_delta=False, no_choices=False, usage=None):
    if no_choices:
        return SimpleNamespace(choices=[], usage=usage)
    delta = None if no_delta else SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()

    async def close(self):
        self.closed = True


class _Client:
    def __init__(self, stream):
        async def create(**_kwargs):
            return stream

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_chunks_without_a_delta_are_skipped_not_fatal():
    from app.core.rag_engine import _stream_answer

    stream = _Stream(
        [
            _chunk(no_delta=True),  # content-filter results arrive like this
            _chunk("The codename "),
            _chunk(None),  # a delta with no text
            _chunk(no_delta=True),
            _chunk("is BLUEBIRD."),
            _chunk(no_choices=True, usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
        ]
    )
    settings = SimpleNamespace(azure_openai_model="m", max_tokens=100, temperature=0)
    into: dict = {}

    async def run():
        return [e async for e in _stream_answer(_Client(stream), [], settings, 1, into)]

    events = asyncio.run(run())
    tokens = [json.loads(e.removeprefix("data: ").strip())["content"] for e in events]
    assert tokens == ["The codename ", "is BLUEBIRD."]
    assert into["text"] == "The codename is BLUEBIRD."
    assert stream.closed
