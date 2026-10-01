"""Greetings, acknowledgements, and questions nothing was retrieved for.

The behaviour these pin down came from two real traces: "Hi" was answered
with "No information is provided in the context.", and "What is n8n?" with
"The context only describes n8n as ...". Both leak machinery the reader
cannot see, and the first is not a reply to a greeting at all.
"""

from __future__ import annotations

import pytest

from app.core import smalltalk


# ── What counts as small talk ────────────────────────────────────────

@pytest.mark.parametrize(
    "message, kind",
    [
        ("Hi", smalltalk.GREETING),
        ("hi", smalltalk.GREETING),
        ("HELLO!", smalltalk.GREETING),
        ("  hey there  ", smalltalk.GREETING),
        ("Good morning!!", smalltalk.GREETING),
        ("hi!! hi!!", smalltalk.GREETING),
        ("Thanks!", smalltalk.THANKS),
        ("thank you so much", smalltalk.THANKS),
        ("ty", smalltalk.THANKS),
        ("bye", smalltalk.FAREWELL),
        ("Take care.", smalltalk.FAREWELL),
        ("ok", smalltalk.ACK),
        ("Got it, thanks" if False else "got it", smalltalk.ACK),
        ("Nice", smalltalk.ACK),
        ("What can you do?", smalltalk.CAPABILITY),
        ("who are you", smalltalk.CAPABILITY),
        ("help", smalltalk.CAPABILITY),
        ("how are you?", smalltalk.WELLBEING),
        ("hows it going", smalltalk.WELLBEING),
    ],
)
def test_small_talk_is_recognised(message, kind):
    assert smalltalk.classify(message) == kind


def test_punctuation_and_emoji_only_messages_read_as_acknowledgement():
    for message in ("👍", "!!", "...", "🙏"):
        assert smalltalk.classify(message) == smalltalk.ACK


@pytest.mark.parametrize(
    "question",
    [
        # The whole message must be small talk. A greeting attached to a real
        # question is a real question.
        "Hi, what does clause 4 say?",
        "hello, summarise the contract",
        "thanks for the last answer, now what is the notice period",
        "What can you do with the Q3 figures?",
        # Words that look like acknowledgements but are plausible search terms.
        "fine",
        "sure",
        "right",
        "good faith",
        "what is the ok rate",
        # Ordinary questions.
        "What is n8n?",
        "What drove the Q3 revenue increase?",
        "revenue",
        "",
        "   ",
    ],
)
def test_real_questions_are_left_alone(question):
    assert smalltalk.classify(question) is None


def test_normalise_collapses_the_things_that_should_not_decide_a_match():
    assert smalltalk.normalise("Good Morning!! 😊") == "good morning"
    assert smalltalk.normalise("How's it going?") == "hows it going"
    assert smalltalk.normalise("  HI   THERE  ") == "hi there"


# ── What it says back ────────────────────────────────────────────────

def test_every_kind_has_a_reply():
    for kind in (smalltalk.GREETING, smalltalk.THANKS, smalltalk.FAREWELL,
                 smalltalk.ACK, smalltalk.CAPABILITY, smalltalk.WELLBEING):
        for count in (0, 1, 5):
            reply = smalltalk.reply_for(kind, count)
            assert reply and len(reply) > 10, f"{kind} at {count} documents"


def test_a_greeting_says_what_is_loaded():
    empty = smalltalk.reply_for(smalltalk.GREETING, 0)
    assert "upload" in empty.lower(), "a new user is not told what to do first"

    loaded = smalltalk.reply_for(smalltalk.GREETING, 3)
    assert "3 documents" in loaded
    assert smalltalk.reply_for(smalltalk.GREETING, 1).count("one document") == 1


def test_no_reply_mentions_machinery_the_reader_cannot_see():
    """The whole point: no "context", no "retrieval", no "embeddings"."""
    banned = ("context", "retriev", "embed", "chunk", "vector", "index",
              "no information is provided", "llm", "prompt")
    replies = [smalltalk.reply_for(k, n)
               for k in (smalltalk.GREETING, smalltalk.THANKS, smalltalk.FAREWELL,
                         smalltalk.ACK, smalltalk.CAPABILITY, smalltalk.WELLBEING)
               for n in (0, 2)]
    replies += [smalltalk.nothing_retrieved_reply("anything", n) for n in (0, 1, 4)]

    for reply in replies:
        low = reply.lower()
        for word in banned:
            assert word not in low, f"{word!r} leaked into: {reply}"


def test_every_reply_points_at_a_next_step():
    """A refusal or a greeting that ends in a dead end is a bad answer."""
    for reply in (smalltalk.reply_for(smalltalk.GREETING, 0),
                  smalltalk.reply_for(smalltalk.CAPABILITY, 0),
                  smalltalk.nothing_retrieved_reply("x", 0),
                  smalltalk.nothing_retrieved_reply("x", 3)):
        low = reply.lower()
        assert any(w in low for w in ("ask", "upload", "add", "try")), reply


def test_nothing_retrieved_distinguishes_an_empty_library_from_a_miss():
    empty = smalltalk.nothing_retrieved_reply("notice period", 0)
    assert "haven't uploaded" in empty

    miss = smalltalk.nothing_retrieved_reply("notice period", 4)
    assert "couldn't find" in miss
    assert "4 documents" in miss
    assert "haven't uploaded" not in miss


def test_an_unknown_kind_returns_nothing_rather_than_guessing():
    assert smalltalk.reply_for("not-a-kind", 2) == ""


# ── Through the pipeline ─────────────────────────────────────────────

def _events(chunks: list[str]) -> list[dict]:
    import json

    out = []
    for chunk in chunks:
        if chunk.startswith("data: "):
            try:
                out.append(json.loads(chunk[6:]))
            except ValueError:
                pass
    return out


@pytest.mark.asyncio
async def test_a_greeting_never_reaches_retrieval_or_the_model(monkeypatch):
    """No search, no Azure call, no gate - and no charge."""
    from app.core import rag_engine

    def unreachable(*_args, **_kwargs):
        raise AssertionError("a greeting reached retrieval")

    monkeypatch.setattr(rag_engine, "hybrid_search", unreachable)
    monkeypatch.setattr(rag_engine, "_get_async_client", unreachable)

    chunks = [c async for c in rag_engine.ask_stream(question="Hi", user_id="u1",
                                                     session_id="s1")]
    events = _events(chunks)
    kinds = [e["type"] for e in events]

    assert "meta" in kinds and "done" in kinds
    assert "eval" not in kinds, "a written-out reply was sent to the gate"

    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert answer.lower().startswith("hello")
    assert "context" not in answer.lower()

    done = next(e for e in events if e["type"] == "done")
    assert done["charged"] is False
    assert done.get("gate") != "pending"


@pytest.mark.asyncio
async def test_an_empty_search_answers_without_calling_the_model(monkeypatch):
    from app.core import rag_engine

    monkeypatch.setattr(rag_engine, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(rag_engine, "plan_followups", lambda *a, **k: [])

    def unreachable(*_args, **_kwargs):
        raise AssertionError("an empty search still called the model")

    monkeypatch.setattr(rag_engine, "_get_async_client", unreachable)

    chunks = [c async for c in rag_engine.ask_stream(question="what is the notice period",
                                                     user_id="u1", session_id="s1")]
    events = _events(chunks)
    answer = "".join(e["content"] for e in events if e["type"] == "token")

    assert "context" not in answer.lower()
    assert "haven't uploaded" in answer, answer
    assert next(e for e in events if e["type"] == "done")["charged"] is False


@pytest.mark.asyncio
async def test_a_real_question_with_results_still_goes_to_the_model(monkeypatch):
    """The short-circuits must not swallow the ordinary path."""
    from langchain_core.documents import Document

    from app.core import rag_engine

    monkeypatch.setattr(rag_engine, "hybrid_search",
                        lambda *a, **k: [Document(page_content="Revenue rose 12%.",
                                                  metadata={"source": "q3.pdf"})])
    monkeypatch.setattr(rag_engine, "plan_followups", lambda *a, **k: [])

    reached = {"model": False}

    def fake_client():
        reached["model"] = True
        raise RuntimeError("stop here - reaching the model is the assertion")

    monkeypatch.setattr(rag_engine, "_get_async_client", fake_client)

    with pytest.raises(RuntimeError):
        async for _ in rag_engine.ask_stream(question="What drove revenue?",
                                             user_id="u1", session_id="s1"):
            pass

    assert reached["model"], "a question with sources was short-circuited"
