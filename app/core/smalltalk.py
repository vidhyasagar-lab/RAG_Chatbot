"""Replies to messages that are not questions about a document.

"Hi" used to go through the whole pipeline: embed the greeting, search the
index, find nothing, and hand an empty context to a model told to answer only
from context. It answered "No information is provided in the context.", which
is both unfriendly and incoherent as a reply to a greeting.

Skipping retrieval for greetings, thanks and acknowledgements is the standard
fix, and it is what every assistant that feels natural does. The replies here
are written out rather than generated: a greeting costs no tokens, arrives
instantly, cannot drift, and can state real facts about the account - how many
documents are loaded - which a model with an empty context cannot.

Matching is deliberately strict. The *whole* message must be small talk, so
"Hi, what does clause 4 say?" is a document question and reaches retrieval
untouched. Single words that could plausibly be search terms - "fine",
"sure", "right" - are left out on purpose.
"""

from __future__ import annotations

import re
import unicodedata

from app.core.logging import get_logger

logger = get_logger(__name__)

# Kinds of non-document message. Each maps to one reply.
GREETING = "greeting"
THANKS = "thanks"
FAREWELL = "farewell"
ACK = "ack"
CAPABILITY = "capability"
WELLBEING = "wellbeing"

#: Exact matches, after normalisation. Chosen over substring matching so a
#: real question containing "hi" or "ok" is never swallowed.
_EXACT: dict[str, str] = {}


def _register(kind: str, *phrases: str) -> None:
    for phrase in phrases:
        _EXACT[phrase] = kind


_register(
    GREETING,
    "hi", "hii", "hiii", "hello", "helo", "hey", "heya", "hiya", "yo",
    "hi there", "hello there", "hey there", "hi hi", "hello hello",
    "good morning", "good afternoon", "good evening", "morning", "evening",
    "namaste", "hola", "greetings", "hi verity", "hello verity", "hey verity",
)
_register(
    THANKS,
    "thanks", "thank you", "thank u", "thanks a lot", "thanks so much",
    "thank you so much", "thanks very much", "thx", "tysm", "ty",
    "cheers", "appreciate it", "much appreciated", "perfect thanks",
    "ok thanks", "okay thanks", "great thanks", "thanks mate",
)
_register(
    FAREWELL,
    "bye", "byebye", "bye bye", "goodbye", "good bye", "see you", "see ya",
    "cya", "later", "good night", "goodnight", "take care",
)
_register(
    ACK,
    "ok", "okay", "k", "kk", "okie", "got it", "gotcha", "understood",
    "makes sense", "cool", "nice", "great", "awesome", "perfect", "lovely",
    "good", "very good", "excellent", "noted", "i see", "ic",
)
_register(
    CAPABILITY,
    "what can you do", "what do you do", "what can i ask",
    "what can i ask you", "what are you", "who are you", "what are you for",
    "how does this work", "how do you work", "how does it work",
    "help", "help me", "what is verity", "who is verity",
    "capabilities", "what are your capabilities", "what can you help with",
    "what can you help me with",
)
_register(
    WELLBEING,
    "how are you", "how are you doing", "how r u", "hows it going",
    "how is it going", "whats up", "sup", "how do you do",
)

#: Emoji-only and punctuation-only messages read as acknowledgements.
_SYMBOL_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


def normalise(message: str) -> str:
    """Lowercase, strip accents, drop punctuation and emoji, collapse spaces.

    "Good Morning!! 😊" and "good morning" must reach the same key, or the
    match is decided by whether someone typed an exclamation mark.
    """
    text = unicodedata.normalize("NFKD", message or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    # Apostrophes are deleted, not spaced, so "how's" becomes "hows" and not
    # "how s". Straight and curly both, since a phone keyboard types curly.
    text = re.sub(r"['‘’ʼ`]", "", text)
    # Everything else that is not a letter, digit or space becomes a gap.
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify(message: str) -> str | None:
    """The kind of small talk this message is, or None if it is a question.

    None is the answer for anything that is not an exact match, which is the
    safe default: a misread question goes to retrieval and gets a real answer,
    while a misread greeting would get a canned reply to a genuine question.
    """
    raw = (message or "").strip()
    if not raw:
        return None

    normalised = normalise(raw)
    if not normalised:
        # Emoji or punctuation only - "👍", "!!", "..."
        return ACK if _SYMBOL_ONLY.match(raw) else None

    kind = _EXACT.get(normalised)
    if kind:
        return kind

    # "hi!! hi!!" and "thanks thanks" collapse to one word repeated.
    words = normalised.split()
    if len(set(words)) == 1 and len(words) <= 3:
        return _EXACT.get(words[0])

    return None


# ── Replies ──────────────────────────────────────────────────────────

def _library(doc_count: int) -> str:
    """One clause describing what is loaded, for replies that orient the user."""
    if doc_count <= 0:
        return ""
    if doc_count == 1:
        return "You have one document loaded"
    return f"You have {doc_count} documents loaded"


def reply_for(kind: str, doc_count: int = 0) -> str:
    """The reply for a kind of small talk, given what the account holds.

    Every reply ends pointing at what the reader can do next, which is what
    separates a useful short answer from a dead end.
    """
    has_docs = doc_count > 0

    if kind == GREETING:
        if has_docs:
            it = "it" if doc_count == 1 else "them"
            return (
                f"Hello. {_library(doc_count)} — ask me anything about {it} and I'll "
                "answer with the page the answer came from."
            )
        return (
            "Hello. I answer questions about documents you upload — reports, contracts, "
            "notes, even scans and screenshots. Add one from the Documents panel and ask "
            "away, and every answer will cite the page it came from."
        )

    if kind == CAPABILITY:
        base = (
            "I answer questions about your own documents. Upload a PDF, Word file, text "
            "file or image, ask in plain language, and each answer cites the page it came "
            "from and is scored against your sources before you read it. I read charts and "
            "tables too, so a figure that only appears in a bar chart is still answerable."
        )
        if has_docs:
            return f"{base}\n\n{_library(doc_count)}, so go ahead and ask."
        return f"{base}\n\nNothing is loaded yet — add a document and ask your first question."

    if kind == THANKS:
        return "You're welcome. Ask away if anything else comes up."

    if kind == FAREWELL:
        return "Goodbye. Your documents and chats will be here when you come back."

    if kind == ACK:
        if has_docs:
            return "Happy to help. Ask away whenever you have another question."
        return (
            "Happy to help. Upload a document whenever you're ready and ask me anything "
            "about it."
        )

    if kind == WELLBEING:
        if has_docs:
            return f"Doing well, thank you. {_library(doc_count)} — what would you like to know?"
        return (
            "Doing well, thank you. Upload a document and I'll answer questions about it."
        )

    # Unknown kind: never guess at a reply, let retrieval handle the message.
    logger.warning("smalltalk_unknown_kind", kind=kind)
    return ""


def nothing_retrieved_reply(question: str, doc_count: int) -> str:
    """What to say when the search came back empty.

    Also written out rather than generated. The model used to be handed an
    empty context and asked to answer from it, which produced "No information
    is provided in the context." - true, unhelpful, and in a vocabulary the
    reader has no way to understand.
    """
    if doc_count <= 0:
        return (
            "You haven't uploaded any documents yet, so there's nothing for me to search. "
            "Add a PDF, Word file, text file or image from the Documents panel and ask "
            "again — I'll answer from it and show you the page."
        )

    where = "your document" if doc_count == 1 else f"any of your {doc_count} documents"
    return (
        f"I couldn't find anything about that in {where}. Try the wording the document "
        "itself uses, or name the section you're thinking of — and if the answer lives in "
        "a file you haven't added yet, upload it and ask again."
    )
