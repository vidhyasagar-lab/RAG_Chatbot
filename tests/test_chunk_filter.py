"""No chunk is only a heading, and the parents the model reads keep their breadcrumb.

On the review document 23 of 204 chunks were a heading with no body, and the
empty `## Composite ranking` ranked first for the question it names, pushing
the table that answers it out of the top five.

Parents are what the model actually reads - retrieval swaps each matched
child for its parent - and they were stored without the
`[Document | Section]` line, so a table's heading never reached the answer.

Every section here stays under 60 tokens so the semantic splitter, which
would need a real embedding model, passes it through untouched.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.document_loader import chunk_documents


def _doc(text: str, content_type: str = "text") -> Document:
    return Document(page_content=text,
                    metadata={"source": "uploads/u1/r.docx", "page": "", "content_type": content_type})


def test_a_heading_with_no_body_produces_no_chunk():
    """`## B` is followed straight by a same-level heading, as `## Composite
    ranking` is once its table has been extracted separately."""
    parents, children = chunk_documents(
        [_doc("# A\n\n## B\n\n## C\n\nA real sentence under C.")])

    assert len(parents) == 1
    assert "A real sentence under C." in parents[0].page_content
    assert all("A real sentence" in c.page_content for c in children)


def test_parents_carry_the_document_and_section_line():
    parents, _ = chunk_documents([_doc("# A\n\n## B\n\nA real sentence under B.")])

    assert parents[0].page_content.startswith("[Document: r.docx | Section: A > B]")


def test_a_short_table_is_kept():
    parents, _ = chunk_documents([_doc("[Table 1]\n| a |", content_type="table")])

    assert len(parents) == 1


def test_a_short_sentence_without_a_heading_is_kept():
    """Short is not empty. Only a chunk with no body at all is dropped."""
    parents, _ = chunk_documents([_doc("Revenue rose 12%.")])

    assert len(parents) == 1
