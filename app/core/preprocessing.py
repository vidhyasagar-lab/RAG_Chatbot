"""Document preprocessing — extract and describe ALL visual elements.

Before the chunking pipeline, every document passes through this
preprocessor which:

  **PDF** (via PyMuPDF):
    - Extracts embedded raster images (photos, screenshots, logos)
    - Renders pages to detect tables (heuristic + vision OCR)
    - Renders pages to detect flowcharts / diagrams / charts
    - Extracts raw text per-page with layout preservation

  **DOCX** (via python-docx):
    - Extracts inline and floating images from media/
    - Extracts tables natively (cell-by-cell) into Markdown
    - Detects SmartArt / embedded charts via relationship inspection

Each extracted element is described by GPT-5.2 vision so it can be
chunked, embedded, and retrieved alongside regular text.

Returns a unified ``PreprocessedDocument`` containing text chunks and
visual element records ready for the chunking pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Vision prompts specialised by element type ───────────────────────

FLOWCHART_PROMPT = """\
You are a document analysis assistant. This image is a flowchart or \
process diagram extracted from a business document. Describe it for a \
searchable knowledge base:
- List every step/node with its label
- Describe the flow direction and all connections/arrows between nodes
- Note any decision points (yes/no branches)
- Describe conditions, loops, and parallel paths
- Mention any colour coding or grouping

Be thorough — your description will be used for search retrieval."""

CHART_PROMPT = """\
You are a document analysis assistant. This image is a chart or graph \
extracted from a business document. Describe it for a searchable \
knowledge base:
- Chart type (bar, line, pie, scatter, etc.)
- Axes labels and units
- All data series with their approximate values
- Title, legend, and annotations
- Key trends, comparisons, or takeaways

Be precise with numbers — your description will be used for search retrieval."""

DIAGRAM_PROMPT = """\
You are a document analysis assistant. This image is a diagram \
(architecture, network, ER, class, sequence, etc.) extracted from a \
document. Describe it for a searchable knowledge base:
- All components/entities and their labels
- Relationships and connections between components
- Directionality of arrows/links
- Groupings, layers, or boundaries
- Any annotations or legends

Be thorough — your description will be used for search retrieval."""

GENERIC_IMAGE_PROMPT = """\
You are a document analysis assistant. Describe this image in detail for \
a searchable knowledge base. Include:
- All visible text (transcribe exactly)
- Data in tables (format as markdown tables)
- Chart/graph descriptions with key data points
- Diagram explanations with relationships
- Any other relevant visual information

Be thorough — your description will be used for search retrieval."""

TABLE_VISION_PROMPT = """\
Extract all data from this table image. Return it as:
1. A markdown table with all rows and columns preserved exactly
2. A brief summary of what the table contains

Be precise with numbers, dates, and labels."""


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class VisualElement:
    """One extracted visual element from a document."""

    content_type: str  # "image" | "table" | "flowchart" | "chart" | "diagram"
    description: str  # GPT-5.2 vision description (searchable text)
    source: str  # original file path
    page: int | str  # page number (1-based) or ""
    image_path: str = ""  # saved image path (for display)
    raw_text: str = ""  # native text extraction (tables in DOCX)


@dataclass
class PreprocessedDocument:
    """Result of preprocessing: text documents + visual element records."""

    text_docs: list[Document] = field(default_factory=list)
    visual_elements: list[VisualElement] = field(default_factory=list)


# ── Deferred (parallel) vision descriptions ──────────────────────────────────
#
# Describing an image costs a ~1-2s Azure round trip. Done inline, an N-image
# document serialises into N round trips. Instead each extraction site appends
# *placeholder* records and registers a _PendingVisual; once extraction is done
# the whole batch is described concurrently and the placeholders are patched.
# Placeholders keep their original positions, so output ordering is unchanged.


@dataclass
class _PendingVisual:
    """One image awaiting its description."""

    image_path: str
    prompt: str
    fallback: str  # used if the vision call fails
    header: str  # bracketed label prefixed to the description in the text doc
    element_index: int  # index into result.visual_elements
    doc_index: int  # index into result.text_docs
    drop_if_blank: bool = False  # discard the record if the description is empty


def _resolve_pending_visuals(
    pending: list[_PendingVisual],
    result: PreprocessedDocument,
    log_event: str,
) -> None:
    """Describe every pending image concurrently and patch the placeholders."""
    if not pending:
        return

    from app.core.vision import describe_images

    outcomes = describe_images([(p.image_path, p.prompt) for p in pending])

    drop_docs: set[int] = set()
    drop_elements: set[int] = set()

    for p, (description, error) in zip(pending, outcomes):
        if error is not None:
            logger.warning(log_event, path=p.image_path, error=str(error))
            description = p.fallback
        description = description or ""

        if p.drop_if_blank and not description.strip():
            drop_docs.add(p.doc_index)
            drop_elements.add(p.element_index)
            continue

        result.visual_elements[p.element_index].description = description
        result.text_docs[p.doc_index].page_content = f"{p.header}\n{description}"

    if drop_docs or drop_elements:
        result.text_docs[:] = [
            d for i, d in enumerate(result.text_docs) if i not in drop_docs
        ]
        result.visual_elements[:] = [
            v for i, v in enumerate(result.visual_elements) if i not in drop_elements
        ]


def _defer_description(
    result: PreprocessedDocument,
    pending: list[_PendingVisual],
    *,
    content_type: str,
    source: str,
    page: int | str,
    image_path: str,
    prompt: str,
    header: str,
    fallback: str,
    metadata: dict,
    drop_if_blank: bool = False,
) -> None:
    """Append placeholder records and queue the image for description."""
    result.visual_elements.append(VisualElement(
        content_type=content_type,
        description="",
        source=source,
        page=page,
        image_path=image_path,
    ))
    result.text_docs.append(Document(page_content="", metadata=metadata))
    pending.append(_PendingVisual(
        image_path=image_path,
        prompt=prompt,
        fallback=fallback,
        header=header,
        element_index=len(result.visual_elements) - 1,
        doc_index=len(result.text_docs) - 1,
        drop_if_blank=drop_if_blank,
    ))


# ── Classify visual element type ─────────────────────────────────────

_FLOWCHART_KEYWORDS = re.compile(
    r"(start|end|begin|process|decision|yes|no|→|➜|⮕|arrow|flow|step\s*\d)",
    re.IGNORECASE,
)
_CHART_KEYWORDS = re.compile(
    r"(axis|legend|x-axis|y-axis|bar|pie|chart|graph|series|%|percent|trend)",
    re.IGNORECASE,
)


def _classify_visual(surrounding_text: str, page_text: str = "") -> str:
    """Guess the type of visual element from surrounding context.

    Returns one of: image, table, flowchart, chart, diagram.
    """
    combined = f"{surrounding_text} {page_text}"

    # Strong signals from surrounding text
    lower = combined.lower()
    if any(kw in lower for kw in ("flowchart", "flow chart", "process flow", "workflow")):
        return "flowchart"
    if any(kw in lower for kw in ("architecture", "network diagram", "er diagram",
                                   "class diagram", "sequence diagram", "system design")):
        return "diagram"
    if _CHART_KEYWORDS.search(combined):
        return "chart"
    if _FLOWCHART_KEYWORDS.search(combined):
        return "flowchart"

    return "image"  # default


def _get_vision_prompt(content_type: str) -> str:
    """Return the specialised vision prompt for a content type."""
    prompts = {
        "flowchart": FLOWCHART_PROMPT,
        "chart": CHART_PROMPT,
        "diagram": DIAGRAM_PROMPT,
        "table": TABLE_VISION_PROMPT,
        "image": GENERIC_IMAGE_PROMPT,
    }
    return prompts.get(content_type, GENERIC_IMAGE_PROMPT)


# ══════════════════════════════════════════════════════════════════════
#  PDF PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess_pdf(file_path: str) -> PreprocessedDocument:
    """Extract all content from a PDF: text, images, tables, diagrams."""
    import pymupdf  # the `fitz` alias is deprecated and slated for removal

    settings = get_settings()
    output_dir = Path(settings.upload_dir) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(file_path)
    result = PreprocessedDocument()
    pending: list[_PendingVisual] = []
    stem = Path(file_path).stem

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_no = page_num + 1
        page_text = page.get_text("text")

        # ── 1. Regular text ──────────────────────────────────────
        if page_text.strip():
            result.text_docs.append(Document(
                page_content=page_text,
                metadata={
                    "source": file_path,
                    "page": page_no,
                    "content_type": "text",
                },
            ))

        # ── 2. Embedded images ───────────────────────────────────
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")

            # Skip tiny images (icons, bullets, decorations)
            if len(image_bytes) < 5_000:
                continue

            img_filename = f"{stem}_p{page_no}_img{img_idx + 1}.{image_ext}"
            img_path = output_dir / img_filename
            img_path.write_bytes(image_bytes)

            # Classify based on surrounding page text
            visual_type = _classify_visual("", page_text)

            # Queue for description; resolved in parallel after extraction.
            # The text doc is what gets chunked & indexed.
            _defer_description(
                result, pending,
                content_type=visual_type,
                source=file_path,
                page=page_no,
                image_path=str(img_path),
                prompt=_get_vision_prompt(visual_type),
                header=f"[{visual_type.title()} from page {page_no}]",
                fallback=f"[{visual_type} from page {page_no}]",
                metadata={
                    "source": file_path,
                    "page": page_no,
                    "content_type": visual_type,
                    "image_path": str(img_path),
                },
            )

        # ── 3. Tables (heuristic detection → vision extraction) ──
        if _page_has_table(page_text):
            pix = page.get_pixmap(dpi=200)
            table_img = output_dir / f"{stem}_p{page_no}_table.png"
            pix.save(str(table_img))

            # Queued like the images above. drop_if_blank preserves the
            # original behaviour: a table that yields no text is not indexed.
            _defer_description(
                result, pending,
                content_type="table",
                source=file_path,
                page=page_no,
                image_path=str(table_img),
                prompt=TABLE_VISION_PROMPT,
                header=f"[Table from page {page_no}]",
                fallback="",
                metadata={
                    "source": file_path,
                    "page": page_no,
                    "content_type": "table",
                    "image_path": str(table_img),
                },
                drop_if_blank=True,
            )

        # ── 4. Flowcharts / diagrams (drawing-heavy pages) ──────
        #   Pages with many vector drawings but little text
        #   likely contain flowcharts, diagrams, or charts.
        drawings = page.get_drawings()
        text_ratio = len(page_text.strip()) / max(len(drawings), 1)
        has_many_drawings = len(drawings) > 15
        is_visual_page = has_many_drawings and text_ratio < 50

        if is_visual_page:
            visual_type = _classify_visual(page_text, page_text)
            if visual_type == "image":
                # Refine: many drawings → likely flowchart or diagram
                visual_type = "flowchart" if len(drawings) > 30 else "diagram"

            pix = page.get_pixmap(dpi=200)
            diag_img = output_dir / f"{stem}_p{page_no}_{visual_type}.png"
            pix.save(str(diag_img))

            _defer_description(
                result, pending,
                content_type=visual_type,
                source=file_path,
                page=page_no,
                image_path=str(diag_img),
                prompt=_get_vision_prompt(visual_type),
                header=f"[{visual_type.title()} from page {page_no}]",
                fallback=f"[{visual_type} from page {page_no}]",
                metadata={
                    "source": file_path,
                    "page": page_no,
                    "content_type": visual_type,
                    "image_path": str(diag_img),
                },
            )

    total_pages = len(doc)
    doc.close()

    # All images are on disk now — describe them concurrently.
    _resolve_pending_visuals(pending, result, "vision_describe_failed")

    logger.info(
        "pdf_preprocessed",
        file=file_path,
        pages=total_pages,
        text_docs=len(result.text_docs),
        images=sum(1 for v in result.visual_elements if v.content_type == "image"),
        tables=sum(1 for v in result.visual_elements if v.content_type == "table"),
        flowcharts=sum(1 for v in result.visual_elements if v.content_type == "flowchart"),
        charts=sum(1 for v in result.visual_elements if v.content_type == "chart"),
        diagrams=sum(1 for v in result.visual_elements if v.content_type == "diagram"),
    )
    return result


def _page_has_table(text: str) -> bool:
    """Heuristic: does this page text look like it contains a table?"""
    if text.count("\t") > 3 or text.count("|") > 3:
        return True
    # Multiple lines with 3+ numbers → likely tabular data
    lines = text.strip().split("\n")
    numeric_lines = 0
    for line in lines:
        numbers = re.findall(r"\d+[\.,]?\d*", line)
        if len(numbers) >= 3:
            numeric_lines += 1
    return numeric_lines >= 2


# ══════════════════════════════════════════════════════════════════════
#  DOCX PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess_docx(file_path: str) -> PreprocessedDocument:
    """Extract all content from a DOCX: text, images, tables, SmartArt."""
    from docx import Document as DocxDocument

    settings = get_settings()
    output_dir = Path(settings.upload_dir) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    docx = DocxDocument(file_path)
    result = PreprocessedDocument()
    pending: list[_PendingVisual] = []
    stem = Path(file_path).stem

    # ── 1. Extract text paragraphs ───────────────────────────────
    text_parts: list[str] = []
    for para in docx.paragraphs:
        text = para.text.strip()
        if text:
            # Preserve heading structure for downstream structural split
            if para.style and para.style.name.startswith("Heading"):
                level = para.style.name.replace("Heading", "").strip()
                prefix = "#" * int(level) if level.isdigit() else "#"
                text_parts.append(f"{prefix} {text}")
            else:
                text_parts.append(text)

    if text_parts:
        result.text_docs.append(Document(
            page_content="\n\n".join(text_parts),
            metadata={
                "source": file_path,
                "page": "",
                "content_type": "text",
            },
        ))

    # ── 2. Extract tables natively ───────────────────────────────
    for tbl_idx, table in enumerate(docx.tables):
        md_table = _docx_table_to_markdown(table)
        if md_table.strip():
            # Also save table as text doc for chunking
            result.text_docs.append(Document(
                page_content=f"[Table {tbl_idx + 1}]\n{md_table}",
                metadata={
                    "source": file_path,
                    "page": "",
                    "content_type": "table",
                },
            ))
            result.visual_elements.append(VisualElement(
                content_type="table",
                description=md_table,
                source=file_path,
                page="",
                raw_text=md_table,
            ))

    # ── 3. Extract images from media/ ────────────────────────────
    img_counter = 0
    for rel in docx.part.rels.values():
        if "image" in rel.reltype:
            img_counter += 1
            try:
                image_part = rel.target_part
                image_bytes = image_part.blob
                content_type = image_part.content_type or "image/png"

                # Skip tiny images (bullets, decorations)
                if len(image_bytes) < 5_000:
                    continue

                ext = _mime_to_ext(content_type)
                img_filename = f"{stem}_img{img_counter}.{ext}"
                img_path = output_dir / img_filename
                img_path.write_bytes(image_bytes)

                # Classify — look at surrounding paragraph text for clues
                surrounding = _get_surrounding_text(docx, img_counter)
                visual_type = _classify_visual(surrounding)

                _defer_description(
                    result, pending,
                    content_type=visual_type,
                    source=file_path,
                    page="",
                    image_path=str(img_path),
                    prompt=_get_vision_prompt(visual_type),
                    header=f"[{visual_type.title()} {img_counter}]",
                    fallback=f"[{visual_type} — image {img_counter}]",
                    metadata={
                        "source": file_path,
                        "page": "",
                        "content_type": visual_type,
                        "image_path": str(img_path),
                    },
                )

            except Exception:
                logger.exception("docx_image_extract_failed", index=img_counter)

    # ── 4. Detect SmartArt / embedded objects ────────────────────
    _extract_docx_smartart(docx, file_path, output_dir, stem, result, pending)

    # All images are on disk now - describe them concurrently.
    _resolve_pending_visuals(pending, result, "docx_image_describe_failed")

    logger.info(
        "docx_preprocessed",
        file=file_path,
        text_docs=len(result.text_docs),
        images=sum(1 for v in result.visual_elements if v.content_type == "image"),
        tables=sum(1 for v in result.visual_elements if v.content_type == "table"),
        flowcharts=sum(1 for v in result.visual_elements if v.content_type == "flowchart"),
        diagrams=sum(1 for v in result.visual_elements if v.content_type == "diagram"),
        charts=sum(1 for v in result.visual_elements if v.content_type == "chart"),
    )
    return result


def _docx_table_to_markdown(table) -> str:
    """Convert a python-docx Table object to a Markdown table string."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")

    if len(rows) < 1:
        return ""

    # Insert header separator after first row
    num_cols = len(table.rows[0].cells) if table.rows else 0
    separator = "| " + " | ".join(["---"] * num_cols) + " |"

    return rows[0] + "\n" + separator + "\n" + "\n".join(rows[1:])


def _mime_to_ext(content_type: str) -> str:
    """Map MIME type to file extension."""
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/webp": "webp",
        "image/x-emf": "emf",
        "image/x-wmf": "wmf",
    }
    return mapping.get(content_type, "png")


def _get_surrounding_text(docx_doc, img_index: int) -> str:
    """Get text around the image position for classification context."""
    paragraphs = [p.text for p in docx_doc.paragraphs if p.text.strip()]
    # Approximate — look at paragraphs around where the image appears
    start = max(0, img_index - 3)
    end = min(len(paragraphs), img_index + 3)
    return " ".join(paragraphs[start:end])


def _extract_docx_smartart(docx_doc, file_path: str, output_dir: Path,
                           stem: str, result: PreprocessedDocument,
                           pending: list[_PendingVisual]) -> None:
    """Inspect DOCX rels for SmartArt, charts, and embedded OLE objects.

    SmartArt in DOCX is stored as diagramXX.xml relationships.
    Charts are stored as chartXX.xml relationships.
    We detect their presence and extract any fallback images.
    """
    smart_counter = 0
    for rel in docx_doc.part.rels.values():
        rel_type_lower = rel.reltype.lower() if rel.reltype else ""

        is_diagram = "diagram" in rel_type_lower
        is_chart = "chart" in rel_type_lower
        is_ole = "oleobject" in rel_type_lower or "package" in rel_type_lower

        if not (is_diagram or is_chart or is_ole):
            continue

        smart_counter += 1

        if is_chart:
            content_type = "chart"
        elif is_diagram:
            content_type = "flowchart"
        else:
            content_type = "diagram"

        # Try to find fallback image for the SmartArt/chart
        try:
            target_part = rel.target_part
            # SmartArt/charts often have image rels of their own
            for sub_rel in getattr(target_part, "rels", {}).values():
                if "image" in (sub_rel.reltype or "").lower():
                    img_part = sub_rel.target_part
                    image_bytes = img_part.blob
                    if len(image_bytes) < 3_000:
                        continue

                    ext = _mime_to_ext(img_part.content_type or "image/png")
                    img_filename = f"{stem}_smartart{smart_counter}.{ext}"
                    img_path = output_dir / img_filename
                    img_path.write_bytes(image_bytes)

                    _defer_description(
                        result, pending,
                        content_type=content_type,
                        source=file_path,
                        page="",
                        image_path=str(img_path),
                        prompt=_get_vision_prompt(content_type),
                        header=f"[{content_type.title()} {smart_counter}]",
                        fallback=f"[{content_type} — SmartArt/Chart {smart_counter}]",
                        metadata={
                            "source": file_path,
                            "page": "",
                            "content_type": content_type,
                            "image_path": str(img_path),
                        },
                    )
                    break  # one image per SmartArt element is enough
        except Exception:
            logger.debug("smartart_fallback_failed", index=smart_counter)


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess_image(file_path: str) -> PreprocessedDocument:
    """Describe a standalone image via GPT-5.2 vision."""
    result = PreprocessedDocument()

    try:
        from app.core.vision import describe_image
        desc = describe_image(file_path)
    except Exception:
        logger.exception("standalone_image_failed", path=file_path)
        desc = f"[Image: {Path(file_path).name}]"

    result.text_docs.append(Document(
        page_content=desc,
        metadata={
            "source": file_path,
            "content_type": "image",
            "image_path": file_path,
            "page": "",
        },
    ))
    result.visual_elements.append(VisualElement(
        content_type="image",
        description=desc,
        source=file_path,
        page="",
        image_path=file_path,
    ))
    return result


# ══════════════════════════════════════════════════════════════════════
#  UNIFIED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def preprocess_document(file_path: str) -> PreprocessedDocument:
    """Preprocess any supported document, extracting all visual elements.

    Call this BEFORE the chunking pipeline. It returns text documents
    ready for chunking plus image/table/flowchart records for indexing.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return preprocess_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return preprocess_docx(file_path)
    elif ext in _IMAGE_EXTENSIONS:
        return preprocess_image(file_path)
    elif ext in (".txt", ".md"):
        # Plain text — no visual elements to extract
        from app.core.document_loader import _load_text
        docs = _load_text(file_path)
        for d in docs:
            d.metadata["source"] = file_path
            d.metadata["content_type"] = "text"
        return PreprocessedDocument(text_docs=docs)
    else:
        raise ValueError(f"Unsupported file type for preprocessing: {ext}")
