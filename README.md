# RAG Chatbot

Production-grade **Multimodal Retrieval-Augmented Generation** chatbot powered by **Azure OpenAI GPT-5.2**, **FastAPI**, and a built-in web UI. Features an **eval-gated answer pipeline** using [RAGAS](https://docs.ragas.io/) that automatically verifies answer quality before showing it to the user, with full **Langfuse** observability.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![uv](https://img.shields.io/badge/managed%20by-uv-261230)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [RAG Pipeline](#rag-pipeline)
- [Eval-Gated Pipeline](#eval-gated-pipeline)
- [Evaluation System](#evaluation-system)
- [Caching](#caching)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Security](#security)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)

---

## Features

| Category | Details |
|---|---|
| **Web UI** | Built-in dark-themed chat interface with Jinja2 + HTMX + Tailwind CSS — no frontend build step |
| **Multimodal RAG** | GPT-5.2 vision extracts and describes images, tables, flowcharts, charts, and diagrams from PDFs; standalone image upload supported |
| **Hybrid Chunking** | 7-step pipeline: text cleaning → structural split → semantic split → token-aware sizing → contextual headers → parent/child hierarchy → deduplication |
| **Hybrid Retrieval** | Dense (FAISS embeddings) + Sparse (BM25 keyword) fused with Reciprocal Rank Fusion (RRF) |
| **Parent/Child Chunks** | Small child chunks for precise retrieval, large parent chunks for rich LLM context |
| **Eval-Gated Answers** | Every streamed answer is verified via RAGAS faithfulness before delivery; low-quality answers are regenerated automatically |
| **RAGAS Evaluation** | Per-query async evaluation (Faithfulness, Answer Relevancy, Context Precision) + batch evaluation with golden datasets |
| **Eval Caching** | SHA-256 hash-based caching of evaluation scores to avoid redundant LLM calls |
| **Document Ingestion** | PDF, DOCX, TXT, Markdown, PNG, JPG, TIFF, BMP, GIF, WebP — up to 50 MB |
| **Multi-turn Chat** | Persistent chat sessions with SQLite-backed history (last 20 messages for LLM context) |
| **Source Citations** | Every answer includes source documents, page numbers, content type tags, and chunk references |
| **Streaming** | Server-Sent Events (SSE) for real-time token-by-token answer delivery |
| **Admin Panel** | User management, Langfuse insights, golden dataset CRUD, batch evaluation triggers |
| **Langfuse Observability** | Full tracing of retrieval, generation, and evaluation spans with cost/usage tracking |
| **User Feedback** | Thumbs up/down scoring pushed to Langfuse traces |
| **Rate Limiting** | 60 req/min per IP (`limits`, enforced in middleware) |
| **Authentication** | HMAC-signed session cookies, PBKDF2-SHA256 password hashing, brute-force protection |
| **Security Headers** | CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy |
| **Structured Logging** | JSON-structured logs via `structlog` |
| **Docker-ready** | Single-command deployment with `docker compose`, non-root container user |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (HTMX + Tailwind)                 │
│   Login/Register ─── Chat UI ─── Document Upload ─── Admin Panel │
└──────────────┬───────────────────────────────────────────────────┘
               │ HTTP / SSE
┌──────────────▼───────────────────────────────────────────────────┐
│                     FastAPI Application                          │
│ ┌─────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐         │
│ │Middleware│ │  API Routes  │ │  Pages   │ │  Static  │         │
│ │• API Key │ │• /chat       │ │• /       │ │• CSS/JS  │         │
│ │• Logging │ │• /documents  │ │• /login  │ │          │         │
│ │• Security│ │• /feedback   │ │• /admin  │ │          │         │
│ │  Headers │ │• /users      │ │• /logout │ │          │         │
│ │• CORS   │ │• /admin      │ │• partials│ │          │         │
│ │• Rate   │ │• /health     │ │          │ │          │         │
│ │  Limit  │ │              │ │          │ │          │         │
│ └─────────┘ └──────┬───────┘ └──────────┘ └──────────┘         │
│                    │                                             │
│ ┌──────────────────▼─────────────────────────────────────┐      │
│ │                    Core Engine                          │      │
│ │ ┌──────────┐ ┌──────────────┐ ┌─────────────────────┐ │      │
│ │ │   RAG    │ │  Evaluator   │ │   Document Loader   │ │      │
│ │ │ Engine   │ │  (RAGAS)     │ │   + Preprocessor    │ │      │
│ │ │• ask()   │ │• faithfulness│ │   + Vision (GPT-5.2)│ │      │
│ │ │• stream  │ │• relevancy   │ │   + Chunking (7-step│ │      │
│ │ │• eval-   │ │• context     │ │     pipeline)       │ │      │
│ │ │  gated   │ │  precision   │ │                     │ │      │
│ │ └─────┬────┘ └──────┬───────┘ └──────────┬──────────┘ │      │
│ │       │             │                     │            │      │
│ │ ┌─────▼─────────────▼─────────────────────▼──────────┐ │      │
│ │ │              Vector Store (Hybrid)                  │ │      │
│ │ │  Dense: FAISS  │  Sparse: BM25  │  RRF Fusion      │ │      │
│ │ │  Parent Cache  │  Image Cache   │  User Isolation   │ │      │
│ │ └────────────────────────────────────────────────────┘ │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐    │
│ │                  SQLite (users.db)                        │    │
│ │  users │ user_documents │ chat_sessions │ chat_messages   │    │
│ │  golden_dataset │ eval_runs │ eval_results │ query_scores │    │
│ │  eval_cache                                               │    │
│ └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│ ┌──────────────────────┐  ┌───────────────────────────────┐    │
│ │  Azure OpenAI        │  │  Langfuse (Observability)     │    │
│ │  • GPT-5.2 (chat)    │  │  • Traces & spans            │    │
│ │  • GPT-5.2 (vision)  │  │  • Generations               │    │
│ │  • text-embedding-   │  │  • Scores (feedback + RAGAS)  │    │
│ │    ada-002           │  │  • Cost tracking              │    │
│ └──────────────────────┘  └───────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| **LLM** | Azure OpenAI GPT-5.2 (chat + vision), text-embedding-ada-002 |
| **RAG Framework** | LangChain, LangChain-OpenAI, LangChain-Experimental (SemanticChunker) |
| **Vector Store** | FAISS (dense), custom BM25 (sparse), Reciprocal Rank Fusion |
| **Evaluation** | RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall), LiteLLM |
| **Observability** | Langfuse (tracing, scores, cost tracking) |
| **Database** | SQLite (WAL mode) — users, documents, chat sessions, eval results, eval cache |
| **Document Processing** | PyPDF, PyMuPDF, python-docx, docx2txt, tiktoken |
| **Frontend** | Jinja2 templates, HTMX 2.0, Tailwind CSS (CDN), marked.js (Markdown rendering) |
| **Auth** | itsdangerous (HMAC-signed cookies), PBKDF2-SHA256 passwords |
| **Rate Limiting** | `limits` (60 req/min per IP, configurable) |
| **Logging** | structlog (structured JSON logs) |
| **Containerisation** | Docker, Docker Compose |

---

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (manages Python and dependencies)
- Azure OpenAI resource with:
  - A **GPT-5.2** (or compatible) chat deployment
  - A **text-embedding-ada-002** embedding deployment

Python itself does **not** need to be pre-installed — uv downloads the pinned
interpreter (3.13.12, see `.python-version`) automatically.

### Option 1: One-Click Launcher

```bash
python starter.py
```

This script syncs the uv environment and starts the FastAPI server with hot-reload at `http://localhost:8000`.

### Option 2: Manual Setup

```bash
# Create .venv with the pinned Python and install locked dependencies
uv sync
```

Create a `.env` file in the project root (see [Configuration](#configuration)), then:

```bash
uv run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Common uv commands:

```bash
uv sync                      # install/update .venv to match uv.lock
uv sync --no-dev             # production install, skip the dev group
uv add <package>             # add a dependency (updates pyproject.toml + uv.lock)
uv remove <package>          # drop a dependency
uv lock --upgrade            # refresh all pinned versions
```

### Pinned dependency

`langchain-community` is held at `<0.4` (see `pyproject.toml`). RAGAS 0.4.3 imports
`langchain_community.chat_models.vertexai`, which was removed in langchain-community
0.4.0, and RAGAS declares no upper bound of its own. Without the pin, `import ragas`
raises `ModuleNotFoundError` and every evaluation path silently degrades to `None`.
This holds the whole langchain stack at 0.3.x; lift it once RAGAS supports
langchain-community >= 0.4.

> **Windows note:** invoke tools as `uv run python -m <tool>` rather than
> `uv run <tool>`. Endpoint-security policies commonly block the generated
> `.exe` console shims in `.venv\Scripts\` with *Access is denied (os error 5)*,
> while the interpreter itself runs fine.

```bash
uv run python -m pytest      # run a command inside the environment
```

Open **http://localhost:8000** for the web UI. API docs at **http://localhost:8000/docs**.

### Option 3: Docker

```bash
docker compose up --build
```

Open `http://localhost:8000` in your browser.

---

## Configuration

All settings are loaded from environment variables or a `.env` file. Create `.env` in the project root:

```env
# ── Azure OpenAI (required) ──────────────────────────
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-5.2
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# ── Embedding-specific overrides (optional) ──────────
# Falls back to main Azure OpenAI values if not set
AZURE_OPENAI_EMBEDDING_API_KEY=
AZURE_OPENAI_EMBEDDING_ENDPOINT=
AZURE_OPENAI_EMBEDDING_API_VERSION=

# ── App ──────────────────────────────────────────────
APP_ENV=production                    # production | development
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
API_KEY=                              # Optional: require X-API-Key header on all requests
SECRET_KEY=change-me-to-a-random-secret  # HMAC signing for session cookies

# ── RAG — Hybrid Chunking (token-based) ──────────────
PARENT_CHUNK_TOKENS=512
PARENT_OVERLAP_TOKENS=50
CHILD_CHUNK_TOKENS=128
CHILD_OVERLAP_TOKENS=16
SEMANTIC_THRESHOLD=85                 # Percentile for semantic boundary detection

# ── RAG — Hybrid Retrieval ───────────────────────────
TOP_K_RESULTS=5
DENSE_WEIGHT=0.5
SPARSE_WEIGHT=0.5

# ── LLM ─────────────────────────────────────────────
MAX_TOKENS=2048
TEMPERATURE=0.3

# ── Vision (multimodal) ─────────────────────────────
VISION_DETAIL=high                    # low | high | auto
VISION_MAX_TOKENS=1024
VISION_MAX_CONCURRENCY=6                # parallel image descriptions during ingest

# ── Eval-Gated Answers ──────────────────────────────
EVAL_GATING_ENABLED=true
EVAL_QUALITY_THRESHOLD=0.5           # Minimum faithfulness score to show answer
EVAL_MAX_RETRIES=1                   # Regeneration attempts on low quality

# ── Langfuse Observability ───────────────────────────
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# ── CORS ─────────────────────────────────────────────
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# ── Paths ────────────────────────────────────────────
VECTORSTORE_DIR=data/vectorstore
UPLOAD_DIR=uploads
```

### Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | *(required)* | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | *(required)* | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_MODEL` | `gpt-5.2` | Chat model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2025-01-01-preview` | API version |
| `AZURE_OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding deployment name |
| `AZURE_OPENAI_EMBEDDING_API_KEY` | *(falls back to main key)* | Separate key for embeddings |
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | *(falls back to main endpoint)* | Separate endpoint for embeddings |
| `APP_ENV` | `production` | Environment (`production` or `development`) |
| `API_KEY` | *(empty = disabled)* | Optional global API key via `X-API-Key` header |
| `PARENT_CHUNK_TOKENS` | `512` | Max tokens per parent chunk |
| `CHILD_CHUNK_TOKENS` | `128` | Max tokens per child chunk |
| `PARENT_OVERLAP_TOKENS` | `50` | Overlap between parent chunks |
| `CHILD_OVERLAP_TOKENS` | `16` | Overlap between child chunks |
| `SEMANTIC_THRESHOLD` | `85` | Percentile for semantic boundary detection |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `DENSE_WEIGHT` | `0.5` | Weight for dense (FAISS) retrieval in RRF |
| `SPARSE_WEIGHT` | `0.5` | Weight for sparse (BM25) retrieval in RRF |
| `MAX_TOKENS` | `2048` | Max tokens for LLM response |
| `TEMPERATURE` | `0.3` | LLM temperature |
| `VISION_DETAIL` | `high` | Vision API detail level (`low`, `high`, `auto`) |
| `VISION_MAX_TOKENS` | `1024` | Max tokens for vision API responses |
| `VISION_MAX_CONCURRENCY` | `6` | Image descriptions run in parallel during ingest. Raise to speed up image-heavy documents; lower if you hit Azure rate limits |
| `SECRET_KEY` | - | **Required.** Signs session cookies; startup refuses the shipped default outside development |
| `RATE_LIMIT` | `60/minute` | Per-IP request budget (health and static are exempt) |
| `RATE_LIMIT_ENABLED` | `true` | Set false to disable rate limiting |
| `DATA_DIR` | `data` | Relational stores (users, chats, evals) |
| `EVAL_GATING_ENABLED` | `true` | Enable eval-gated answer pipeline |
| `EVAL_QUALITY_THRESHOLD` | `0.5` | Minimum faithfulness to accept an answer |
| `EVAL_MAX_RETRIES` | `1` | Regeneration attempts when faithfulness is too low |
| `LANGFUSE_ENABLED` | `true` | Enable Langfuse tracing |
| `LANGFUSE_PUBLIC_KEY` | *(empty)* | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | *(empty)* | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse server URL |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Comma-separated allowed origins |
| `VECTORSTORE_DIR` | `data/vectorstore` | Path to persist FAISS index and SQLite DBs |
| `UPLOAD_DIR` | `uploads` | Path for uploaded documents |

---

## RAG Pipeline

The RAG pipeline follows a retrieve → augment → generate pattern with extensive preprocessing:

### Document Ingestion (Upload)

```
Upload → Save File → Multimodal Preprocessing → 7-Step Chunking → Index (FAISS + BM25)
                                                                   → Auto-Generate Golden Dataset
```

**Multimodal Preprocessing** (`preprocessing.py`):
1. **PDF**: Extracts text per-page, embedded raster images, tables (heuristic + vision OCR), and flowcharts/diagrams/charts via PyMuPDF
2. **DOCX**: Extracts inline/floating images, tables natively into Markdown, SmartArt/embedded charts
3. **Images**: Standalone images (PNG, JPG, etc.) are described by GPT-5.2 vision
4. Each visual element is classified (image/table/flowchart/chart/diagram) and described with specialised vision prompts

**7-Step Hybrid Chunking Pipeline** (`document_loader.py`):

| Step | Operation | Details |
|---|---|---|
| 1 | **Text Cleaning** | Normalise whitespace, collapse excessive blank lines |
| 2 | **Structural Split** | Split by Markdown/document headers to respect section boundaries |
| 3 | **Semantic Split** | Embedding cosine-distance based splitting within sections (percentile threshold) |
| 4 | **Token-Aware Sizing** | Enforce token budgets using tiktoken (cl100k_base); produces **parent chunks** |
| 5 | **Contextual Headers** | Prepend `[Document: filename | Section: header]` to every chunk for self-contained retrieval |
| 6 | **Child Splitting** | Fine-grained recursive split of header-enriched parents → **child chunks** with `parent_id` links |
| 7 | **Deduplication** | Remove near-duplicate chunks (>90% token-set overlap) |

### Query Pipeline

```
Question → Hybrid Search → Context Assembly → LLM Generation → Response
              │                    │                │
              ├─ Dense (FAISS)     │                └─ Langfuse tracing
              ├─ Sparse (BM25)    │
              └─ RRF Fusion       └─ Parent Expansion
                  + User filtering
```

**Hybrid Retrieval** (`vector_store.py`):
1. **Dense retrieval**: FAISS embedding similarity search (over-fetch 3× top_k)
2. **Sparse retrieval**: BM25 Okapi keyword search (over-fetch 3× top_k)
3. **User isolation**: Filter results to only documents belonging to the requesting user
4. **Reciprocal Rank Fusion**: Merge both lists with weighted RRF scores: `score(d) = Σ weight_i / (k_rrf + rank_i(d))`
5. **Parent expansion**: Replace matched child chunks with their parent chunks for richer LLM context

---

## Eval-Gated Pipeline

The eval-gated pipeline (`ask_with_eval()` in `rag_engine.py`) ensures answer quality before delivery by running RAGAS metrics in-line:

```
Question
    │
    ▼
┌─── Retrieve Context ───┐
│                        │
▼                        ▼
Generate Answer     Context Precision
(async, parallel)   (sync in thread, parallel)
│                        │
└────────┬───────────────┘
         ▼
   Faithfulness Gate
   (needs the answer)
         │
    ┌────┴────┐
    │ PASS    │ FAIL
    ▼         ▼
  Stream    Regenerate with
  Answer    stricter prompt
              │
              ▼
         Re-check Faithfulness
              │
         ┌────┴────┐
         │ PASS    │ FAIL (use if better)
         ▼         ▼
       Stream    Stream best answer
       Answer    with warning
```

**Latency-optimised flow:**
1. **Parallel step**: Generate answer (async) + evaluate context precision (in threadpool) — concurrently
2. **Sequential step**: Run faithfulness check on the generated answer
3. **Gate check**: If `faithfulness >= threshold` → stream the answer
4. **Regeneration**: On failure, regenerate with a stricter system prompt (lower temperature, explicit faithfulness instructions) and re-check
5. **Fallback**: Use the better-scoring answer (original or regenerated)

**SSE Event Stream:**
- `{"type": "meta", ...}` — sources, images, trace_id
- `{"type": "eval", "scores": {...}}` — quality gate result (pass/fail + scores)
- `{"type": "token", "content": "..."}` — each token of the verified answer
- `{"type": "done", "usage": {...}}` — final token usage stats

---

## Evaluation System

### Per-Query Evaluation (Background)

After every chat response, a full 3-metric RAGAS evaluation runs in a background thread:
- **Faithfulness** — Are claims in the answer supported by the retrieved context?
- **Answer Relevancy** — Is the answer relevant to the question?
- **Context Precision** (without reference) — Are retrieved chunks relevant to the question?

Scores are pushed to Langfuse as trace scores and stored locally in SQLite `query_scores` table. The UI can poll `/api/v1/chat/scores/{trace_id}` to display quality badges.

### Golden Dataset

Golden datasets (question + ground truth answer pairs) are generated:
1. **Automatically on document upload** — background thread generates Q&A pairs from parent chunks via GPT-5.2
2. **Manually via admin panel** — add/edit/delete individual samples
3. **Synthetically on demand** — admin triggers generation from all indexed documents

### Batch Evaluation

Admin can trigger a batch evaluation run that:
1. Iterates through the golden dataset
2. Runs each question through the full RAG pipeline (retrieve + generate)
3. Evaluates with 4 RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
4. Stores per-sample results and computes averages
5. Pushes aggregate scores to Langfuse

---

## Caching

### Eval Cache

Evaluation scores are cached in SQLite keyed by a SHA-256 hash of `question + sorted(contexts)`. Both `evaluate_context_precision_sync()` and `evaluate_faithfulness_sync()` check the cache before making LLM calls, avoiding redundant RAGAS evaluations for repeated queries with the same context.

### Other Caches

| Cache | Type | Purpose |
|---|---|---|
| **Settings** | `@lru_cache` (singleton) | Environment config loaded once |
| **Embeddings client** | `@lru_cache` (singleton) | AzureOpenAIEmbeddings instance |
| **FAISS store** | Thread-safe singleton | Vector index loaded/created once |
| **BM25 index** | In-memory singleton | Keyword index rebuilt on add/delete |
| **Parent chunk store** | In-memory `dict[str, Document]` | Maps `chunk_id` → parent document |
| **Image metadata store** | In-memory `dict[str, list[dict]]` | Maps source → image records |
| **Langfuse client** | Lazy singleton | Initialised on first use |
| **tiktoken encoder** | Module-level singleton | `cl100k_base` encoder |

---

## API Reference

All API endpoints are prefixed with `/api/v1`. Page routes are served at the root.

### Health

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | None | Health check — returns status, version, environment |

### Chat

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/chat/` | Cookie | Send a question, receive a RAG-augmented answer |
| `POST` | `/api/v1/chat/stream` | Cookie | Stream an eval-gated RAG answer via SSE |
| `GET` | `/api/v1/chat/sessions` | Cookie | List chat sessions for the authenticated user |
| `GET` | `/api/v1/chat/sessions/{id}` | Cookie | Get all messages for a chat session |
| `PATCH` | `/api/v1/chat/sessions/{id}` | Cookie | Rename a chat session |
| `DELETE` | `/api/v1/chat/sessions/{id}` | Cookie | Delete a session and all its messages |
| `GET` | `/api/v1/chat/scores/{trace_id}` | None | Poll RAGAS quality scores (204 if not ready) |

### Documents

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/documents/upload` | Cookie | Upload, chunk, and index a document (multipart) |
| `GET` | `/api/v1/documents/history` | Cookie | List documents uploaded by authenticated user |
| `DELETE` | `/api/v1/documents/{doc_id}` | Cookie | Delete a document and all its chunks |
| `GET` | `/api/v1/documents/stats` | None | Vector store collection stats |

### Feedback

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/feedback/` | None | Submit thumbs up/down feedback (pushed to Langfuse) |

### Users

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/users/login` | None | Login or auto-create user by username |

### Admin (requires admin role)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/admin/users` | Admin | List all users with doc counts |
| `POST` | `/api/v1/admin/users` | Admin | Create a user with role |
| `DELETE` | `/api/v1/admin/users/{id}` | Admin | Delete a user and their documents |
| `PATCH` | `/api/v1/admin/users/{id}/role` | Admin | Change a user's role (user/admin) |
| `GET` | `/api/v1/admin/stats` | Admin | System-wide dashboard stats |
| `GET` | `/api/v1/admin/langfuse/traces` | Admin | Fetch recent Langfuse traces |
| `GET` | `/api/v1/admin/langfuse/summary` | Admin | Aggregated cost/usage summary |
| `GET` | `/api/v1/admin/golden` | Admin | List golden dataset samples |
| `POST` | `/api/v1/admin/golden` | Admin | Add a manual golden sample |
| `DELETE` | `/api/v1/admin/golden/{id}` | Admin | Delete a golden sample |
| `DELETE` | `/api/v1/admin/golden` | Admin | Clear entire golden dataset |
| `POST` | `/api/v1/admin/golden/generate` | Admin | Auto-generate golden dataset from documents |
| `POST` | `/api/v1/admin/evaluate` | Admin | Start a batch evaluation run |
| `GET` | `/api/v1/admin/evaluate/runs` | Admin | List evaluation runs |
| `GET` | `/api/v1/admin/evaluate/runs/{id}` | Admin | Get evaluation run detail with per-sample results |

### Pages (Server-Rendered)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | Session cookie | Login page or chat app (depending on auth state) |
| `POST` | `/login` | None | Handle login form submission |
| `POST` | `/register` | None | Handle registration form submission |
| `GET` | `/logout` | None | Clear session and redirect to login |
| `GET` | `/admin` | Admin cookie | Admin dashboard page |
| `GET` | `/partials/stats` | None | HTMX partial — sidebar stats fragment |
| `GET` | `/partials/doc-history` | Cookie | HTMX partial — document history list |

---

## Frontend

The UI is server-rendered with **Jinja2 templates** and uses **HTMX** for dynamic interactions without a JavaScript framework.

### Pages

| Page | Template | Description |
|---|---|---|
| **Login/Register** | `templates/login.html` | Tabbed sign-in / create account form with brute-force protection feedback |
| **Chat App** | `templates/app.html` | Full chat interface with sidebar (upload, doc history, sessions), message area, streaming responses |
| **Admin Dashboard** | `templates/admin.html` | User management, Langfuse insights, golden dataset management, evaluation runner |
| **Base Layout** | `templates/base.html` | Dark theme base with Tailwind config, HTMX, animations, page loader |

### Design

- **Dark theme** with purple accent (`#7c5cfc`) and glass morphism effects
- Inter font family, custom scrollbar styling
- Page loader with spinner animation
- Responsive layout

---

## Security

| Feature | Implementation |
|---|---|
| **Password Hashing** | PBKDF2-SHA256 with 260,000 iterations and random 16-byte salt |
| **Session Cookies** | HMAC-signed via `itsdangerous.URLSafeTimedSerializer`, HttpOnly, Secure (in production), SameSite=Lax, 7-day expiry |
| **Brute-Force Protection** | 5 failed attempts → 5-minute lockout per username |
| **API Key Auth** | Optional `X-API-Key` header with constant-time comparison (`hmac.compare_digest`) |
| **CORS** | Configurable origins; wildcard with credentials explicitly denied |
| **Security Headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`, Content Security Policy |
| **File Upload Validation** | Extension allowlist, 50 MB size limit, magic bytes verification, path traversal prevention |
| **User Isolation** | Documents and retrieval results are scoped to the authenticated user's `user_id` |
| **Error Handling** | Global exception handler prevents stack trace leakage |
| **Non-Root Container** | Docker runs as `appuser` (non-root) |
| **Input Validation** | Pydantic schemas with field constraints (min/max length, regex patterns, value ranges) |

---

## Docker Deployment

### Build and Run

```bash
docker compose up --build -d
```

### Details

- **Port**: `8000:8000`
- **Env file**: `.env` loaded automatically
- **Volumes**: `data/vectorstore` and `uploads` are persisted to host
- **Health check**: Polls `/api/v1/health` every 30s
- **Restart policy**: `unless-stopped`
- **Base image**: `python:3.13-slim` with uv and a non-root `appuser`
- **Health check**: probes `/api/v1/health` with the Python interpreter (the slim image has no `curl`)
- **Workers**: 1 Uvicorn worker. The FAISS index, BM25 index, parent-chunk map and login counters are per-process state, so additional workers would each hold a divergent copy. Scaling out requires moving that state to a shared store first.

---

## Project Structure

```
.
├── starter.py                  # One-click launcher (uv sync + server)
├── pyproject.toml              # Project metadata + dependencies
├── uv.lock                     # Fully pinned, reproducible dependency lock
├── .python-version             # Pinned interpreter (3.13.12)
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Docker Compose orchestration
├── .env                        # Environment variables (create manually)
│
├── app/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings — all env vars centralised
│   ├── main.py                 # FastAPI app factory, middleware, routes, lifespan
│   │
│   ├── api/
│   │   ├── middleware.py        # SecurityHeaders, RequestLogging, APIKey, global exception handler
│   │   └── routes/
│   │       ├── health.py        # GET /api/v1/health
│   │       ├── chat.py          # POST /chat, /chat/stream, session CRUD, score polling
│   │       ├── documents.py     # POST /upload, GET /history, DELETE /{doc_id}, GET /stats
│   │       ├── feedback.py      # POST /feedback (Langfuse score)
│   │       ├── users.py         # POST /users/login
│   │       ├── admin.py         # Admin: users, Langfuse, golden dataset, evaluation
│   │       └── pages.py         # Server-rendered pages (login, app, admin, HTMX partials)
│   │
│   ├── core/
│   │   ├── rag_engine.py        # RAG orchestrator: ask(), ask_stream(), ask_with_eval()
│   │   ├── vector_store.py      # FAISS + BM25 hybrid store, RRF fusion, parent expansion
│   │   ├── document_loader.py   # File I/O, multimodal loading, 7-step chunking pipeline
│   │   ├── preprocessing.py     # PDF/DOCX visual extraction, content classification, vision prompts
│   │   ├── embeddings.py        # Azure OpenAI Embeddings wrapper (singleton)
│   │   ├── vision.py            # GPT-5.2 vision: image description with Langfuse tracing
│   │   ├── evaluator.py         # RAGAS evaluation: per-query, batch, synthetic golden generation
│   │   ├── eval_store.py        # SQLite: golden dataset, eval runs/results, query scores, eval cache
│   │   ├── chat_store.py        # SQLite: chat sessions and messages
│   │   ├── user_store.py        # SQLite: users, documents, admin ops, PBKDF2 passwords
│   │   ├── auth.py              # Signed cookies, brute-force protection, FastAPI dependencies
│   │   ├── observability.py     # Langfuse client singleton, trace/span/score helpers
│   │   └── logging.py           # structlog configuration
│   │
│   └── models/
│       └── schemas.py           # Pydantic request/response models for all endpoints
│
├── templates/
│   ├── base.html                # Base layout: dark theme, Tailwind config, HTMX, animations
│   ├── login.html               # Login/register page with tabs
│   ├── app.html                 # Main chat application page
│   ├── admin.html               # Admin dashboard
│   └── partials/
│       ├── stats.html           # Vector store stats fragment
│       └── doc_history.html     # Document history list fragment
│
├── static/
│
├── data/vectorstore/            # Persisted FAISS index
├── uploads/                     # User-uploaded documents (per-user subdirectories)
│   └── extracted/               # Extracted images/tables from PDFs
└── tests/                       # Test directory (empty — tests not yet implemented)
```

---

## License

MIT
