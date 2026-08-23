FROM python:3.13-slim

# uv, pinned for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/

WORKDIR /app

# System deps for document processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libmagic1 && \
    rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install dependencies in their own cached layer, before the source is copied,
# so code changes don't invalidate the dependency install.
# Note: .python-version is deliberately NOT copied here — it pins the exact
# patch (3.13.12) for local dev, while in the image we use the base image's
# interpreter, which satisfies requires-python = ">=3.13".
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

COPY . .

RUN mkdir -p data/vectorstore uploads/extracted && \
    chown -R appuser:appuser /app

USER appuser

# Put the project venv on PATH so its interpreter is the default `python`
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Single worker, deliberately. The FAISS index, BM25 index, parent-chunk map
# and login-attempt counters are all per-process state, so multiple workers
# would each hold a divergent copy: a document uploaded through one worker is
# invisible to the others, and the login lockout weakens by a factor of N.
# Raising this requires moving that state into a shared store first.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
