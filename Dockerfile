FROM python:3.13-slim

# uv, pinned for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/

WORKDIR /app

# System deps for document processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# UV_CONCURRENT_DOWNLOADS / UV_HTTP_TIMEOUT: uv defaults to ~50 parallel
# downloads and a 30s per-request timeout. This dependency set pulls several
# very large wheels (pyarrow 48MB, scipy 34MB, pymupdf 25MB, litellm 23MB), and
# on a constrained uplink the parallel streams starve each other until small
# wheels time out mid-flight — the build fails on something tiny like
# s3transfer while the big ones succeed. Fewer streams each get usable
# bandwidth; the longer timeout absorbs the rest. Slightly slower on a fast
# link, but deterministic on a slow one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=180

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
# --proxy-headers so the unauthenticated rate-limit path can tell callers
# apart: it falls back to request.client.host (app/api/rate_limit.py), which
# without this is the same address for everyone behind the proxy, letting one
# caller exhaust the login budget for all.
#
# The trust list is "*", not "127.0.0.1". Requests do NOT arrive from loopback:
# the container is bridge-networked, so Docker's userland proxy forwards them
# in and uvicorn sees the bridge gateway. Verified on the VM - every proxied
# request logs 172.18.0.1, and only in-container healthchecks log 127.0.0.1.
# A loopback-only list therefore never matches and the rewrite never happens.
#
# "*" is safe here because the trust boundary is the port binding, not this
# list: docker-compose.yml publishes to 127.0.0.1:8000, so nothing off-host can
# reach this process at all. Only Caddy can, and Caddy overwrites
# X-Forwarded-For with the real peer. Widening this WOULD be unsafe if the
# published port were ever moved back to 0.0.0.0.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
