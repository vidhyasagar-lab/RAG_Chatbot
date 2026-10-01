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
# The trust list is the private ranges, and it must not be "127.0.0.1" or
# "*". Both of those have been wrong here, for opposite reasons.
#
# Not loopback: requests do NOT arrive from 127.0.0.1. The container is
# bridge-networked, so Docker's userland proxy forwards them in and uvicorn
# sees the bridge gateway. Verified on the VM - every proxied request logs
# 172.18.0.1, and only in-container healthchecks log 127.0.0.1. A
# loopback-only list never matches, so the rewrite never happens and every
# caller shares the gateway's rate-limit bucket.
#
# Not "*" either, which is what this said before. Trusting every host makes
# uvicorn take the LEFTMOST X-Forwarded-For entry:
#
#     if self.always_trust:
#         return _parse_host_port(x_forwarded_for_hosts[0])
#
# That was justified here by the claim that Caddy overwrites the header, so
# there would only ever be one entry. The measured part was the gateway
# address above; the overwrite was an assumption, and Caddy's documented
# default is to APPEND the peer. If it appends, a caller who sends their own
# X-Forwarded-For puts a value of their choosing to the left of the real one,
# uvicorn believes it, and request.client.host - which the rate limiter keys
# on - becomes theirs to pick. A fresh budget on every request.
#
# A real list makes uvicorn walk the header from the right and return the
# first untrusted hop, which is correct whether Caddy appends or overwrites,
# and stops depending on which. The whole private range rather than
# 172.18.0.1 alone because Compose chooses that subnet itself and naming one
# would break the same silent way a loopback list does. A public client
# address is never private, so nothing real is skipped.
#
# The port binding in docker-compose.yml is still the outer boundary, but it
# is no longer the only thing standing between a caller and the limiter.
# Keep this in step with TRUSTED_PROXY_IPS in app/api/rate_limit.py;
# tests/test_rate_limit_proxy.py fails if they drift.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"]
