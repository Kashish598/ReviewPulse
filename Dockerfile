# syntax=docker/dockerfile:1

# ---------- build stage: Python packages go into a virtualenv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Requirements are copied before the source, so this layer is cached until the pins change.
COPY requirements-serve.txt .

# CPU-only torch keeps the image far smaller than the default CUDA wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_PIN="$(grep -E '^torch==' requirements-serve.txt)" \
    && pip install --default-timeout=1000 --retries 10 "${TORCH_PIN}" --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries 10 -r requirements-serve.txt


# ---------- run stage ----------
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000

# Run as a normal user, not root.
RUN useradd --system --no-create-home --uid 10001 appuser

WORKDIR /app
RUN mkdir -p /app/models /app/logs /app/.cache \
    && chown -R appuser /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser src ./src
COPY --chown=appuser scripts/fetch_model.py scripts/healthcheck.py ./scripts/

USER appuser

# Optional: bake the fine-tuned model into the image so the container starts fast and
# an older image tag brings its model with it. Leave these empty to fetch at start instead.
# Build arguments stay visible in the image history, so never put a secret in MODEL_URL.
#   docker build --build-arg MODEL_HF_REPO=<user>/<repo> -t mlops-app:v1 .
ARG MODEL_HF_REPO=""
ARG MODEL_HF_REVISION=""
ARG MODEL_URL=""
ARG MODEL_SHA256=""
RUN if [ -n "${MODEL_HF_REPO}${MODEL_URL}" ]; then python scripts/fetch_model.py; fi

EXPOSE 8000

# /health is the course-required check. The start period covers the first model load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

# Make sure a model is in place (instant when one is baked in), then listen on all
# interfaces on $PORT, which hosting platforms set and which defaults to 8000.
CMD ["sh", "-c", "python scripts/fetch_model.py && exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
