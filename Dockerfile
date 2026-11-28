# PURPOSE:
# Builds a Docker image for ReviewPulse, packaging the application,
# Python dependencies, model and configuration needed to run it.
# Uses a multi-stage build to keep the final image smaller.
# The Dockerfile tells Docker how to package ReviewPulse into a container so that the application can run consistently on different machines and hosting platforms.
# Without Docker, the application would need to be installed and configured on each machine, which can lead to inconsistencies and errors. Docker ensures that the application runs in the same environment regardless of where it is deployed.

# ============================================================
# STAGE 01: BUILDER
# ============================================================
# Installs all Python dependencies inside a virtual environment.
# This stage is used to prepare dependencies, not run the app.

# ENV: Sets environment variables.
# RUN: Executes commands during image building.
# WORKDIR: Sets the working directory.
# COPY: Copies requirements into the image.
# venv: Creates an isolated Python environment.
# pip install: Installs dependencies.
# CPU-only PyTorch reduces image size by avoiding CUDA packages.
# Cache mount speeds up repeated dependency installations.

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

# ============================================================
# STAGE 02: RUNTIME
# ============================================================
# Creates the final image that runs ReviewPulse.
# Copies only the required dependencies, source code and files
# from the builder stage and prepares the application to run.

# ENV: Configures Python, Hugging Face cache and application port.
# RUN: Creates directories and a non-root application user.
# mkdir: Creates required directories.
# chown: Changes directory ownership.
# COPY --from=builder: Copies the virtual environment from Stage 01.
# COPY: Copies application source code and required scripts.
# USER: Runs the application as a non-root user for security.
# ARG: Defines build-time model configuration.
# HEALTHCHECK: Periodically checks whether the API is responding.
# EXPOSE: Documents the application's listening port.
# CMD: Fetches the model if necessary and starts FastAPI using Uvicorn.

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
COPY --chown=appuser data/reference ./data/reference
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
