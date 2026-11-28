"""Put the model weights where the API looks for them.

Run by the Docker image, once while it is built (optional) and again every time
the container starts. It does nothing when a model is already in place.

Where the model comes from, in order:
  1. MODEL_URL       a zip of the model folder (optional MODEL_SHA256 to verify it)
  2. MODEL_HF_REPO   a Hugging Face Hub repo (optional MODEL_HF_REVISION to pin it)
  3. neither         the public pretrained nlptown model, the same fallback the API uses

Where it goes:
  REVIEWPULSE_MODEL_DIR if set, else models/nlptown_finetuned_v2 for 1 and 2,
  else models/nlptown_sentiment. These are the folders src/bert_model.py looks in.

Exit code 0 when a model is in place, 1 when it could not be fetched.
"""

# fetch_model.py runs again at every start. When the model is baked in, it does nothing.
# If an image was ever built without the model, it is a safety net that downloads it,
# so the container still serves the right weights.

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

logger = logging.getLogger("reviewpulse.model_fetch")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Keep in step with src/bert_model.py (tests/test_fetch_model.py checks this).
FINETUNED_DIR_NAME = "nlptown_finetuned_v2"
PRETRAINED_DIR_NAME = "nlptown_sentiment"
MODEL_DIR_ENV = "REVIEWPULSE_MODEL_DIR"
PRETRAINED_REPO = "nlptown/bert-base-multilingual-uncased-sentiment"

# Weight formats the API never loads. Skipping them saves several hundred MB.
SKIP_PATTERNS = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "*.tflite"]
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Plan:
    dest: Path
    source: str  # "url", "hf", "pretrained" or "none"


def has_model(folder: Path) -> bool:
    return (folder / "config.json").is_file()


def make_plan(env: Mapping[str, str], models_dir: Path = MODELS_DIR) -> Plan:
    override = (env.get(MODEL_DIR_ENV) or "").strip()
    url = (env.get("MODEL_URL") or "").strip()
    repo = (env.get("MODEL_HF_REPO") or "").strip()

    if url:
        source = "url"
    elif repo:
        source = "hf"
    else:
        source = "pretrained"

    if override:
        # With no source there is nothing to download: the folder must already hold a model.
        return Plan(Path(override), "none" if source == "pretrained" else source)
    if source in ("url", "hf"):
        return Plan(models_dir / FINETUNED_DIR_NAME, source)

    # No source given. Keep a fine-tuned model that was baked into the image.
    finetuned = models_dir / FINETUNED_DIR_NAME
    if has_model(finetuned):
        return Plan(finetuned, source)
    return Plan(models_dir / PRETRAINED_DIR_NAME, source)


def redact(url: str) -> str:
    """Drop the query string, which may carry a token, before a URL is logged."""
    parts = urlparse(url)
    return parts._replace(query="", fragment="").geturl()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https", "file"}:
        raise ValueError(f"MODEL_URL must be http, https or file, got {scheme!r}")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        with target.open("wb") as out:
            shutil.copyfileobj(response, out, 1024 * 1024)


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip, refusing absolute paths, '..' parts and symlinks."""
    dest = dest.resolve()
    for info in archive.infolist():
        name = info.filename
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or "\\" in name:
            raise ValueError(f"Unsafe path in zip: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Symlink in zip: {name!r}")
        if not (dest / name).resolve().is_relative_to(dest):
            raise ValueError(f"Unsafe path in zip: {name!r}")
    archive.extractall(dest)


def find_model_root(base: Path) -> Path:
    """The folder that holds config.json: the base itself or the shallowest folder below it."""
    configs = sorted(base.rglob("config.json"), key=lambda p: len(p.parts))
    if not configs:
        raise ValueError("The download has no config.json, so it is not a model folder")
    return configs[0].parent


def fetch_from_url(url: str, staging: Path, expected_sha256: str) -> Path:
    archive_path = staging / "download.zip"
    logger.info("Downloading model zip from %s", redact(url))
    download_file(url, archive_path)

    expected = expected_sha256.strip().lower()
    if expected:
        actual = sha256_of(archive_path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
        logger.info("SHA-256 matches")
    else:
        logger.warning("MODEL_SHA256 not set, the download was not verified")

    unpacked = staging / "unpacked"
    unpacked.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        safe_extract(archive, unpacked)
    return find_model_root(unpacked)


def download_hf(repo_id: str, dest: Path, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        ignore_patterns=SKIP_PATTERNS,
    )
    shutil.rmtree(dest / ".cache", ignore_errors=True)


def fetch_from_hf(repo_id: str, revision: str | None, staging: Path) -> Path:
    logger.info("Downloading model from Hugging Face repo %s (revision %s)", repo_id, revision)
    target = staging / "model"
    target.mkdir()
    download_hf(repo_id, target, revision)
    return find_model_root(target)


def fetch_pretrained(staging: Path) -> Path:
    import json

    logger.info("No model source set, downloading the pretrained %s", PRETRAINED_REPO)
    target = staging / "model"
    target.mkdir()
    download_hf(PRETRAINED_REPO, target, None)
    root = find_model_root(target)
    metadata = {"model_name": PRETRAINED_REPO, "fine_tuned": False}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return root


def run(env: Mapping[str, str], models_dir: Path = MODELS_DIR) -> int:
    plan = make_plan(env, models_dir)

    if has_model(plan.dest):
        logger.info("Model already in place at %s, nothing to do", plan.dest)
        return 0
    if plan.source == "none":
        logger.error(
            "%s=%s holds no model and no MODEL_URL or MODEL_HF_REPO is set",
            MODEL_DIR_ENV,
            plan.dest,
        )
        return 1

    plan.dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=plan.dest.parent))
    try:
        if plan.source == "url":
            root = fetch_from_url(env["MODEL_URL"].strip(), staging, env.get("MODEL_SHA256", ""))
        elif plan.source == "hf":
            revision = (env.get("MODEL_HF_REVISION") or "").strip() or None
            root = fetch_from_hf(env["MODEL_HF_REPO"].strip(), revision, staging)
        else:
            root = fetch_pretrained(staging)

        if not (root / "metadata.json").is_file():
            logger.warning(
                "No metadata.json in the model folder, /health will not report it as fine-tuned"
            )
        if plan.dest.exists():
            shutil.rmtree(plan.dest)
        os.replace(root, plan.dest)
        logger.info("Model ready at %s", plan.dest)
        return 0
    except Exception:
        logger.exception("Could not fetch the model")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return run(os.environ)


if __name__ == "__main__":
    sys.exit(main())
