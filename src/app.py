"""FastAPI service. Serves the fine-tuned BERT model when it is present, else the pretrained one.

Model folder: `models/nlptown_finetuned_v2/` if it exists, otherwise `models/nlptown_sentiment/`;
set REVIEWPULSE_MODEL_DIR to serve a different folder. GET /health says which model is loaded.

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs

Logs go to logs/reviewpulse.log and the console (see src/logging_config.py). Set LOG_LEVEL=DEBUG
to also log what each request carried and what the model returned.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.common import MODELS_DIR, Predictor
from src.logging_config import setup_logging

MAX_TEXT_CHARS = 5000
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS, examples=["Great product, works well."])

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class PredictResponse(BaseModel):
    sentiment: Literal["negative", "positive"]
    score: float | None = None


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    from src.bert_model import BertPredictor, resolve_model_dir

    model_path = resolve_model_dir(MODELS_DIR)
    logger.info("Loading sentiment model from %s", model_path)

    return BertPredictor(model_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log_file = setup_logging()
    logger.info("ReviewPulse API starting, log file: %s", log_file)
    yield
    logger.info("ReviewPulse API shutting down")


app = FastAPI(title="ReviewPulse", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def log_unhandled_error(request: Request, error: Exception) -> JSONResponse:
    """Log the full traceback, answer with a generic 500, and keep the server running."""
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=error)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health(predictor: Annotated[Predictor, Depends(get_predictor)]) -> dict:
    info = {"status": "ok", "model": getattr(predictor, "name", type(predictor).__name__)}
    if hasattr(predictor, "fine_tuned"):
        info["fine_tuned"] = bool(predictor.fine_tuned)
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]):
    logger.debug("predict: received %d chars: %r", len(request.text), request.text[:200])
    result = predictor.predict([request.text])[0]
    logger.debug(
        "predict: model=%s label=%s score=%s",
        getattr(predictor, "name", type(predictor).__name__),
        result.label,
        result.score,
    )
    return PredictResponse(sentiment=result.label, score=result.score)
