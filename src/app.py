"""FastAPI service. Serves the fine-tuned BERT model when it is present, else the pretrained one.

Model folder: `models/nlptown_finetuned_v2/` if it exists, otherwise `models/nlptown_sentiment/`;
set REVIEWPULSE_MODEL_DIR to serve a different folder. GET /health says which model is loaded.

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs

Logs go to logs/reviewpulse.log and the console (see src/logging_config.py). Set LOG_LEVEL=DEBUG
to also log what each request carried and what the model returned.

Every prediction is saved (src/feedback_store.py) and a background thread labels the saved
requests with an LLM (src/labeler.py). Both run after the response is sent, so they add nothing to
the request time, and a failure in either one never fails /predict. FEEDBACK_STORE=off turns saving
off; without GROQ_API_KEY the labeller stays off.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src import feedback_store, labeler
from src.common import MODELS_DIR, Predictor
from src.logging_config import setup_logging

MAX_TEXT_CHARS = 5000
PRIVACY_NOTE = (
    "Privacy: the review text you submit is stored together with the model's answer and may be "
    "used to retrain the model. Do not send personal data."
)
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
    _app.state.labeler = labeler.start_labeler()  # None without a store or a GROQ_API_KEY
    yield
    if _app.state.labeler is not None:
        _app.state.labeler.stop()
    logger.info("ReviewPulse API shutting down")


app = FastAPI(
    title="ReviewPulse",
    version="0.1.0",
    description=f"Review sentiment API. {PRIVACY_NOTE}",
    lifespan=lifespan,
)
app.state.labeler = None


def save_request(text: str, label: str, score: float | None, model_name: str | None) -> None:
    """Runs after the response: saves the request and wakes the labeller. Never raises."""
    saved = feedback_store.record_prediction(text, label, score, model_name)
    background_labeler = getattr(app.state, "labeler", None)
    if saved is None or background_labeler is None:
        return
    try:
        background_labeler.wake()
    except Exception:
        logger.exception("Could not wake the labeller, the request stays pending")


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


@app.post("/predict", response_model=PredictResponse, description=PRIVACY_NOTE)
def predict(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    predictor: Annotated[Predictor, Depends(get_predictor)],
):
    logger.debug("predict: received %d chars: %r", len(request.text), request.text[:200])
    result = predictor.predict([request.text])[0]
    model_name = getattr(predictor, "name", type(predictor).__name__)
    logger.debug("predict: model=%s label=%s score=%s", model_name, result.label, result.score)
    background_tasks.add_task(save_request, request.text, result.label, result.score, model_name)
    return PredictResponse(sentiment=result.label, score=result.score)
