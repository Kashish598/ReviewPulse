"""FastAPI service. Serves the fine-tuned BERT model when it is present, else the pretrained one.

Model folder: `models/nlptown_finetuned_v2/` if it exists, otherwise `models/nlptown_sentiment/`;
set REVIEWPULSE_MODEL_DIR to serve a different folder. GET /health says which model is loaded.

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs

Health: GET /health says the process is up and never loads the model (the container is healthy
within seconds). GET /ready says whether the model has finished loading (200) or not (503). The
model loads in a background thread at startup. GET /stats and GET /metrics return request and
model timings (src/monitoring.py).

Drift: after each prediction the review's simple features are added to a rolling window and
compared with the reference snapshot (src/drift.py). When the window has drifted, an email alert is
sent (src/alerts.py, with a cooldown). GET /drift shows the current state; the status is "disabled"
when there is no reference file. Both run after the response is sent and never fail /predict.

MODEL_PRELOAD=off skips the startup load (the first /predict loads the model instead).

Logs go to logs/reviewpulse.log and the console (see src/logging_config.py). Set LOG_LEVEL=DEBUG
to also log what each request carried and what the model returned.

Every prediction is saved (src/feedback_store.py) and a background thread labels the saved
requests with an LLM (src/labeler.py). Both run after the response is sent, so they add nothing to
the request time, and a failure in either one never fails /predict. FEEDBACK_STORE=off turns saving
off; without GROQ_API_KEY the labeller stays off.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src import alerts, drift, feedback_store, labeler
from src.common import MODELS_DIR, Predictor
from src.logging_config import setup_logging
from src.monitoring import stats

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


_load_lock = threading.Lock()
_loaded_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Load the model once and return it. The lock makes a request that arrives while the startup
    thread is still loading wait for that load instead of starting a second one."""
    global _loaded_predictor
    with _load_lock:
        if _loaded_predictor is None:
            from src.bert_model import BertPredictor, resolve_model_dir

            model_path = resolve_model_dir(MODELS_DIR)
            logger.info("Loading sentiment model from %s", model_path)
            _loaded_predictor = BertPredictor(model_path)
        return _loaded_predictor


def _clear_loaded_predictor() -> None:
    global _loaded_predictor
    with _load_lock:
        _loaded_predictor = None


get_predictor.cache_clear = _clear_loaded_predictor  # same name the old lru_cache offered


def _load_model_into_state(_app: FastAPI) -> None:
    """Load the model and publish it on app.state. Never raises: /ready reports the failure."""
    loader = _app.dependency_overrides.get(get_predictor, get_predictor)
    started = time.perf_counter()
    try:
        _app.state.predictor = loader()
        _app.state.model_error = None
        logger.info("Model ready after %.1f s", time.perf_counter() - started)
    except Exception as error:
        _app.state.model_error = f"{type(error).__name__}: {error}"
        logger.exception("Model failed to load; /ready will report 503")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log_file = setup_logging()
    logger.info("ReviewPulse API starting, log file: %s", log_file)
    _app.state.labeler = labeler.start_labeler()  # None without a store or a GROQ_API_KEY
    _app.state.predictor = None
    _app.state.model_error = None
    drift.set_on_drift(alert_on_drift)
    try:
        drift.get_detector()  # logs at startup whether drift detection is on
    except Exception:
        logger.exception("Could not start drift detection; the API runs without it")
    if get_predictor in _app.dependency_overrides:
        # A test (or an embedding program) supplied its own predictor: it is quick, load it now.
        _load_model_into_state(_app)
    elif os.environ.get("MODEL_PRELOAD", "on").lower() == "off":
        logger.info("MODEL_PRELOAD=off: the model loads on the first /predict call")
    else:
        # The real model takes many seconds. Load it in the background so /health answers at once.
        threading.Thread(
            target=_load_model_into_state, args=(_app,), name="model-loader", daemon=True
        ).start()
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
app.state.predictor = None
app.state.model_error = None


@app.middleware("http")
async def time_predict_requests(request: Request, call_next):
    """Time every /predict request, from arrival to the start of the response (K2)."""
    started = time.perf_counter()
    response = await call_next(request)
    if request.url.path == "/predict" and response.status_code == 200:
        stats.record_request((time.perf_counter() - started) * 1000)
    return response


def alert_on_drift(report: drift.DriftReport) -> None:
    """Called by the detector when the window has drifted. The alert module logs it and sends the
    email in the background, at most once per cooldown. Never raises."""
    alerts.alert_drift(report.to_dict())


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


def _model_info(predictor: Predictor) -> dict:
    info = {"model": getattr(predictor, "name", type(predictor).__name__)}
    if hasattr(predictor, "fine_tuned"):
        info["fine_tuned"] = bool(predictor.fine_tuned)
    return info


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness: the process is up. Never loads the model, so it answers in milliseconds.
    Names the model once it has loaded."""
    info = {"status": "ok"}
    predictor = request.app.state.predictor
    if predictor is not None:
        info.update(_model_info(predictor))
    return info


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: 200 once the model is loaded, 503 while it loads or if loading failed."""
    predictor = request.app.state.predictor
    if predictor is not None:
        return JSONResponse(status_code=200, content={"status": "ready", **_model_info(predictor)})
    error = request.app.state.model_error
    if error:
        return JSONResponse(status_code=503, content={"status": "error", "detail": error})
    return JSONResponse(status_code=503, content={"status": "loading"})


@app.get("/drift")
def get_drift() -> dict:
    """Is the traffic still like the training data? Status: ok, drift, insufficient_data (the
    window is still filling) or disabled (no reference snapshot)."""
    return drift.current_report().to_dict()


@app.get("/stats")
@app.get("/metrics")
def get_stats() -> dict:
    """Request count and average API and model latency, in total and per model."""
    return stats.snapshot()


def predict_with_timing(predictor: Predictor, text: str) -> tuple:
    """One prediction and its stage times. Predictors without `predict_timed` (the mock) are
    timed as a whole."""
    timed = getattr(predictor, "predict_timed", None)
    if timed is not None:
        predictions, stages = timed([text])
        return predictions[0], stages
    started = time.perf_counter()
    prediction = predictor.predict([text])[0]
    return prediction, {"total_ms": (time.perf_counter() - started) * 1000}


@app.post("/predict", response_model=PredictResponse, description=PRIVACY_NOTE)
def predict(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    predictor: Annotated[Predictor, Depends(get_predictor)],
):
    logger.debug("predict: received %d chars: %r", len(request.text), request.text[:200])
    model_name = getattr(predictor, "name", type(predictor).__name__)
    result, stages = predict_with_timing(predictor, request.text)
    stats.record_model(model_name, stages["total_ms"], stages)
    logger.debug("predict: model=%s label=%s score=%s", model_name, result.label, result.score)
    background_tasks.add_task(drift.observe, request.text, result.label)
    background_tasks.add_task(save_request, request.text, result.label, result.score, model_name)
    return PredictResponse(sentiment=result.label, score=result.score)
