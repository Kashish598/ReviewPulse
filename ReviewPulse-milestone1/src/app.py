"""FastAPI service skeleton. Runs on MockPredictor until the BERT adapter is merged.

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field, field_validator

from src.common import MockPredictor, Predictor

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
    # TODO (after BERT PR): load the saved BERT artifact here instead of the mock.
    logger.warning("Serving MockPredictor. This is not a real model.")
    return MockPredictor()


app = FastAPI(title="ReviewPulse", version="0.1.0")


@app.get("/health")
def health(predictor: Annotated[Predictor, Depends(get_predictor)]) -> dict:
    return {"status": "ok", "model": getattr(predictor, "name", type(predictor).__name__)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]):
    result = predictor.predict([request.text])[0]
    return PredictResponse(sentiment=result.label, score=result.score)
