"""Request and response models for the API."""

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

from backend.config import get_settings


def check_text(text: str) -> str:
    """Strip a post and reject it if it is empty or too long.

    Request models are module-level, so the limit comes from the global
    settings (environment / ``.env``), not from a ``Settings`` passed to ``create_app``.
    """
    text = text.strip()
    if not text:
        raise ValueError("Text must not be empty.")
    limit = get_settings().max_text_chars
    if len(text) > limit:
        raise ValueError(f"Text is too long ({len(text)} characters); the limit is {limit}.")
    return text


PostText = Annotated[str, AfterValidator(check_text)]


class PredictRequest(BaseModel):
    """A single post to classify."""

    text: PostText = Field(..., examples=["I can't sleep, my rent is due and I have no idea how to pay it."])


class BatchPredictRequest(BaseModel):
    """Several posts to classify at once."""

    texts: list[PostText] = Field(..., min_length=1, examples=[["I feel calm today.", "I'm panicking."]])

    @field_validator("texts")
    @classmethod
    def _check_batch_size(cls, texts: list[str]) -> list[str]:
        limit = get_settings().max_batch_size
        if len(texts) > limit:
            raise ValueError(f"Too many texts ({len(texts)}); the limit is {limit} per request.")
        return texts


class PredictionOut(BaseModel):
    """One prediction."""

    label: Literal["stress", "no stress"]
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted label")
    stress_probability: float = Field(..., ge=0, le=1)


class PredictResponse(PredictionOut):
    """Response of ``POST /predict``."""

    model_name: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    """Response of ``POST /predict/batch``."""

    predictions: list[PredictionOut]
    model_name: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: Literal["ok"]
    model_name: str


class ModelInfoResponse(BaseModel):
    """Response of ``GET /model-info``."""

    model_name: str
    display_name: str
    is_fallback: bool = Field(..., description="True when RoBERTa is missing and TF-IDF is used")
    test_f1: float | None
    test_f1_ci: tuple[float, float] | None
    trained_at: str | None
    max_text_chars: int


class MetricsResponse(BaseModel):
    """Response of ``GET /metrics``: everything the Results page shows."""

    metrics: list[dict[str, Any]]
    api_check: dict[str, Any] | None = None
    latency: dict[str, Any] | None = None
    errors: list[dict[str, Any]] | None = None
