"""FastAPI app: stress detection for Reddit-style posts.

Run from the project root:
    uvicorn backend.main:app --reload
Then open http://localhost:8000/docs
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Settings, get_settings
from backend.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EmotionsResponse,
    ExplainResponse,
    HealthResponse,
    LabelSheet,
    LabelUpdate,
    MetricsResponse,
    ModelInfoResponse,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    SegmentOut,
)
from backend.service import load_metrics, model_info, timed_predict
from ml.annotation import annotator_items, set_label
from ml.emotions import load_emotion_predictor
from ml.predictors import Prediction, Predictor, load_predictor

logger = logging.getLogger("textlab.api")


def _prediction_out(p: Prediction) -> PredictionOut:
    """Convert an internal prediction to the API model."""
    return PredictionOut(label=p.label, confidence=round(p.confidence, 4),
                         stress_probability=round(p.stress_probability, 4))


def _validation_message(exc: RequestValidationError) -> str:
    """Turn Pydantic's error list into one readable sentence."""
    error = exc.errors()[0]
    message = str(error.get("msg", "Invalid request.")).removeprefix("Value error, ")
    field = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
    return message if message.endswith(".") or not field else f"{field}: {message}"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app; the model is loaded once when the app starts."""
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = load_predictor(settings.roberta_dir, settings.tfidf_dir, settings.onnx_dir)
        logger.info("Loaded model %r from %s", app.state.predictor.name, app.state.predictor.model_dir)
        app.state.emotions = load_emotion_predictor(settings.emotion_roberta_dir, settings.emotion_tfidf_dir)
        logger.info("Emotion model: %s", app.state.emotions.name if app.state.emotions else "none")
        yield

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan,
                  description="Research demo that detects stress in Reddit posts. "
                              "Not a medical diagnosis.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    def predictor(request: Request) -> Predictor:
        return request.app.state.predictor

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    def health(request: Request) -> HealthResponse:
        """Check that the API is running and a model is loaded."""
        return HealthResponse(status="ok", model_name=predictor(request).name)

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["status"])
    def get_model_info(request: Request) -> ModelInfoResponse:
        """Name, test F1 and training date of the loaded model."""
        info = model_info(predictor(request), settings.results_dir)
        emotions = request.app.state.emotions
        return ModelInfoResponse(**info, emotion_model=emotions.name if emotions else None,
                                 max_text_chars=settings.max_text_chars)

    @app.post("/emotions", response_model=EmotionsResponse, tags=["prediction"])
    def emotions(body: PredictRequest, request: Request) -> EmotionsResponse:
        """Extension: probabilities of 27 emotions + neutral (GoEmotions), if an emotion model is installed."""
        model = request.app.state.emotions
        if model is None:
            return EmotionsResponse(available=False)
        start = time.perf_counter()
        rows = model.predict(body.text)
        return EmotionsResponse(available=True, model_name=model.name, display_name=model.display_name,
                                emotions=rows, latency_ms=round((time.perf_counter() - start) * 1000, 2))

    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        """Classify one post as stress or no stress."""
        model = predictor(request)
        (result,), latency = timed_predict(model, [body.text])
        logger.info("predict label=%s latency_ms=%.1f", result.label, latency)
        return PredictResponse(**_prediction_out(result).model_dump(),
                               model_name=model.name, latency_ms=round(latency, 2))

    @app.post("/explain", response_model=ExplainResponse, tags=["prediction"])
    def explain(body: PredictRequest, request: Request) -> ExplainResponse:
        """Classify one post and show which words (TF-IDF) or sentences (RoBERTa) drove the result."""
        model = predictor(request)
        start = time.perf_counter()
        (result,) = model.predict([body.text])
        explanation = model.explain(body.text)
        latency = (time.perf_counter() - start) * 1000
        logger.info("explain label=%s segments=%d latency_ms=%.1f",
                    result.label, len(explanation.segments), latency)
        return ExplainResponse(
            **_prediction_out(result).model_dump(),
            text=body.text,
            unit=explanation.unit,
            method=explanation.method,
            segments=[SegmentOut(**vars(s)) for s in explanation.segments],
            model_name=model.name,
            latency_ms=round(latency, 2),
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
    def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
        """Classify several posts in one request."""
        model = predictor(request)
        results, latency = timed_predict(model, body.texts)
        logger.info("predict_batch n=%d latency_ms=%.1f", len(results), latency)
        return BatchPredictResponse(predictions=[_prediction_out(p) for p in results],
                                    model_name=model.name, latency_ms=round(latency, 2))

    @app.get("/metrics", response_model=MetricsResponse, tags=["results"])
    def metrics() -> MetricsResponse:
        """Evaluation results saved in results/ (for the Results page)."""
        return MetricsResponse(**load_metrics(
            settings.results_dir, settings.error_groups_file,
            settings.label_check_file if settings.labeling_enabled else None,
        ))

    label_lock = Lock()

    def _label_sheet(annotator: str) -> LabelSheet:
        if not settings.labeling_enabled:
            raise HTTPException(status_code=403, detail="Labelling is switched off on this server.")
        if not settings.label_check_file.exists():
            raise HTTPException(status_code=404, detail="No label check file. Run `python -m scripts.make_label_check`.")
        try:
            items = annotator_items(pd.read_csv(settings.label_check_file, dtype={"id": str}), annotator)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown annotator {annotator!r}.") from None
        done = sum(item["label"] is not None for item in items)
        return LabelSheet(annotator=annotator, items=items, n_done=done, n_total=len(items))

    @app.get("/label-check/{annotator}", response_model=LabelSheet, tags=["label check"])
    def get_labels(annotator: str) -> LabelSheet:
        """One team member's label-check posts and labels (the dataset label is not shown)."""
        return _label_sheet(annotator)

    @app.put("/label-check/{annotator}/{post_id}", response_model=LabelSheet, tags=["label check"])
    def put_label(annotator: str, post_id: str, body: LabelUpdate) -> LabelSheet:
        """Save one label: 1 = stress, 0 = no stress, null = clear."""
        _label_sheet(annotator)  # validates settings and annotator
        with label_lock:
            try:
                set_label(settings.label_check_file, annotator, post_id, body.label)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Unknown post {post_id!r}.") from None
        return _label_sheet(annotator)

    if (settings.frontend_dist / "index.html").exists():
        _serve_frontend(app, settings.frontend_dist)

    return app


def _serve_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built React app; unknown paths return index.html (client-side routing).

    Registered after the API routes, so those always take priority.
    """
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        file = (root / path).resolve()
        if path and file.is_file() and file.is_relative_to(root):
            return FileResponse(file)
        return FileResponse(root / "index.html")

    logger.info("Serving the website from %s", dist)


app = create_app()
