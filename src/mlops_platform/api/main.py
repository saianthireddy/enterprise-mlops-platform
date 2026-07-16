"""FastAPI service over the platform: predictions, registry, drift checks.

Run locally:  uvicorn mlops_platform.api.main:app --reload

If the registry is empty the service bootstraps itself by running one
training cycle, so it always boots — locally, in Docker, or on k8s.
"""

from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException

from .. import __version__
from ..config import get_settings
from ..monitoring.alerts import alert_on_drift, build_alerter
from ..monitoring.drift import detect_drift
from ..registry.model_registry import ModelRegistry
from ..training.pipeline import make_synthetic_dataset, run_training
from .schemas import (
    DriftCheckRequest,
    DriftCheckResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="Enterprise MLOps Platform", version=__version__)


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    settings = get_settings()
    registry = ModelRegistry(settings.registry_root)
    if registry.get_version(settings.model_name, stage="production") is None:
        run_training(registry)  # bootstrap so the service always starts
    return registry


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    mv = get_registry().get_version(settings.model_name, stage="production")
    return HealthResponse(
        status="ok",
        version=__version__,
        production_model=f"{settings.model_name}/v{mv.version}" if mv else "none",
    )


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    settings = get_settings()
    return [
        ModelInfo(name=mv.name, version=mv.version, stage=mv.stage, metrics=mv.metrics)
        for mv in get_registry().list_versions(settings.model_name)
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    settings = get_settings()
    try:
        model, mv = get_registry().load_model(settings.model_name, stage="production")
    except LookupError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    row = np.asarray([request.features])
    return PredictResponse(
        prediction=int(model.predict(row)[0]),
        probability=round(float(model.predict_proba(row)[0, 1]), 4),
        model_name=mv.name,
        model_version=mv.version,
    )


@app.post("/drift/check", response_model=DriftCheckResponse)
def drift_check(request: DriftCheckRequest) -> DriftCheckResponse:
    settings = get_settings()
    reference, _ = make_synthetic_dataset(seed=settings.random_state)
    report = detect_drift(
        reference,
        np.asarray(request.current),
        psi_threshold=settings.drift_psi_threshold,
        ks_threshold=settings.drift_ks_threshold,
    )
    alerted = alert_on_drift(
        report, build_alerter(settings.slack_webhook_url), settings.model_name
    )
    return DriftCheckResponse(
        has_drift=report.has_drift,
        drifted_features=report.drifted_features,
        alerted=alerted,
    )
