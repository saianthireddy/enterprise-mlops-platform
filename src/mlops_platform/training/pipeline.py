"""Automated training with champion/challenger promotion.

`run_training` trains a challenger, scores it on a holdout split, registers it
as a new version, and promotes it to production only if it beats the incumbent
champion on the configured metric. The first model ever trained is promoted
unconditionally, since there is nothing to compare against.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import get_settings
from ..registry.model_registry import ModelRegistry, ModelVersion

N_FEATURES = 6
DEFAULT_N_SAMPLES = 2000

MODELS = {
    "logistic": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
}


@dataclass
class TrainingResult:
    """Outcome of one training cycle."""

    version: ModelVersion
    challenger_metrics: dict
    champion_metrics: dict | None
    promoted: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "model": self.version.name,
            "version": self.version.version,
            "stage": self.version.stage,
            "promoted": self.promoted,
            "reason": self.reason,
            "challenger_metrics": self.challenger_metrics,
            "champion_metrics": self.champion_metrics,
        }


def make_synthetic_dataset(
    n_samples: int = DEFAULT_N_SAMPLES, seed: int = 42, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, learnable 6-feature binary classification dataset.

    *shift* moves the feature means, which is how the drift tests and the
    `check_drift` CLI simulate a distribution change in the live window.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=shift, scale=1.0, size=(n_samples, N_FEATURES))

    # Fixed weights keep the decision boundary identical across seeds, so a
    # model trained on one sample generalises to another.
    weights = np.array([1.4, -1.1, 0.8, 0.0, 0.6, -0.5])
    logits = X @ weights + rng.normal(0.0, 0.35, size=n_samples)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (probabilities > 0.5).astype(int)
    return X, y


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Holdout metrics for a fitted model."""
    predictions = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }
    if len(np.unique(y)) > 1 and hasattr(model, "predict_proba"):
        metrics["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return {k: round(float(v), 6) for k, v in metrics.items()}


def build_pipeline(model: str = "gradient_boosting", seed: int = 42) -> Pipeline:
    """Scaler + classifier behind one estimator."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {sorted(MODELS)}")
    return Pipeline(
        [("scaler", StandardScaler()), ("classifier", MODELS[model](seed))]
    )


def run_training(
    registry: ModelRegistry,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    model: str = "gradient_boosting",
    name: str | None = None,
) -> TrainingResult:
    """Train a challenger and promote it only if it beats the champion."""
    settings = get_settings()
    name = name or settings.model_name
    metric = settings.promotion_metric

    if X is None or y is None:
        X, y = make_synthetic_dataset(seed=settings.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=settings.random_state, stratify=y
    )

    pipeline = build_pipeline(model, seed=settings.random_state)
    pipeline.fit(X_train, y_train)
    challenger_metrics = evaluate(pipeline, X_test, y_test)

    champion = registry.get_version(name, stage="production")
    champion_metrics = champion.metrics if champion else None

    version = registry.register(
        name, pipeline, challenger_metrics, tags={"model": model, "role": "challenger"}
    )

    if champion is None:
        promoted, reason = True, "first model — promoted by default"
    else:
        challenger_score = challenger_metrics.get(metric, 0.0)
        champion_score = champion_metrics.get(metric, 0.0)
        promoted = challenger_score > champion_score
        reason = (
            f"challenger {metric}={challenger_score:.4f} "
            f"{'>' if promoted else '<='} champion {metric}={champion_score:.4f}"
        )

    if promoted:
        version = registry.transition(name, version.version, "production")

    return TrainingResult(
        version=version,
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        promoted=promoted,
        reason=reason,
    )
