"""Central configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    registry_root: str = field(
        default_factory=lambda: os.getenv("REGISTRY_ROOT", "artifacts/registry")
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "churn-classifier"))
    random_state: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))
    promotion_metric: str = field(default_factory=lambda: os.getenv("PROMOTION_METRIC", "f1"))
    drift_psi_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    )
    drift_ks_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_KS_THRESHOLD", "0.1"))
    )
    slack_webhook_url: str = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", ""))
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", ""))


def get_settings() -> Settings:
    return Settings()
