"""File-backed model registry with versioning and stage transitions.

Mirrors MLflow Model Registry semantics (versions, stages, metadata) while
staying dependency-free so it runs anywhere — swap in the MLflow adapter via
MLFLOW_TRACKING_URI for a managed backend.
"""

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

import joblib

STAGES = ("none", "staging", "production", "archived")


@dataclass
class ModelVersion:
    name: str
    version: int
    stage: str
    metrics: dict
    tags: dict = field(default_factory=dict)
    created_at: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    def __init__(self, root: str | Path = "artifacts/registry"):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # -- internals ---------------------------------------------------------
    def _model_dir(self, name: str) -> Path:
        return self._root / name

    def _version_dir(self, name: str, version: int) -> Path:
        return self._model_dir(name) / f"v{version}"

    def _meta_path(self, name: str, version: int) -> Path:
        return self._version_dir(name, version) / "meta.json"

    def _write_meta(self, mv: ModelVersion) -> None:
        self._meta_path(mv.name, mv.version).write_text(json.dumps(mv.as_dict(), indent=2))

    def _read_meta(self, name: str, version: int) -> ModelVersion:
        data = json.loads(self._meta_path(name, version).read_text())
        return ModelVersion(**data)

    # -- public API --------------------------------------------------------
    def register(self, name: str, model, metrics: dict, tags: dict | None = None) -> ModelVersion:
        """Persist *model* as the next version of *name* (stage: staging)."""
        version = len(self.list_versions(name)) + 1
        vdir = self._version_dir(name, version)
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        mv = ModelVersion(
            name=name,
            version=version,
            stage="staging",
            metrics={k: round(float(v), 6) for k, v in metrics.items()},
            tags=tags or {},
            created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
        self._write_meta(mv)
        return mv

    def list_versions(self, name: str) -> list[ModelVersion]:
        mdir = self._model_dir(name)
        if not mdir.exists():
            return []
        versions = []
        for vdir in sorted(mdir.glob("v*"), key=lambda p: int(p.name[1:])):
            versions.append(self._read_meta(name, int(vdir.name[1:])))
        return versions

    def transition(self, name: str, version: int, stage: str) -> ModelVersion:
        """Move a version to a new stage; demotes current production if needed."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Choose from {STAGES}")
        if stage == "production":
            current = self.get_version(name, stage="production")
            if current and current.version != version:
                current.stage = "archived"
                self._write_meta(current)
        mv = self._read_meta(name, version)
        mv.stage = stage
        self._write_meta(mv)
        return mv

    def get_version(self, name: str, stage: str = "production") -> ModelVersion | None:
        for mv in reversed(self.list_versions(name)):
            if mv.stage == stage:
                return mv
        return None

    def load_model(self, name: str, stage: str = "production"):
        mv = self.get_version(name, stage)
        if mv is None:
            raise LookupError(f"No model '{name}' in stage '{stage}'")
        return joblib.load(self._version_dir(name, mv.version) / "model.joblib"), mv

    def delete_model(self, name: str) -> None:
        shutil.rmtree(self._model_dir(name), ignore_errors=True)
