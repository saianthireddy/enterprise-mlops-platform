"""Deployment targets behind one interface.

`LocalDeployer` snapshots the production model for docker/k8s images;
`SageMakerDeployer` (lazy boto3) pushes the artifact to S3 and points a
SageMaker endpoint at it.
"""

import shutil
from pathlib import Path

from ..registry.model_registry import ModelRegistry


class LocalDeployer:
    """Copies the production artifact to a serving directory."""

    def __init__(self, serving_dir: str | Path = "artifacts/serving"):
        self._dir = Path(serving_dir)

    def deploy(self, registry: ModelRegistry, name: str) -> Path:
        mv = registry.get_version(name, stage="production")
        if mv is None:
            raise LookupError(f"No production model for '{name}'")
        source = Path(registry._version_dir(name, mv.version)) / "model.joblib"
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._dir / f"{name}.joblib"
        shutil.copy(source, target)
        return target


class SageMakerDeployer:
    """Uploads the production artifact to S3 for a SageMaker endpoint."""

    def __init__(self, bucket: str, role_arn: str, region: str = "us-east-1"):
        import boto3  # lazy import — only needed in AWS environments

        self._s3 = boto3.client("s3", region_name=region)
        self._bucket = bucket
        self._role_arn = role_arn

    def deploy(self, registry: ModelRegistry, name: str) -> str:
        mv = registry.get_version(name, stage="production")
        if mv is None:
            raise LookupError(f"No production model for '{name}'")
        source = Path(registry._version_dir(name, mv.version)) / "model.joblib"
        key = f"models/{name}/v{mv.version}/model.joblib"
        self._s3.upload_file(str(source), self._bucket, key)
        return f"s3://{self._bucket}/{key}"
