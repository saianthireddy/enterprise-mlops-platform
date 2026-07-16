import pytest
from sklearn.linear_model import LogisticRegression

from mlops_platform.registry.model_registry import ModelRegistry
from mlops_platform.training.pipeline import make_synthetic_dataset


def make_model():
    X, y = make_synthetic_dataset(300)
    return LogisticRegression(max_iter=200).fit(X, y)


def test_register_assigns_incrementing_versions(tmp_path):
    registry = ModelRegistry(tmp_path)
    m = make_model()
    assert registry.register("m", m, {"f1": 0.9}).version == 1
    assert registry.register("m", m, {"f1": 0.91}).version == 2


def test_transition_to_production_archives_previous(tmp_path):
    registry = ModelRegistry(tmp_path)
    m = make_model()
    registry.register("m", m, {"f1": 0.9})
    registry.register("m", m, {"f1": 0.92})
    registry.transition("m", 1, "production")
    registry.transition("m", 2, "production")
    versions = {v.version: v.stage for v in registry.list_versions("m")}
    assert versions == {1: "archived", 2: "production"}


def test_invalid_stage_rejected(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register("m", make_model(), {"f1": 0.9})
    with pytest.raises(ValueError):
        registry.transition("m", 1, "galaxy")


def test_load_production_roundtrip(tmp_path):
    registry = ModelRegistry(tmp_path)
    m = make_model()
    mv = registry.register("m", m, {"f1": 0.9})
    registry.transition("m", mv.version, "production")
    loaded, meta = registry.load_model("m")
    X, _ = make_synthetic_dataset(50)
    assert (loaded.predict(X) == m.predict(X)).all()
    assert meta.stage == "production"


def test_load_missing_raises(tmp_path):
    with pytest.raises(LookupError):
        ModelRegistry(tmp_path).load_model("ghost")
