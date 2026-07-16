from mlops_platform.registry.model_registry import ModelRegistry
from mlops_platform.training.pipeline import make_synthetic_dataset, run_training


def test_first_training_promotes_to_production(tmp_path):
    registry = ModelRegistry(tmp_path)
    result = run_training(registry, model="logistic")
    assert result.promoted
    assert result.version.stage == "production"
    assert result.challenger_metrics["f1"] > 0.7


def test_second_run_registers_new_version(tmp_path):
    registry = ModelRegistry(tmp_path)
    run_training(registry, model="logistic")
    result = run_training(registry, model="gradient_boosting")
    assert result.version.version == 2
    assert result.champion_metrics is not None


def test_weak_challenger_not_promoted(tmp_path):
    registry = ModelRegistry(tmp_path)
    run_training(registry, model="gradient_boosting")
    X, y = make_synthetic_dataset(400, seed=9)
    y_noisy = y.copy()
    y_noisy[::2] = 1 - y_noisy[::2]  # heavy label noise -> weak challenger
    result = run_training(registry, X=X, y=y_noisy, model="logistic")
    assert result.promoted is False
    assert result.version.stage == "staging"
