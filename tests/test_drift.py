from mlops_platform.monitoring.alerts import ConsoleAlerter, alert_on_drift
from mlops_platform.monitoring.drift import detect_drift, population_stability_index
from mlops_platform.training.pipeline import make_synthetic_dataset


def test_no_drift_on_same_distribution():
    ref, _ = make_synthetic_dataset(2000, seed=1)
    cur, _ = make_synthetic_dataset(2000, seed=2)
    report = detect_drift(ref, cur)
    assert not report.has_drift


def test_drift_detected_on_shifted_distribution():
    ref, _ = make_synthetic_dataset(2000, seed=1)
    cur, _ = make_synthetic_dataset(2000, seed=2, shift=1.5)
    report = detect_drift(ref, cur)
    assert report.has_drift
    assert len(report.drifted_features) > 0


def test_psi_zero_for_identical_sample():
    ref, _ = make_synthetic_dataset(1000, seed=3)
    assert population_stability_index(ref[:, 0], ref[:, 0]) < 0.01


def test_alert_fires_only_on_drift():
    ref, _ = make_synthetic_dataset(1500, seed=1)
    same, _ = make_synthetic_dataset(1500, seed=4)
    shifted, _ = make_synthetic_dataset(1500, seed=5, shift=1.5)
    alerter = ConsoleAlerter()
    assert alert_on_drift(detect_drift(ref, same), alerter, "m") is False
    assert alert_on_drift(detect_drift(ref, shifted), alerter, "m") is True
    assert len(alerter.sent) == 1
