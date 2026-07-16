"""CLI: simulate a live window, run drift detection, alert if drifted.

Usage: python scripts/check_drift.py [--shift 0.8]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_platform.config import get_settings
from mlops_platform.monitoring.alerts import alert_on_drift, build_alerter
from mlops_platform.monitoring.drift import detect_drift
from mlops_platform.training.pipeline import make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", type=float, default=0.0, help="mean shift for the live window")
    args = parser.parse_args()

    settings = get_settings()
    reference, _ = make_synthetic_dataset(seed=settings.random_state)
    current, _ = make_synthetic_dataset(seed=settings.random_state + 1, shift=args.shift)

    report = detect_drift(
        reference, current,
        psi_threshold=settings.drift_psi_threshold,
        ks_threshold=settings.drift_ks_threshold,
    )
    alerted = alert_on_drift(report, build_alerter(settings.slack_webhook_url), settings.model_name)
    print(json.dumps({**report.as_dict(), "alerted": alerted}, indent=2))
    sys.exit(1 if report.has_drift else 0)


if __name__ == "__main__":
    main()
