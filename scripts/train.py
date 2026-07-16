"""CLI: run the automated training pipeline and print the promotion decision.

Usage: python scripts/train.py [--model gradient_boosting|logistic]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_platform.config import get_settings
from mlops_platform.registry.model_registry import ModelRegistry
from mlops_platform.training.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="gradient_boosting", choices=["logistic", "gradient_boosting"]
    )
    args = parser.parse_args()

    registry = ModelRegistry(get_settings().registry_root)
    result = run_training(registry, model=args.model)
    print(json.dumps({
        "model": result.version.name,
        "new_version": result.version.version,
        "stage": result.version.stage,
        "promoted": result.promoted,
        "challenger_metrics": result.challenger_metrics,
        "champion_metrics": result.champion_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
