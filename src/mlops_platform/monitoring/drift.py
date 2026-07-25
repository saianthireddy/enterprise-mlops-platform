"""Feature-level drift detection using PSI and the two-sample KS statistic.

Both statistics are computed with numpy only — no scipy dependency — so the
drift gate runs in the same slim image as the API.

PSI compares binned population shares between a reference window and a live
window. The conventional reading is: < 0.1 stable, 0.1-0.2 moderate shift,
> 0.2 significant shift. The KS statistic is the maximum distance between the
two empirical CDFs and catches shape changes that binning can smooth over.

A feature is flagged when *either* statistic crosses its threshold.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_KS_THRESHOLD = 0.1
_EPSILON = 1e-6


@dataclass
class FeatureDrift:
    """Per-feature drift statistics."""

    feature: str
    psi: float
    ks: float
    drifted: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    """Result of comparing a live window against the reference window."""

    features: list[FeatureDrift] = field(default_factory=list)
    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    ks_threshold: float = DEFAULT_KS_THRESHOLD

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    @property
    def has_drift(self) -> bool:
        return bool(self.drifted_features)

    @property
    def max_psi(self) -> float:
        return max((f.psi for f in self.features), default=0.0)

    def as_dict(self) -> dict:
        return {
            "has_drift": self.has_drift,
            "drifted_features": self.drifted_features,
            "max_psi": round(self.max_psi, 6),
            "psi_threshold": self.psi_threshold,
            "ks_threshold": self.ks_threshold,
            "features": [f.as_dict() for f in self.features],
        }


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index between two 1-D samples.

    Bin edges come from the reference quantiles, so the reference is uniformly
    distributed across bins by construction and PSI is 0 for identical input.
    """
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    if reference.size == 0 or current.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:  # constant reference feature — nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(reference, bins=edges)[0] / reference.size
    cur_share = np.histogram(current, bins=edges)[0] / current.size

    ref_share = np.clip(ref_share, _EPSILON, None)
    cur_share = np.clip(cur_share, _EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max gap between empirical CDFs."""
    reference = np.sort(np.asarray(reference, dtype=float).ravel())
    current = np.sort(np.asarray(current, dtype=float).ravel())
    if reference.size == 0 or current.size == 0:
        return 0.0

    grid = np.concatenate([reference, current])
    ref_cdf = np.searchsorted(reference, grid, side="right") / reference.size
    cur_cdf = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_threshold: float = DEFAULT_KS_THRESHOLD,
    feature_names: list[str] | None = None,
) -> DriftReport:
    """Compare every column of *current* against *reference*."""
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    current = np.atleast_2d(np.asarray(current, dtype=float))
    if reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"Feature count mismatch: reference has {reference.shape[1]}, "
            f"current has {current.shape[1]}"
        )

    names = feature_names or [f"feature_{i}" for i in range(reference.shape[1])]
    features = []
    for index, name in enumerate(names):
        ref_col, cur_col = reference[:, index], current[:, index]
        psi = population_stability_index(ref_col, cur_col)
        ks = ks_statistic(ref_col, cur_col)
        features.append(
            FeatureDrift(
                feature=name,
                psi=round(psi, 6),
                ks=round(ks, 6),
                drifted=bool(psi > psi_threshold or ks > ks_threshold),
            )
        )

    return DriftReport(
        features=features, psi_threshold=psi_threshold, ks_threshold=ks_threshold
    )
