"""The parts that are ours: staying honest after the sensor changes.

1. TrustScore: "do I trust myself right now?" Distance of the current feature
   vector from what the model saw in training, turned into a percentile.
2. recalibrate: refit a cheap calibrator on a short labelled clip from the new skin
   (the "30 seconds of new-skin data" experiment).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from .calibration import CALIBRATORS


class TrustScore:
    def fit(self, F: np.ndarray, max_rows: int = 20000, seed: int = 0):
        rng = np.random.default_rng(seed)
        if len(F) > max_rows:
            F = F[rng.choice(len(F), max_rows, replace=False)]
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-9
        Z = (F - self.mu) / self.sd
        self.cov = LedoitWolf().fit(Z)
        d = self._dist(F)
        self.ref = np.sort(d)
        return self

    def _dist(self, F):
        Z = (F - self.mu) / self.sd
        return np.sqrt(np.maximum(self.cov.mahalanobis(Z), 0))

    def percentile(self, F) -> np.ndarray:
        """0..1: share of training samples that were closer than this one.
        Values near 1 mean 'this looks unlike anything I trained on'."""
        return np.searchsorted(self.ref, self._dist(F)) / len(self.ref)

    def flag_rate(self, F, level: float = 0.99) -> float:
        return float((self.percentile(F) > level).mean())


def recalibrate(p: np.ndarray, y: np.ndarray, method: str = "platt"):
    if len(np.unique(y)) < 2:
        return CALIBRATORS["raw"]()
    return CALIBRATORS[method]().fit(p, y)
