"""Causal windowed features. Every feature at sample t uses only samples <= t,
so the same code can run inside a real-time reflex loop."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MAG_COLS

X_COLS = MAG_COLS[0::3]
Y_COLS = MAG_COLS[1::3]
Z_COLS = MAG_COLS[2::3]


def tare(df: pd.DataFrame, n_baseline: int = 10) -> pd.DataFrame:
    """Subtract each trial's pre-contact baseline (median of its first samples).
    This mirrors what a real controller does when it zeroes the skin before a grasp."""
    out = df.copy()
    base = df.groupby("trial", sort=False)[MAG_COLS].transform(
        lambda s: s.iloc[:n_baseline].median())
    out[MAG_COLS] = df[MAG_COLS] - base
    return out


def _trial_features(g: pd.DataFrame, w: int) -> pd.DataFrame:
    x = g[MAG_COLS]
    f = {}
    for c in MAG_COLS:
        f[f"{c}"] = x[c]
    d1 = x.diff().fillna(0.0)
    delta = (x - x.shift(w)).fillna(0.0)
    vib = d1.rolling(w, min_periods=2).std().fillna(0.0)
    d2 = d1.diff().fillna(0.0)
    hf = (d2 ** 2).rolling(w, min_periods=1).mean()
    for c in MAG_COLS:
        f[f"{c}_delta"] = delta[c]
        f[f"{c}_vib"] = vib[c]
    shear = np.sqrt((x[X_COLS].to_numpy() ** 2).sum(1) + (x[Y_COLS].to_numpy() ** 2).sum(1))
    normal = np.abs(x[Z_COLS].to_numpy()).sum(1)
    f["shear_mag"] = shear
    f["normal_mag"] = normal
    f["shear_ratio"] = shear / (normal + 1.0)
    f["shear_ratio_delta"] = pd.Series(f["shear_ratio"], index=g.index).diff(w).fillna(0.0).to_numpy()
    f["hf_energy"] = hf.sum(axis=1)
    f["hf_energy_log"] = np.log1p(f["hf_energy"])
    f["vib_total"] = vib.sum(axis=1)
    return pd.DataFrame(f, index=g.index)


def build_features(df: pd.DataFrame, window: int = 10, tared: bool = False) -> pd.DataFrame:
    """Return a feature frame aligned with df's index."""
    d = df if tared else tare(df)
    parts = [_trial_features(g, window) for _, g in d.groupby("trial", sort=False)]
    return pd.concat(parts).loc[df.index]


def warn_labels(df: pd.DataFrame, horizon_samples: int) -> np.ndarray:
    """1 if slip starts within the next `horizon_samples` (or is already happening)."""
    out = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("trial", sort=False):
        s = g["slip"].to_numpy()
        fut = pd.Series(s[::-1]).rolling(horizon_samples + 1, min_periods=1).max().to_numpy()[::-1]
        out[df.index.get_indexer(g.index)] = fut.astype(int)
    return out
