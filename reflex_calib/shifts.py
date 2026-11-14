"""Sensor shifts applied to raw magnetometer signals ("ShiftBench for touch").

Each shift stands in for something that really happens to a magnetic skin:
  gain      a new skin instance: magnet strength differs channel to channel
  offset    a reseated skin: constant field offset (taring should remove this)
  rotation  a skin seated a few degrees rotated: x/y axes mix
  drift     temperature drift during a grasp: slow ramp per channel
  noise     electrical noise (motor nearby, loose cable)
  dead_mag  one magnetometer fails and reads a constant
  wear      worn elastomer: weaker response, damped vibration, more noise
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .data import MAG_COLS

SHIFTS = ["gain", "offset", "rotation", "drift", "noise", "dead_mag", "wear"]


def apply_shift(df: pd.DataFrame, kind: str, severity: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    X = out[MAG_COLS].to_numpy(dtype=float).copy()
    nch = len(MAG_COLS)
    scale = np.std(X, axis=0) + 1e-6
    if kind == "none":
        pass
    elif kind == "gain":
        X *= 1.0 + rng.normal(0, 0.15 * severity, nch)
    elif kind == "offset":
        X += rng.normal(0, 0.5 * severity, nch) * scale
    elif kind == "rotation":
        a = np.deg2rad(10.0 * severity)
        c, s = np.cos(a), np.sin(a)
        for m in range(nch // 3):
            xi, yi = 3 * m, 3 * m + 1
            x, y = X[:, xi].copy(), X[:, yi].copy()
            X[:, xi], X[:, yi] = c * x - s * y, s * x + c * y
    elif kind == "drift":
        slope = rng.normal(0, 0.15 * severity, nch) * scale  # per second
        t = out.groupby("trial", sort=False)["t"].transform(lambda s: s - s.iloc[0]).to_numpy()
        X += t[:, None] * slope[None, :]
    elif kind == "noise":
        X += rng.normal(0, 0.05 * severity, X.shape) * scale
    elif kind == "dead_mag":
        m = rng.integers(nch // 3)
        X[:, 3 * m:3 * m + 3] = np.median(X[:, 3 * m:3 * m + 3], axis=0)
    elif kind == "wear":
        X *= 1.0 - min(0.5, 0.15 * severity)  # weaker magnetic response
        alpha = min(0.6, 0.25 * severity)      # damped vibration: one-pole low-pass per trial
        for idx in out.groupby("trial", sort=False).indices.values():
            X[idx] = _lowpass(X[idx], alpha)
        X += rng.normal(0, 0.03 * severity, X.shape) * scale
    else:
        raise ValueError(f"unknown shift {kind}")
    out[MAG_COLS] = X
    return out


def _lowpass(seg: np.ndarray, alpha: float) -> np.ndarray:
    # y[i] = alpha*y[i-1] + (1-alpha)*x[i], starting at y[0] = x[0]
    zi = (alpha * seg[0])[None, :]
    y, _ = lfilter([1 - alpha], [1, -alpha], seg, axis=0, zi=zi)
    return y
