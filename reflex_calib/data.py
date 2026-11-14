"""Loading AnySkin slip data.

Real data: the public pollen-robotics/anyskin_slip_detection dataset on Hugging Face
(Reachy 2 gripper, one AnySkin with 5 magnetometers, 16 objects, operator-labelled slip).
Each CSV holds one object, with columns:

    log_time, mag1_x, mag1_y, mag1_z, ..., mag5_x, mag5_y, mag5_z, slip

We split each file into trials wherever the time gap between samples is large
(the gripper was opened and the sensor reset between pulls).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_MAGS = 5
MAG_COLS = [f"mag{i}_{a}" for i in range(1, N_MAGS + 1) for a in "xyz"]
HF_REPO = "pollen-robotics/anyskin_slip_detection"


def download_anyskin(dest: str | Path = "data/anyskin") -> Path:
    """Download the dataset with huggingface_hub (needs internet access to huggingface.co)."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=dest,
                      allow_patterns=["*.csv", "data/*.csv"])
    return dest


def _parse_time(col: pd.Series) -> np.ndarray:
    """Return time in seconds from whatever format log_time uses."""
    if np.issubdtype(col.dtype, np.number):
        t = col.to_numpy(dtype=float)
        span = t.max() - t.min()
        # Heuristic unit detection: ns, us, ms, or s.
        for scale in (1e9, 1e6, 1e3):
            if span > scale * 5 and np.median(np.diff(t)) > scale * 1e-4:
                return t / scale
        return t
    ts = pd.to_datetime(col, errors="coerce")
    if ts.isna().mean() > 0.5:
        raise ValueError("Could not parse log_time column")
    return (ts - ts.iloc[0]).dt.total_seconds().to_numpy()


def load_csv(path: str | Path, object_name: str | None = None,
             gap_factor: float = 20.0) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in MAG_COLS + ["slip"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if "log_time" in df.columns:
        t = _parse_time(df["log_time"])
    else:
        t = np.arange(len(df)) / 100.0
    order = np.argsort(t, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    t = t[order]
    t = t - t[0]
    dt = np.diff(t, prepend=t[0])
    med = np.median(dt[dt > 0]) if (dt > 0).any() else 0.01
    trial = np.cumsum(dt > gap_factor * med)
    out = pd.DataFrame({"t": t})
    out[MAG_COLS] = df[MAG_COLS].astype(float).to_numpy()
    out["slip"] = df["slip"].astype(str).str.lower().isin(["1", "true", "1.0"]).astype(int)
    out["object"] = object_name or path.stem
    out["trial"] = [f"{out['object'].iloc[0]}_{k}" for k in trial]
    out["skin"] = df["skin"].astype(str) if "skin" in df.columns else "skin0"
    return out


def load_dir(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {root}")
    frames = [load_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def sample_rate(df: pd.DataFrame) -> float:
    rates = []
    for _, g in df.groupby("trial"):
        d = np.diff(g["t"].to_numpy())
        d = d[d > 0]
        if len(d):
            rates.append(1.0 / np.median(d))
    return float(np.median(rates)) if rates else 100.0
