"""Event-level evaluation: what the reflex actually experiences.

Sample-level accuracy hides what matters for a gripper: how early the alarm fires
before a slip, and how often it fires when nothing is slipping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def alarm_episodes(alarm: np.ndarray, trial_ids: np.ndarray) -> int:
    """Count rising edges of a boolean alarm signal, never joining across trials."""
    alarm = np.asarray(alarm, bool)
    trial_ids = np.asarray(trial_ids)
    prev = np.r_[False, alarm[:-1]]
    new_trial = np.r_[True, trial_ids[1:] != trial_ids[:-1]]
    prev[new_trial] = False
    return int((alarm & ~prev).sum())


def onset_index(slip: np.ndarray):
    idx = np.flatnonzero(slip)
    return int(idx[0]) if len(idx) else None


def event_metrics(df: pd.DataFrame, p: np.ndarray, thr: float, fs: float,
                  max_lead_s: float = 1.0, max_late_s: float = 0.3) -> dict:
    """For each trial: did the alarm catch the slip, and with what lead time?

    An alarm counts as a catch if it first fires within [onset - max_lead_s, onset + max_late_s].
    Alarms on samples outside that window and before onset (or in no-slip trials) are false alarms.
    Lead time > 0 means the alarm fired before slip onset (early warning);
    < 0 means after (detection latency)."""
    leads, n_slip_trials, caught = [], 0, 0
    fa, quiet_samples = 0, 0
    pos = 0
    p = np.asarray(p)
    for _, g in df.groupby("trial", sort=False):
        n = len(g)
        pg = p[pos:pos + n]
        pos += n
        s = g["slip"].to_numpy()
        a = pg >= thr
        on = onset_index(s)
        if on is None:
            quiet = np.ones(n, bool)
        else:
            n_slip_trials += 1
            lo = max(0, on - int(max_lead_s * fs))
            hi = min(n, on + int(max_late_s * fs) + 1)
            hits = np.flatnonzero(a[lo:hi])
            if len(hits):
                caught += 1
                leads.append((on - (lo + hits[0])) / fs)
            quiet = np.zeros(n, bool)
            quiet[:lo] = True
        fa += alarm_episodes(a & quiet, np.zeros(n))
        quiet_samples += quiet.sum()
    minutes = quiet_samples / fs / 60.0
    leads = np.array(leads)
    return dict(
        threshold=float(thr),
        catch_rate=caught / max(n_slip_trials, 1),
        median_lead_ms=float(np.median(leads) * 1000) if len(leads) else float("nan"),
        p10_lead_ms=float(np.percentile(leads, 10) * 1000) if len(leads) else float("nan"),
        fa_per_min=fa / max(minutes, 1e-9),
        n_slip_trials=n_slip_trials,
    )


def lead_fa_curve(df, p, fs, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    return pd.DataFrame([event_metrics(df, p, t, fs) for t in thresholds])
