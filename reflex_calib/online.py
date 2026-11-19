"""Real-time slip prediction: the bridge from this benchmark to the Reflex controller.

    pred = SlipPredictor.train(df)          # or SlipPredictor.load("slip_model.pkl")
    pred.save("slip_model.pkl")

    s = pred.stream()                       # one per grasp
    s.reset(baseline)                       # (15,) median of pre-contact samples (tare)
    for sample in sensor:                   # (15,) raw mag1_x .. mag5_z
        out = s.step(sample)                # {"p_slip", "trust", "alarm"}

StreamingFeatures reproduces features.build_features exactly, one sample at a time,
given the same baseline (tests/test_online.py checks this). Pure numpy, no pandas,
so it fits inside a 100 Hz loop.
"""
from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive import TrustScore
from .benchmark import make_model
from .calibration import CALIBRATORS
from .data import MAG_COLS, sample_rate
from .events import event_metrics
from .features import X_COLS, Y_COLS, Z_COLS, build_features

FORMAT_VERSION = 1
_XI = [MAG_COLS.index(c) for c in X_COLS]
_YI = [MAG_COLS.index(c) for c in Y_COLS]
_ZI = [MAG_COLS.index(c) for c in Z_COLS]


def feature_names() -> list[str]:
    names = list(MAG_COLS)
    for c in MAG_COLS:
        names += [f"{c}_delta", f"{c}_vib"]
    return names + ["shear_mag", "normal_mag", "shear_ratio", "shear_ratio_delta",
                    "hf_energy", "hf_energy_log", "vib_total"]


class StreamingFeatures:
    """Causal per-sample features, identical to build_features(tare(df))."""

    def __init__(self, window: int = 10):
        self.w = window
        self.reset(np.zeros(len(MAG_COLS)))

    def reset(self, baseline: np.ndarray):
        self.base = np.asarray(baseline, float)
        self.X = deque(maxlen=self.w + 1)
        self.D1 = deque(maxlen=self.w)
        self.D2SQ = deque(maxlen=self.w)
        self.R = deque(maxlen=self.w + 1)
        self.prev_d1 = None

    def step(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, float) - self.base
        d1 = x - self.X[-1] if self.X else np.zeros_like(x)
        d2 = d1 - self.prev_d1 if self.prev_d1 is not None else np.zeros_like(x)
        self.prev_d1 = d1
        self.X.append(x)
        self.D1.append(d1)
        self.D2SQ.append(d2 ** 2)
        w = self.w
        delta = self.X[-1] - self.X[0] if len(self.X) == w + 1 else np.zeros_like(x)
        vib = np.std(np.array(self.D1), axis=0, ddof=1) if len(self.D1) >= 2 else np.zeros_like(x)
        hf = float(np.mean(np.array(self.D2SQ), axis=0).sum())
        shear = float(np.sqrt((x[_XI] ** 2).sum() + (x[_YI] ** 2).sum()))
        normal = float(np.abs(x[_ZI]).sum())
        ratio = shear / (normal + 1.0)
        self.R.append(ratio)
        r_delta = self.R[-1] - self.R[0] if len(self.R) == w + 1 else 0.0
        inter = np.empty(2 * len(x))
        inter[0::2], inter[1::2] = delta, vib
        return np.concatenate([x, inter, [shear, normal, ratio, r_delta, hf, np.log1p(hf),
                                          float(vib.sum())]])


@dataclass
class SlipPredictor:
    model: object
    calibrator: object
    trust: TrustScore
    cols: list
    window: int = 10
    threshold: float = 0.5
    fs: float = 100.0
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, df: pd.DataFrame, model_name: str = "gbdt", calibrator: str = "temperature",
              window: int = 10, n_calib_objects: int = 3, target_fa_per_min: float = 2.0,
              horizon_ms: float = 0.0, seed: int = 0) -> "SlipPredictor":
        """Fit model on most objects; fit calibrator and alarm threshold on held-out objects.
        horizon_ms > 0 trains an early-warning model (label = slip within the horizon)."""
        from .features import warn_labels

        df = df.reset_index(drop=True)
        fs = sample_rate(df)
        F = build_features(df, window)
        cols = feature_names() if model_name != "threshold" else ["hf_energy_log"]
        h = int(round(horizon_ms * fs / 1000))
        y = warn_labels(df, h) if h > 0 else df["slip"].to_numpy()
        objs = sorted(df["object"].unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(objs)
        cal_objs = objs[:n_calib_objects]
        m_cal = df["object"].isin(cal_objs).to_numpy()
        model = make_model(model_name, seed).fit(F.loc[~m_cal, cols], y[~m_cal])
        p_cal = model.predict_proba(F.loc[m_cal, cols])[:, 1]
        cal = CALIBRATORS[calibrator]().fit(p_cal, y[m_cal])
        p_cal = cal.transform(p_cal)
        df_cal = df.loc[m_cal].reset_index(drop=True)
        thr, best_fa, met = 0.95, np.inf, False
        for t in np.linspace(0.05, 0.95, 37):  # lowest threshold that meets the FA budget
            fa = event_metrics(df_cal, p_cal, t, fs)["fa_per_min"]
            if fa <= target_fa_per_min:
                thr, best_fa, met = float(t), fa, True
                break
            if fa < best_fa:
                thr, best_fa = float(t), fa
        trust = TrustScore().fit(F.loc[~m_cal, cols].to_numpy(), seed=seed)
        meta = dict(format=FORMAT_VERSION, model=model_name, calibrator=calibrator,
                    horizon_ms=horizon_ms, calib_objects=list(cal_objs),
                    train_objects=[o for o in objs if o not in cal_objs],
                    target_fa_per_min=target_fa_per_min,
                    fa_budget_met=met, calib_fa_per_min=float(best_fa))
        return cls(model, cal, trust, cols, window, thr, fs, meta)

    # ---------------------------------------------------------------- inference
    def predict_proba(self, F: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(F, np.ndarray):
            F = pd.DataFrame(F, columns=feature_names())
        return self.calibrator.transform(self.model.predict_proba(F[self.cols])[:, 1])

    def stream(self) -> "StreamingPredictor":
        return StreamingPredictor(self)

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "SlipPredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"model format {obj.meta.get('format')} != {FORMAT_VERSION}")
        return obj


class StreamingPredictor:
    def __init__(self, pred: SlipPredictor):
        self.pred = pred
        self.feat = StreamingFeatures(pred.window)
        names = feature_names()
        self._idx = [names.index(c) for c in pred.cols]

    def reset(self, baseline: np.ndarray):
        self.feat.reset(baseline)

    def step(self, raw: np.ndarray) -> dict:
        f = self.feat.step(raw)
        row = pd.DataFrame([f[self._idx]], columns=self.pred.cols)
        p = float(self.pred.calibrator.transform(self.pred.model.predict_proba(row)[:, 1])[0])
        trust = float(self.pred.trust.percentile(f[self._idx][None, :])[0])
        return {"p_slip": p, "trust": trust, "alarm": p >= self.pred.threshold}
