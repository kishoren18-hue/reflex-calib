"""Calibration metrics and post-hoc calibrators."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece(y, p, n_bins: int = 15, adaptive: bool = False) -> float:
    """Expected calibration error for a binary event (confidence = P(slip))."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y, p, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        conf.append(p[m].mean() if m.any() else np.nan)
        acc.append(y[m].mean() if m.any() else np.nan)
        cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def nll(y, p) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = dict(
        n=int(len(y)), pos_rate=float(y.mean()),
        acc=float(((p >= threshold) == y).mean()),
        f1=float(f1_score(y, p >= threshold, zero_division=0)),
        brier=float(brier_score_loss(y, p)), nll=nll(y, p),
        ece=ece(y, p), ece_adaptive=ece(y, p, adaptive=True),
    )
    out["auroc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    return out


class Identity:
    name = "raw"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p)


class Temperature:
    """Single-parameter temperature scaling on the logit."""
    name = "temperature"

    def fit(self, p, y):
        z = logit(p)
        res = minimize_scalar(lambda lt: nll(y, sigmoid(z / np.exp(lt))),
                              bounds=(-3, 3), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def transform(self, p):
        return sigmoid(logit(p) / self.T)


class Platt:
    """Two-parameter logistic recalibration (a*logit + b). Can fix bias, not just sharpness."""
    name = "platt"

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=100.0).fit(logit(p)[:, None], y)
        return self

    def transform(self, p):
        return self.lr.predict_proba(logit(p)[:, None])[:, 1]


class Isotonic:
    name = "isotonic"

    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def transform(self, p):
        return self.iso.predict(p)


CALIBRATORS = {c.name: c for c in (Identity, Temperature, Platt, Isotonic)}
