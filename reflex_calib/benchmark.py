"""The four Phase 0 studies.

  detect  Reproduce slip detection on unseen objects (leave-one-object-out), with calibration.
  shift   Break the sensor on purpose (new skin, worn skin, ...) and measure what calibration,
          30 s recalibration and the trust score do about it, and what happens to the alarms.
  warn    Predict slip 50/100/200 ms ahead; trace lead time against false alarms per minute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .adaptive import TrustScore, recalibrate
from .calibration import CALIBRATORS, metrics
from .events import event_metrics, lead_fa_curve
from .features import build_features, warn_labels
from .shifts import SHIFTS, apply_shift


@dataclass
class Config:
    window: int = 10            # samples (100 ms at 100 Hz)
    subsample: int = 2          # train on every k-th sample to save time
    n_calib_objects: int = 3    # training objects held back to fit calibrators
    recal_seconds: float = 30.0
    target_fa_per_min: float = 2.0
    severities: tuple = (1.0, 2.0)
    horizons_ms: tuple = (0, 50, 100, 200)
    seed: int = 0
    models: tuple = ("gbdt", "logreg", "threshold")
    extra: dict = field(default_factory=dict)


def make_model(name: str, seed: int = 0):
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=seed)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    if name == "threshold":
        # "Simple threshold detector" baseline: one vibration-energy feature, logistic link.
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    raise ValueError(name)


def feature_cols(F: pd.DataFrame, model: str) -> list[str]:
    return ["hf_energy_log"] if model == "threshold" else list(F.columns)


def _split_calib(objects: list[str], n: int, rng) -> tuple[list, list]:
    objs = list(objects)
    rng.shuffle(objs)
    return objs[n:], objs[:n]


def fit_with_calibrators(df_tr, F_tr, y_tr, model_name, cfg: Config, rng):
    """Fit the model on some training objects and every calibrator on the rest."""
    fit_objs, cal_objs = _split_calib(sorted(df_tr["object"].unique()), cfg.n_calib_objects, rng)
    m_fit = df_tr["object"].isin(fit_objs).to_numpy()
    cols = feature_cols(F_tr, model_name)
    sub = np.zeros(len(df_tr), bool)
    sub[::cfg.subsample] = True
    model = make_model(model_name, cfg.seed).fit(F_tr.loc[m_fit & sub, cols], y_tr[m_fit & sub])
    p_cal = model.predict_proba(F_tr.loc[~m_fit, cols])[:, 1]
    cals = {k: c().fit(p_cal, y_tr[~m_fit]) for k, c in CALIBRATORS.items()}
    model.calib_objects_ = cal_objs
    return model, cols, cals


# ---------------------------------------------------------------- detect
def study_detect(df: pd.DataFrame, fs: float, cfg: Config):
    F = build_features(df, cfg.window)
    y = df["slip"].to_numpy()
    rng = np.random.default_rng(cfg.seed)
    rows, preds = [], []
    plot_model = "gbdt" if "gbdt" in cfg.models else cfg.models[0]
    for obj in sorted(df["object"].unique()):
        te = (df["object"] == obj).to_numpy()
        for mname in cfg.models:
            model, cols, cals = fit_with_calibrators(df[~te], F[~te], y[~te], mname, cfg, rng)
            p_raw = model.predict_proba(F.loc[te, cols])[:, 1]
            for cname, cal in cals.items():
                p = cal.transform(p_raw)
                r = metrics(y[te], p)
                r.update(object=obj, model=mname, calibrator=cname)
                rows.append(r)
                if mname == plot_model:
                    preds.append(pd.DataFrame({"object": obj, "calibrator": cname, "y": y[te], "p": p}))
    per_obj = pd.DataFrame(rows)
    pooled = []
    P = pd.concat(preds)
    for (mname, cname), g in per_obj.groupby(["model", "calibrator"]):
        r = dict(model=mname, calibrator=cname,
                 acc_mean=g["acc"].mean(), f1_mean=g["f1"].mean(), auroc_mean=g["auroc"].mean(),
                 ece_mean=g["ece"].mean(), ece_worst=g["ece"].max(),
                 worst_object=g.loc[g["ece"].idxmax(), "object"], brier_mean=g["brier"].mean())
        pooled.append(r)
    return pd.DataFrame(pooled), per_obj, P


# ---------------------------------------------------------------- shift
def _group_folds(objects, k, rng):
    objs = list(objects)
    rng.shuffle(objs)
    return [objs[i::k] for i in range(k)]


def _recal_split(df_te, fs, seconds):
    """First trials adding up to `seconds` go to recalibration, the rest to evaluation."""
    lens = df_te.groupby("trial", sort=False).size()
    cum = lens.cumsum() / fs
    recal_trials = set(cum.index[cum <= seconds]) or {cum.index[0]}
    return df_te["trial"].isin(recal_trials).to_numpy()


def tune_threshold(df_l, p, fs, target_fa_per_min):
    """Lowest threshold whose event-level false-alarm rate is <= target (needs labels)."""
    for thr in np.linspace(0.05, 0.99, 95):
        if event_metrics(df_l, p, thr, fs)["fa_per_min"] <= target_fa_per_min:
            return float(thr)
    return 0.99


def study_shift(df: pd.DataFrame, fs: float, cfg: Config, k_folds: int = 4):
    rng = np.random.default_rng(cfg.seed + 1)
    folds = _group_folds(sorted(df["object"].unique()), k_folds, rng)
    F_all = build_features(df, cfg.window)
    y_all = df["slip"].to_numpy()
    rows = []
    for fi, test_objs in enumerate(folds):
        te = df["object"].isin(test_objs).to_numpy()
        df_tr = df[~te]
        model, cols, cals = fit_with_calibrators(df_tr, F_all[~te], y_all[~te], "gbdt", cfg, rng)
        trust = TrustScore().fit(F_all.loc[~te, cols].to_numpy())
        # Alarm threshold: tuned on the clean, labelled calibration objects so the
        # false-alarm rate hits the target there (the usual practice).
        cm = df["object"].isin(model.calib_objects_).to_numpy()
        df_cal = df[cm].reset_index(drop=True)
        p_cal = cals["isotonic"].transform(model.predict_proba(F_all.loc[cm, cols])[:, 1])
        thr_fixed = tune_threshold(df_cal, p_cal, fs, cfg.target_fa_per_min)
        df_te0 = df[te].reset_index(drop=True)
        conds = [("none", 0.0)] + [(s, v) for s in SHIFTS for v in cfg.severities]
        for kind, sev in conds:
            d = apply_shift(df_te0, kind, sev, seed=cfg.seed + fi) if kind != "none" else df_te0
            F = build_features(d, cfg.window)[cols]
            y = d["slip"].to_numpy()
            p_raw = model.predict_proba(F)[:, 1]
            rmask = _recal_split(d, fs, cfg.recal_seconds)
            ev = ~rmask
            base = dict(fold=fi, shift=kind, severity=sev, trust_flag_rate=trust.flag_rate(F.to_numpy()[ev]))
            for cname in ("raw", "temperature", "isotonic"):
                p = cals[cname].transform(p_raw)
                r = metrics(y[ev], p[ev])
                rows.append({**base, "method": cname, **r})
            # 30 s recalibration on the shifted skin (needs labels for those 30 s).
            p_iso = cals["isotonic"].transform(p_raw)
            rc = recalibrate(p_iso[rmask], y[rmask], "platt")
            rows.append({**base, "method": "recal_30s", **metrics(y[ev], rc.transform(p_iso[ev]))})
            # Alarm behaviour under shift, with the threshold tuned on clean data.
            d_ev = d[ev].reset_index(drop=True)
            em = event_metrics(d_ev, p_iso[ev], thr_fixed, fs)
            rows.append({**base, "method": "alarm_fixed", **{f"ev_{k}": v for k, v in em.items()}})
            # TODO(Phase 0 step 5): a label-free threshold that keeps fa_per_min on target
            # under shift. Matching the alarm share on an unlabelled 30 s clip (quantile
            # transfer) was tried and failed: slip prevalence in a short clip varies too much.
    res = pd.DataFrame(rows)
    return res


def summarize_shift(res: pd.DataFrame) -> pd.DataFrame:
    cal = res[res["method"].isin(["raw", "temperature", "isotonic", "recal_30s"])]
    t1 = cal.pivot_table(index=["shift", "severity"], columns="method", values="ece", aggfunc="mean")
    t1.columns = [f"ece_{c}" for c in t1.columns]
    auc = cal[cal["method"] == "isotonic"].groupby(["shift", "severity"])["auroc"].mean().rename("auroc")
    tr = cal[cal["method"] == "isotonic"].groupby(["shift", "severity"])["trust_flag_rate"].mean()
    al = res[res["method"].str.startswith("alarm")]
    t2 = al.pivot_table(index=["shift", "severity"], columns="method",
                        values=["ev_fa_per_min", "ev_catch_rate"], aggfunc="mean")
    t2.columns = [a.replace("ev_", "") for a, _ in t2.columns]
    return pd.concat([t1, auc, tr, t2], axis=1).reset_index()


# ---------------------------------------------------------------- warn
def study_warn(df: pd.DataFrame, fs: float, cfg: Config, k_folds: int = 4):
    rng = np.random.default_rng(cfg.seed + 2)
    folds = _group_folds(sorted(df["object"].unique()), k_folds, rng)
    F = build_features(df, cfg.window)
    curves, cal_rows = [], []
    for h_ms in cfg.horizons_ms:
        h = int(round(h_ms / 1000 * fs))
        y = warn_labels(df, h) if h > 0 else df["slip"].to_numpy()
        P = np.zeros(len(df))
        for test_objs in folds:
            te = df["object"].isin(test_objs).to_numpy()
            model, cols, cals = fit_with_calibrators(df[~te], F[~te], y[~te], "gbdt", cfg, rng)
            P[te] = cals["isotonic"].transform(model.predict_proba(F.loc[te, cols])[:, 1])
        r = metrics(y, P)
        r["horizon_ms"] = h_ms
        cal_rows.append(r)
        c = lead_fa_curve(df, P, fs)
        c["horizon_ms"] = h_ms
        curves.append(c)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(cal_rows)
