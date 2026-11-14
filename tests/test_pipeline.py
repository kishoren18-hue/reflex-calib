import numpy as np
import pandas as pd

from reflex_calib import benchmark as B
from reflex_calib.calibration import CALIBRATORS, ece
from reflex_calib.data import MAG_COLS, load_dir, sample_rate
from reflex_calib.events import alarm_episodes, event_metrics
from reflex_calib.features import build_features, warn_labels
from reflex_calib.shifts import SHIFTS, apply_shift
from reflex_calib.synth import generate, write_csvs


def small():
    return generate(n_trials_per_object=2, seed=1).reset_index(drop=True)


def test_csv_roundtrip(tmp_path):
    df = small()
    write_csvs(df, tmp_path)
    back = load_dir(tmp_path)
    assert set(back.columns) >= set(MAG_COLS + ["slip", "object", "trial"])
    assert back["object"].nunique() == df["object"].nunique()
    assert back["trial"].nunique() == df["trial"].nunique()  # gaps re-split trials
    assert abs(sample_rate(back) - 100.0) < 1.0
    assert back["slip"].sum() == df["slip"].sum()


def test_features_are_causal():
    df = small()
    F1 = build_features(df)
    df2 = df.copy()
    tr = df2["trial"] == df2["trial"].iloc[0]
    idx = df2.index[tr]
    cut = idx[len(idx) // 2]
    df2.loc[idx[idx > cut], MAG_COLS] += 1000.0  # change the future only
    F2 = build_features(df2)
    pd.testing.assert_frame_equal(F1.loc[idx[idx <= cut]], F2.loc[idx[idx <= cut]])


def test_warn_labels_lead_onset():
    df = pd.DataFrame({"trial": ["a"] * 10, "slip": [0] * 6 + [1] * 4})
    y = warn_labels(df, 2)
    assert list(y) == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p * 0.3 + 0.7, 0, 1)) > 0.2


def test_calibrators_fix_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    over = 1 / (1 + np.exp(-3 * np.log(p / (1 - p) + 1e-9)))
    for name in ("temperature", "platt", "isotonic"):
        c = CALIBRATORS[name]().fit(over[:10000], y[:10000])
        assert ece(y[10000:], c.transform(over[10000:])) < ece(y[10000:], over[10000:])


def test_shifts_run_and_offset_is_tared_away():
    df = small()
    F0 = build_features(df)
    for s in SHIFTS:
        d = apply_shift(df, s, 1.0)
        assert d[MAG_COLS].notna().all().all()
    Fo = build_features(apply_shift(df, "offset", 1.0))
    np.testing.assert_allclose(Fo.to_numpy(), F0.to_numpy(), atol=1e-6)


def test_alarm_episodes_do_not_join_trials():
    a = np.array([1, 1, 1, 1], bool)
    assert alarm_episodes(a, np.array(["x", "x", "y", "y"])) == 2


def test_event_metrics_lead_time():
    df = pd.DataFrame({"trial": ["a"] * 100, "slip": [0] * 60 + [1] * 40})
    p = np.zeros(100)
    p[50:] = 1.0  # fires 10 samples (100 ms) before onset
    m = event_metrics(df, p, 0.5, fs=100)
    assert m["catch_rate"] == 1.0 and abs(m["median_lead_ms"] - 100) < 1e-6
    assert m["fa_per_min"] == 0


def test_detect_study_smoke():
    df = generate(n_trials_per_object=2, seed=2).reset_index(drop=True)
    cfg = B.Config(models=("threshold",))
    summ, per_obj, _ = B.study_detect(df, 100.0, cfg)
    assert len(per_obj) == 16 * 4 and summ["auroc_mean"].notna().all()
