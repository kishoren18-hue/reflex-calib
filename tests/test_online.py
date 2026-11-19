import numpy as np

from reflex_calib.data import MAG_COLS
from reflex_calib.features import build_features
from reflex_calib.online import SlipPredictor, StreamingFeatures, feature_names
from reflex_calib.synth import generate


def test_streaming_matches_batch():
    df = generate(n_trials_per_object=1, seed=3).reset_index(drop=True)
    F = build_features(df, window=10)
    assert list(F.columns) == feature_names()
    for _, g in list(df.groupby("trial", sort=False))[:3]:
        raw = g[MAG_COLS].to_numpy()
        s = StreamingFeatures(10)
        s.reset(np.median(raw[:10], axis=0))
        got = np.array([s.step(r) for r in raw])
        np.testing.assert_allclose(got, F.loc[g.index].to_numpy(), rtol=1e-6, atol=1e-6)


def test_predictor_roundtrip(tmp_path):
    df = generate(n_trials_per_object=3, seed=4).reset_index(drop=True)
    pred = SlipPredictor.train(df)
    path = tmp_path / "m.pkl"
    pred.save(path)
    pred2 = SlipPredictor.load(path)
    g = df[df["trial"] == df["trial"].iloc[0]]
    raw = g[MAG_COLS].to_numpy()
    s = pred2.stream()
    s.reset(np.median(raw[:10], axis=0))
    outs = [s.step(r) for r in raw]
    p_stream = np.array([o["p_slip"] for o in outs])
    p_batch = pred.predict_proba(build_features(g.reset_index(drop=True)))
    np.testing.assert_allclose(p_stream, p_batch, atol=1e-6)
    assert all(0 <= o["trust"] <= 1 for o in outs)
    assert 0 < pred.threshold < 1
