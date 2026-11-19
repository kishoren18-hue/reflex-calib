#!/usr/bin/env python3
"""Train a deployable slip predictor and save it for the Reflex controller.

  python export_model.py --download                     # real AnySkin data
  python export_model.py --data data/anyskin --horizon-ms 50
  python export_model.py --synthetic --out models/synthetic.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflex_calib.data import download_anyskin, load_dir
from reflex_calib.online import SlipPredictor
from reflex_calib.synth import generate


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data")
    src.add_argument("--download", action="store_true")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "logreg", "threshold"])
    ap.add_argument("--calibrator", default="temperature")
    ap.add_argument("--horizon-ms", type=float, default=0.0, help=">0 trains an early-warning model")
    ap.add_argument("--fa-per-min", type=float, default=2.0, help="false-alarm budget for the threshold")
    ap.add_argument("--out", default="models/slip_model.pkl")
    a = ap.parse_args()

    if a.download:
        df = load_dir(download_anyskin("data/anyskin"))
    elif a.data:
        df = load_dir(a.data)
    else:
        df = generate(n_trials_per_object=6)
    pred = SlipPredictor.train(df, a.model, a.calibrator, horizon_ms=a.horizon_ms,
                               target_fa_per_min=a.fa_per_min)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pred.save(a.out)
    print(json.dumps({**pred.meta, "threshold": pred.threshold, "fs": pred.fs, "out": a.out}))


if __name__ == "__main__":
    main()
