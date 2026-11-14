#!/usr/bin/env python3
"""Phase 0 benchmark for Reflex: slip detection, calibration under shift, early warning.

  python run_benchmark.py --download            # fetch the AnySkin dataset, run everything
  python run_benchmark.py --data data/anyskin   # run on CSVs you already have
  python run_benchmark.py --synthetic           # synthetic data (pipeline check only)
  python run_benchmark.py --synthetic --quick   # fast smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from reflex_calib import benchmark as B
from reflex_calib import plots
from reflex_calib.data import download_anyskin, load_dir, sample_rate
from reflex_calib.synth import generate


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data", type=str, help="folder of AnySkin-format CSVs")
    src.add_argument("--download", action="store_true", help="download the AnySkin dataset first")
    src.add_argument("--synthetic", action="store_true", help="use synthetic data")
    ap.add_argument("--studies", default="detect,shift,warn")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="smaller run for smoke testing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.download:
        df = load_dir(download_anyskin("data/anyskin"))
        source = "anyskin"
    elif args.data:
        df = load_dir(args.data)
        source = Path(args.data).name
    else:
        df = generate(n_trials_per_object=4 if args.quick else 12, seed=args.seed)
        source = "synthetic"
    df = df.reset_index(drop=True)
    fs = sample_rate(df)
    cfg = B.Config(seed=args.seed)
    if args.quick:
        cfg.models = ("gbdt", "threshold")
        cfg.severities = (2.0,)
        cfg.horizons_ms = (0, 100)
        cfg.recal_seconds = 10.0
    out = Path(args.out) / source
    out.mkdir(parents=True, exist_ok=True)
    info = dict(source=source, rows=len(df), objects=df["object"].nunique(),
                trials=df["trial"].nunique(), slip_rate=float(df["slip"].mean()), fs_hz=round(fs, 1))
    print(json.dumps(info))
    if source == "synthetic":
        print("NOTE: synthetic data. These numbers check the pipeline; they are not results.")
    studies = args.studies.split(",")
    report = {"data": info}

    if "detect" in studies:
        t = time.time()
        summ, per_obj, P = B.study_detect(df, fs, cfg)
        summ.to_csv(out / "detect_summary.csv", index=False)
        per_obj.to_csv(out / "detect_per_object.csv", index=False)
        plots.reliability_plot(P, out / "reliability.png")
        print(f"\n[detect] leave-one-object-out ({time.time() - t:.0f}s)")
        print(summ.round(3).to_string(index=False))
        report["detect"] = summ.round(4).to_dict("records")

    if "shift" in studies:
        t = time.time()
        res = B.study_shift(df, fs, cfg)
        res.to_csv(out / "shift_raw.csv", index=False)
        summ = B.summarize_shift(res)
        summ.to_csv(out / "shift_summary.csv", index=False)
        plots.shift_plot(summ, out / "shift_ece.png")
        print(f"\n[shift] grouped 4-fold by object ({time.time() - t:.0f}s)")
        print(summ.round(3).to_string(index=False))
        report["shift"] = summ.round(4).to_dict("records")

    if "warn" in studies:
        t = time.time()
        curves, cal = B.study_warn(df, fs, cfg)
        curves.to_csv(out / "warn_curves.csv", index=False)
        cal.to_csv(out / "warn_calibration.csv", index=False)
        plots.warn_plot(curves, out / "warn_curve.png")
        print(f"\n[warn] ({time.time() - t:.0f}s)")
        print(cal[["horizon_ms", "auroc", "ece", "brier"]].round(3).to_string(index=False))
        at = curves[(curves["fa_per_min"] <= cfg.target_fa_per_min)]
        if len(at):
            best = at.sort_values("median_lead_ms", ascending=False).groupby("horizon_ms").head(1)
            print(f"\nBest lead time at <= {cfg.target_fa_per_min} false alarms/min:")
            print(best[["horizon_ms", "threshold", "catch_rate", "median_lead_ms", "fa_per_min"]]
                  .round(3).to_string(index=False))
        report["warn"] = cal.round(4).to_dict("records")

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nWrote results to {out}/")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
