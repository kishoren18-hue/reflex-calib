# reflex-calib

Can a robot gripper tell when an object is about to slip, and is its confidence still honest after the sensor changes?

This repo is an open benchmark for **calibrated slip prediction from magnetic tactile skins** (AnySkin). Slip detection itself is well studied: AnySkin reports about 92% accuracy on unseen objects. What hasn't been measured is whether the predicted *probability* of slip stays trustworthy when the skin is swapped, reseated, worn down or partly broken. A controller that acts on those probabilities needs that to hold.

## Quick start

```bash
pip install -r requirements.txt
python run_benchmark.py --download        # fetch pollen-robotics/anyskin_slip_detection, run all studies
python run_benchmark.py --synthetic       # no download: synthetic data, checks the pipeline (~2 min)
python -m pytest -q
```

Results go to `results/<source>/`: CSV tables, `reliability.png`, `shift_ece.png`, `warn_curve.png` and `report.json`.

**After the first real download, check the printed line** (`objects`, `trials`, `fs_hz`). Each CSV is split into trials wherever there's a large time gap. If the trial count or sample rate looks wrong, change `gap_factor` in `reflex_calib/data.py`.

## Use it in a controller

```bash
python export_model.py --download --out models/slip_model.pkl        # detection model
python export_model.py --download --horizon-ms 50 --out models/warn50.pkl  # early warning
```

```python
from reflex_calib.online import SlipPredictor
s = SlipPredictor.load("models/slip_model.pkl").stream()
s.reset(baseline)            # median of ~10 pre-contact samples (tare)
out = s.step(sample)         # sample: 15 raw values mag1_x..mag5_z -> {"p_slip", "trust", "alarm"}
```

The alarm threshold is tuned on held-out objects to a false-alarm budget (default 2/min). `meta["fa_budget_met"]` says whether it got there. Streaming features match the batch pipeline exactly (`tests/test_online.py`). About 4 ms per step on a laptop, inside a 100 Hz budget.

## Data

[`pollen-robotics/anyskin_slip_detection`](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection) (Apache-2.0) was recorded on a Reachy 2 gripper with one AnySkin (5 magnetometers × 3 axes), across 16 objects. The operator pulled each object up, down, left or right, and labelled slip by hand. Any folder of CSVs with the columns `log_time, mag1_x … mag5_z, slip` works with `--data`.

`reflex_calib/synth.py` generates data in the same format. It's a simple physics-flavoured model (friction cone, incipient-slip vibration, per-skin gain and offset). **Numbers from synthetic data are not results.** It exists so the code can be built and tested before real data and hardware are in hand.

## Studies

| Study | Question | Protocol |
|---|---|---|
| `detect` | How well do we detect slip on objects never seen in training, and how calibrated is P(slip)? | Leave one object out. GBDT vs logistic regression vs a single-feature vibration-threshold baseline. Calibrators: raw, temperature, Platt, isotonic, fit on 3 held-back training objects. |
| `shift` | What breaks calibration? | Train on clean data. Test on held-out objects with simulated sensor shifts: `gain` (new skin), `offset` (reseated), `rotation`, `drift`, `noise`, `dead_mag`, `wear`. We report ECE, AUROC, the trust-score flag rate, the effect of recalibrating on 30 s of labelled new-skin data, and alarm catch rate and false alarms per minute at a threshold tuned on clean data. |
| `warn` | How early can we warn? | Train to predict slip within 0/50/100/200 ms. Sweep the alarm threshold and plot median lead time against false alarms per minute of contact. |

Metrics: ECE (15 equal-width bins, plus an adaptive-bin version), Brier, NLL and AUROC at the sample level. At the event level we report catch rate, lead time (positive means the alarm fired before slip onset) and false alarms per minute.

All features are causal: each one uses only past samples, so the same code can run in a real-time loop. A test checks this.

## Code map

```
reflex_calib/
  data.py         load / download AnySkin CSVs, split trials, estimate sample rate
  synth.py        synthetic AnySkin-format data
  features.py     taring + causal window features (deltas, vibration, high-frequency energy, shear ratio)
  calibration.py  ECE / Brier / NLL, temperature, Platt and isotonic calibrators
  shifts.py       the seven sensor shifts
  adaptive.py     TrustScore (Mahalanobis percentile vs training) and few-shot recalibration
  events.py       event-level alarm metrics
  benchmark.py    the three studies
  online.py       streaming features + exportable SlipPredictor (used by the reflex controller)
run_benchmark.py  CLI
export_model.py   train and save a deployable model
```

## Phase 0 checklist

- [x] Pipeline: loader, features, models, calibrators, shifts, event metrics, plots, tests
- [x] Real-time predictor: streaming features, exportable model, false-alarm-tuned threshold
- [ ] **Reproduce** on real AnySkin data: does the GBDT reach about 92% on unseen objects?
- [ ] **Calibration study**: which objects is the model overconfident on, and why?
- [ ] **Shift study** on real data: which shifts break calibration, and does 30 s recalibration fix it?
- [ ] **Early warning** on real data: the lead time vs false-alarm curve
- [ ] **Own method**: a label-free threshold that holds the false-alarm rate under shift, and a trust score that catches wear (see the notes below)
- [ ] **Second dataset** (Waseda uSkin, Bielefeld 1 kHz slip, or Bristol TacTip) to check the conclusions aren't sensor-specific
- [ ] 4-page write-up

## Notes from the synthetic pipeline check (hypotheses to test on real data)

- **Wear is the dangerous shift.** It roughly triples ECE and drops alarm catch rate from about 86% to under 10%, yet the trust score flags almost none of it. A worn skin gives weaker, smoother signals, which look like an ordinary light grasp. Recalibrating on 30 s helps, but only partly. If this holds on real data, it's the central finding.
- **Offset doesn't matter once you tare** before each grasp. That's a useful sanity check.
- **Electrical noise mostly causes false alarms** (about 2 → 21 per minute) rather than misses.
- **Gain and rotation shifts are mild**, and 30 s of recalibration can make them slightly *worse* because the clip is so small. Fewer recalibration parameters (temperature only) might be the better default.
- **A failed idea:** setting the alarm threshold on a new skin by matching the share of samples that fire, using an unlabelled 30 s clip. Slip prevalence in a short clip varies too much, and false alarms went from 2 to about 50 per minute. A label-free threshold needs a better anchor. Candidates: quiet contact detected from low shear ratio, or matching the score distribution on no-contact samples only.

## Published numbers to compare against

- AnySkin: 92% slip accuracy on unseen objects (LSTM, 100 Hz)
- PapillArray learning-based incipient slip: 95.6% (offline)
- SlipSense: 96.7% macro F1, false-positive rate under 1.6%, 95/100 closed-loop catches, about 50 ms from detection to peak force
- FORTE: 0.91 F1 slip detection

Check these against the original papers before citing them.

## License

MIT
