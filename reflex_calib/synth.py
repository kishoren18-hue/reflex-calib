"""Synthetic AnySkin-like slip data, in the same format as the real dataset.

This exists so the whole pipeline can be built and tested without the real data.
Numbers produced on synthetic data are NOT findings. Use them to check that code
runs and that each analysis behaves sensibly, then rerun everything on real data.

Model (deliberately simple):
  * A grasp produces normal force N(t) and a pull produces shear S(t) in one of four
    directions (up/down/left/right, as in the real capture protocol).
  * When |S| exceeds mu * N the object slips: shear partially drops and stick-slip
    vibration appears. Before that, "incipient slip" micro-vibration ramps up over
    ~50-250 ms, which is what makes early warning possible at all.
  * Each of 5 magnetometers sees a fixed linear mix of (Sx, Sy, N), with a soft
    saturation that depends on object stiffness, plus a skin-specific gain/offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MAG_COLS, N_MAGS

OBJECT_NAMES = [
    "bottle", "can", "sponge", "tissue_box", "apple", "cup_paper", "cup_glass", "foam",
    "wood_block", "tape_roll", "marker", "egg_carton", "plush", "phone", "orange", "soap",
]


def make_objects(rng: np.random.Generator) -> dict:
    objs = {}
    for name in OBJECT_NAMES:
        objs[name] = dict(
            mu=rng.uniform(0.35, 1.1),           # friction coefficient
            stiffness=rng.uniform(0.3, 1.5),     # soft (foam) .. hard (glass)
            texture_hz=rng.uniform(8, 35),       # dominant stick-slip frequency
            vib_gain=rng.uniform(0.5, 2.0),
        )
    for soft in ("sponge", "foam", "plush"):
        objs[soft]["stiffness"] = rng.uniform(0.15, 0.35)
        objs[soft]["vib_gain"] = rng.uniform(0.2, 0.5)   # soft objects damp vibration
    return objs


def make_skin(rng: np.random.Generator, name: str = "skin0", spread: float = 1.0) -> dict:
    return dict(
        name=name,
        gain=1.0 + spread * rng.normal(0, 0.04, len(MAG_COLS)),
        offset=spread * rng.normal(0, 15.0, len(MAG_COLS)),
        noise=2.5,
    )


def _mixing(rng: np.random.Generator) -> np.ndarray:
    # Fixed sensor geometry: each magnetometer axis responds to (Sx, Sy, N).
    m = rng.normal(0, 1, (len(MAG_COLS), 3))
    m[2::3, 2] += 3.0  # z axes mostly see normal force
    m[0::3, 0] += 2.0  # x axes see shear-x
    m[1::3, 1] += 2.0  # y axes see shear-y
    return m * 12.0


GEOMETRY_SEED = 12345


def simulate_trial(rng, obj: dict, skin: dict, mix: np.ndarray, base: np.ndarray,
                   fs: float = 100.0, slip_trial: bool = True) -> tuple[np.ndarray, np.ndarray]:
    dur = rng.uniform(3.0, 5.0)
    n = int(dur * fs)
    t = np.arange(n) / fs
    t_grip, t_pull = rng.uniform(0.3, 0.6), rng.uniform(1.0, 1.6)
    grip = rng.uniform(0.7, 1.3)
    N = grip * np.clip((t - t_grip) / 0.25, 0, 1)
    direction = rng.integers(4)
    ang = direction * np.pi / 2 + rng.normal(0, 0.15)
    rate = rng.uniform(0.25, 0.9) * (1.0 if slip_trial else 0.25)
    S_mag = rate * np.clip(t - t_pull, 0, None)
    if not slip_trial:
        S_mag = np.minimum(S_mag, 0.6 * obj["mu"] * grip)

    limit = obj["mu"] * N
    slip = (S_mag > limit) & (N > 0.1)
    # Once slipping, keep slipping until the pull ends (operator keeps pulling).
    if slip.any():
        first = np.argmax(slip)
        slip[first:] = True
        S_eff = S_mag.copy()
        S_eff[first:] = limit[first:] * (0.75 + 0.1 * rng.random())
    else:
        first = None
        S_eff = S_mag
    # Incipient slip: micro-vibration ramping up before gross slip.
    ratio = np.where(limit > 1e-6, S_mag / np.maximum(limit, 1e-6), 0)
    lead = rng.uniform(0.6, 0.9)
    incipient = np.clip((ratio - lead) / (1 - lead), 0, 1) ** 2
    f = obj["texture_hz"] * (1 + 0.1 * rng.normal())
    vib_amp = obj["vib_gain"] * (0.05 * incipient + 0.12 * slip)
    vib = vib_amp * (np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + 0.5 * rng.normal(0, 1, n))

    Sx = S_eff * np.cos(ang) + vib * np.cos(ang)
    Sy = S_eff * np.sin(ang) + vib * np.sin(ang)
    Nn = N + 0.3 * vib
    forces = np.stack([Sx, Sy, Nn], axis=1)
    # Soft objects spread contact: weaker, saturating response.
    k = obj["stiffness"]
    resp = np.tanh(forces * k) / max(k, 1e-3) * (0.6 + 0.4 * k)
    sig = base + resp @ mix.T
    sig = sig * skin["gain"] + skin["offset"]
    sig += rng.normal(0, skin["noise"], sig.shape)
    labels = slip.astype(int)
    return np.column_stack([t, sig]), labels


def generate(n_trials_per_object: int = 12, fs: float = 100.0, seed: int = 0,
             skins: list[dict] | None = None, static_fraction: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(GEOMETRY_SEED)
    mix = _mixing(geo)
    base = geo.normal(0, 40, len(MAG_COLS))
    objs = make_objects(np.random.default_rng(GEOMETRY_SEED + 1))
    skins = skins or [make_skin(rng)]
    rows = []
    for skin in skins:
        for name, obj in objs.items():
            for k in range(n_trials_per_object):
                slip_trial = rng.random() > static_fraction
                arr, lab = simulate_trial(rng, obj, skin, mix, base, fs, slip_trial)
                d = pd.DataFrame(arr[:, 1:], columns=MAG_COLS)
                d.insert(0, "t", arr[:, 0])
                d["slip"] = lab
                d["object"] = name
                d["trial"] = f"{skin['name']}_{name}_{k}"
                d["skin"] = skin["name"]
                rows.append(d)
    return pd.concat(rows, ignore_index=True)


def write_csvs(df: pd.DataFrame, out_dir) -> None:
    """Write in the real dataset's CSV layout (one file per object, log_time in seconds)."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, g in df.groupby("object"):
        g = g.copy()
        # Offset each trial in time so the loader's gap detection re-splits them.
        offs, t0 = [], 0.0
        for _, tr in g.groupby("trial", sort=False):
            offs.append(tr["t"] + t0)
            t0 += tr["t"].max() + 5.0
        g["log_time"] = pd.concat(offs)
        g["slip"] = g["slip"].astype(bool)
        g[["log_time"] + MAG_COLS + ["slip"]].to_csv(out / f"{name}.csv", index=False)


assert N_MAGS * 3 == len(MAG_COLS)
