from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import reliability  # noqa: E402


def reliability_plot(P, path, title="Slip probability calibration (unseen objects)"):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", c="#999", lw=1)
    for cname, g in P.groupby("calibrator"):
        if cname == "platt":
            continue
        conf, acc, cnt = reliability(g["y"].to_numpy(), g["p"].to_numpy())
        m = cnt > 20
        ax.plot(conf[m], acc[m], marker="o", ms=4, label=cname)
    ax.set(xlabel="Predicted P(slip)", ylabel="Observed slip frequency", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shift_plot(summary, path):
    import pandas as pd

    clean = summary[summary["shift"] == "none"]
    worst = summary[(summary["shift"] != "none") & (summary["severity"] == summary["severity"].max())]
    s = pd.concat([clean, worst])
    labels = s["shift"].tolist()
    x = np.arange(len(labels))
    cols = [c for c in ("ece_raw", "ece_isotonic", "ece_recal_30s") if c in s]
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, c in enumerate(cols):
        ax.bar(x + i * w - 0.4 + w / 2, s[c], w, label=c.replace("ece_", ""))
    ax.set_xticks(x, labels, rotation=20)
    ax.set(ylabel="ECE (lower is better)", title="Calibration under sensor shift (max severity)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def warn_plot(curves, path):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for h, g in curves.groupby("horizon_ms"):
        g = g.sort_values("fa_per_min")
        ax.plot(g["fa_per_min"].clip(lower=0.1), g["median_lead_ms"], marker="o", ms=3, label=f"trained {h} ms ahead")
    ax.axhline(0, c="#999", lw=1)
    ax.set(xscale="log", xlim=(0.1, None), xlabel="False alarms per minute of contact (0 shown as 0.1)",
           ylabel="Median lead time before slip (ms)", title="Early warning trade-off")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
