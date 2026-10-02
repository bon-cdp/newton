#!/usr/bin/env python3
"""
Plot simulated vs measured discharge curves for calibration results.

    <python with matplotlib> calibration/soybean_bucket/plot_compare.py out.png \
        runs/calib/<tag>/<label> [runs/calib/<tag2>/<label2> ...]

Each directory holds run*/result.json (calibrate.py).  Solid: video (flow_curves.json),
dashed: simulations.  Horizontal lines mark the weighed discharge (m0 - m_end).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
runs = json.load(open(os.path.join(HERE, "runs.json")))["runs"]
curves = json.load(open(os.path.join(HERE, "flow_curves.json")))["runs"]
order = ["run1", "run4", "run3", "run2"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
for ax, rid in zip(axes, order):
    r = runs[rid]
    if rid in curves:
        ax.plot(curves[rid]["t_s"], curves[rid]["discharged_g"], "k-", lw=2.5, label="video")
    if r.get("choked"):
        ax.axvline(r["t_choke"] - r["t_open"], color="k", ls=":", label="choke (video)")
    ax.axhline(r["m0_g"] - r["m_end_g"], color="k", lw=0.8, ls="--")
    for d in sys.argv[2:]:
        for f in glob.glob(os.path.join(d, rid, "result.json")):
            res = json.load(open(f))
            s = res["sim"]
            ax.plot(s["curve_t"], s["curve_out_g"], "--",
                    label=f"{os.path.basename(d)[:40]}  res {s['residual_g']:.0f} g")
    split = "FIT" if rid in ("run1", "run4") else "TEST"
    ax.set_title(f"{rid} [{split}]  m0 {r['m0_g']:.0f} g, drop {r['drop_mm']} mm, "
                 f"residual {r['m_end_g']:.0f} g", fontsize=9)
    ax.set_xlabel("s after plug pulled")
    ax.set_ylabel("discharged g")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(sys.argv[1], dpi=80)
