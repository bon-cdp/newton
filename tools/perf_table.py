#!/usr/bin/env python3
"""
One row per DEM run: cost, every BFA observable, rms log error, and pile height.

    python tools/perf_table.py runs/dem/verify_best runs/dem/perf_*

Scores exactly as compare_bfa_dem.py does (same steady window, same rms over the seven
observables) so numbers are comparable with everything recorded before; adds the
frame-averaged top-of-slow-material height from tools/pileavg.py, which is what the eye
reads as the pile and what the visual match was tuned on.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_bfa_dem import bfa_reference, dem_run  # noqa: E402
from compare_bfa_mpm import DEM_MASS, read_bfa_frames, read_mpm_frame  # noqa: E402

XMAX, NF = -4.8, 30


def pile_top(pos, vel):
    m = pos[:, 0] < XMAX
    p, v = pos[m], np.linalg.norm(vel[m], axis=1)
    slow = v < 1.5
    return np.percentile(p[slow][:, 1], 98) if slow.sum() > 20 else np.nan


def main():
    ref = bfa_reference()
    bf = list(read_bfa_frames())[-NF:]
    tops = [pile_top(f["pos"], f["vel"]) for f in bf]
    hdr = (f"{'run':<22} {'dt us':>6} {'E':>7} {'s/sim-s':>8} {'vs BFA':>7} | {'hold':>6} {'KE':>6} "
           f"{'tube':>5} {'@m/s':>5} {'casc':>5} {'@m/s':>5} {'kg/s':>5} | {'rms':>6} | {'pile top (m)':>15}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'BFA':<22} {24.3:6.1f} {1.42e8:7.1e} {579.6:8.1f} {1.0:7.1f} | {ref['holdup']:6.2f} "
          f"{ref['ke']:6.1f} {ref['tube_kg']:5.2f} {ref['tube_ms']:5.2f} {4.59:5.2f} {2.39:5.2f} "
          f"{8.61:5.2f} | {'':>6} | {np.mean(tops):+.3f}+-{np.std(tops):.3f}")
    for run in sys.argv[1:]:
        if not os.path.exists(os.path.join(run, "history.csv")):
            continue
        d = dem_run(run)
        st = d["st"]
        if not st.sum():
            print(f"{os.path.basename(run):<22} (no steady window yet, t = {d['t'][-1]:.2f})")
            continue
        m = d["meta"]
        rows = [(d["holdup"][st].mean(), ref["holdup"]), (d["ke"][st].mean(), ref["ke"]),
                (d["tube_kg"][st].mean(), ref["tube_kg"]), (d["tube_ms"][st].mean(), ref["tube_ms"]),
                (d["rate"], 8.6111), (d["casc_kg"][st].mean(), 4.59), (d["casc_ms"][st].mean(), 2.39)]
        rms = np.sqrt(np.mean([np.log(a / b) ** 2 for a, b in rows]))
        per_s = d["wall"][-1] / max(d["t"][-1], 1e-9)
        fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))[-NF:]
        pt = [pile_top(*read_mpm_frame(f)) for f in fs] if len(fs) >= 5 else [np.nan]
        youngs = m.get("youngs") or float("nan")
        v = [a for a, _b in rows]
        print(f"{os.path.basename(run)[:22]:<22} {m['dt']*1e6:6.1f} {youngs:7.1e} {per_s:8.1f} "
              f"{579.6 / per_s:6.0f}x | {v[0]:6.2f} {v[1]:6.1f} {v[2]:5.2f} {v[3]:5.2f} {v[5]:5.2f} "
              f"{v[6]:5.2f} {v[4]:5.2f} | {rms:6.3f} | {np.nanmean(pt):+.3f}+-{np.nanstd(pt):.3f}"
              + ("" if d["t"][-1] >= 9.99 else f"   (t = {d['t'][-1]:.1f})"))


if __name__ == "__main__":
    main()
