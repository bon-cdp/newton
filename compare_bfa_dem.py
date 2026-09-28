#!/usr/bin/env python3
"""
Compare a SolverGranularDEM run against the BulkFlowAnalyst DEM reference.

Unlike the MPM comparison this is like-for-like: both sides are 12 mm spheres of the
same mass, so grain counts compare directly and no continuum mapping is involved.

Usage:  python compare_bfa_dem.py dem_full [more_run_dirs...]
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

from compare_bfa_mpm import (
    DEM_MASS, TUBE_HI, TUBE_LO, read_bfa_frames, read_bfa_history, read_mpm_frame,
    tube_stats, y_profile,
)

STEADY_FROM = 3.0


def bfa_reference():
    his = read_bfa_history()
    steady = his[:, 0] >= STEADY_FROM
    frames = list(read_bfa_frames())
    edges = np.arange(-2.0, 4.5, 0.5)
    m = np.zeros(len(edges) - 1)
    s = np.zeros(len(edges) - 1)
    tm, tv = [], []
    for f in frames[-30:]:
        a, b = y_profile(f["pos"], f["vel"], DEM_MASS, edges)
        m += a
        s += b
        a2, b2 = tube_stats(f["pos"], f["vel"], DEM_MASS)
        tm.append(a2)
        tv.append(b2)
    return dict(
        holdup=float(his[steady, 1].mean()), ke=float(his[steady, 2].mean()),
        tube_kg=float(np.mean(tm)), tube_ms=float(np.mean(tv)),
        prof_kg=m / 30.0, prof_ms=s / 30.0, edges=edges,
        grains=float(his[steady, 1].mean() / DEM_MASS),
    )


def dem_run(run):
    rows = list(csv.DictReader(open(os.path.join(run, "history.csv"))))
    t = np.array([float(r["time_s"]) for r in rows])
    meta = json.load(open(os.path.join(run, "run.json")))
    st = t >= STEADY_FROM
    g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    out = g("discharged_kg")
    rate = np.polyfit(t[st], out[st], 1)[0] if st.sum() > 2 else float("nan")
    d = dict(meta=meta, t=t, holdup=g("mass_kg"), ke=g("kinetic_energy_J"),
             tube_kg=g("tube_mass_kg"), tube_ms=g("tube_speed_ms"), rate=rate, st=st,
             grains=g("n_grains"), wall=g("wallclock_s"),
             casc_kg=g("cascade_mass_kg") if "cascade_mass_kg" in rows[0] else None,
             casc_ms=g("cascade_speed_ms") if "cascade_speed_ms" in rows[0] else None)
    return d


def main():
    runs = sys.argv[1:]
    if not runs:
        print(__doc__)
        sys.exit(1)
    ref = bfa_reference()

    print("=" * 88)
    print("BulkFlowAnalyst DEM reference  (steady state, t > %.0f s)" % STEADY_FROM)
    print("=" * 88)
    print(f"  {ref['grains']:,.0f} grains = {ref['holdup']:.2f} kg   KE {ref['ke']:.1f} J   "
          f"chute tube {ref['tube_kg']:.2f} kg at {ref['tube_ms']:.2f} m/s")
    print(f"  discharge 8.61 kg/s     wall clock 5796 s on 24 CPU cores (411,245 steps)")

    for run in runs:
        d = dem_run(run)
        st = d["st"]
        print("\n" + "=" * 88)
        print(f"{run}   ({d['meta'].get('solver', '?')})")
        print("=" * 88)
        m = d["meta"]
        print(f"  grain r {m['grain_radius']*1e3:.1f} mm, m {m['grain_mass']*1e3:.4f} g;  "
              f"dt {m['dt']*1e6:.2f} us;  ke {m['ke']:.1e}  mu {m['mu']:.4f}  wall_mu {m['wall_mu']}")
        if not st.sum():
            print(f"  reached t = {d['t'][-1]:.2f} s -- no steady window yet")
            continue
        rows = [("hold-up (kg)", d["holdup"][st].mean(), ref["holdup"]),
                ("kinetic energy (J)", d["ke"][st].mean(), ref["ke"]),
                ("chute tube (kg)", d["tube_kg"][st].mean(), ref["tube_kg"]),
                ("chute speed (m/s)", d["tube_ms"][st].mean(), ref["tube_ms"]),
                ("discharge (kg/s)", d["rate"], 8.6111)]
        if d["casc_kg"] is not None:
            rows += [("cascade (kg)", d["casc_kg"][st].mean(), 4.59),
                     ("cascade speed (m/s)", d["casc_ms"][st].mean(), 2.39)]
        print(f"  {'quantity':<22} {'Newton':>10} {'BFA':>10} {'ratio':>8}")
        for label, a, b in rows:
            print(f"  {label:<22} {a:10.2f} {b:10.2f} {a/b:8.2f}")
        vals = [np.log(max(a, 1e-9) / b) ** 2 for _l, a, b in rows if np.isfinite(a)]
        err = np.sqrt(np.mean(vals)) if vals else float("nan")
        print(f"  {'rms log error':<22} {'':>10} {'':>10} {err:8.3f}")
        # normalise: BFA took 5796 s for 10 s of simulation on 24 CPU cores
        per_s = d["wall"][-1] / max(d["t"][-1], 1e-9)
        print(f"  reached t = {d['t'][-1]:.2f} s;  wall clock {d['wall'][-1]:.0f} s "
              f"= {per_s:.1f} s per simulated second")
        print(f"  BFA: 579.6 s per simulated second on 24 CPU cores  ->  "
              f"{579.6/max(per_s,1e-9):.1f}x faster on one GPU")

        frs = sorted(glob.glob(os.path.join(run, "frame_*_particles.vtk")))[-30:]
        if not frs:
            continue
        edges = ref["edges"]
        pm = np.zeros(len(edges) - 1)
        ps = np.zeros(len(edges) - 1)
        for fn in frs:
            p, v = read_mpm_frame(fn)
            a, b = y_profile(p, v, m["grain_mass"], edges)
            pm += a
            ps += b
        pm /= len(frs)
        ps /= len(frs)
        print(f"\n  vertical profile (last {len(frs)} frames)")
        print(f"  {'y band (m)':>14} | {'BFA kg':>7} {'BFA m/s':>8} | {'DEM kg':>7} {'DEM m/s':>8} | delta")
        for k in range(len(edges) - 2, -1, -1):
            print(f"  {edges[k]:6.1f}..{edges[k+1]:5.1f} | {ref['prof_kg'][k]:7.2f} "
                  f"{ref['prof_ms'][k]:8.2f} | {pm[k]:7.2f} {ps[k]:8.2f} | "
                  f"{pm[k]-ref['prof_kg'][k]:+6.2f}")
        print(f"  {'total':>14} | {ref['prof_kg'].sum():7.2f} {'':>8} | {pm.sum():7.2f}")


if __name__ == "__main__":
    main()
