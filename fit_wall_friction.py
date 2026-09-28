#!/usr/bin/env python3
"""
Calibrate the MPM wall friction against the BulkFlowAnalyst DEM run.

The BFA project specifies mu_wall = 0.50 for all seven boundary components, and
DEM uses that value directly.  Newton's MPM cannot: it resolves the chute stream
on a grid whose cells are wider than the stream is thick, so the whole stream
sits inside the wall-friction band and is over-braked.  mu_wall therefore has to
be treated as an *effective* coefficient that absorbs that sub-grid error.

This script fits it against four independent steady-state observables rather
than the one it is most obviously tuned to:

    hold-up (kg)          total material in the system
    kinetic energy (J)    sum 1/2 m v^2, directly comparable to the BFA .his log
    chute tube (kg, m/s)  mass and mean speed in the constant-section chute
    discharge rate (kg/s) must equal the 8.61 kg/s feed at steady state

A fit that matches only the tube speed is meaningless -- one scalar tuned to one
scalar.  A single mu_eff that matches all four is evidence.

Usage:
    python fit_wall_friction.py sweep_mu000 sweep_mu005 sweep_mu010 ...
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from compare_bfa_mpm import DEM_MASS, read_bfa_frames, read_bfa_history, tube_stats

STEADY_FROM = 3.0  # s; DEM reaches steady state at ~2.5 s


def dem_targets():
    his = read_bfa_history()
    steady = his[:, 0] >= STEADY_FROM
    tm, tv = [], []
    for f in list(read_bfa_frames())[-30:]:
        a, b = tube_stats(f["pos"], f["vel"], DEM_MASS)
        tm.append(a)
        tv.append(b)
    return {
        "holdup_kg": float(his[steady, 1].mean()),
        "kinetic_J": float(his[steady, 2].mean()),
        "tube_kg": float(np.mean(tm)),
        "tube_ms": float(np.mean(tv)),
        "discharge_kgs": 8.6111,  # steady state must equal the feed rate
    }


def run_metrics(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "history.csv"))))
    t = np.array([float(r["time_s"]) for r in rows])
    steady = t >= STEADY_FROM
    if steady.sum() < 3:
        return None
    get = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    out = get("discharged_kg")
    # discharge rate from a least-squares slope over the steady window
    rate = np.polyfit(t[steady], out[steady], 1)[0] if steady.sum() > 2 else 0.0
    meta = json.load(open(os.path.join(run_dir, "run.json")))
    return {
        "mu": meta["wall_mu"],
        "voxel": meta["voxel"],
        "holdup_kg": float(get("mass_kg")[steady].mean()),
        "kinetic_J": float(get("kinetic_energy_J")[steady].mean()),
        "tube_kg": float(get("tube_mass_kg")[steady].mean()),
        "tube_ms": float(get("tube_speed_ms")[steady].mean()),
        "discharge_kgs": float(rate),
        "wall_s": float(get("wallclock_s")[-1]),
        "t_end": float(t[-1]),
    }


KEYS = ["holdup_kg", "kinetic_J", "tube_kg", "tube_ms", "discharge_kgs"]


def rms_error(m, target):
    """RMS of the log-ratio across the observables -- scale-free, symmetric."""
    e = [np.log(max(m[k], 1e-9) / target[k]) for k in KEYS]
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    runs = sys.argv[1:]
    if not runs:
        print(__doc__)
        sys.exit(1)
    tgt = dem_targets()

    print("=" * 96)
    print("BFA DEM steady-state targets (t > %.1f s)" % STEADY_FROM)
    print("=" * 96)
    print("  hold-up %.2f kg   KE %.1f J   chute tube %.2f kg at %.2f m/s   discharge %.2f kg/s"
          % (tgt["holdup_kg"], tgt["kinetic_J"], tgt["tube_kg"], tgt["tube_ms"], tgt["discharge_kgs"]))

    ms = [m for m in (run_metrics(r) for r in runs) if m]
    ms.sort(key=lambda m: m["mu"])
    if not ms:
        print("\nno run reached steady state yet")
        return

    print("\n" + "=" * 96)
    print("%-7s %-7s %10s %10s %10s %10s %11s %8s" %
          ("mu_wall", "voxel", "holdup kg", "KE J", "tube kg", "tube m/s", "out kg/s", "rms err"))
    print("-" * 96)
    print("%-7s %-7s %10.2f %10.1f %10.2f %10.2f %11.2f %8s" %
          ("DEM", "-", tgt["holdup_kg"], tgt["kinetic_J"], tgt["tube_kg"], tgt["tube_ms"],
           tgt["discharge_kgs"], "-"))
    for m in ms:
        print("%-7.3f %-7.0f %10.2f %10.1f %10.2f %10.2f %11.2f %8.3f" %
              (m["mu"], m["voxel"] * 1e3, m["holdup_kg"], m["kinetic_J"], m["tube_kg"],
               m["tube_ms"], m["discharge_kgs"], rms_error(m, tgt)))

    print("\nratio to DEM (1.00 = exact):")
    print("%-7s %10s %10s %10s %10s %11s" % ("mu_wall", "holdup", "KE", "tube kg", "tube m/s", "discharge"))
    for m in ms:
        print("%-7.3f %10.2f %10.2f %10.2f %10.2f %11.2f"
              % (m["mu"], *[m[k] / tgt[k] for k in KEYS]))

    # parabolic fit of rms error vs mu, if we have a bracketed minimum
    if len(ms) >= 3:
        mu = np.array([m["mu"] for m in ms])
        err = np.array([rms_error(m, tgt) for m in ms])
        i = int(err.argmin())
        print("\nbest sampled: mu_wall = %.3f  (rms error %.3f)" % (mu[i], err[i]))
        if 0 < i < len(ms) - 1:
            a, b, c = np.polyfit(mu[i - 1:i + 2], err[i - 1:i + 2], 2)
            if a > 0:
                print("parabolic minimum: mu_wall = %.3f" % (-b / (2 * a)))
        else:
            print("minimum is at the edge of the sweep -- extend the range")

    voxels = {m["voxel"] for m in ms}
    if len(voxels) > 1:
        print("\nTRANSFERABILITY: this sweep spans %d voxel sizes.  If the best mu_eff differs\n"
              "between them, it is absorbing a discretisation error, not a physical one, and\n"
              "the quarter-grain run at 4x finer resolution will need a different value."
              % len(voxels))


if __name__ == "__main__":
    main()
