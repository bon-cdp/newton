#!/usr/bin/env python3
"""
Compare flow through BFA portals with a dem_run result, window by window.

    python tools/portal_compare.py 20060-CM-552/scenario.json runs/dem/<run> [--settle 2.0]

BFA writes one row per grain crossing a portal ("PMA Analyses/<sim> <portal>-portal.csv":
time, radius mm, position mm, velocity m/s, density).  dem_run writes the cumulative mass
through each flow plane per frame (history.csv, flow_<name>_kg).  Time windows come from
the scenario's parts: every on/off time of a part that is not always active starts a
window (in the conveyor project, the five deflectors are switched in turn), and each
window is labelled by the timed parts active in it.  The first --settle seconds after
a switch are skipped: the stream needs time to settle onto a new deflector.
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bfa_import import safe_name  # noqa: E402
from dem_scenario import INF, Scenario  # noqa: E402


def bfa_crossings(path):
    """(times, masses) of every grain crossing in a BFA portal CSV."""
    t, m = [], []
    with open(path, encoding="latin1") as fh:
        for ln in fh:
            f = ln.split(",")
            if len(f) != 9:
                continue
            try:
                vals = [float(x) for x in f]
            except ValueError:
                continue                           # the header line
            r = vals[1] * 1e-3
            t.append(vals[0])
            m.append(4.0 / 3.0 * np.pi * r ** 3 * vals[8])
    return np.array(t), np.array(m)


def windows(sc):
    edges = {0.0, sc.output.duration}
    dur = sc.output.duration
    # parts switched on or off during the run (not ones active throughout)
    timed = [p for p in sc.parts if p.active[0] > 0.0 or p.active[1] < dur]
    for p in timed:
        for e in p.active:
            if 0.0 < e < sc.output.duration:
                edges.add(e)
    edges = sorted(edges)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        on = [p.name for p in timed if p.active[0] <= mid < p.active[1]]
        out.append((a, b, ", ".join(on) if on else "(none)"))
    return out


def main():
    scen, run = sys.argv[1], sys.argv[2]
    settle = float(sys.argv[sys.argv.index("--settle") + 1]) if "--settle" in sys.argv else 2.0
    sc = Scenario.load(scen)
    proj = sc.base_dir
    rows = list(csv.DictReader(open(os.path.join(run, "history.csv"))))
    th = np.array([float(r["time_s"]) for r in rows])
    t_end = th[-1]
    planes = [fp.name for fp in sc.flow_planes]
    ours = {n: np.array([float(r[f"flow_{n}_kg"]) for r in rows]) for n in planes
            if f"flow_{n}_kg" in rows[0]}
    bfa = {}
    for n in planes:
        hits = glob.glob(os.path.join(proj, "Local Output", "PMA Analyses", "*-portal.csv"))
        for h in hits:
            if safe_name(os.path.basename(h).split(" ", 1)[1].rsplit("-portal", 1)[0]) == n:
                bfa[n] = bfa_crossings(h)
    both = [n for n in planes if n in bfa and n in ours]
    left = next((n for n in both if "left" in n.lower()), None)
    right = next((n for n in both if "right" in n.lower()), None)

    def rate_ours(n, a, b):
        return (np.interp(b, th, ours[n]) - np.interp(a, th, ours[n])) / (b - a)

    def rate_bfa(n, a, b):
        t, m = bfa[n]
        return m[(t >= a) & (t < b)].sum() / (b - a)

    print(f"flow through portals, kg/s, averaged over each window after a {settle:g} s settle"
          f"   (run reached t = {t_end:.1f} s)\n")
    hdr = f"{'window':>11}  {'active':<22}"
    for n in both:
        hdr += f" | {n[:18]:>18} BFA / ours"
    if left and right:
        hdr += " | left share BFA / ours"
    print(hdr)
    print("-" * len(hdr))
    for a, b, label in windows(sc):
        a2 = min(a + settle, b)
        if a2 >= min(b, t_end):
            continue
        b2 = min(b, t_end)
        line = f"{a:5.1f}-{b:4.1f}  {label[:22]:<22}"
        rb, ro = {}, {}
        for n in both:
            rb[n], ro[n] = rate_bfa(n, a2, b2), rate_ours(n, a2, b2)
            line += f" | {rb[n]:12.1f} / {ro[n]:7.1f}"
        if left and right:
            sb = rb[left] / max(rb[left] + rb[right], 1e-9)
            so = ro[left] / max(ro[left] + ro[right], 1e-9)
            line += f" |    {sb:6.3f} / {so:6.3f}"
        print(line)


if __name__ == "__main__":
    main()
