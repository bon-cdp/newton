#!/usr/bin/env python3
"""
Issue #13 test: do grains sink into the concave corner of a single-part trough?

The wall kernels keep ONE contact per collider part (the closest triangle).  A grain in a
concave corner of one part touches two faces, but only the nearer can push back.  This
builds a troughed-belt cross-section -- flat bottom and two wings at `--angle` -- runs a
pour-and-settle through dem_run, and measures every settled grain's overlap with each
plane analytically.  Two builds:

  one-part    bottom + both wings in ONE STL          (suspect)
  three-part  bottom and each wing as separate parts  (control: contacts summed)

A correct contact law gives overlaps at the Hertz level (micrometres) in both.

    python tools/trough_test.py [--angle 35] [--preset fast|reference]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dem_run  # noqa: E402
from dem_scenario import INF, Injector, Material, Output, Part, Region, Scenario, Solver  # noqa: E402

L = 0.60          # trough length (x)
W = 0.12          # bottom width (z)
LW = 0.15         # wing length along the slope


def planes(angle_deg):
    """(name, point, unit normal, in-plane extent test) for bottom and wings."""
    a = math.radians(angle_deg)
    wing_dir_r = np.array([0.0, math.sin(a), math.cos(a)])      # up and out (+z side)
    wing_dir_l = np.array([0.0, math.sin(a), -math.cos(a)])
    p_r, p_l = np.array([0.0, 0.0, W / 2]), np.array([0.0, 0.0, -W / 2])
    n_r = np.array([0.0, math.cos(a), -math.sin(a)])             # points into the trough
    n_l = np.array([0.0, math.cos(a), math.sin(a)])
    return [("bottom", np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), None),
            ("wing+", p_r, n_r, (p_r, wing_dir_r)), ("wing-", p_l, n_l, (p_l, wing_dir_l))]


def quad(p0, du, dv):
    v = np.array([p0, p0 + du, p0 + du + dv, p0 + dv])
    return v, np.array([[0, 1, 2], [0, 2, 3]])


def write_geometry(d, angle_deg, split):
    a = math.radians(angle_deg)
    x = np.array([L, 0.0, 0.0])
    bottom = quad(np.array([0.0, 0.0, -W / 2]), x, np.array([0.0, 0.0, W]))
    wr = quad(np.array([0.0, 0.0, W / 2]), x, LW * np.array([0.0, math.sin(a), math.cos(a)]))
    wl = quad(np.array([0.0, 0.0, -W / 2]), LW * np.array([0.0, math.sin(a), -math.cos(a)]), x)
    zmax = W / 2 + LW * math.cos(a) + 0.02
    ymax = LW * math.sin(a) + 0.02
    ends = [quad(np.array([xe, -0.01, -zmax]), np.array([0.0, 0.0, 2 * zmax]), np.array([0.0, ymax + 0.01, 0.0]))
            for xe in (0.0, L)]

    def save(name, pieces):
        vs, fs, off = [], [], 0
        for v, f in pieces:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False).export(os.path.join(d, name))

    if split:
        save("bottom.stl", [bottom])
        save("wing_r.stl", [wr])
        save("wing_l.stl", [wl])
        names = ["bottom", "wing_r", "wing_l"]
    else:
        save("trough.stl", [bottom, wr, wl])
        names = ["trough"]
    save("ends.stl", ends)
    save("inject.stl", [quad(np.array([0.04, ymax + 0.15, -W / 2 - 0.03]),
                             np.array([L - 0.08, 0.0, 0.0]), np.array([0.0, 0.0, W + 0.06]))])
    return names + ["ends"], zmax, ymax


def scenario(d, angle_deg, split, preset, corners=False):
    names, zmax, ymax = write_geometry(d, angle_deg, split)
    parts = [Part(name=n, stl=f"{n}.stl", two_sided=True, friction=0.5, corners=corners)
             for n in names]
    mat = Material(name="corn", radius=0.006, density=994.05, youngs=1.4220405e8, poisson=0.3,
                   restitution=0.2, friction=0.11, rolling_friction=0.3,
                   wall_rolling_friction=0.5, tangential_ratio=1.0, contact="hertz")
    solver = Solver(dt=2.4316429e-5, youngs_divisor=1.0, neighbor_every=0, expected_holdup_kg=6.0)
    if preset == "fast":
        solver = Solver(dt="auto", youngs_divisor=10.0, neighbor_every=4, skin_speed=6.0,
                        expected_holdup_kg=6.0)
    inj = Injector(name="pour", face_stl="inject.stl", mass_rate=6.0, velocity=[0.0, -1.0, 0.0],
                   stop=0.6)
    sc = Scenario(name=f"trough_{angle_deg:g}_{'split' if split else 'onepart'}", material=mat,
                  parts=parts, injectors=[inj], domain_lo=[-0.1, -0.2, -zmax - 0.1],
                  domain_hi=[L + 0.1, ymax + 0.5, zmax + 0.1], solver=solver,
                  output=Output(duration=1.6, fps=15, vtk=True, checkpoint_at=1.6),
                  base_dir=d, regions=[Region("trough")])
    sc.validate()
    return sc


def overlaps(q, r, angle_deg):
    """Per plane: overlaps of grains within reach of the plane's finite extent."""
    out = {}
    for name, p0, n, wing in planes(angle_deg):
        d = (q - p0) @ n                          # signed distance, +ve on the trough side
        if wing is None:
            inside = np.abs(q[:, 2]) <= W / 2
        else:
            s = (q - wing[0]) @ wing[1]
            inside = (s >= 0) & (s <= LW)
        inside &= (q[:, 0] > 0.02) & (q[:, 0] < L - 0.02)
        ov = r - d[inside]
        out[name] = ov[ov > -0.5 * r]             # grains touching or nearly touching
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angle", type=float, default=35.0)
    ap.add_argument("--preset", choices=["fast", "reference"], default="fast")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "runs", "trough"))
    args = ap.parse_args()
    rows = []
    for split, corners in ((False, False), (False, True), (True, False)):
        tag = (f"{args.angle:g}deg_{args.preset}_{'three-part' if split else 'one-part'}"
               + ("_corners" if corners else ""))
        d = os.path.join(args.out, tag)
        os.makedirs(d, exist_ok=True)
        sc = scenario(d, args.angle, split, args.preset, corners)
        S = dem_run.build(sc, out_dir=d, quiet=True)
        dem_run.run(S)
        ck = np.load(os.path.join(d, "checkpoint.npz"))
        act = (ck["flags"] & 1) == 1
        q = ck["q"][act]
        ov = overlaps(q, sc.material.radius, args.angle)
        rows.append((tag, len(q), ov))
    print(f"\n#13 trough test, wings at {args.angle:g} deg, preset {args.preset}")
    print(f"{'build':34s} {'grains':>6s}   plane    touching   overlap p50 / p99 / max (um)")
    for tag, n, ov in rows:
        for name, o in ov.items():
            t = o[o > 0]
            if len(t):
                print(f"{tag:34s} {n:6d}   {name:7s} {len(t):8d}   {np.median(t)*1e6:9.1f} / "
                      f"{np.percentile(t, 99)*1e6:8.1f} / {t.max()*1e6:8.1f}")
            else:
                print(f"{tag:34s} {n:6d}   {name:7s} {0:8d}")


if __name__ == "__main__":
    main()
