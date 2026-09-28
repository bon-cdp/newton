#!/usr/bin/env python3
"""
Compare a Newton implicit-MPM run against the BulkFlowAnalyst DEM reference in
23087-25sim/.

Reference channels pulled straight out of the BFA output:
  * Local Output/*.his  -- particle count and translational kinetic energy at 15 fps
  * Local Output/*.por  -- per-particle position / velocity / radius, 149 frames

Both are converted to mass (kg) so they can be compared against MPM material
points of a completely different size.

Usage:
  python compare_bfa_mpm.py bfa_mpm_replication_v1 [more_run_dirs...]
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BFA_OUT = os.path.join(SCRIPT_DIR, "23087-25sim", "Local Output")

IN = 0.0254
RHO_GRAIN = 994.05  # kg/m3, BFA "Material Intr Density" 62.055188 lb/ft3
DEM_RADIUS = 0.006  # m
DEM_MASS = 4.0 / 3.0 * np.pi * DEM_RADIUS**3 * RHO_GRAIN  # 8.9954e-4 kg
INLET_Y = 3.994
# Constant-section part of the inclined chute (0.19 x 0.14 m).  DEM runs a thin
# stream through here; hold-up and mean speed in this band are the sharpest
# single discriminator between a correct MPM solution and an over-braked one.
TUBE_LO, TUBE_HI = 0.0, 2.5


def tube_stats(pos, vel, mass_per_point):
    """(mass_kg, mass-weighted mean speed) in the constant-section tube."""
    m = (pos[:, 1] > TUBE_LO) & (pos[:, 1] < TUBE_HI)
    if not m.any():
        return 0.0, 0.0
    return m.sum() * mass_per_point, float(np.linalg.norm(vel[m], axis=1).mean())


# ---------------------------------------------------------------------------
# BFA readers
# ---------------------------------------------------------------------------
def read_bfa_history(path=None):
    """(time_s, mass_kg, kinetic_energy_J) from the BFA .his log."""
    path = path or os.path.join(BFA_OUT, "New Simulation (1).his")
    rows = []
    text = open(path, "rb").read().decode("latin1")
    for line in text.split("\r\n"):
        m = re.match(r"Time \(minutes\): ([\d.]+)\t([\d.]+)\s+(\d+)\s+([\d.eE+-]+)\s*$", line)
        if m:
            rows.append((float(m.group(2)), int(m.group(3)) * DEM_MASS, float(m.group(4))))
    return np.array(rows)


def read_bfa_frames(path=None):
    """Yield dicts of BFA particle state.

    .por layout (reverse engineered, verified against the .his kinetic energy):
      32-byte file header, then per frame: int32 count, 2x float32, count x 128 B.
      Record float32 slots: 0 = id (int32), 5..7 = position (in),
      8..10 = velocity (in/s), 17 = radius (in).
    """
    path = path or os.path.join(BFA_OUT, "New Simulation (1).por")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    off = 32
    while off + 12 <= mm.size:
        n = int(np.frombuffer(mm[off:off + 4].tobytes(), "<i4")[0])
        if n <= 0 or off + 12 + 128 * n > mm.size:
            break
        rec = np.frombuffer(mm[off + 12:off + 12 + 128 * n].tobytes(), "<f4").reshape(n, 32)
        yield dict(n=n, id=rec[:, 0].copy().view(np.int32),
                   pos=rec[:, 5:8] * IN, vel=rec[:, 8:11] * IN, rad=rec[:, 17] * IN)
        off += 12 + 128 * n


# ---------------------------------------------------------------------------
# MPM readers
# ---------------------------------------------------------------------------
def read_mpm_history(run_dir):
    path = os.path.join(run_dir, "history.csv")
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return dict(
        t=np.array([float(r["time_s"]) for r in rows]),
        mass=np.array([float(r["mass_kg"]) for r in rows]),
        mass_below=np.array([float(r["mass_below_inlet_kg"]) for r in rows]),
        ke=np.array([float(r["kinetic_energy_J"]) for r in rows]),
        vmax=np.array([float(r["max_speed_ms"]) for r in rows]),
        out=np.array([float(r["discharged_kg"]) for r in rows]),
        wall=np.array([float(r["wallclock_s"]) for r in rows]),
    )


def read_mpm_frame(path):
    """Positions and velocities from one frame_XXXX_particles.vtk."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("POINTS"))
    n = int(lines[i].split()[1])
    pos = np.array([[float(x) for x in ln.split()] for ln in lines[i + 1:i + 1 + n]])
    j = next(k for k, ln in enumerate(lines) if ln.startswith("VECTORS velocity"))
    vel = np.array([[float(x) for x in ln.split()] for ln in lines[j + 1:j + 1 + n]])
    return pos, vel


# ---------------------------------------------------------------------------
def y_profile(pos, vel, mass_per_point, edges):
    """Mass and mean speed in horizontal slabs."""
    idx = np.digitize(pos[:, 1], edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    speed = np.linalg.norm(vel, axis=1)
    mass = np.bincount(idx[ok], minlength=len(edges) - 1) * mass_per_point
    ssum = np.bincount(idx[ok], weights=speed[ok], minlength=len(edges) - 1)
    cnt = np.bincount(idx[ok], minlength=len(edges) - 1)
    return mass, np.divide(ssum, cnt, out=np.zeros_like(ssum), where=cnt > 0)


def bar(v, vmax, width=22):
    n = 0 if vmax <= 0 else int(round(width * min(v / vmax, 1.0)))
    return "#" * n + " " * (width - n)


def main():
    runs = sys.argv[1:]
    if not runs:
        print(__doc__)
        sys.exit(1)

    his = read_bfa_history()
    print("=" * 78)
    print("BulkFlowAnalyst DEM reference")
    print("=" * 78)
    steady = his[:, 0] >= 3.0
    print(f"  grain            {DEM_RADIUS * 2e3:.0f} mm sphere, {RHO_GRAIN} kg/m3, "
          f"{DEM_MASS * 1e3:.4f} g each")
    print(f"  holdup (t>3 s)   {his[steady, 1].mean():.2f} +/- {his[steady, 1].std():.2f} kg")
    print(f"  kinetic energy   {his[steady, 2].mean():.1f} +/- {his[steady, 2].std():.1f} J")
    print(f"  peak holdup      {his[:, 1].max():.2f} kg at t = {his[his[:, 1].argmax(), 0]:.2f} s")
    tm, tv = [], []
    for f in list(read_bfa_frames())[-30:]:
        a, b = tube_stats(f["pos"], f["vel"], DEM_MASS)
        tm.append(a)
        tv.append(b)
    dem_tube = (float(np.mean(tm)), float(np.mean(tv)))
    print(f"  chute tube       {dem_tube[0]:.2f} kg at {dem_tube[1]:.2f} m/s "
          f"(y {TUBE_LO}..{TUBE_HI} m)")
    print(f"  wall clock       5796 s on 24 CPU cores (411,245 DEM steps)")

    # BFA steady-state vertical profile, averaged over the last 2 s
    frames = list(read_bfa_frames())
    edges = np.arange(-2.0, 4.5, 0.5)
    bm = np.zeros(len(edges) - 1)
    bs = np.zeros(len(edges) - 1)
    used = 0
    for f in frames[-30:]:
        m, s = y_profile(f["pos"], f["vel"], DEM_MASS, edges)
        bm += m
        bs += s
        used += 1
    bm /= used
    bs /= used

    print("\n" + "=" * 78)
    print("MPM runs")
    print("=" * 78)
    results = []
    for run in runs:
        h = read_mpm_history(run)
        import json
        meta = json.load(open(os.path.join(run, "run.json")))
        st = h["t"] >= 3.0
        print(f"\n  {run}")
        print(f"    voxel {meta['voxel'] * 1e3:.1f} mm, spacing {meta['spacing'] * 1e3:.2f} mm, "
              f"dt {meta['sim_dt'] * 1e3:.3f} ms, grain scale {meta['grain_scale']}")
        if st.sum():
            print(f"    holdup (t>3 s)   {h['mass'][st].mean():.2f} +/- {h['mass'][st].std():.2f} kg"
                  f"   ({h['mass'][st].mean() / his[steady, 1].mean() * 100:.0f}% of DEM)")
            print(f"    kinetic energy   {h['ke'][st].mean():.1f} +/- {h['ke'][st].std():.1f} J"
                  f"   ({h['ke'][st].mean() / his[steady, 2].mean() * 100:.0f}% of DEM)")
        if "tube_speed_ms" in open(os.path.join(run, "history.csv")).readline():
            import csv as _csv
            rows = list(_csv.DictReader(open(os.path.join(run, "history.csv"))))
            tmv = np.array([(float(r["tube_mass_kg"]), float(r["tube_speed_ms"])) for r in rows])
            last = tmv[-min(30, len(tmv)):]
            print(f"    chute tube       {last[:, 0].mean():.2f} kg at {last[:, 1].mean():.2f} m/s"
                  f"   (DEM {dem_tube[0]:.2f} kg at {dem_tube[1]:.2f} m/s)")
        print(f"    reached t = {h['t'][-1]:.2f} s, discharged {h['out'][-1]:.1f} kg, "
              f"wall clock {h['wall'][-1]:.0f} s")

        frs = sorted(f for f in os.listdir(run) if f.endswith("_particles.vtk"))
        mm_ = np.zeros(len(edges) - 1)
        ms_ = np.zeros(len(edges) - 1)
        used = 0
        for fn in frs[-30:]:
            p, v = read_mpm_frame(os.path.join(run, fn))
            m, s = y_profile(p, v, meta["point_mass"], edges)
            mm_ += m
            ms_ += s
            used += 1
        if used:
            results.append((run, mm_ / used, ms_ / used))

    print("\n" + "=" * 78)
    print("Steady-state vertical profile (average of the last 2 s)")
    print("=" * 78)
    hdr = f"{'y band (m)':>14} | {'DEM kg':>7} {'DEM m/s':>8} |"
    for run, _, _ in results:
        hdr += f" {os.path.basename(run)[:16]:>16} kg   m/s |"
    print(hdr)
    mmax = max([bm.max()] + [r[1].max() for r in results]) if results else bm.max()
    for k in range(len(edges) - 2, -1, -1):
        line = f"{edges[k]:6.1f}..{edges[k + 1]:5.1f} | {bm[k]:7.2f} {bs[k]:8.2f} |"
        for _, m, s in results:
            line += f" {m[k]:16.2f} {s[k]:5.2f} |"
        print(line + "  " + bar(bm[k], mmax, 14))
    print(f"{'total':>14} | {bm.sum():7.2f} {'':>8} |" +
          "".join(f" {m.sum():16.2f} {'':>5} |" for _, m, _ in results))


if __name__ == "__main__":
    main()
