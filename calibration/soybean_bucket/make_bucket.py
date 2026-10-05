#!/usr/bin/env python3
"""
Build the DEM scenario for one soybean bucket-discharge test (runs.json).

Geometry (metres, y up, belt surface at y = 0):
  wall    open cylinder, 190 mm dia x 240 mm, from the bucket bottom up
  bottom  the bottom plate with the drilled hole (hole_contour.json, traced from the photo)
  plug    a disc under the hole, switched off at t_open (the pulled paper towel)
  belt    a 1.2 m square plane at y = 0 (a stopped conveyor belt)

Sequence: fill from a box inside the bucket at fill_rate kg/s until m0 is in, let it
settle, pull the plug at t_open, run until t_open + discharge_s.  history.csv carries the
bucket mass (region "bucket") every frame.

    .venv/bin/python calibration/soybean_bucket/make_bucket.py run1 --out runs/calib/x \
        --set friction=0.4 rolling_friction=0.1
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from dem_scenario import (Injector, Material, Output, Part, Region, Scenario,  # noqa: E402
                          Solver)

R_BUCKET = 0.095
H_BUCKET = 0.240
BELT_HALF = 0.6

# Starting point for soybean (literature ranges; the calibration moves the frictions).
# Material fields go straight into dem_scenario.Material; wall_friction / belt_friction
# are the per-part Coulomb coefficients.
DEFAULTS = dict(
    radius=0.0025, density=1200.0, youngs=1.0e8, poisson=0.35, restitution=0.5,
    friction=0.35, rolling_friction=0.05, wall_rolling_friction=0.05,
    tangential_ratio=1.0, contact="hertz",
    wall_friction=0.30, belt_friction=0.50,
    # REFERENCE preset (true E, dt 0.35 Rayleigh, no neighbour list): the fast preset
    # (E/10, Verlet list) was seen to throw stray grains
    youngs_divisor=1.0, neighbor_every=0, vtk=False,
    fill_rate=4.0, t_open=1.0, discharge_s=15.0, fps=30.0,
)
MATERIAL_KEYS = {f for f in Material.__dataclass_fields__}

# Multi-sphere soybean shapes (dem_clumps.py / granular_clumps.py, issue #10), selected with
# the "clump" parameter.  Soybeans are near-ellipsoids ~6 x 5.3 x 4.8 mm for this size;
# the video measured 5.0-5.4 mm projected.  Offsets and radii in mm.
CLUMPS = {
    # two spheres along the long axis: 6.1 x 4.9 x 4.9 mm (aspect 1.24)
    "soy2": {"offsets_mm": [[-0.6, 0, 0], [0.6, 0, 0]], "radii_mm": [2.45, 2.45]},
    # unequal pair (one end slightly smaller, like a bean's germ end): 6.05 x 5.1 x 5.1 mm
    "soy2a": {"offsets_mm": [[-0.6, 0, 0], [0.6, 0, 0]], "radii_mm": [2.55, 2.3]},
    # flatter, longer: 6.4 x 4.6 x 4.6 mm (aspect 1.39)
    "soy2b": {"offsets_mm": [[-0.9, 0, 0], [0.9, 0, 0]], "radii_mm": [2.3, 2.3]},
    # triaxial: 4 spheres in a flat rhombus, ~6.2 x 5.4 x 4.6 mm
    "soy4": {"offsets_mm": [[-0.8, 0, 0], [0.8, 0, 0], [0, 0.4, 0], [0, -0.4, 0]],
             "radii_mm": [2.3, 2.3, 2.3, 2.3]},
}


def load_runs():
    return json.load(open(os.path.join(HERE, "runs.json")))


def bucket_meshes(h):
    """(wall, bottom, plug) trimeshes for a bucket whose bottom is at height h."""
    hole = json.load(open(os.path.join(HERE, "hole_contour.json")))
    q = np.asarray(hole["contour_mm"]) * 1e-3              # (x, z) in the plate plane
    c = q.mean(axis=0)
    d = q - c
    ang = np.arctan2(d[:, 1], d[:, 0])
    order = np.argsort(ang)                                 # rays sorted by angle
    q, d, ang = q[order], d[order], ang[order]
    rho = np.linalg.norm(d, axis=1)
    u = d / rho[:, None]
    # where each ray from the hole centroid meets the bucket circle
    b = u @ c
    t_out = -b + np.sqrt(b * b - (c @ c - R_BUCKET ** 2))
    n_ray, n_ring = len(q), 16
    rings = []
    for k in range(n_ring + 1):
        s = rho + (t_out - rho) * (k / n_ring)
        rings.append(c + u * s[:, None])
    rings = np.array(rings)                                 # (ring, ray, 2)
    xz = rings.reshape(-1, 2)
    v_bot = np.stack([xz[:, 0], np.full(len(xz), h), xz[:, 1]], 1)
    f_bot = []
    for k in range(n_ring):
        for i in range(n_ray):
            j = (i + 1) % n_ray
            a, bb = k * n_ray + i, k * n_ray + j
            cc, dd = (k + 1) * n_ray + j, (k + 1) * n_ray + i
            f_bot += [[a, bb, cc], [a, cc, dd]]
    bottom = trimesh.Trimesh(v_bot, np.array(f_bot), process=False)

    # wall shares the plate's outer ring so plate and wall meet without a crack
    outer = rings[-1]
    n_lev = 10
    v_wall = np.concatenate([np.stack([outer[:, 0], np.full(n_ray, h + H_BUCKET * L / n_lev),
                                       outer[:, 1]], 1) for L in range(n_lev + 1)])
    f_wall = []
    for L in range(n_lev):
        for i in range(n_ray):
            j = (i + 1) % n_ray
            a, bb = L * n_ray + i, L * n_ray + j
            cc, dd = (L + 1) * n_ray + j, (L + 1) * n_ray + i
            f_wall += [[a, bb, cc], [a, cc, dd]]
    wall = trimesh.Trimesh(v_wall, np.array(f_wall), process=False)

    # plug: a disc 1 mm under the plate, covering the hole with margin
    r_plug = float(rho.max()) + 0.008
    n = 48
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_plug = np.vstack([[c[0], h - 0.001, c[1]],
                        np.stack([c[0] + r_plug * np.cos(th), np.full(n, h - 0.001),
                                  c[1] + r_plug * np.sin(th)], 1)])
    f_plug = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    plug = trimesh.Trimesh(v_plug, np.array(f_plug), process=False)
    return wall, bottom, plug


def belt_mesh(n=24):
    g = np.linspace(-BELT_HALF, BELT_HALF, n + 1)
    X, Z = np.meshgrid(g, g, indexing="ij")
    v = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    f = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            f += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return trimesh.Trimesh(v, np.array(f), process=False)


def make(run_id, out_dir, params=None):
    p = dict(DEFAULTS)
    p.update(params or {})
    r = load_runs()["runs"][run_id]
    h = r["drop_mm"] * 1e-3
    m0 = r["m0_g"] * 1e-3
    os.makedirs(out_dir, exist_ok=True)
    wall, bottom, plug = bucket_meshes(h)
    for name, m in (("wall", wall), ("bottom", bottom), ("plug", plug), ("belt", belt_mesh())):
        m.export(os.path.join(out_dir, f"{name}.stl"))

    mat = Material(name="soybean", **{k: v for k, v in p.items() if k in MATERIAL_KEYS})
    t_fill = m0 / p["fill_rate"]
    if t_fill > 0.7 * p["t_open"]:
        raise ValueError(f"fill takes {t_fill:.2f} s: raise fill_rate or t_open")
    parts = [
        Part("wall", "wall.stl", friction=p["wall_friction"]),
        Part("bottom", "bottom.stl", friction=p["wall_friction"]),
        Part("plug", "plug.stl", friction=p["wall_friction"], active=[0.0, p["t_open"]]),
        Part("belt", "belt.stl", friction=p["belt_friction"]),
    ]
    inj = Injector(
        # small, frequent batches: the fill overshoots by at most one batch (~12 g)
        name="fill", mass_rate=p["fill_rate"], velocity=[0.0, -0.5, 0.0], start=0.0, stop=t_fill,
        max_interval=0.003,
        box={"reference": "plane", "plane_height": h, "center": [0.0, 0.0, 0.0],
             "clearance": 0.10, "length": 0.12, "width": 0.12, "height": 0.10})
    sc = Scenario(
        name=f"soy_{run_id}", material=mat, parts=parts, injectors=[inj],
        domain_lo=[-0.8, -0.2, -0.8], domain_hi=[0.8, h + H_BUCKET + 0.15, 0.8],
        regions=[Region("bucket", lo=[-0.1, h - 0.0005, -0.1], hi=[0.1, h + H_BUCKET + 0.1, 0.1]),
                 Region("belt", lo=[-BELT_HALF, -0.05, -BELT_HALF], hi=[BELT_HALF, h - 0.0005, BELT_HALF])],
        solver=Solver(youngs_divisor=p["youngs_divisor"], neighbor_every=p["neighbor_every"],
                      expected_holdup_kg=m0),
        output=Output(duration=p["t_open"] + p["discharge_s"], fps=p["fps"], vtk=bool(p["vtk"])),
        notes=dict(run=run_id, params=p, measured=r),
    )
    if p.get("clump"):
        # dem_clumps.py: the fill is placed at t = 0 as a lattice of clumps in the bucket
        sc.notes["clump"] = CLUMPS[p["clump"]]
        sc.notes["fill"] = {"shape": "cylinder", "center": [0.0, 0.0], "radius": R_BUCKET,
                            "y0": h, "mass": m0, "clearance": 0.0015}
        sc.domain_lo = [-0.8, -0.15, -0.8]     # bodies falling off the belt are retired here
    path = os.path.join(out_dir, "scenario.json")
    sc.save(path)
    return path


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = float(v) if any(ch in v for ch in ".e") else int(v)
        except ValueError:
            pass
        out[k] = v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", help="key=value overrides of DEFAULTS")
    a = ap.parse_args()
    print(make(a.run, a.out, parse_set(a.set)))
