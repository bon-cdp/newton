#!/usr/bin/env python3
"""
A/B test: stock Newton DEM vs SolverGranularDEM on the BFA chute geometry.

Drops a non-overlapping lattice of 12 mm grains into the bottom cascade -- the region
where the stock contact generator was measured to invert 44% of Mid's normals and 15%
of Def's -- and counts how many are still inside the geometry afterwards.

An inverted contact normal makes `fn = n*c*ke` push the grain into the wall instead of
out of it, so failures show up as particles leaving the geometry envelope, usually at
high speed.

Usage:  python test_granular_dem.py [stock|new|both]
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import newton
from bfa_replication_mpm import COLLIDER_PARTS, load_part
from granular_dem import SolverGranularDEM, build_collider

wp.init()

RADIUS = 0.006
MASS = 0.8994e-3
KE, KD, KF, MU_WALL = 1.0e4, 5.0, 5.0, 0.5
DT = 1.0e-5
STEPS = 12000


def seed_lattice(lo, hi, spacing):
    """Non-overlapping seed points -- uniform random seeding makes DEM explode."""
    ax = np.arange(lo[0], hi[0], spacing)
    ay = np.arange(lo[1], hi[1], spacing)
    az = np.arange(lo[2], hi[2], spacing)
    return np.array([[x, y, z] for x in ax for y in ay for z in az])


def build(device):
    parts = [(n, *load_part(n, fx, fl)) for n, ts, fx, fl in COLLIDER_PARTS]
    allv = np.vstack([v for _n, v, _f in parts])
    return parts, allv.min(0), allv.max(0)


def run(which: str, device="cuda:0"):
    parts, lo, hi = build(device)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    for name, v, f in parts:
        builder.add_shape_mesh(
            body=-1, mesh=newton.Mesh(v, f.flatten()),
            cfg=newton.ModelBuilder.ShapeConfig(mu=MU_WALL, ke=KE, kd=KD, kf=KF), key=name)

    # a block of grains sitting just above the cascade assembly
    pts = seed_lattice((-5.52, -1.42, 4.02), (-5.30, -1.20, 4.16), 0.016)
    for p in pts:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=MASS, radius=RADIUS)

    model = builder.finalize(device=device)
    model.particle_ke, model.particle_kd = KE, KD
    model.particle_kf, model.particle_mu = KF, 0.43
    model.particle_cohesion, model.particle_adhesion = 0.0, 0.0
    model.set_gravity((0.0, -9.81, 0.0))
    model.particle_grid = wp.HashGrid(64, 64, 64, device=device)

    n = model.particle_count
    if which == "new":
        collider, meshes = build_collider(
            parts,
            two_sided=[p[1] for p in COLLIDER_PARTS],
            friction=[MU_WALL] * len(parts),
            ke=[KE] * len(parts), kd=[KD] * len(parts), kf=[KF] * len(parts),
            thickness=[0.002 if p[1] else 0.0 for p in COLLIDER_PARTS],
            max_dist=0.05, device=device)
        solver = SolverGranularDEM(model, collider, grid_cell=0.024, keepalive=meshes)
        pipeline = None
    else:
        solver = newton.solvers.SolverSemiImplicit(model)
        pipeline = newton.CollisionPipeline.from_model(model, soft_contact_max=n * 8)

    s0, s1 = model.state(), model.state()
    print(f"\n--- {which}: {n} grains, dt {DT*1e6:.0f} us, {STEPS} steps")
    print(f"{'step':>7} {'t (ms)':>8} {'inside':>8} {'escaped':>8} {'max|v|':>9} {'mean|v|':>9} {'mean y':>9}")

    for step in range(STEPS + 1):
        if pipeline is not None:
            model.particle_grid.build(s0.particle_q, 0.024)
            contacts = pipeline.collide(model, s0)
            s0.clear_forces()
            solver.step(s0, s1, None, contacts, DT)
        else:
            solver.step(s0, s1, None, None, DT)
        s0, s1 = s1, s0

        if step % 1500 == 0:
            q = s0.particle_q.numpy()
            v = s0.particle_qd.numpy()
            esc = ((q[:, 2] < lo[2] - 0.01) | (q[:, 2] > hi[2] + 0.01) |
                   (q[:, 0] < lo[0] - 0.01) | (q[:, 0] > hi[0] + 0.01) |
                   (q[:, 1] > hi[1] + 0.01)).sum()
            sp = np.linalg.norm(v, axis=1)
            print(f"{step:7d} {step*DT*1e3:8.1f} {n-esc:8d} {esc:8d} "
                  f"{sp.max():9.2f} {sp.mean():9.3f} {q[:,1].mean():9.3f}")

    q = s0.particle_q.numpy()
    esc = ((q[:, 2] < lo[2] - 0.01) | (q[:, 2] > hi[2] + 0.01) |
           (q[:, 0] < lo[0] - 0.01) | (q[:, 0] > hi[0] + 0.01) |
           (q[:, 1] > hi[1] + 0.01)).sum()
    sp = np.linalg.norm(s0.particle_qd.numpy(), axis=1)
    print(f"    -> {esc}/{n} escaped ({100*esc/n:.1f}%);  final speed mean {sp.mean():.3f} "
          f"max {sp.max():.2f} m/s  (a settled pile should be near zero)")
    return esc, n, float(sp.mean()), float(sp.max())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    res = {}
    for m in (["stock", "new"] if mode == "both" else [mode]):
        res[m] = run(m)
    if len(res) == 2:
        print("\n" + "=" * 56)
        for m, (e, n, sm, sx) in res.items():
            print(f"  {m:>6}: {e:5d}/{n} escaped ({100*e/n:5.1f}%)   "
                  f"final speed mean {sm:6.3f}  max {sx:7.2f} m/s")
