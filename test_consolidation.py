#!/usr/bin/env python3
"""Checks for consolidation strength (simplified EEPA) in SolverGranularDEM.

A contact remembers the largest elastic normal force it has carried, F_max, and holds with
lam * F_max while it lasts.  So a grain pressed hard against a ceiling (gravity turned
UP, k times normal) and then let go (normal gravity, down) stays stuck iff lam * k > 1:

  1. wall, lam 0.2, pressed at 20 g: holds 4 x its weight -> stays
  2. wall, lam 0.2, pressed at  2 g: holds 0.4 x its weight -> falls
  3. wall, lam 0: falls however hard it was pressed
  4-6. the same with a grain pressed onto a fixed grain and then pulled away from it

    .venv/bin/python test_consolidation.py
"""
import math
import sys

import numpy as np
import warp as wp

import newton
from granular_dem import SolverGranularDEM, build_collider, build_wall_grid

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
R, RHO, E, NU = 0.005, 1000.0, 1.0e7, 0.3
M = RHO * 4.0 / 3.0 * math.pi * R ** 3
DT = 2.0e-6
ok = True


def check(name, cond, detail):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok &= bool(cond)


def make(particles, plane_y=None, lam=0.0, lam_w=0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    if plane_y is None:
        v = np.array([[50, 50, 50], [51, 50, 50], [50, 50, 51]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
    else:
        v = np.array([[-1, plane_y, -1], [1, plane_y, -1], [1, plane_y, 1], [-1, plane_y, 1]], dtype=np.float64)
        f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    parts = [("p", v, f)]
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()), key="p")
    for pos, mass in particles:
        builder.add_particle(pos=wp.vec3(*pos), vel=wp.vec3(0.0), mass=mass, radius=R)
    model = builder.finalize(device=DEV)
    model.particle_ke, model.particle_kd, model.particle_kf, model.particle_mu = 2e4, 0.1, 3.0, 0.5
    model.particle_max_velocity = 30.0
    col, meshes = build_collider(parts, two_sided=[True], friction=[0.5], ke=[2e4], kd=[0.1],
                                 kf=[3.0], thickness=[0.0], max_dist=0.03, device=DEV)
    wg, _ = build_wall_grid(parts, meshes, col.lower, col.upper, R + 1.1e-3, 0.01, DEV)
    sv = SolverGranularDEM(model, col, grid_cell=2.0 * R, keepalive=meshes, wall_grid=wg,
                           rotation=True, hertz=True, youngs=E, poisson=NU, restitution=0.1,
                           consolidation=lam, wall_consolidation=[lam_w])
    return model, sv, model.state()


def press_then_release(model, sv, st, g_press, seconds=0.04):
    """Gravity g_press (signed, y) for `seconds`, then normal gravity (-9.81) as long."""
    for g in (g_press, -9.81):
        model.set_gravity((0.0, g, 0.0))
        for _ in range(int(seconds / DT)):
            sv.step(st, st, None, None, DT)
    return st.particle_q.numpy()


# wall: a grain just under a ceiling at y = 0, pressed up into it, then let go
for lam, k, expect in ((0.2, 20.0, True), (0.2, 2.0, False), (0.0, 20.0, False)):
    model, sv, st = make([([0.0, -R + 1e-7, 0.0], M)], plane_y=0.0, lam_w=lam)
    q = press_then_release(model, sv, st, +9.81 * k)
    stuck = q[0][1] > -R - 0.001
    check(f"wall: lam {lam:g}, pressed at {k:g} g -> {'stays' if expect else 'falls'}",
          stuck == expect, f"{'stuck' if stuck else 'fell'}, y = {q[0][1]*1e3:.2f} mm (lam x k = {lam*k:g})")

# pair: a grain on top of a fixed grain (mass 0), pressed down onto it, then pulled up
for lam, k, expect in ((0.2, 20.0, True), (0.2, 2.0, False), (0.0, 20.0, False)):
    model, sv, st = make([([0.0, 0.0, 0.0], 0.0), ([0.0, 2 * R - 1e-7, 0.0], M)], lam=lam)
    model.set_gravity((0.0, -9.81 * k, 0.0))
    for _ in range(int(0.04 / DT)):
        sv.step(st, st, None, None, DT)
    model.set_gravity((0.0, +9.81, 0.0))             # now pulling it away
    for _ in range(int(0.04 / DT)):
        sv.step(st, st, None, None, DT)
    q = st.particle_q.numpy()
    gap = np.linalg.norm(q[1] - q[0]) - 2 * R
    stuck = gap < 0.001
    check(f"pair: lam {lam:g}, pressed at {k:g} g -> {'stays' if expect else 'pulls off'}",
          stuck == expect, f"{'stuck' if stuck else 'pulled off'}, gap {gap*1e3:.2f} mm (lam x k = {lam*k:g})")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
