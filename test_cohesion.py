#!/usr/bin/env python3
"""Checks for cohesion in SolverGranularDEM, against closed forms.

Cohesion c is a fraction of grain weight (BulkFlowAnalyst's convention): touching grains
also pull together with F_c = c m* g (g = 9.81, m* = 2 mi mj / (mi + mj) = m for equal
grains; a grain on a wall: its own m).  With Hertz repulsion F_h = (4/3) E* sqrt(R*) d^1.5:

  1. grain-grain, no gravity: two touching grains settle where the two balance,
     d_eq = (3 c m g / (4 E* sqrt(R*)))^(2/3)   (R* = r/2, grain-grain E*)
  2. grain-wall, no gravity: the same with the wall's E* and R* = r
  3. a grain under a ceiling, with gravity, stays stuck iff c > 1: at 1.1 it stays, at
     0.9 it falls
  4. cohesion 0: nothing changes (no pull, grains do not stick)

    .venv/bin/python test_cohesion.py
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
E_PP = E / (2.0 * (1.0 - NU * NU))
E_W = E / (1.0 - NU * NU)
ok = True


def check(name, cond, detail):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok &= bool(cond)


def make(positions, plane_y=None, k_pp=0.0, k_w=0.0, gravity=0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=gravity)
    if plane_y is None:
        v = np.array([[50, 50, 50], [51, 50, 50], [50, 50, 51]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
    else:
        v = np.array([[-1, plane_y, -1], [1, plane_y, -1], [1, plane_y, 1], [-1, plane_y, 1]],
                     dtype=np.float64)
        f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    parts = [("p", v, f)]
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()), key="p")
    for p in positions:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=M, radius=R)
    model = builder.finalize(device=DEV)
    model.particle_ke, model.particle_kd, model.particle_kf, model.particle_mu = 2e4, 0.1, 3.0, 0.5
    model.particle_max_velocity = 30.0
    col, meshes = build_collider(parts, two_sided=[True], friction=[0.5], ke=[2e4], kd=[0.1],
                                 kf=[3.0], thickness=[0.0], max_dist=0.03, device=DEV)
    wg, _ = build_wall_grid(parts, meshes, col.lower, col.upper, R + 1.1e-3, 0.01, DEV)
    sv = SolverGranularDEM(model, col, grid_cell=2.0 * R, keepalive=meshes, wall_grid=wg,
                           rotation=True, hertz=True, youngs=E, poisson=NU, restitution=0.1,
                           cohesion=k_pp, wall_cohesion=[k_w])
    return model, sv, model.state()


def run(model, sv, st, seconds, dt=2.0e-6):
    for _ in range(int(seconds / dt)):
        sv.step(st, st, None, None, dt)
    return st.particle_q.numpy()


# 1. grain-grain equilibrium overlap
c = 3.0
d_eq = (3.0 * c * M * 9.81 / (4.0 * E_PP * math.sqrt(R / 2))) ** (2.0 / 3.0)
model, sv, st = make([[0.0, 0.0, 0.0], [2 * R - 1e-7, 0.0, 0.0]], k_pp=c)
q = run(model, sv, st, 0.05)
d = 2 * R - np.linalg.norm(q[1] - q[0])
check("grain-grain equilibrium overlap", abs(d / d_eq - 1) < 0.02,
      f"{d*1e6:.2f} um vs closed form {d_eq*1e6:.2f} um")

# 2. grain-wall equilibrium overlap (plate at y = 0, grain above it)
d_eq_w = (3.0 * c * M * 9.81 / (4.0 * E_W * math.sqrt(R))) ** (2.0 / 3.0)
model, sv, st = make([[0.0, R - 1e-7, 0.0]], plane_y=0.0, k_w=c)
q = run(model, sv, st, 0.05)
d = R - q[0][1]
check("grain-wall equilibrium overlap", abs(d / d_eq_w - 1) < 0.02,
      f"{d*1e6:.2f} um vs closed form {d_eq_w*1e6:.2f} um")

# 3. hanging under a ceiling: the pull holds the grain's weight iff c > 1
for cc, should_stick in ((1.1, True), (0.9, False)):
    model, sv, st = make([[0.0, -R + 1e-7, 0.0]], plane_y=0.0, k_w=cc, gravity=-9.81)
    q = run(model, sv, st, 0.05)
    stuck = q[0][1] > -R - 0.001
    check(f"hangs under a ceiling at cohesion {cc:.1f}",
          stuck == should_stick, f"{'stuck' if stuck else 'fell'} (expected "
          f"{'stuck' if should_stick else 'fell'}), y = {q[0][1]*1e3:.2f} mm")

# 4. no cohesion: the hanging grain falls
model, sv, st = make([[0.0, -R + 1e-7, 0.0]], plane_y=0.0, gravity=-9.81)
q = run(model, sv, st, 0.05)
check("cohesion 0: the hanging grain falls", q[0][1] < -R - 0.001, f"y = {q[0][1]*1e3:.2f} mm")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
