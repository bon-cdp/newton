#!/usr/bin/env python3
"""Checks for granular_clumps.py (issue #10).

  1. template mass properties: one sphere (m, I = 0.4 m r^2) and two touching spheres
     (mass = 2 spheres, I about the long axis = 2 x 0.4 m r^2, transverse by parallel axes)
  2. torque-free asymmetric clump in zero gravity: angular momentum exact, kinetic
     energy drift small over 0.5 s
  3. a 2-sphere clump dropped flat onto a plane comes to rest at its radius above it

    .venv/bin/python test_clumps.py
"""
import math
import sys

import numpy as np
import warp as wp

import newton
from granular_clumps import ClumpTemplate, SolverGranularClumps, quat_to_R
from granular_dem import build_collider, build_wall_grid

wp.init()
wp.set_module_options({"enable_backward": False})
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
RHO = 1200.0
ok = True


def check(name, cond, detail):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok &= bool(cond)


# 1. mass properties --------------------------------------------------------------
r = 0.0025
t1 = ClumpTemplate([[0, 0, 0]], [r], RHO, voxels=128)
m1 = RHO * 4 / 3 * math.pi * r ** 3
check("sphere mass", abs(t1.mass / m1 - 1) < 0.01, f"{t1.mass/m1:.4f} of analytic")
check("sphere inertia", np.all(np.abs(t1.inertia / (0.4 * m1 * r * r) - 1) < 0.02),
      f"{(t1.inertia/(0.4*m1*r*r)).round(4)}")
t2 = ClumpTemplate([[-r, 0, 0], [r, 0, 0]], [r, r], RHO, voxels=128)
i_long = 2 * 0.4 * m1 * r * r
i_tr = 2 * (0.4 * m1 * r * r + m1 * r * r)
check("touching pair mass", abs(t2.mass / (2 * m1) - 1) < 0.01, f"{t2.mass/(2*m1):.4f}")
check("touching pair inertia", abs(t2.inertia[0] / i_long - 1) < 0.02 and
      np.all(np.abs(t2.inertia[1:] / i_tr - 1) < 0.02), f"{t2.inertia.round(12)} vs {i_long:.3g}, {i_tr:.3g}")


def make(n_part, plane_y=None):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    if plane_y is None:          # a far-away triangle: the solver needs some collider
        v = np.array([[50, 50, 50], [51, 50, 50], [50, 50, 51]], dtype=np.float64)
    else:
        v = np.array([[-1, plane_y, -1], [1, plane_y, -1], [1, plane_y, 1], [-1, plane_y, 1]],
                     dtype=np.float64)
    f = np.array([[0, 1, 2]] if plane_y is None else [[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    parts = [("p", v, f)]
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()), key="p")
    for _ in range(n_part):
        builder.add_particle(pos=wp.vec3(0, 40, 0), vel=wp.vec3(0.0), mass=1e-4, radius=r, flags=0)
    model = builder.finalize(device=DEV)
    model.particle_ke, model.particle_kd, model.particle_kf, model.particle_mu = 2e4, 0.1, 3.0, 0.5
    model.particle_max_velocity = 30.0
    col, meshes = build_collider(parts, two_sided=[True], friction=[0.5], ke=[2e4], kd=[0.1],
                                 kf=[3.0], thickness=[0.0], max_dist=0.03, device=DEV)
    wg, _ = build_wall_grid(parts, meshes, col.lower, col.upper, r + 1.1e-3, 0.006, DEV)
    return model, col, meshes, wg


# 2. torque-free asymmetric body ---------------------------------------------------
tpl = ClumpTemplate([[-0.0015, 0, 0], [0.0015, 0.0008, 0], [0.0004, -0.0006, 0.0012]],
                    [r, 0.0022, 0.002], RHO)
model, col, meshes, wg = make(tpl.n)
model.set_gravity((0.0, 0.0, 0.0))
sv = SolverGranularClumps(model, col, keepalive=meshes, wall_grid=wg, rotation=True, hertz=True,
                          youngs=1e8, poisson=0.35, restitution=0.5)
st = model.state()
sv.bind_state(st)
q0 = np.array([[0.0, 0.0, 0.0, 1.0]])
sv.set_clumps(tpl, [[0, 0, 0]], q0)
w0 = np.array([0.3, 20.0, 0.4])               # near the intermediate axis: tumbles
I = tpl.inertia
sv.clump_L.assign(np.array([I * w0], dtype=np.float32))
L0 = I * w0
E0 = 0.5 * np.sum(I * w0 ** 2)
dt = 2e-5
Es = []
for k in range(25000):
    sv.step(st, st, None, None, dt)
    if k % 2500 == 0:
        q = sv.clump_q.numpy()[0]
        R = quat_to_R(q)
        L = sv.clump_L.numpy()[0]
        wb = R.T @ L / I
        Es.append(0.5 * np.sum(I * wb ** 2))
L = sv.clump_L.numpy()[0]
check("angular momentum conserved", np.allclose(L, L0, rtol=1e-5), f"{L} vs {L0}")
drift = max(abs(np.array(Es) / E0 - 1))
check("kinetic energy drift < 1% over 0.5 s", drift < 0.01, f"max {drift*100:.3f}%")
xb = st.particle_q.numpy()[: tpl.n]
q = sv.clump_q.numpy()[0]
expect = (quat_to_R(q) @ tpl.offsets.T).T
check("sub-spheres follow the body", np.allclose(xb, expect, atol=1e-6),
      f"max err {np.abs(xb-expect).max():.2e} m")

# 3. drop onto a plane ------------------------------------------------------------
tpl2 = ClumpTemplate([[-0.0006, 0, 0], [0.0006, 0, 0]], [r, r], RHO)
model, col, meshes, wg = make(tpl2.n, plane_y=0.0)
sv = SolverGranularClumps(model, col, keepalive=meshes, wall_grid=wg, rotation=True, hertz=True,
                          youngs=1e8, poisson=0.35, restitution=0.5, tangential_ratio=1.0)
st = model.state()
sv.bind_state(st)
sv.set_clumps(tpl2, [[0, 0.02, 0]], np.array([[0.0, 0.0, 0.0, 1.0]]))
for k in range(40000):                        # 0.8 s
    sv.step(st, st, None, None, dt)
y = sv.clump_x.numpy()[0][1]
v = np.linalg.norm(sv.clump_v.numpy()[0])
check("rests on the plane at its radius", abs(y - r) < 2e-4 and v < 1e-2,
      f"centre at {y*1e3:.3f} mm (radius {r*1e3} mm), speed {v:.2e} m/s")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
