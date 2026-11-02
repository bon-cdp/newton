#!/usr/bin/env python3
"""Checks for moving parts (motion manager, #14) in SolverGranularDEM.

  1. kinematics: motion_pose / motion_tables against closed forms -- a translation moves
     the part v t, a rotation of 90 deg about a world line sends a point where it should,
     and the tabulated wall velocity v + w x (y - T) matches the motion over the step
  2. a grain resting on a plate lifted at 0.1 m/s rises with it (stays seated on it)
  3. a plate sliding sideways at 0.2 m/s sets a grain rolling: it ends at 2/7 of the
     plate's speed (rolling without slip, the textbook result for a free sphere)
  4. a turntable rotating at 30 deg/s about a vertical line carries a grain round at 2/7
     of its rate (the same rolling result)

    .venv/bin/python test_moving.py
"""
import math
import sys

import numpy as np
import warp as wp

import newton
from dem_run import motion_pose, motion_tables
from granular_dem import SolverGranularDEM, build_collider, build_wall_grid, set_part_poses

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


# 1. kinematics ------------------------------------------------------------------------------
mo = {"type": "moving", "steps": [
    {"kind": "translate", "start": 0.5, "end": 1.5, "velocity": [0.0, 0.2, 0.0]},
    {"kind": "rotate", "start": 2.0, "end": 5.0, "point": [1.0, 0.0, 0.0], "axis": [0, 0, 1],
     "deg_per_s": 30.0}]}
Rm, Tm = motion_pose(mo, np.array([0.0, 1.0, 1.5, 5.0]))
x = np.array([2.0, 0.0, 0.0])
y = np.einsum("nij,j->ni", Rm, x) + Tm
check("translation", np.allclose(y[:3], [[2, 0, 0], [2, 0.1, 0], [2, 0.2, 0]]),
      f"{np.round(y[:3], 6).tolist()}")
# after the lift, 90 deg about z through (1,0,0): (2, 0.2) -> (1 - 0.2, 1)
check("rotation about a world line", np.allclose(y[3], [0.8, 1.0, 0.0], atol=1e-9),
      f"{np.round(y[3], 6).tolist()} vs [0.8, 1.0, 0.0]")
Rt, Tt, vt, wt = motion_tables(mo, 3000, 1e-3)
k = 2500                                         # t = 2.5 s, rotating
yk = Rt[k] @ x + Tt[k]
y1 = Rt[k + 1] @ x + Tt[k + 1]
v_pt = vt[k] + np.cross(wt[k], yk - Tt[k])
# the chord over the step differs from v + w x r by the centripetal term w^2 r dt / 2
# (2.7e-4 m/s at this coarse 1 ms step; ~1e-6 at a run's 10 us)
check("wall velocity = motion over the step", np.allclose(v_pt, (y1 - yk) / 1e-3, atol=(math.radians(30) ** 2 * 2.0 * 1e-3 / 2) * 1.2),
      f"{np.round(v_pt, 6).tolist()} vs {np.round((y1 - yk) / 1e-3, 6).tolist()}")


# contact helpers ----------------------------------------------------------------------------
def plate(y0, half=0.2):
    v = np.array([[-half, y0, -half], [half, y0, -half], [half, y0, half], [-half, y0, half]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return v, f


def make(motion, positions, friction=0.5, seconds=0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    far_v = np.array([[50, 50, 50], [51, 50, 50], [50, 50, 51]], dtype=np.float64)   # static dummy
    far_f = np.array([[0, 1, 2]], dtype=np.int32)
    pv, pf = plate(0.0)
    parts = [("far", far_v, far_f), ("plate", pv, pf)]
    for name, v, f in parts:
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()), key=name)
    for p in positions:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=M, radius=R)
    model = builder.finalize(device=DEV)
    model.particle_ke, model.particle_kd, model.particle_kf, model.particle_mu = 2e4, 0.1, 3.0, 0.5
    model.particle_max_velocity = 30.0
    col, meshes = build_collider(parts, two_sided=[True, True], friction=[0.5, friction], ke=[2e4] * 2,
                                 kd=[0.1] * 2, kf=[3.0] * 2, thickness=[0.0, 0.0], max_dist=0.03,
                                 device=DEV, motions=[None, (3, (0, 0, 0), (0, 0, 0), 0.0)])
    lo, hi = col.lower.numpy(), col.upper.numpy()

    def sub(idx):
        return build_wall_grid([parts[k] for k in idx], [meshes[k] for k in idx],
                               wp.array(lo[idx], dtype=wp.vec3, device=DEV),
                               wp.array(hi[idx], dtype=wp.vec3, device=DEV),
                               R + 1.1e-3, 0.01, DEV, part_ids=idx)[0]
    sv = SolverGranularDEM(model, col, grid_cell=2.0 * R, keepalive=meshes, wall_grid=sub([0]),
                           rotation=True, hertz=True, youngs=E, poisson=NU, restitution=0.3)
    sv.set_moving([sub([1])], wp.array(np.eye(3, dtype=np.float32)[None], dtype=wp.mat33, device=DEV),
                  wp.zeros(1, dtype=wp.vec3, device=DEV))
    n = int(seconds / DT)
    tb = motion_tables(motion, n, DT)
    tabs = [wp.array(tb[j][:, None].astype(np.float32), dtype=wp.mat33 if j == 0 else wp.vec3, device=DEV)
            for j in range(4)]
    part = wp.array([1], dtype=int, device=DEV)
    st = model.state()
    off = sv.step_count
    for _ in range(n):
        wp.launch(set_part_poses, dim=1, device=DEV, inputs=[sv.step_arr, off, *tabs, part, sv.pose_R,
                                                              sv.pose_T, sv.collider])
        sv.step(st, st, None, None, DT)
    return st.particle_q.numpy(), st.particle_qd.numpy(), tb


# 2. lift --------------------------------------------------------------------------------------
lift = {"type": "moving", "steps": [{"kind": "translate", "start": 0.02, "end": 1.0, "velocity": [0, 0.1, 0]}]}
q, qd, _tb = make(lift, [[0.0, R - 2e-6, 0.0]], seconds=0.12)
y_plate = 0.1 * (0.12 - 0.02)
check("grain rides a lifted plate", abs(q[0][1] - (y_plate + R)) < 2e-4 and abs(qd[0][1] - 0.1) < 0.01,
      f"y {q[0][1]*1e3:.2f} mm vs plate + r {(y_plate + R)*1e3:.2f} mm, vy {qd[0][1]:.3f} m/s")

# 3. sliding plate carries the grain by friction ---------------------------------------------
slide = {"type": "moving", "steps": [{"kind": "translate", "start": 0.02, "end": 1.0, "velocity": [0.2, 0, 0]}]}
q, qd, _tb = make(slide, [[0.0, R - 2e-6, 0.0]], seconds=0.15)
# a free sphere set rolling by a surface moving at V ends rolling without slip at 2/7 V
check("sliding plate carries the grain (rolling: 2/7 of its speed)", abs(qd[0][0] - 0.2 * 2 / 7) < 0.003,
      f"vx {qd[0][0]:.4f} m/s vs 2/7 x 0.2 = {0.2 * 2 / 7:.4f} m/s")

# 4. turntable -------------------------------------------------------------------------------
spin = {"type": "moving", "steps": [{"kind": "rotate", "start": 0.0, "end": 1.0, "point": [0, 0, 0],
                                     "axis": [0, 1, 0], "deg_per_s": 30.0}]}
r0 = 0.1
q, qd, _tb = make(spin, [[r0, R - 2e-6, 0.0]], seconds=0.2)
ang = math.degrees(math.atan2(-q[0][2], q[0][0]))          # right-hand about +y: x -> -z
# rolling again: a ball on a turntable circles at 2/7 of the table's rate
check("turntable carries the grain round (rolling: 2/7 of its rate)", abs(ang - 6.0 * 2 / 7) < 0.1,
      f"{ang:.3f} deg vs 2/7 x 6 = {6.0 * 2 / 7:.3f} deg, radius {math.hypot(q[0][0], q[0][2])*1e3:.1f} mm")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
