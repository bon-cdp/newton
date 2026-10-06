#!/usr/bin/env python3
"""Checks for granular_ellipsoids.py.

  1. pair overlap: the normal force of eval_ellipsoid_pairs for random poses equals the
     Hertz force of the brute-force overlap min_n [h_i(n) + h_j(n) - n.(x_i - x_j)]
     (40k directions), within 2%
  2. sphere limit: a = b = c gives the sphere overlap exactly
  3. an ellipsoid dropped at a random orientation onto a plane comes to rest lying flat
     (shortest axis vertical) at height c

    .venv/bin/python test_ellipsoids.py
"""
import sys

import numpy as np
import warp as wp

import newton
from granular_clumps import quat_to_R, random_quats
from granular_dem import build_collider, build_wall_grid
from granular_ellipsoids import EllipsoidShape, SolverGranularEllipsoids, eval_ellipsoid_pairs

wp.init()
wp.set_module_options({"enable_backward": False})
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
E, NU, RHO = 1.0e8, 0.35, 1200.0
ok = True


def check(name, cond, detail):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok &= bool(cond)


def make(n, plane_y=None, mu_roll_wall=0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    if plane_y is None:
        v = np.array([[50, 50, 50], [51, 50, 50], [50, 50, 51]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
    else:
        v = np.array([[-1, plane_y, -1], [1, plane_y, -1], [1, plane_y, 1], [-1, plane_y, 1]], dtype=np.float64)
        f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    parts = [("p", v, f)]
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()), key="p")
    for _ in range(n):
        builder.add_particle(pos=wp.vec3(0, 40, 0), vel=wp.vec3(0.0), mass=1e-4, radius=0.003, flags=0)
    model = builder.finalize(device=DEV)
    model.particle_ke, model.particle_kd, model.particle_kf, model.particle_mu = 2e4, 0.1, 3.0, 0.4
    model.particle_max_velocity = 30.0
    col, meshes = build_collider(parts, two_sided=[True], friction=[0.4], ke=[2e4], kd=[0.1],
                                 kf=[3.0], thickness=[0.0], max_dist=0.03, device=DEV)
    wg, _ = build_wall_grid(parts, meshes, col.lower, col.upper, 0.0035 + 1.1e-3, 0.006, DEV)
    sv = SolverGranularEllipsoids(model, col, keepalive=meshes, wall_grid=wg, rotation=True,
                                  hertz=True, youngs=E, poisson=NU, restitution=0.5,
                                  tangential_ratio=1.0, mu_roll=0.0, mu_roll_wall=mu_roll_wall)
    st = model.state()
    sv.bind_state(st)
    return model, sv, st


def support_h(M, N):
    return np.sqrt(np.einsum("ki,ij,kj->k", N, M, N))


def brute_overlap(Mi, Mj, u, nd=40000, rng=np.random.default_rng(1)):
    N = rng.normal(size=(nd, 3))
    N /= np.linalg.norm(N, axis=1)[:, None]
    g = support_h(Mi, N) + support_h(Mj, N) - N @ u
    k = int(np.argmin(g))
    # refine around the best direction
    best = N[k]
    for scale in (0.05, 0.01, 0.002):
        M2 = best + scale * rng.normal(size=(4000, 3))
        M2 /= np.linalg.norm(M2, axis=1)[:, None]
        g2 = support_h(Mi, M2) + support_h(Mj, M2) - M2 @ u
        best = M2[int(np.argmin(g2))]
        gbest = g2.min()
    return gbest


# 1 & 2 -------------------------------------------------------------------------------
rng = np.random.default_rng(0)
e_star = E / (2.0 * (1.0 - NU * NU))
for label, axes in (("bean 3.1x2.7x2.4 mm", (0.0031, 0.0027, 0.0024)),
                    ("sphere 2.5 mm", (0.0025, 0.0025, 0.0025))):
    shp = EllipsoidShape(*axes, RHO)
    errs = []
    for trial in range(12):
        model, sv, st = make(2)
        q = random_quats(2, rng)
        dirn = rng.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        R = quat_to_R(q)
        Ms = [R[k] @ np.diag(shp.axes ** 2) @ R[k].T for k in range(2)]
        # place j along dirn at a distance giving a modest overlap (~2-6% of size)
        lo, hi = 0.0, 0.02
        for _ in range(60):          # bisection on the brute-force contact distance
            mid = 0.5 * (lo + hi)
            if brute_overlap(Ms[0], Ms[1], mid * dirn, nd=3000) > 0:
                lo = mid
            else:
                hi = mid
        dist = lo - rng.uniform(0.0001, 0.0003)
        com = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        com[0] = dist * dirn                   # u = x_i - x_j = dist * dirn
        sv.set_ellipsoids(shp, com, q)
        model.particle_grid.build(st.particle_q, sv.grid_cell)
        wp.launch(eval_ellipsoid_pairs, dim=2, inputs=[
            model.particle_grid.id, st.particle_q, st.particle_qd, sv.particle_w,
            model.particle_radius, model.particle_flags, model.particle_inv_mass, sv.clump_q,
            sv.ax2, sv.r_eq, 0.4, 0.0, 0.0, model.particle_max_radius, sv.k_t, sv.e_star_pp,
            sv.g_star_pp, sv.beta_pp, 1e-5, sv.step_arr, sv.tang_partner, sv.tang_stamp,
            sv.tang_xi], outputs=[st.particle_f, sv.particle_t], device=DEV)
        fk = np.linalg.norm(st.particle_f.numpy()[0])
        ov = brute_overlap(Ms[0], Ms[1], com[0] - com[1])
        f_ref = 4.0 / 3.0 * e_star * np.sqrt(0.5 * shp.equiv_radius) * ov ** 1.5
        errs.append(abs(fk / f_ref - 1.0))
    check(f"pair force vs brute-force overlap ({label})", max(errs) < 0.02,
          f"max error {max(errs)*100:.2f}% over {len(errs)} random poses")

# 3 ---------------------------------------------------------------------------------
shp = EllipsoidShape(0.0031, 0.0027, 0.0024, RHO)
model, sv, st = make(1, plane_y=0.0, mu_roll_wall=0.05)
q = random_quats(1, np.random.default_rng(5))
sv.set_ellipsoids(shp, [[0.0, 0.012, 0.0]], q)
dt = 1.5e-5
for k in range(int(2.5 / dt)):
    sv.step(st, st, None, None, dt)
y = sv.clump_x.numpy()[0][1]
R = quat_to_R(sv.clump_q.numpy()[0])
short_axis_world = R[:, 2]                    # body z = shortest semi-axis
v = np.linalg.norm(sv.clump_v.numpy()[0])
check("rests flat on a plane", abs(y - shp.axes[2]) < 1e-4 and abs(abs(short_axis_world[1]) - 1) < 0.02
      and v < 1e-2, f"centre {y*1e3:.3f} mm (c = {shp.axes[2]*1e3:.1f} mm), "
      f"short axis . up = {abs(short_axis_world[1]):.4f}, speed {v:.1e}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
