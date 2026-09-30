#!/usr/bin/env python3
"""
Issue #11 check: a sphere dropped at rest onto a moving belt, against the analytic answer.

Friction at the contact drives the sphere forward (a = mu g) and spins it up
(alpha = 5 mu g / (2 r)).  The contact slips until the grain's contact-point velocity
equals the belt's:

    t* = 2 v_b / (7 mu g),   v(t*) = (2/7) v_b

after which it rolls at (2/7) v_b if there is no rolling resistance, and creeps on
toward v_b if there is.  The belt is a stationary plane whose SURFACE moves.

    python tools/belt_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import newton  # noqa: E402
from dem_run import surface_motion  # noqa: E402
from granular_dem import SolverGranularDEM, build_collider, build_wall_grid  # noqa: E402

R, RHO, G = 0.006, 994.05, 9.81
MASS = 4.0 / 3.0 * np.pi * R ** 3 * RHO


def run(v_belt, mu, mu_roll, t_end=0.25, dt=2.0e-5):
    wp.init()
    v = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2]])
    parts = [("belt", v, f)]
    b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-G)
    b.add_particle(pos=wp.vec3(0.0, R - 2e-6, 0.0), vel=wp.vec3(0.0), mass=MASS, radius=R,
                   flags=int(newton.ParticleFlags.ACTIVE))
    m = b.finalize()
    m.particle_ke, m.particle_kd, m.particle_kf, m.particle_mu = 2e4, 1.0, 3.0, 0.1
    m.set_gravity((0.0, -G, 0.0))
    col, meshes = build_collider(parts, [True], [mu], [2e4], [1.0], [3.0], [0.0], 0.03, "cuda:0",
                                 motions=[surface_motion({"type": "belt", "velocity": [v_belt, 0, 0]},
                                                         (0, -G, 0))])
    g, _ = build_wall_grid(parts, meshes, col.lower, col.upper, R + 1.1e-3, 0.006, "cuda:0")
    sv = SolverGranularDEM(m, col, grid_cell=2 * R, keepalive=meshes, wall_grid=g, hertz=True,
                           youngs=1.42e7, tangential_ratio=1.0, mu_roll_wall=mu_roll,
                           rot_damp_wall=0.0, restitution=0.2)
    s = m.state()
    ts, vs, ws = [], [], []
    for k in range(int(t_end / dt)):
        sv.step(s, s, None, None, dt)
        if k % 25 == 0:
            ts.append((k + 1) * dt)
            vs.append(s.particle_qd.numpy()[0].copy())
            ws.append(sv.particle_w.numpy()[0].copy())
    return np.array(ts), np.array(vs), np.array(ws)


def main():
    vb, mu = 2.0, 0.5
    t_star = 2.0 * vb / (7.0 * mu * G)
    ok = True
    print(f"belt {vb} m/s, mu {mu}:  analytic slip ends at t* = {t_star*1e3:.1f} ms, "
          f"v(t*) = 2/7 v_b = {2*vb/7:.4f} m/s")

    t, v, w = run(vb, mu, mu_roll=0.0)
    # slip phase: slope of v_x
    early = (t > 0.1 * t_star) & (t < 0.8 * t_star)
    a = np.polyfit(t[early], v[early, 0], 1)[0]
    after = t > 1.5 * t_star
    v_roll = v[after, 0].mean()
    slip = v[after, 0] + w[after, 2] * R          # contact-point speed: v + w x (-r y) = v_x + w_z r
    print(f"  no rolling resistance:  a = {a:.4f} m/s2 (mu g = {mu*G:.4f}; ratio {a/(mu*G):.3f})"
          f"   rolling speed {v_roll:.4f} m/s (2/7 v_b ratio {v_roll/(2*vb/7):.3f})"
          f"   contact speed after slip {slip.mean():.4f} (belt {vb})")
    ok &= abs(a / (mu * G) - 1) < 0.03 and abs(v_roll / (2 * vb / 7) - 1) < 0.03 and abs(slip.mean() / vb - 1) < 0.03

    t2, v2, _w2 = run(vb, mu, mu_roll=0.1, t_end=1.5)
    print(f"  rolling resistance 0.1: v at 0.25 / 0.75 / 1.5 s = {np.interp(0.25, t2, v2[:,0]):.3f} / "
          f"{np.interp(0.75, t2, v2[:,0]):.3f} / {v2[-1,0]:.3f} m/s  (creeps from 2/7 v_b toward v_b)")
    ok &= v2[-1, 0] > np.interp(0.25, t2, v2[:, 0]) and v2[-1, 0] <= vb * 1.01
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
