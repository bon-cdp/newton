#!/usr/bin/env python3
"""Where does SolverGranularDEM actually spend its time?

Times each kernel separately with device synchronisation, at a few particle counts,
so the cost model is measured rather than assumed.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

import newton
from bfa_dem import GRAIN_MASS, GRAIN_RADIUS
from bfa_replication_mpm import COLLIDER_PARTS, load_part
from granular_dem import SolverGranularDEM, build_collider, eval_shell_contact_forces
from newton._src.solvers.semi_implicit.kernels_contact import eval_particle_contact

wp.init()
DEVICE = "cuda:0"
DT = 2.4316429e-5


def timed(label, fn, reps=60):
    fn()
    wp.synchronize_device(DEVICE)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize_device(DEVICE)
    return label, (time.perf_counter() - t0) / reps * 1e3  # ms


def bench(n_active: int, pool_pad: int):
    parts = [(n, *load_part(n, fx, fl)) for n, ts, fx, fl in COLLIDER_PARTS]
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    for name, v, f in parts:
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()),
                               cfg=newton.ModelBuilder.ShapeConfig(mu=0.5), key=name)

    # active grains packed into the chute at realistic density, plus parked pool slots
    side = int(np.ceil(n_active ** (1 / 3)))
    g = 0.014
    pts = np.array([[-1.60 + i * g, 3.30 + j * g, 4.00 + k * g]
                    for i in range(side) for j in range(side) for k in range(side)])[:n_active]
    for p in pts:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0, -3.0, 0.0),
                             mass=GRAIN_MASS, radius=GRAIN_RADIUS)
    for _ in range(pool_pad):
        builder.add_particle(pos=wp.vec3(-1.5, 6.0, 4.1), vel=wp.vec3(0.0),
                             mass=GRAIN_MASS, radius=GRAIN_RADIUS, flags=0)

    model = builder.finalize(device=DEVICE)
    model.particle_ke, model.particle_kd, model.particle_kf = 2.0e4, 3.0, 3.0
    model.particle_mu, model.particle_cohesion = 0.4307, 0.0
    model.set_gravity((0.0, -9.81, 0.0))

    collider, meshes = build_collider(
        parts, two_sided=[p[1] for p in COLLIDER_PARTS], friction=[0.5] * len(parts),
        ke=[2.0e4] * len(parts), kd=[3.0] * len(parts), kf=[3.0] * len(parts),
        thickness=[0.002 if p[1] else 0.0 for p in COLLIDER_PARTS],
        max_dist=0.03, device=DEVICE)
    solver = SolverGranularDEM(model, collider, keepalive=meshes)
    s0, s1 = model.state(), model.state()
    N = model.particle_count

    rows = []
    rows.append(timed("hash grid build",
                      lambda: model.particle_grid.build(s0.particle_q, solver.grid_cell)))
    rows.append(timed("particle-particle", lambda: wp.launch(
        eval_particle_contact, dim=N, device=DEVICE,
        inputs=[model.particle_grid.id, s0.particle_q, s0.particle_qd, model.particle_radius,
                model.particle_flags, model.particle_ke, model.particle_kd, model.particle_kf,
                model.particle_mu, model.particle_cohesion, model.particle_max_radius],
        outputs=[s0.particle_f])))
    rows.append(timed("wall (cache warm)", lambda: wp.launch(
        eval_shell_contact_forces, dim=N, device=DEVICE,
        inputs=[s0.particle_q, s0.particle_qd, model.particle_radius, model.particle_flags,
                collider, solver.wall_slack, DT],
        outputs=[s0.particle_f, solver.contact_count])))

    def wall_cold():
        solver.wall_slack.zero_()
        wp.launch(eval_shell_contact_forces, dim=N, device=DEVICE,
                  inputs=[s0.particle_q, s0.particle_qd, model.particle_radius,
                          model.particle_flags, collider, solver.wall_slack, DT],
                  outputs=[s0.particle_f, solver.contact_count])
    rows.append(timed("wall (cache cold)", wall_cold))
    rows.append(timed("integrate", lambda: solver.integrate_particles(model, s0, s1, DT)))
    rows.append(timed("full step", lambda: solver.step(s0, s1, None, None, DT)))

    print(f"\n  active {n_active:,}  parked {pool_pad:,}  total {N:,}")
    for label, ms in rows:
        print(f"    {label:<20} {ms:8.3f} ms" + ("   <- 41,124 steps/sim-s => %.0f s/sim-s"
                                                 % (ms * 41.124) if label == "full step" else ""))


if __name__ == "__main__":
    print("SolverGranularDEM kernel profile (P5000)")
    bench(3000, 0)
    bench(3000, 46000)      # same active count, realistic parked pool
    bench(20000, 30000)     # steady-state-like
