#!/usr/bin/env python3
"""
Minimal repro: do particles collide with an open-shell STL under the stock solvers?

Drops a handful of particles inside the Spout tube, well clear of the walls, with a
sideways velocity aimed at a wall.  With working contacts they should bounce or slide
and stay inside.  If they end up outside the tube, particle-vs-mesh contact is not
working for this geometry.

The suspicion is ``create_soft_contacts`` (newton/_src/geometry/kernels.py): its mesh
branch takes the sign from ``wp.mesh_query_point_sign_normal``, a winding-based
inside/outside test that assumes a closed, correctly wound mesh.  The BFA chute parts
are open shells -- Spout has 164 open boundary edges, Mid has 634, only Def is
watertight -- so the sign is unreliable, and an inverted sign makes the contact normal
point into the wall, driving the particle through it.

Usage:  python repro_mesh_passthrough.py [semi_implicit|xpbd]
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import newton
from bfa_replication_mpm import load_part

wp.init()


@wp.kernel
def nearest_on_mesh(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float,
                    out_cp: wp.array(dtype=wp.vec3), out_d: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if r.result:
        out_cp[i] = wp.mesh_eval_position(mesh, r.face, r.u, r.v)
        out_d[i] = wp.length(pts[i] - out_cp[i])
    else:
        out_d[i] = 1.0e9


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "semi_implicit"
    device = "cuda:0"

    verts, faces = load_part("Spout", False, True)  # same winding the MPM path uses
    lo, hi = verts.min(0), verts.max(0)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(verts, faces.flatten()),
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=1.0e5, kd=1.0e3, kf=1.0e3),
        key="Spout",
    )

    # seed particles on the chute axis, high up where the duct is widest,
    # moving sideways at 2 m/s straight at a wall
    # DEM is unforgiving of initial overlap: two grains seeded 2 mm apart overlap by
    # 10 mm and, at k_contact = 1000 N/m, take 10 N on a 0.9 g mass -- they explode.
    # Seed on a lattice at 20 mm > one diameter so nothing starts in contact.
    cx, cz = -1.52, 4.09          # centre of the vertical inlet duct
    if "--overlap" in sys.argv:
        rng = np.random.default_rng(0)
        n = 200
        pts = np.stack([cx + rng.uniform(-0.05, 0.05, n),
                        rng.uniform(3.30, 3.60, n),
                        cz + rng.uniform(-0.05, 0.05, n)], axis=1)
    else:
        g = 0.020
        ax = cx + np.arange(-2, 3) * g
        az = cz + np.arange(-2, 3) * g
        ay = 3.30 + np.arange(8) * g
        pts = np.array([[x, y, z] for x in ax for y in ay for z in az])
        n = len(pts)
    for p in pts:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(2.0, 0.0, 0.0),
                             mass=0.8994e-3, radius=0.006)

    wmesh = wp.Mesh(wp.array(verts, dtype=wp.vec3, device=device),
                    wp.array(faces.flatten(), dtype=int, device=device))
    model = builder.finalize(device=device)
    model.particle_mu = 0.43
    model.set_gravity((0.0, -9.81, 0.0))
    model.particle_grid = wp.HashGrid(64, 64, 64, device=device)

    if which == "xpbd":
        solver = newton.solvers.SolverXPBD(model, iterations=4)
    else:
        solver = newton.solvers.SolverSemiImplicit(model)

    pipeline = newton.CollisionPipeline.from_model(model, soft_contact_max=n * 4)

    state_0, state_1 = model.state(), model.state()
    dt = 1.0e-5
    print(f"solver: {which}   dt {dt*1e6:.0f} us   {n} particles, radius 6 mm")
    print(f"Spout bbox  x {lo[0]:.3f}..{hi[0]:.3f}  y {lo[1]:.3f}..{hi[1]:.3f}  z {lo[2]:.3f}..{hi[2]:.3f}")
    print(f"{'step':>7} {'t (ms)':>8} {'inside':>8} {'outside z':>10} {'outside x':>10} {'max|v|':>8}")

    for step in range(4001):
        # the hash grid MUST be rebuilt every step or particle-particle contact
        # silently no-ops (eval_particle_contact returns on hash_grid_point_id == -1)
        model.particle_grid.build(state_0.particle_q, 0.012)
        contacts = pipeline.collide(model, state_0)
        state_0.clear_forces()
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

        if step % 500 == 0:
            q = state_0.particle_q.numpy()
            v = state_0.particle_qd.numpy()
            outz = ((q[:, 2] < lo[2] - 0.002) | (q[:, 2] > hi[2] + 0.002)).sum()
            outx = ((q[:, 0] < lo[0] - 0.002) | (q[:, 0] > hi[0] + 0.002)).sum()
            inside = n - max(outz, outx)
            sp = np.linalg.norm(v, axis=1)
            j = int(sp.argmax())
            wq = wp.array(q[j:j+1].astype(np.float32), dtype=wp.vec3, device=device)
            wcp = wp.zeros(1, dtype=wp.vec3, device=device); wd = wp.zeros(1, dtype=float, device=device)
            wp.launch(nearest_on_mesh, dim=1, inputs=[wmesh.id, wq, 1.0, wcp, wd], device=device)
            nc = int(contacts.soft_contact_count.numpy()[0])
            print(f"{step:7d} {step*dt*1e3:8.1f} {inside:8d} {outz:10d} {outx:10d} "
                  f"{sp.max():8.2f}   fastest at ({q[j,0]:+.3f},{q[j,1]:+.3f},{q[j,2]:+.3f}) "
                  f"wall_dist {wd.numpy()[0]*1e3:7.2f} mm   contacts {nc}")

    q = state_0.particle_q.numpy()
    escaped = ((q[:, 2] < lo[2] - 0.002) | (q[:, 2] > hi[2] + 0.002) |
               (q[:, 0] < lo[0] - 0.002) | (q[:, 0] > hi[0] + 0.002)).sum()
    print(f"\n{escaped} of {n} particles left the tube "
          f"({'PASS-THROUGH CONFIRMED' if escaped > n * 0.1 else 'contacts appear to work'})")
    print(f"soft contacts generated on the last step: {int(contacts.soft_contact_count.numpy()[0])}")


if __name__ == "__main__":
    main()
