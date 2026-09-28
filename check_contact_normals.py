#!/usr/bin/env python3
"""
Are the particle-vs-mesh contact normals pointing out of the wall, or into it?

``create_soft_contacts`` derives the mesh contact normal from
``wp.mesh_query_point_sign_normal``, a winding-based inside/outside test.  On a closed,
correctly wound mesh that is fine.  On an open shell -- which every BFA chute part
except Def is -- the sign is unreliable.  An inverted sign makes the contact normal
point *into* the wall, and ``fn = n * c * ke`` then drives the particle through the
surface and adds energy instead of removing it.

This places particles at a known small distance inside the chute wall, runs one
collision pass, and checks each contact normal against the direction that actually
points from the wall back toward the particle.

Usage:  python check_contact_normals.py [PartName]
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import newton
from bfa_replication_mpm import COLLIDER_PARTS, load_part

wp.init()


@wp.kernel
def nearest_on_mesh(
    mesh: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    max_dist: float,
    out_cp: wp.array(dtype=wp.vec3),
    out_d: wp.array(dtype=float),
):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if r.result:
        out_cp[i] = wp.mesh_eval_position(mesh, r.face, r.u, r.v)
        out_d[i] = wp.length(pts[i] - out_cp[i])
    else:
        out_d[i] = 1.0e9


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Spout"
    flip = {n: fl for n, ts, fx, fl in COLLIDER_PARTS}[name]
    fixn = {n: fx for n, ts, fx, fl in COLLIDER_PARTS}[name]
    verts, faces = load_part(name, fixn, flip)
    device = "cuda:0"

    # Sample points a little way off the surface, on the side the material flows on.
    # Take face centroids and step along the face normal by 3 mm (half a grain).
    tri = verts[faces]
    cent = tri.mean(axis=1)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    nrm = np.cross(e1, e2)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(cent), size=min(400, len(cent)), replace=False)
    offset = 0.003
    pts = cent[pick] + nrm[pick] * offset       # 3 mm off the surface, on the +normal side

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(verts, faces.flatten()),
                           cfg=newton.ModelBuilder.ShapeConfig(mu=0.5), key=name)
    for p in pts:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.8994e-3, radius=0.006)

    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline.from_model(model, soft_contact_max=len(pts) * 4)
    contacts = pipeline.collide(model, state)

    cnt = int(contacts.soft_contact_count.numpy()[0])
    print(f"part {name}: {len(faces)} tris, {len(pts)} probe particles 3 mm off the +normal side")
    print(f"soft contacts generated: {cnt}")
    if cnt == 0:
        print("  -> no contacts at all: particles would fall straight through")
        return

    cp = contacts.soft_contact_body_pos.numpy()[:cnt]
    cn = contacts.soft_contact_normal.numpy()[:cnt]
    pi = contacts.soft_contact_particle.numpy()[:cnt]
    q = state.particle_q.numpy()

    # the normal should point from the contact point back toward the particle
    to_particle = q[pi] - cp
    ln = np.linalg.norm(to_particle, axis=1, keepdims=True)
    to_particle = to_particle / np.maximum(ln, 1e-12)
    align = np.einsum("ij,ij->i", cn, to_particle)

    good = (align > 0.5).sum()
    bad = (align < -0.5).sum()
    print(f"  normal points OUT of the wall (correct) : {good:5d}  ({100*good/cnt:.1f}%)")
    print(f"  normal points INTO the wall (inverted)  : {bad:5d}  ({100*bad/cnt:.1f}%)")
    print(f"  alignment  p5 {np.percentile(align,5):+.2f}  p50 {np.percentile(align,50):+.2f}"
          f"  p95 {np.percentile(align,95):+.2f}")
    if bad > cnt * 0.05:
        print("\n  -> INVERTED NORMALS CONFIRMED.  fn = n*c*ke drives these particles")
        print("     through the wall and adds energy: the gas-like ejection.")
    else:
        print("\n  -> normals look correct; the instability is elsewhere (stiffness/dt).")


if __name__ == "__main__":
    main()
