#!/usr/bin/env python3
"""
Validate wall-contact distances against float64 brute force.

    python tools/wall_grid_check.py --ckpt runs/perf/ckpt/checkpoint.npz [bfa_dem config flags]

For every active grain within 6.5 mm of a collider part, compares the nearest distance
found by (a) the baked wall grid and (b) Warp's native BVH query against an exact float64
closest-point computation over every triangle of the part.

Why this exists: the BVH path's float32 closest-point test loses precision on the Spout's
3 m sliver panels -- measured p99 error 47 um, max 194 um, against Hertz overlaps of
~50 um -- and on 12% of Spout contacts the sign test found no face and dropped the
contact entirely.  The grid path works in per-triangle orthonormal frames and should
stay at float32 resolution (~0.3 um) everywhere.  Any change to wall contact must keep
the grid column at that level.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bfa_dem  # noqa: E402
import granular_dem as G  # noqa: E402
from dem_bench import load_checkpoint  # noqa: E402

REACH = 0.0071


@wp.kernel
def _nearest(g: G.WallGrid, meshes: wp.array(dtype=wp.uint64), x: wp.array(dtype=wp.vec3),
             dg: wp.array2d(dtype=float), db: wp.array2d(dtype=float)):
    i = wp.tid()
    p = x[i]
    for m in range(meshes.shape[0]):
        r = wp.mesh_query_point_no_sign(meshes[m], p, REACH)
        if r.result:
            db[i, m] = wp.length(p - wp.mesh_eval_position(meshes[m], r.face, r.u, r.v))
    cc = G._cell_of(g, p)
    if cc[0] < 0 or cc[1] < 0 or cc[2] < 0 or cc[0] >= g.nx or cc[1] >= g.ny or cc[2] >= g.nz:
        return
    cell = (cc[2] * g.ny + cc[1]) * g.nx + cc[0]
    for k in range(g.cell_start[cell], g.cell_start[cell + 1]):
        rec = g.rec[k]
        sq, _cp = G._tri_closest(rec.v0, rec.u, rec.w, rec.n, rec.t2, p)
        d = wp.sqrt(sq)
        if d < REACH:
            wp.atomic_min(dg, i, rec.part, d)


def exact_nearest(P, v, f, chunk=256):
    """float64 Ericson closest point, vectorised over (points x triangles)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    out = np.full(len(P), 1.0)
    ab, ac = b - a, c - a
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        ap, bp, cp = p - a, p - b, p - c
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        with np.errstate(all="ignore"):
            den = 1.0 / (va + vb + vc)
            q = a + ab * (vb * den)[..., None] + ac * (vc * den)[..., None]
            regions = [
                ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
                 b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]),
                ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[..., None]),
                ((d6 >= 0) & (d5 <= d6), c + 0 * p),
                ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[..., None]),
                ((d3 >= 0) & (d4 <= d3), b + 0 * p),
                ((d1 <= 0) & (d2 <= 0), a + 0 * p),
            ]
        for mask, pts in regions:          # later entries win: same precedence as Ericson
            q = np.where(mask[..., None], pts, q)
        out[s:s + chunk] = np.linalg.norm(p - q, axis=-1).min(1)
    return out


def main():
    argv = sys.argv[1:]
    k = argv.index("--ckpt")
    ckpt = argv[k + 1]
    del argv[k:k + 2]
    args = bfa_dem.parse_args(argv + ["--no-vtk", "--out", os.path.join(os.path.dirname(ckpt), "check_out")])
    wp.init()
    S = bfa_dem.build(args, quiet=True)
    ck = load_checkpoint(S, ckpt)
    g = S.solver.wall_grid
    act = (ck["flags"] & 1) == 1
    X = ck["q"][act].astype(np.float64)
    nparts = len(S.parts)
    dg = wp.full((len(X), nparts), 1.0, dtype=float)
    db = wp.full((len(X), nparts), 1.0, dtype=float)
    wp.launch(_nearest, dim=len(X), inputs=[g, S.collider.mesh,
                                           wp.array(X.astype(np.float32), dtype=wp.vec3), dg, db])
    dg, db = dg.numpy(), db.numpy()
    lo, hi = S.collider.lower.numpy(), S.collider.upper.numpy()
    print(f"{'part':6s} {'grains':>6s}   |d - exact| (um)   grid: median   p99    max"
          f"   |  native BVH: median    p99     max")
    worst = 0.0
    for m, (name, v, f) in enumerate(S.parts):
        sel = np.all(X > lo[m] - 0.008, 1) & np.all(X < hi[m] + 0.008, 1)
        de = np.full(len(X), 1.0)
        de[sel] = exact_nearest(X[sel], np.asarray(v, np.float64), np.asarray(f).reshape(-1, 3))
        near = de < 0.0065
        if not near.any():
            continue
        eg = np.abs(dg[near, m] - de[near]) * 1e6
        eb = np.abs(db[near, m] - de[near]) * 1e6
        worst = max(worst, eg.max())
        print(f"{name:6s} {near.sum():6d}   {'':18s} {np.median(eg):6.2f} {np.percentile(eg, 99):6.2f} "
              f"{eg.max():6.2f}   |  {np.median(eb):16.2f} {np.percentile(eb, 99):7.2f} {eb.max():7.2f}")
    print(f"\ngrid worst error {worst:.2f} um -> {'PASS' if worst < 1.0 else 'FAIL'} (limit 1 um)")
    sys.exit(0 if worst < 1.0 else 1)


if __name__ == "__main__":
    main()
