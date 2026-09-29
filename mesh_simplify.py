#!/usr/bin/env python3
"""
Collapse sub-grain-scale CAD detail out of collider meshes.

The BFA STL parts carry CAD detail a 12 mm grain can never resolve: Top has 520 triangles
with every edge under 1.5 mm (0.07% of its area) packed into fillets and chamfers.
For wall contact that is pure cost -- up to 95 triangles lie within 9 mm of one grain
there, and a GPU warp runs as long as its longest candidate walk, so those few grains set
the wall kernel's run time.

`collapse_short_edges` merges each edge shorter than `eps` into its midpoint, refusing any
collapse that would flip an adjacent face or leave a sliver, and repeats until no short
edge remains.  `surface_deviation` measures the result in float64 against the original,
both ways (Hausdorff), so the geometric cost is a number, not a hope.
"""

from __future__ import annotations

import numpy as np


def _face_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def collapse_short_edges(v, f, eps, tol=None, max_passes=50, min_cos=0.2):
    """Return (vertices, faces) with every edge >= eps where a safe collapse exists.

    Measured at eps = 2 mm with tol=None (the setting in use): Top 6372 -> 2285 tris,
    Mid 4708 -> 1175, SS 1285 -> 302, Baf 536 -> 239, Def 1332 -> 416; surface deviation
    max 1.35 mm, p99 <= 1.0 mm, all at plate rims and fillets.

    A collapse of edge (a, b) to its midpoint is rejected if
      * (only when tol is given) the midpoint is farther than `tol` from the ORIGINAL plane of any face around a or
        b -- the quadric-error idea: flat or gently curved patches collapse freely, but a
        sharp feature does not.  Without this, the thin plates' rims (short edges joining
        top and bottom faces) collapsed and moved both faces ~1 mm;
      * or any surviving face would turn by more than acos(min_cos) or lose (nearly) all
        its area -- the two ways edge collapse folds a surface.
    """
    v = np.asarray(v, dtype=np.float64).copy()
    f = np.asarray(f, dtype=np.int64).reshape(-1, 3).copy()
    for _ in range(max_passes):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        L = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
        short = np.flatnonzero(L < eps)
        if not len(short):
            break
        short = short[np.argsort(L[short])]
        # faces incident to each vertex
        inc = [[] for _ in range(len(v))]
        for fi, tri in enumerate(f):
            for x in tri:
                inc[x].append(fi)
        n0 = _face_normals(v, f)
        p0 = v[f[:, 0]].copy()           # a point on each face's current plane
        touched = np.zeros(len(v), dtype=bool)
        remap = np.arange(len(v))
        done = 0
        for k in short:
            a, b = e[k]
            if touched[a] or touched[b]:
                continue
            mid = 0.5 * (v[a] + v[b])
            ok = True
            around = set(inc[a]) | set(inc[b])
            for fi in around:
                if tol is not None and abs(np.dot(mid - p0[fi], n0[fi])) > tol:
                    ok = False
                    break
            if not ok:
                continue
            for fi in around:
                tri = f[fi]
                if a in tri and b in tri:
                    continue                       # this face disappears
                p = np.array([mid if x in (a, b) else v[x] for x in tri])
                nn = np.cross(p[1] - p[0], p[2] - p[0])
                ln = np.linalg.norm(nn)
                area0 = np.linalg.norm(np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]]))
                if ln < 1e-3 * area0 or np.dot(nn / max(ln, 1e-30), n0[fi]) < min_cos:
                    ok = False
                    break
            if not ok:
                continue
            v[a] = mid
            remap[b] = a
            touched[[a, b]] = True
            for fi in set(inc[a]) | set(inc[b]):
                for x in f[fi]:
                    touched[x] = True             # neighbours wait for the next pass
            done += 1
        if not done:
            break
        f = remap[f]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    # drop unreferenced vertices
    used = np.unique(f)
    new_idx = -np.ones(len(v), dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return v[used], new_idx[f]


def _closest_dist(P, v, f, chunk=128):
    """float64 point-to-mesh distance (brute force, Ericson), P: (n, 3)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    ab, ac = b - a, c - a
    out = np.empty(len(P))
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
        for mask, pts in regions:
            q = np.where(mask[..., None], pts, q)
        out[s:s + chunk] = np.linalg.norm(p - q, axis=-1).min(1)
    return out


def _surface_samples(v, f, per_tri=6, seed=0):
    rng = np.random.default_rng(seed)
    t = v[f]
    r = rng.random((len(f), per_tri, 2))
    flip = r.sum(-1) > 1
    r[flip] = 1 - r[flip]
    pts = t[:, None, 0] + r[..., :1] * (t[:, None, 1] - t[:, None, 0]) + r[..., 1:] * (t[:, None, 2] - t[:, None, 0])
    return np.concatenate([pts.reshape(-1, 3), v])          # vertices too: corners matter


def surface_deviation(v0, f0, v1, f1):
    """Two-sided Hausdorff distance between two triangle meshes (sampled, float64)."""
    d01 = _closest_dist(_surface_samples(v0, f0), v1, f1)
    d10 = _closest_dist(_surface_samples(v1, f1), v0, f0)
    return float(max(d01.max(), d10.max())), float(np.percentile(np.concatenate([d01, d10]), 99))
