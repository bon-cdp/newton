#!/usr/bin/env python3
"""
Seal the cracks between the separately tessellated BFA collider STLs.

The seven parts (Spout/Top/Mid/Bot/SS/Baf/Def) were exported independently and do
not share vertices, so their seams leave gaps of 2-7 mm:

    SS -> Bot   2.1 mm      Mid -> Def  2.5 mm      Baf -> SS  5.0 mm
    SS -> Baf   6.6 mm

A 12 mm DEM grain cannot fit through those, which is why the DEM run leaks nothing.
An MPM material point is a dimensionless quadrature point and walks straight through.
Nothing in the boundary code can stop it: ``rasterize_collider`` samples one SDF value
per grid node, so a sub-voxel crack is invisible to the momentum solve, and
``project_outside_collider`` only fires when a particle is *inside* a collider -- in a
crack it is in genuine free space and correctly left alone.

So the fix is geometric.  Rather than welding vertices (which moves the original
surfaces and can distort the flow region), this builds a *gasket*: for every open
boundary edge that sits close to another part's surface, it emits a two-triangle skirt
bridging the edge to its projection on that surface.  It only adds geometry inside the
cracks and never modifies the seven parts.

Correctness check: the BFA DEM particle cloud is 19,607 positions per frame that are
known to be free space.  If the gasket swallows any of them it has plugged a real flow
path, and the script says so.

Usage:
    python build_gasket.py [max_gap_mm] [--vtk gasket.vtk]
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

from bfa_replication_mpm import COLLIDER_PARTS, load_part

wp.init()


@wp.kernel
def _closest(
    mesh: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    max_dist: float,
    out_pos: wp.array(dtype=wp.vec3),
    out_dist: wp.array(dtype=float),
):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if r.result:
        cp = wp.mesh_eval_position(mesh, r.face, r.u, r.v)
        out_pos[i] = cp
        out_dist[i] = wp.length(pts[i] - cp)
    else:
        out_pos[i] = pts[i]
        out_dist[i] = 1.0e9


def closest_points(mesh: wp.Mesh, pts: np.ndarray, max_dist: float):
    p = wp.array(pts.astype(np.float32), dtype=wp.vec3)
    cp = wp.zeros(len(pts), dtype=wp.vec3)
    d = wp.zeros(len(pts), dtype=float)
    wp.launch(_closest, dim=len(pts), inputs=[mesh.id, p, max_dist, cp, d])
    return cp.numpy(), d.numpy()


def open_edges(faces: np.ndarray) -> np.ndarray:
    """Edges used by exactly one face -- the mesh's boundary."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return uniq[cnt == 1]


def build_gasket(parts: dict[str, tuple[np.ndarray, np.ndarray]],
                 max_gap: float = 0.008, min_gap: float = 3.0e-4, verbose: bool = True):
    """Bridge every open boundary edge that lies within ``max_gap`` of another part.

    Edges closer than ``min_gap`` are already touching and need no bridge; edges with
    nothing within ``max_gap`` are genuine openings (the inlet rim, the outlet) and are
    deliberately left alone -- which is what keeps the gasket from plugging the chute.
    """
    names = list(parts)
    meshes = {
        n: wp.Mesh(wp.array(v, dtype=wp.vec3), wp.array(np.ascontiguousarray(f).flatten(), dtype=int))
        for n, (v, f) in parts.items()
    }

    g_verts: list[np.ndarray] = []
    g_faces: list[list[int]] = []
    report = []

    for a in names:
        va, fa = parts[a]
        oe = open_edges(fa)
        if len(oe) == 0:
            report.append((a, 0, 0, 0.0, 0.0, "watertight"))
            continue

        v0 = va[oe[:, 0]]
        v1 = va[oe[:, 1]]
        mid = 0.5 * (v0 + v1)

        # pick, per edge, the nearest OTHER part
        best_d = np.full(len(oe), np.inf)
        best_b = np.full(len(oe), -1)
        for j, b in enumerate(names):
            if b == a:
                continue
            _, d = closest_points(meshes[b], mid, max_gap * 3.0)
            better = d < best_d
            best_d[better] = d[better]
            best_b[better] = j

        sel = (best_d > min_gap) & (best_d <= max_gap)
        made = 0
        for j in np.unique(best_b[sel]):
            if j < 0:
                continue
            b = names[int(j)]
            m = sel & (best_b == j)
            p0, d0 = closest_points(meshes[b], v0[m], max_gap * 3.0)
            p1, d1 = closest_points(meshes[b], v1[m], max_gap * 3.0)
            ok = (d0 < max_gap * 2.0) & (d1 < max_gap * 2.0)
            if not ok.any():
                continue
            q0, q1, r0, r1 = v0[m][ok], v1[m][ok], p0[ok], p1[ok]
            base = sum(len(x) for x in g_verts)
            g_verts.append(np.concatenate([q0, q1, r1, r0]))
            n = len(q0)
            idx = base + np.arange(n)
            # quad (q0, q1, r1, r0) -> two triangles
            g_faces.append(np.stack([idx, idx + n, idx + 2 * n], axis=1))
            g_faces.append(np.stack([idx, idx + 2 * n, idx + 3 * n], axis=1))
            made += n
        gaps = best_d[sel]
        report.append((a, len(oe), made,
                       float(gaps.min()) if len(gaps) else 0.0,
                       float(gaps.max()) if len(gaps) else 0.0, ""))

    if g_verts:
        verts = np.vstack(g_verts).astype(np.float64)
        faces = np.vstack(g_faces).astype(np.int64)
    else:
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)

    if verbose:
        print("  %-6s %9s %9s %16s" % ("part", "open edg", "bridged", "gap range (mm)"))
        for a, n_open, made, gmin, gmax, note in report:
            if note:
                print("  %-6s %9d %9s %16s" % (a, n_open, "-", note))
            else:
                print("  %-6s %9d %9d %7.1f .. %-7.1f" % (a, n_open, made, gmin * 1e3, gmax * 1e3))
        print("  gasket: %d triangles, %d vertices" % (len(faces), len(verts)))
    return verts, faces


def write_vtk(path, verts, faces):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nseam gasket\nASCII\nDATASET POLYDATA\n\n")
        f.write(f"POINTS {len(verts)} float\n")
        f.writelines(f"{x} {y} {z}\n" for x, y, z in verts)
        f.write(f"\nPOLYGONS {len(faces)} {len(faces) * 4}\n")
        f.writelines(f"3 {a} {b} {c}\n" for a, b, c in faces)


def main():
    max_gap = float(sys.argv[1]) / 1000.0 if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 0.008
    parts = {n: load_part(n, fx, fl) for n, ts, fx, fl in COLLIDER_PARTS}

    print("=" * 72)
    print("Seam gasket, max gap %.1f mm" % (max_gap * 1e3))
    print("=" * 72)
    verts, faces = build_gasket(parts, max_gap=max_gap)
    if len(faces) == 0:
        print("\nnothing to bridge")
        return

    # ---- correctness check against the DEM cloud ---------------------------
    from compare_bfa_mpm import read_bfa_frames
    gm = wp.Mesh(wp.array(verts, dtype=wp.vec3), wp.array(faces.flatten(), dtype=int))
    frames = list(read_bfa_frames())
    P = np.vstack([f["pos"] for f in frames[-10::3]])
    _, d = closest_points(gm, P, 0.05)
    near = d < 1e8
    print("\ncorrectness check -- %d BFA DEM particle positions (known free space):" % len(P))
    print("  within 6 mm of the gasket : %d (%.3f%%)" % ((d < 0.006).sum(), 100 * (d < 0.006).mean()))
    print("  within 2 mm of the gasket : %d (%.3f%%)" % ((d < 0.002).sum(), 100 * (d < 0.002).mean()))
    if near.any():
        print("  closest DEM particle      : %.2f mm" % (d[near].min() * 1e3))
    print("  (the gasket sits in cracks a 12 mm grain cannot enter, so DEM particles")
    print("   should stay a grain radius -- 6 mm -- clear of it)")

    out = "gasket.vtk"
    if "--vtk" in sys.argv:
        out = sys.argv[sys.argv.index("--vtk") + 1]
    write_vtk(out, verts, faces)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
