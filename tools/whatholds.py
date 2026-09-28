"""In the band where BFA has mass we lack (y -1.65..-1.40), what supports it?

If BFA's grains there sit ~one radius from a collider surface, they are resting on a
wall -- and if our grains in the same band are NOT near a wall, we are mismodelling a
retention feature.  If instead they are far from every surface, the layer is held by
grain-grain contact and geometry is not the story.
"""
import numpy as np, warp as wp, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from bfa_replication_mpm import COLLIDER_PARTS, load_part
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame
wp.init()
R = 0.006

@wp.kernel
def nearest(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), maxd: float,
            out: wp.array(dtype=float), who: wp.array(dtype=int), pid: int):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
    if r.result:
        d = wp.length(pts[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v))
        if d < out[i]:
            out[i] = d
            who[i] = pid

meshes = []
for k, (name, ts, fx, fl) in enumerate(COLLIDER_PARTS):
    v, f = load_part(name, fx, fl)
    meshes.append((name, wp.Mesh(wp.array(v, dtype=wp.vec3),
                                 wp.array(f.flatten(), dtype=int))))

def support(P, label):
    p = wp.array(P.astype(np.float32), dtype=wp.vec3)
    d = wp.full(len(P), 1.0e9, dtype=float)
    w = wp.full(len(P), -1, dtype=int)
    for k, (nm, m) in enumerate(meshes):
        wp.launch(nearest, dim=len(P), inputs=[m.id, p, 0.20, d, w, k])
    d = d.numpy(); w = w.numpy()
    near = d < 0.012          # within two radii of a wall = wall-supported
    print(f"  {label:<12} n={len(P):5d}  wall-supported {near.mean():6.3f}   "
          f"median dist {np.median(d)*1e3:6.1f} mm")
    if near.sum():
        import collections
        c = collections.Counter(w[near])
        tot = near.sum()
        parts = "  ".join(f"{meshes[k][0]}:{v/tot:.2f}" for k, v in c.most_common(4))
        print(f"               supported by -> {parts}")

BAND = (-1.65, -1.40)
bf = list(read_bfa_frames())[-1]
m = (bf["pos"][:, 1] > BAND[0]) & (bf["pos"][:, 1] < BAND[1])
print(f"band y {BAND[0]}..{BAND[1]}  (where BFA has ~0.6 kg we do not)")
support(bf["pos"][m], "BFA")
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))
    if not fs:
        continue
    p, _ = read_mpm_frame(fs[-1])
    mm = (p[:, 1] > BAND[0]) & (p[:, 1] < BAND[1])
    support(p[mm], run)
