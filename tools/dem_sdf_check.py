"""Does our DEM collider occupy space that BFA's grains actually use?

Evaluates the exact SDF the DEM solver uses (unsigned distance, average-normal sign,
per-part two-sided, 2 mm shells) at the BFA grain positions in the cascade.  A BFA grain
centre can never be closer than one radius (6 mm) to a real wall, so anything reading
below +6 mm means our collider is intruding into space the real material occupies.
"""
import numpy as np, warp as wp, sys
sys.path.insert(0, '/home/s/Documents/newton-dem')
from bfa_replication_mpm import COLLIDER_PARTS, load_part
from compare_bfa_mpm import read_bfa_frames
wp.init()

SHELL = 0.002
R = 0.006

@wp.kernel
def sdf_k(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), maxd: float, two_sided: int,
          thick: float, out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
    if not r.result:
        return
    cp = wp.mesh_eval_position(mesh, r.face, r.u, r.v)
    n = wp.mesh_eval_face_normal(mesh, r.face)
    off = pts[i] - cp
    d = wp.length(off)
    if two_sided == 1:
        s = d - thick
    else:
        s = d * wp.where(wp.dot(n, off) > 0.0, 1.0, -1.0) - thick
    wp.atomic_min(out, i, s)

bf = list(read_bfa_frames())[-1]
lo, hi = np.array([-5.62, -2.00, 3.95]), np.array([-5.20, -1.40, 4.25])
m = np.all((bf["pos"] > lo) & (bf["pos"] < hi), axis=1)
P = bf["pos"][m].astype(np.float32)
print(f"{len(P)} BFA grain centres in the cascade; a real wall keeps them >= 6 mm clear\n")
print(f"  {'part':<7} {'mode':<10} {'<0 (inside)':>12} {'<6mm':>8}  worst mm {'FORCED':>10}")
print("  (FORCED = fraction in -6mm..0, the band that actually produces a spurious force)")
pts = wp.array(P, dtype=wp.vec3)
allmin = wp.full(len(P), 1.0e9, dtype=float)
for name, ts, fx, fl in COLLIDER_PARTS:
    v, f = load_part(name, fx, fl)
    mesh = wp.Mesh(wp.array(v, dtype=wp.vec3), wp.array(f.flatten(), dtype=int))
    o = wp.full(len(P), 1.0e9, dtype=float)
    th = SHELL if ts else 0.0
    wp.launch(sdf_k, dim=len(P), inputs=[mesh.id, pts, R + th + 0.03, int(ts), th, o])
    wp.launch(sdf_k, dim=len(P), inputs=[mesh.id, pts, R + th + 0.03, int(ts), th, allmin])
    d = o.numpy(); seen = d < 1e8
    if seen.sum() == 0:
        continue
    # the solver only reacts to overlaps shallower than one radius (deeper is treated
    # as a wrong-side artefact and skipped), so that band is what actually applies force
    acting = seen & (d > -R) & (d < 0)
    print(f"  {name:<7} {'two' if ts else 'one':<10} {(d[seen]<0).mean():12.3f} "
          f"{(d[seen]<R).mean():8.3f}  {d[seen].min()*1e3:8.1f} {acting.mean():10.3f}")
d = allmin.numpy(); seen = d < 1e8
print(f"\n  {'COMBINED':<7} {'':<10} {(d[seen]<0).mean():12.3f} {(d[seen]<R).mean():8.3f} "
      f" {d[seen].min()*1e3:8.1f}")
print(f"\n  -> {100*(d[seen]<0).mean():.1f}% of BFA's cascade grains sit INSIDE our collider,")
print(f"     {100*(d[seen]<R).mean():.1f}% are closer than one grain radius to it.")
