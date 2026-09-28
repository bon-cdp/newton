"""Where do escapees come from, and are they hotter than the bulk?

Escape is defined geometrically, not by a bounding box: a point whose signed distance
to the collider set puts it on the *outside* of the flow volume.  We track points
frame to frame by index and catch them on the step they cross.
"""
import numpy as np, warp as wp, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from bfa_replication_mpm import COLLIDER_PARTS, load_part
from compare_bfa_mpm import read_mpm_frame
wp.init()

@wp.kernel
def sdf_k(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), maxd: float, two_sided: int,
          shell: float, out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
    if not r.result:
        out[i] = 1.0e9
        return
    cp = wp.mesh_eval_position(mesh, r.face, r.u, r.v)
    n = wp.mesh_eval_face_normal(mesh, r.face)
    off = pts[i] - cp
    d = wp.length(off)
    if two_sided == 1:
        out[i] = d - shell
    else:
        out[i] = d * wp.where(wp.dot(n, off) > 0.0, 1.0, -1.0)

parts = [(n, *load_part(n, fx, fl), ts) for n, ts, fx, fl in COLLIDER_PARTS]
meshes = [(n, wp.Mesh(wp.array(v, dtype=wp.vec3), wp.array(f.flatten(), dtype=int)), ts)
          for n, v, f, ts in parts]

def spout_sdf(P):
    """Signed distance to the Spout alone: negative = behind the chute wall = escaped."""
    p = wp.array(P.astype(np.float32), dtype=wp.vec3)
    o = wp.zeros(len(P), dtype=float)
    for n, m, ts in meshes:
        if n != 'Spout':
            continue
        wp.launch(sdf_k, dim=len(P), inputs=[m.id, p, 0.6, 0, 0.0, o])
    return o.numpy()

run = sys.argv[1]
fs = sorted(glob.glob(run + '/frame_*_particles.vtk'))[-25:]
esc_pos, esc_spd, bulk_spd = [], [], []
for fn in fs:
    P, V = read_mpm_frame(fn)
    s = np.linalg.norm(V, axis=1)
    d = spout_sdf(P)
    seen = d < 1e8
    out = seen & (d < -0.001)          # behind the chute wall
    inn = seen & (d > 0.02)
    esc_pos.append(P[out]); esc_spd.append(s[out]); bulk_spd.append(s[inn])
esc_pos = np.vstack(esc_pos) if len(esc_pos) else np.zeros((0, 3))
esc_spd = np.concatenate(esc_spd); bulk_spd = np.concatenate(bulk_spd)

print('run: %s   frames %d' % (run, len(fs)))
print('points behind the Spout wall: %d   (bulk sample %d)' % (len(esc_spd), len(bulk_spd)))
if len(esc_spd):
    print('  escapee speed  mean %.2f  p50 %.2f  p95 %.2f  max %.2f m/s'
          % (esc_spd.mean(), np.median(esc_spd), np.percentile(esc_spd, 95), esc_spd.max()))
    print('  bulk    speed  mean %.2f  p50 %.2f  p95 %.2f  max %.2f m/s'
          % (bulk_spd.mean(), np.median(bulk_spd), np.percentile(bulk_spd, 95), bulk_spd.max()))
    print('  ratio of means: %.2fx' % (esc_spd.mean() / max(bulk_spd.mean(), 1e-9)))
    print('\n  escape locations by height:')
    for lo in np.arange(-2.0, 4.5, 0.5):
        m = (esc_pos[:, 1] >= lo) & (esc_pos[:, 1] < lo + 0.5)
        if m.sum() > len(esc_spd) * 0.01:
            print('    y %5.1f..%5.1f  %6d (%4.1f%%)  x %6.2f..%6.2f  mean speed %.2f'
                  % (lo, lo + 0.5, m.sum(), 100 * m.mean(),
                     esc_pos[m][:, 0].min(), esc_pos[m][:, 0].max(), esc_spd[m].mean()))
