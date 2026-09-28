"""Quantify particle escape and measure the seam gaps between collider parts."""
import numpy as np, trimesh, warp as wp, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from bfa_replication_mpm import COLLIDER_PARTS, load_part
from compare_bfa_mpm import read_mpm_frame, read_bfa_frames
wp.init()

# ---- 1. where does escaped material go? ---------------------------------
run = sys.argv[1]
meta = json.load(open(run + '/run.json')); pm = meta.get('point_mass') or meta['grain_mass']
parts = {n: load_part(n, fx, fl) for n, ts, fx, fl in COLLIDER_PARTS}
allv = np.vstack([v for v, f in parts.values()])
lo, hi = allv.min(0), allv.max(0)
print('geometry bbox  min %s  max %s' % (np.round(lo, 3), np.round(hi, 3)))

fs = sorted(glob.glob(run + '/frame_*_particles.vtk'))
tot = esc = 0
zout = xout = yout = 0
for fn in fs[-20:]:
    p, v = read_mpm_frame(fn)
    tot += len(p)
    # z is the narrow axis: the chute is only 0.375 m deep, so anything outside
    # the geometry's z-range has left through a wall, not through the outlet.
    # material below the geometry has legitimately left through the outlet
    inflow = p[:, 1] > lo[1] + 0.002
    mz = ((p[:, 2] < lo[2] - 0.002) | (p[:, 2] > hi[2] + 0.002)) & inflow
    mx = ((p[:, 0] < lo[0] - 0.002) | (p[:, 0] > hi[0] + 0.002)) & inflow
    my = p[:, 1] > hi[1] + 0.002
    zout += mz.sum(); xout += mx.sum(); yout += my.sum()
    esc += (mz | mx | my).sum()
n = len(fs[-20:])
print('\nescaped material (avg over last %d frames):' % n)
print('  through a z wall : %8.3f kg  (%.2f%% of resident mass)' % (zout/n*pm, 100*zout/max(tot,1)))
print('  past x bounds    : %8.3f kg  (%.2f%%)' % (xout/n*pm, 100*xout/max(tot,1)))
print('  above the inlet  : %8.3f kg  (%.2f%%)' % (yout/n*pm, 100*yout/max(tot,1)))
print('  any              : %8.3f kg  (%.2f%%)' % (esc/n*pm, 100*esc/max(tot,1)))
# DEM for comparison
bf = list(read_bfa_frames())[-1]['pos']
binf = bf[:, 1] > lo[1] + 0.002
dz = (((bf[:, 2] < lo[2]-0.002) | (bf[:, 2] > hi[2]+0.002)) & binf).sum()
dx = (((bf[:, 0] < lo[0]-0.002) | (bf[:, 0] > hi[0]+0.002)) & binf).sum()
print('  DEM past x       : %d of %d grains (%.2f%%)' % (dx, len(bf), 100*dx/len(bf)))
print('  DEM through z    : %d of %d grains (%.2f%%)' % (dz, len(bf), 100*dz/len(bf)))

# ---- 2. how wide are the seams between parts? ---------------------------
@wp.kernel
def nearest(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), maxd: float, out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], maxd)
    if r.result:
        cp = wp.mesh_eval_position(mesh, r.face, r.u, r.v)
        out[i] = wp.length(pts[i] - cp)
    else:
        out[i] = 1.0e9

def open_edge_verts(v, f):
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[cnt == 1].ravel()), (cnt == 1).sum()

print('\nseam gaps -- distance from each part\'s OPEN boundary edges to the other parts:')
names = list(parts)
wm = {}
for n_ in names:
    v, f = parts[n_]
    wm[n_] = wp.Mesh(wp.array(v, dtype=wp.vec3), wp.array(f.flatten(), dtype=int))
print('  %-6s %7s  %s' % ('part', 'open e', 'nearest other part (min gap, mm)'))
for a in names:
    va, fa = parts[a]
    idx, nopen = open_edge_verts(va, fa)
    if len(idx) == 0:
        print('  %-6s %7d  watertight' % (a, 0)); continue
    pts = wp.array(va[idx].astype(np.float32), dtype=wp.vec3)
    best = []
    for b in names:
        if b == a: continue
        o = wp.zeros(len(idx), dtype=float)
        wp.launch(nearest, dim=len(idx), inputs=[wm[b].id, pts, 0.5, o])
        o = o.numpy()
        if (o < 1e8).any(): best.append((float(o.min()), b, float(np.median(o[o < 1e8]))))
    best.sort()
    s = '  '.join('%s %.1f' % (b, g*1e3) for g, b, _ in best[:3])
    print('  %-6s %7d  %s' % (a, nopen, s))
