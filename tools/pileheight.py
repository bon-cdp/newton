"""How high does the PILE reach, as distinct from mass per band?

Mass-per-band can match while the dense/slow region -- what the eye reads as the pile --
sits at a different height.  Two definitions, both over the cascade end of the machine:
  slow material  (|v| < 1.5 m/s)  = piled rather than flowing
  dense material (local phi > 0.30 in 24 mm voxels) = packed rather than a curtain
"""
import numpy as np, sys, glob
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame, DEM_MASS

XMAX = -4.8          # cascade end of the machine
D = 0.012
VS = 4/3*np.pi*(D/2)**3

def profile(pos, vel, gm, label):
    m = pos[:, 0] < XMAX
    p, v = pos[m], np.linalg.norm(vel[m], axis=1)
    slow = v < 1.5
    h = 0.024
    idx = np.floor(p/h).astype(int)
    uniq, inv, cnt = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    phi = cnt*VS/h**3
    dense_cell = phi > 0.30
    dense = dense_cell[inv]
    def top(mask, frac=0.02):
        if mask.sum() < 20: return float('nan')
        return np.percentile(p[mask][:, 1], 100*(1-frac))
    print(f"  {label:<12} n={len(p):6d} ({len(p)*gm:5.2f} kg)  "
          f"slow {slow.mean():.2f}  top-of-slow {top(slow):+.3f} m  "
          f"top-of-dense {top(dense):+.3f} m  max y {p[:,1].max():+.3f}")
    return p, v, slow, dense

print(f"cascade end of machine (x < {XMAX});  'top' = 98th percentile height\n")
bf = list(read_bfa_frames())[-1]
profile(bf["pos"], bf["vel"], DEM_MASS, "BFA")
import json
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))
    if not fs: continue
    gm = json.load(open(f"{run}/run.json"))["grain_mass"]
    p, v = read_mpm_frame(fs[-1])
    profile(p, v, gm, run)

print("\nmass vs height in the cascade end (kg per 100 mm):")
edges = np.arange(-2.0, -0.4, 0.1)
rows = {}
bm = bf["pos"][bf["pos"][:, 0] < XMAX]
rows["BFA"] = np.histogram(bm[:, 1], edges)[0]*DEM_MASS
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))
    if not fs: continue
    gm = json.load(open(f"{run}/run.json"))["grain_mass"]
    p, _ = read_mpm_frame(fs[-1])
    rows[run] = np.histogram(p[p[:, 0] < XMAX][:, 1], edges)[0]*gm
hdr = "  y band    " + "".join(f"{k[:11]:>12}" for k in rows)
print(hdr)
for i in range(len(edges)-1):
    line = f"  {edges[i]:+5.2f}..{edges[i+1]:+5.2f}"
    for k in rows: line += f"{rows[k][i]:12.2f}"
    print(line)
