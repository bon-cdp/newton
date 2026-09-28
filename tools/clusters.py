"""Did BFA integrate single spheres or multi-sphere clusters?

The prj defines 7 sizes with 6 typed "Cluster", so it matters.  Two signatures:
  * cluster sub-spheres OVERLAP, so nearest-neighbour distance has a population well
    below one diameter, where free spheres pile at exactly 1.0 D;
  * sub-spheres of one rigid clump move together, so their velocities are near-identical
    while independent grains in contact are not.
"""
import numpy as np, sys
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames
from scipy.spatial import cKDTree

D = 0.012
f = list(read_bfa_frames())[-1]
p, v = f["pos"], f["vel"]
print(f"{len(p)} BFA grains, radius uniq = {np.unique(np.round(f['rad'],6))}")

t = cKDTree(p)
d, idx = t.query(p, k=2)
nn = d[:, 1] / D
print(f"\nnearest-neighbour separation / diameter")
for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.02), (1.02, 1.5), (1.5, 99)]:
    frac = np.mean((nn >= lo) & (nn < hi))
    tag = "  <- cluster sub-spheres would live here" if hi <= 0.95 else ""
    print(f"  {lo:4.2f}..{hi:5.2f} : {frac:6.3f}{tag}")
print(f"  min {nn.min():.4f}   p1 {np.percentile(nn,1):.4f}")

# velocity agreement with the nearest neighbour
vn = np.linalg.norm(v - v[idx[:, 1]], axis=1)
sp = np.linalg.norm(v, axis=1)
rel = vn / np.maximum(sp, 1e-9)
print(f"\nvelocity difference to nearest neighbour, relative to own speed")
print(f"  p5 {np.percentile(rel,5):.4f}  p50 {np.percentile(rel,50):.4f}  "
      f"frac < 0.01 (rigid) {np.mean(rel<0.01):.4f}")
print(f"\nmass check: flow 8.6111 kg/s / single-sphere mass 0.89943 g = "
      f"{8.6111/0.89943e-3:.1f} grains/s;  est Prt Count says 9575")
