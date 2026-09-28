"""Local packing fraction and ordering in the cascade: BFA vs Newton.

Monodisperse spheres cannot randomly pack above ~0.64.  If a run is jamming because the
grains have crystallised, the nearest-neighbour distance distribution collapses onto a
sharp peak at one diameter instead of the broad RCP distribution.
"""
import numpy as np, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame
from scipy.spatial import cKDTree

D = 0.012
VS = 4.0 / 3.0 * np.pi * (D / 2) ** 3


def analyse(label, pos, region):
    lo, hi = region
    m = np.all((pos > lo) & (pos < hi), axis=1)
    p = pos[m]
    if len(p) < 200:
        print(f"  {label:<22} only {len(p)} grains in region"); return
    # local packing via 24 mm voxels, reported over well-occupied cells
    h = 0.024
    idx = np.floor(p / h).astype(int)
    _, inv, cnt = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    occ = cnt[cnt >= 4]
    phi = occ * VS / h**3
    # nearest-neighbour spacing: sharp peak at 1.0 D => ordered
    t = cKDTree(p)
    d, _ = t.query(p, k=2)
    nn = d[:, 1] / D
    print(f"  {label:<22} n={len(p):6d}  phi p50={np.median(phi):.3f} p90={np.percentile(phi,90):.3f}"
          f"   NN/D: p5={np.percentile(nn,5):.3f} p50={np.median(nn):.3f} "
          f"frac<1.02={np.mean(nn<1.02):.2f}")


# cascade region (the deflector assembly)
region = (np.array([-5.60, -1.95, 3.98]), np.array([-5.23, -1.00, 4.20]))
print("cascade packing (region x -5.60..-5.23, y -1.95..-1.00, z 3.98..4.20)")
print("  reference: random close packing phi=0.64, fcc=0.74; BFA spec needs 0.714")
print()
bf = list(read_bfa_frames())[-1]
analyse("BFA DEM", bf["pos"], region)
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))
    if not fs:
        print(f"  {run:<22} no VTK"); continue
    p, _v = read_mpm_frame(fs[-1])
    mu = json.load(open(f"{run}/run.json")).get("mu", "?")
    analyse(f"{run} (mu={mu})", p, region)
