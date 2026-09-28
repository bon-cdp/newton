"""Where, and what kind, is the mass BFA has in the cascade that we don't?

Two hypotheses make opposite predictions about the SPEED distribution:
  static pile (needs true static friction / history spring)
     -> BFA has a population near 0 m/s that we lack; our grains all keep moving.
  geometric (shells filling a pocket, or a retention feature we've closed off)
     -> both have similar speed distributions, we simply have fewer grains,
        and the deficit localises to a particular place rather than a speed band.
"""
import numpy as np, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame, DEM_MASS

CASC = (np.array([-5.62, -2.00, 3.95]), np.array([-5.20, -1.40, 4.25]))


def grab(pos, vel):
    lo, hi = CASC
    m = np.all((pos > lo) & (pos < hi), axis=1)
    return pos[m], np.linalg.norm(vel[m], axis=1)


bf = list(read_bfa_frames())[-6:]
bp = np.vstack([grab(f["pos"], f["vel"])[0] for f in bf])
bs = np.concatenate([grab(f["pos"], f["vel"])[1] for f in bf])
nb = len(bf)

run = sys.argv[1]
fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))[-6:]
meta = json.load(open(f"{run}/run.json"))
gm = meta["grain_mass"]
np_, ns = [], []
for fn in fs:
    p, v = read_mpm_frame(fn)
    a, b = grab(p, v)
    np_.append(a); ns.append(b)
np_ = np.vstack(np_); ns = np.concatenate(ns)

print(f"cascade region, averaged over {nb} frames each")
print(f"  BFA   {len(bp)/nb:7.0f} grains = {len(bp)/nb*DEM_MASS:5.2f} kg")
print(f"  {run:<5} {len(np_)/len(fs):7.0f} grains = {len(np_)/len(fs)*gm:5.2f} kg")

print("\nSPEED DISTRIBUTION (fraction of grains in each band)")
edges = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 100]
print(f"  {'speed m/s':>12} | {'BFA':>8} {'kg':>7} | {run[:10]:>8} {'kg':>7} | delta kg")
for a, b in zip(edges[:-1], edges[1:]):
    fb = np.mean((bs >= a) & (bs < b)) * len(bp) / nb * DEM_MASS
    fn_ = np.mean((ns >= a) & (ns < b)) * len(np_) / len(fs) * gm
    print(f"  {a:5.2f}..{b:5.2f} | {np.mean((bs>=a)&(bs<b)):8.3f} {fb:7.2f} | "
          f"{np.mean((ns>=a)&(ns<b)):8.3f} {fn_:7.2f} | {fn_-fb:+8.2f}")

print("\nSPATIAL: mass per 50 mm slab in y (kg)")
ye = np.arange(-2.00, -1.35, 0.05)
for a, b in zip(ye[:-1], ye[1:]):
    mb = np.sum((bp[:, 1] >= a) & (bp[:, 1] < b)) / nb * DEM_MASS
    mn = np.sum((np_[:, 1] >= a) & (np_[:, 1] < b)) / len(fs) * gm
    bar = "#" * int(mb * 20)
    print(f"  y {a:6.2f}..{b:6.2f} | BFA {mb:5.2f}  {run[:8]} {mn:5.2f}  delta {mn-mb:+5.2f}  {bar}")
