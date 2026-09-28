"""WHERE is the missing 0.2 m of material?  Not how much -- where in x,z and on what.

The deficit band y -1.60..-1.40 sits at the Spout exit (-1.579) / Top deflector
(-1.683..-1.472).  If BFA's extra mass shares our x,z footprint it is a taller pile on
the same plate; if it sits elsewhere it is resting on geometry we are not retaining.
"""
import numpy as np, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame, DEM_MASS
from bfa_replication_mpm import COLLIDER_PARTS, load_part

LO, HI = -1.60, -1.40

def show(pos, vel, gm, label):
    m = (pos[:,1] > LO) & (pos[:,1] < HI)
    p, v = pos[m], np.linalg.norm(vel[m], axis=1)
    if len(p) < 5:
        print(f"  {label:<12} n={len(p)}"); return
    print(f"  {label:<12} n={len(p):5d} ({len(p)*gm:5.2f} kg)  slow {(v<1.5).mean():.2f}  "
          f"x {np.percentile(p[:,0],5):+.2f}..{np.percentile(p[:,0],95):+.2f}  "
          f"z {np.percentile(p[:,2],5):+.2f}..{np.percentile(p[:,2],95):+.2f}  "
          f"|v| p50 {np.median(v):.2f}")

print(f"material in the deficit band y {LO}..{HI}\n")
bf = list(read_bfa_frames())[-1]
show(bf["pos"], bf["vel"], DEM_MASS, "BFA")
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))
    if not fs: continue
    gm = json.load(open(f"{run}/run.json"))["grain_mass"]
    p, v = read_mpm_frame(fs[-1])
    show(p, v, gm, run)

print("\ncollider parts overlapping that band (y extent of each STL):")
for name, ts, fx, fl in COLLIDER_PARTS:
    v, f = load_part(name, fx, fl)
    ylo, yhi = v[:,1].min(), v[:,1].max()
    if yhi > LO and ylo < HI:
        print(f"  {name:<8} y {ylo:+.3f}..{yhi:+.3f}  x {v[:,0].min():+.2f}..{v[:,0].max():+.2f}"
              f"  z {v[:,2].min():+.2f}..{v[:,2].max():+.2f}  two_sided={ts}")
