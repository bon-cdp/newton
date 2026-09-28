"""Pile height averaged over frames, with scatter.

A single frame moves the 98th-percentile height by ~4 cm, which is the same size as the
differences between runs -- so a one-frame comparison cannot rank them.  Average over the
steady window and report the standard deviation so a difference can be called real or not.
"""
import numpy as np, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame, DEM_MASS

XMAX, NF = -4.8, 30

def stats(pos, vel, gm):
    m = pos[:, 0] < XMAX
    p, v = pos[m], np.linalg.norm(vel[m], axis=1)
    slow = v < 1.5
    tos = np.percentile(p[slow][:, 1], 98) if slow.sum() > 20 else np.nan
    return len(p)*gm, slow.mean(), tos

def report(label, rows):
    a = np.array(rows)
    print(f"  {label:<12} kg {a[:,0].mean():5.2f}+-{a[:,0].std():.2f}   "
          f"slow {a[:,1].mean():.3f}+-{a[:,1].std():.3f}   "
          f"top-of-slow {a[:,2].mean():+.3f} +- {a[:,2].std():.3f} m   (n={len(a)} frames)")

print(f"averaged over the last {NF} frames; +- is the frame-to-frame standard deviation\n")
bf = list(read_bfa_frames())[-NF:]
report("BFA", [stats(f["pos"], f["vel"], DEM_MASS) for f in bf])
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))[-NF:]
    if len(fs) < 5: continue
    gm = json.load(open(f"{run}/run.json"))["grain_mass"]
    report(run, [stats(*read_mpm_frame(f), gm) for f in fs])
