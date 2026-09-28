"""Speed along the whole spout, BFA vs ours.  Where does the divergence START?

The chute tube (y 0..2.5) already matches.  The deficit band y -1.60..-1.40 is the
LAST 0.18 m of the spout (its exit is y = -1.579), and there BFA is 4x slower than us.
Something between y=0 and the exit brakes BFA's stream and not ours.
"""
import numpy as np, sys, glob, json
sys.path.insert(0, '/home/s/Documents/newton-dem')
from compare_bfa_mpm import read_bfa_frames, read_mpm_frame, DEM_MASS

edges = np.arange(-1.6, 3.0, 0.2)

def prof(pos, vel, gm):
    v = np.linalg.norm(vel, axis=1)
    m, s, sl = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (pos[:,1] >= a) & (pos[:,1] < b)
        m.append(k.sum()*gm)
        s.append(v[k].mean() if k.sum() > 5 else np.nan)
        sl.append((v[k] < 1.5).mean() if k.sum() > 5 else np.nan)
    return np.array(m), np.array(s), np.array(sl)

bf = list(read_bfa_frames())[-1]
bm, bs, bl = prof(bf["pos"], bf["vel"], DEM_MASS)
cols = {}
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f"{run}/frame_*_particles.vtk"))
    gm = json.load(open(f"{run}/run.json"))["grain_mass"]
    p, v = read_mpm_frame(fs[-1])
    cols[run] = prof(p, v, gm)

print("  y band          BFA kg  BFA m/s  BFA slow |" + "".join(f"{k[:10]:>11} m/s" for k in cols))
for i in range(len(edges)-2, -1, -1):
    line = f"  {edges[i]:+5.2f}..{edges[i+1]:+5.2f}  {bm[i]:6.2f} {bs[i]:8.2f} {bl[i]:9.2f} |"
    for kk in cols:
        m, s, _l = cols[kk]
        line += "  %5.2f kg %6.2f" % (m[i], s[i])
    print(line)
