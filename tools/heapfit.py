"""Re-measure every heap the same way, from the saved grains.

The in-run print reported two estimators that disagreed by up to 3 deg on some cases,
which means one of them is picking up something that is not the free surface.  Here the
profile is printed in full so the disagreement can be seen rather than averaged over.
"""
import glob, json, math, os, sys
import numpy as np

BASE_R = 0.20

def analyse(tag):
    f = os.path.join(tag, "heap.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    p, v = d["pos"], d["vel"]
    still = np.linalg.norm(v, axis=1) < 0.05
    p = p[still & (p[:, 1] > -0.01)]
    rr = np.hypot(p[:, 0], p[:, 2])
    bins = np.linspace(0.0, BASE_R, 11)
    surf, mid = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        k = (rr >= a) & (rr < b)
        if k.sum() < 15:
            surf.append(np.nan); mid.append(0.5*(a+b)); continue
        surf.append(np.percentile(p[k][:, 1], 95)); mid.append(0.5*(a+b))
    surf, mid = np.array(surf), np.array(mid)
    ok = np.isfinite(surf)
    # fit the flank only: drop the flat cap and the toe
    sel = ok & (mid > 0.25*BASE_R) & (mid < 0.90*BASE_R)
    slope = np.polyfit(mid[sel], surf[sel], 1)[0] if sel.sum() >= 4 else np.nan
    apex = float(np.nanmax(surf[ok & (mid < 0.2*BASE_R)])) if (ok & (mid < 0.2*BASE_R)).any() else np.nan
    return dict(tag=tag, n=len(p), apex=apex,
                a_apex=math.degrees(math.atan(apex/BASE_R)) if apex == apex else np.nan,
                a_fit=math.degrees(math.atan(-slope)) if slope == slope else np.nan,
                mid=mid, surf=surf)

rows = [r for r in (analyse(t) for t in sorted(sys.argv[1:])) if r]
print(f"  {'case':<26} {'n':>5} {'apex':>7} {'apex/R':>8} {'flank fit':>10}   BFA 23.3 deg")
for r in rows:
    print(f"  {r['tag']:<26} {r['n']:5d} {r['apex']:7.3f} {r['a_apex']:8.2f} {r['a_fit']:10.2f}")
print("\n  surface height (m) vs radius -- a conical heap falls on a straight line")
print("  r (m)   " + "".join(f"{r['tag'].replace('aor_',''):>16}" for r in rows))
for i in range(len(rows[0]["mid"])):
    line = f"  {rows[0]['mid'][i]:5.3f}   "
    for r in rows:
        line += f"{r['surf'][i]:16.4f}" if np.isfinite(r["surf"][i]) else f"{'--':>16}"
    print(line)
