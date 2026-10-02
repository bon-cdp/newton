#!/usr/bin/env python3
"""
Rank the run-1 points of a calibration tag by a combined error and print the best
parameter set as calibrate.py --set arguments.

  score = |residual - meas| / 20 g  +  |r50 - video| / 40 mm  +  |pile - video| / 10 mm
          + curve rms / 20 g

r50 and pile come from render.py (frames.json); points without a render are skipped.
    python calibration/soybean_bucket/pick_best.py <tag> [--table]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render import summarize  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
tag = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(ROOT, "runs", "calib", tag, "*", "run1", "frames.json")):
    d = os.path.dirname(f)
    res = json.load(open(os.path.join(d, "result.json")))
    fr = summarize(json.load(open(f)))
    s, m = res["sim"], res["meas"]
    score = (abs(s["residual_g"] - m["residual_g"]) / 20 + abs(fr["r50_sim_mm"] - fr["r50_video_mm"]) / 40
             + abs(fr["pile_sim_mm"] - fr["pile_video_mm"]) / 10 + s.get("curve_rms_g", 99) / 20)
    rows.append((score, res["params"], s, fr))
rows.sort(key=lambda r: r[0])
if "--table" in sys.argv:
    for sc, p, s, fr in rows:
        print(f"{sc:6.2f}  res {s['residual_g']:5.0f}  rms {s.get('curve_rms_g', 0):5.1f}  "
              f"r50 {fr['r50_sim_mm']:4.0f}  pile {fr['pile_sim_mm']:5.1f}  {p}", file=sys.stderr)
best = {k: v for k, v in rows[0][1].items() if k != "vtk"}
print(" ".join(f"{k}={v}" for k, v in best.items()))
