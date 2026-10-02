#!/usr/bin/env python3
"""
Markdown table of calibration results: one row per (parameter point, run), simulation vs
measurement, from result.json (calibrate.py) and frames.json (render.py) when present.

    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/report.py <tag> [<tag> ...] [--grep TEXT]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render import summarize  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
pat = sys.argv[sys.argv.index("--grep") + 1] if "--grep" in sys.argv else ""
if pat in args:
    args.remove(pat)

SKIP = {"vtk"}


def short(params):
    return ", ".join(f"{k} {v}" for k, v in sorted(params.items()) if k not in SKIP) or "defaults"


print("| point | run | residual g (meas) | t50 s (meas) | t90 s (meas) | curve rms g | "
      "pile mm (video) | r50 mm (video) | stops? |")
print("|---|---|---|---|---|---|---|---|---|")
for tag in args:
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "calib", tag, "*", "run*", "result.json"))):
        if pat and pat not in f:
            continue
        res = json.load(open(f))
        s, m = res["sim"], res["meas"]
        fr = os.path.join(os.path.dirname(f), "frames.json")
        sm = summarize(json.load(open(fr))) if os.path.exists(fr) else {}

        def pair(a, b, fmt="{:.0f}"):
            if a is None or (isinstance(a, float) and a != a):
                return "–"
            return fmt.format(a) + (f" ({fmt.format(b)})" if b is not None else "")

        print(f"| {short(res['params'])} | {res['run']} | {pair(s['residual_g'], m['residual_g'])} | "
              f"{pair(s.get('t50_s'), m.get('t50_s'), '{:.1f}')} | {pair(s.get('t90_s'), m.get('t90_s'), '{:.1f}')} | "
              f"{pair(s.get('curve_rms_g'), None)} | "
              f"{pair(sm.get('pile_sim_mm'), sm.get('pile_video_mm'))} | "
              f"{pair(sm.get('r50_sim_mm'), sm.get('r50_video_mm'))} | "
              f"{'no' if s.get('still_flowing') else 'yes'} |")
