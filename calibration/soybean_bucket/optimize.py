#!/usr/bin/env python3
"""
Tune the ellipsoidal soybean against the fit runs (1 and 4) by Bayesian optimisation.

Each evaluation simulates both fit runs (reference preset, VTK frames), renders them
through the tracked cameras, and scores the mismatch (lower is better, summed over runs):

  residual      ((sim - meas) / (3% of meas))^2        the weighed residual mass
  curve         (discharge-curve rms / 20 g)^2           video discharge curve
  t90           ((sim - meas) / 0.8 s)^2                 time to 90% discharged
  pile          ((sim - video) / 6 mm)^2                 final heap height (same pixels)
  spread  0.25 x ((min(r50,200) - min(r50v,200)) / 60 mm)^2
                belt coverage radius, capped at 200 mm and down-weighted: the real belt is
                finite and beans that roll off it are lost in the footage

Shape: semi-axes from the volume-equivalent diameter (fixed, measured 5.4 mm) and two
aspect ratios, length/width and width/thickness.  Also tuned: grain rolling friction and
restitution.  Fixed: sliding friction 0.35, wall rolling 0.01, steel 0.3, belt 0.5, rot_damp 0.

Runs in the OpenCV venv (scikit-optimize, render.py); simulations use the project venv.
Resumable: evaluations are appended to runs/calib/opt/log.jsonl and re-told on restart.

    setsid nohup runs/perf/vtkcheck/bin/python calibration/soybean_bucket/optimize.py \
        --calls 24 > runs/logs/optimize.log 2>&1 &
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import calibrate  # noqa: E402
from render import summarize  # noqa: E402

calibrate.PY = "/home/s/Documents/newton-dem/.venv/bin/python"   # simulations need warp
CV = sys.executable
VIDEOS = "/home/s/Documents/newton-dem/video-runs"
TAG = "opt"
LOG = os.path.join(ROOT, "runs", "calib", TAG, "log.jsonl")
D_EQ_MM = 5.4
FIT_RUNS = ("run1", "run4")
FIXED = dict(vtk=1, rot_damp=0.0, rot_damp_wall=0.0, wall_rolling_friction=0.01, friction=0.35)
SPACE = [("aspect_lw", 1.0, 1.4), ("aspect_wt", 1.0, 1.3),
         ("rolling_friction", 0.0, 0.08), ("restitution", 0.6, 0.9)]


def axes_of(lw, wt):
    """Semi-axes (mm) with a/b = lw, b/c = wt and volume-equivalent diameter D_EQ_MM."""
    r = D_EQ_MM / 2.0
    c = r / (lw * wt * wt) ** (1.0 / 3.0)
    b = wt * c
    a = lw * b
    return round(a, 3), round(b, 3), round(c, 3)


SHAPE = None          # (length/width, width/thickness) when the shape is fixed (--fix-shape)
FIXED_E = None        # restitution when held (--fix-restitution), e.g. chosen by eye from the
                      # early-impact footage, which the scores above cannot see


def params_of(x):
    d = dict(zip([s[0] for s in SPACE], x))
    lw, wt = SHAPE if SHAPE else (d["aspect_lw"], d["aspect_wt"])
    a, b, c = axes_of(lw, wt)
    p = dict(FIXED)
    e = FIXED_E if FIXED_E is not None else d["restitution"]
    p.update(ell_a=a, ell_b=b, ell_c=c, rolling_friction=round(d["rolling_friction"], 4),
             restitution=round(e, 3))
    if "friction" in d:
        p["friction"] = round(d["friction"], 3)
    return p


def score_run(res, fr):
    s, m = res["sim"], res["meas"]
    terms = {}
    terms["residual"] = ((s["residual_g"] - m["residual_g"]) / (0.03 * m["residual_g"])) ** 2
    terms["curve"] = (s.get("curve_rms_g", 60.0) / 20.0) ** 2
    t90 = s.get("t90_s")
    terms["t90"] = 9.0 if t90 is None or (isinstance(t90, float) and math.isnan(t90)) \
        else ((t90 - m["t90_s"]) / 0.8) ** 2
    terms["pile"] = ((fr["pile_sim_mm"] - fr["pile_video_mm"]) / 6.0) ** 2
    terms["spread"] = 0.25 * ((min(fr["r50_sim_mm"], 200) - min(fr["r50_video_mm"], 200)) / 60.0) ** 2
    return terms


def evaluate(x):
    p = params_of(x)
    label = calibrate.label_of(p)
    total, detail = 0.0, {}
    for rid in FIT_RUNS:
        res = calibrate.run_one(TAG, rid, p, label)
        d = os.path.join(ROOT, "runs", "calib", TAG, label, rid)
        if not os.path.exists(os.path.join(d, "frames.json")):
            subprocess.run([CV, os.path.join(HERE, "render.py"), VIDEOS, d], cwd=HERE, check=True,
                           stdout=subprocess.DEVNULL)
        fr = summarize(json.load(open(os.path.join(d, "frames.json"))))
        t = score_run(res, fr)
        detail[rid] = dict(terms={k: round(v, 3) for k, v in t.items()},
                           residual_g=res["sim"]["residual_g"], curve_rms_g=res["sim"].get("curve_rms_g"),
                           t90_s=res["sim"].get("t90_s"), pile_mm=fr["pile_sim_mm"], r50_mm=fr["r50_sim_mm"])
        total += sum(t.values())
    return total, p, detail


def main():
    from skopt import Optimizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--initial", type=int, default=6)
    ap.add_argument("--tag", default="opt", help="runs/calib/<tag>, with its own log.jsonl")
    ap.add_argument("--fix-shape", nargs=2, type=float, metavar=("LW", "WT"),
                    help="hold length/width and width/thickness; tune rolling friction, "
                         "restitution and grain sliding friction instead")
    ap.add_argument("--fix-restitution", type=float, default=None,
                    help="hold restitution (with --fix-shape: tune rolling and sliding friction only)")
    a = ap.parse_args()
    global TAG, LOG, SPACE, SHAPE, FIXED_E
    TAG = a.tag
    LOG = os.path.join(ROOT, "runs", "calib", TAG, "log.jsonl")
    start = [1.13, 1.125, 0.025, 0.8]
    if a.fix_shape:
        SHAPE = tuple(a.fix_shape)
        SPACE = [("rolling_friction", 0.0, 0.10), ("restitution", 0.6, 0.95), ("friction", 0.25, 0.6)]
        start = [0.045, 0.8, 0.35]
        if a.fix_restitution is not None:
            FIXED_E = a.fix_restitution
            SPACE = [("rolling_friction", 0.0, 0.10), ("friction", 0.2, 0.6)]
            start = [0.05, 0.30]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    opt = Optimizer([(lo, hi) for _n, lo, hi in SPACE], base_estimator="GP", acq_func="EI",
                    n_initial_points=a.initial, initial_point_generator="lhs", random_state=0)
    done = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    for rec in done:
        opt.tell(rec["x"], rec["score"])
    # the hand-picked starting guess first (soybean-like proportions, the clump fit's values)
    queue = [start] if not done else []
    while len(done) < a.calls:
        x = queue.pop(0) if queue else opt.ask()
        t0 = time.time()
        score, p, detail = evaluate(x)
        rec = dict(n=len(done) + 1, x=[float(v) for v in x], score=round(score, 3), params=p,
                   detail=detail, minutes=round((time.time() - t0) / 60, 1))
        opt.tell(rec["x"], rec["score"])
        done.append(rec)
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        best = min(done, key=lambda r: r["score"])
        print(f"{time.strftime('%a %H:%M')} eval {rec['n']}: score {rec['score']:.2f} "
              f"x {np.round(x, 3).tolist()} ({rec['minutes']} min) | best {best['score']:.2f} "
              f"at eval {best['n']} x {np.round(best['x'], 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
