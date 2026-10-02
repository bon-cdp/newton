#!/usr/bin/env python3
"""
Run bucket-discharge simulations for a set of parameter points and compare them with the
measured tests (runs.json).

Observables, all measured from the plug being pulled (t_open):
  residual_g      mass left in the bucket when the flow has stopped (scale)
  t_steady_s      end of the steady stream: the smoothed discharge rate first falls below
                  half its early (plateau) value (video: where the stream turns intermittent)
  t_last_s        last grain out: discharge within 2 grains of its final value (video)
  rate_gs         plateau discharge rate (no direct measurement; reported for reference)
  choke_s         run2 only: flow stopped with the bucket still holding most of the fill

    .venv/bin/python calibration/soybean_bucket/calibrate.py --tag base --runs run1 run4 \
        --set friction=0.35 --jobs 3
    .venv/bin/python calibration/soybean_bucket/calibrate.py --tag scan --runs run4 \
        --grid friction=0.2,0.35,0.5 rolling_friction=0.02,0.1
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from make_bucket import DEFAULTS, load_runs, make, parse_set  # noqa: E402

PY = sys.executable
TAIL_S = 2.5          # simulate this long past the measured last grain


def observables(hist_csv, t_open, gmass):
    rows = list(csv.DictReader(open(hist_csv)))
    t = np.array([float(r["time_s"]) for r in rows])
    bucket = np.array([float(r["bucket_mass_kg"]) for r in rows]) * 1e3
    i0 = int(np.searchsorted(t, t_open))
    m0 = bucket[i0 - 1]
    out = m0 - bucket[i0:]
    tt = t[i0:] - t_open
    final = out[-1]
    g = gmass * 1e3
    dt = np.median(np.diff(tt))
    w = max(1, int(round(0.5 / dt)))
    rate = np.convolve(np.gradient(out, tt), np.ones(w) / w, mode="same")
    k50 = int(np.searchsorted(out, 0.5 * final))
    plateau = float(np.median(rate[w:max(k50, w + 1)]))
    k_low = np.flatnonzero((rate < 0.5 * plateau) & (np.arange(len(rate)) > k50))
    t_steady = float(tt[k_low[0]]) if len(k_low) else float("nan")
    k_last = np.flatnonzero(out >= final - 2 * g)
    t_last = float(tt[k_last[0]])
    # still flowing at the end? (discharge rose in the last second)
    still = bool(out[-1] - out[np.searchsorted(tt, tt[-1] - 1.0)] > 2 * g)
    return dict(m0_g=round(float(m0), 1), residual_g=round(float(bucket[-1]), 1),
                discharged_g=round(float(final), 1), rate_gs=round(plateau, 1),
                t_steady_s=round(t_steady, 2), t_last_s=round(t_last, 2),
                still_flowing=still, curve_t=tt[::3].round(3).tolist(),
                curve_out_g=out[::3].round(1).tolist())


def flow_curve(run_id):
    """Measured (t, discharged g) from the video (flow_curves.json), or None."""
    p = os.path.join(HERE, "flow_curves.json")
    c = json.load(open(p))["runs"].get(run_id) if os.path.exists(p) else None
    return (np.array(c["t_s"]), np.array(c["discharged_g"])) if c else None


def t_frac(t, m, f, total=None):
    total = m[-1] if total is None else total
    k = np.flatnonzero(m >= f * total)
    return round(float(t[k[0]]), 2) if len(k) else float("nan")


def measured(r, run_id):
    m = dict(residual_g=r["m_end_g"])
    if r.get("choked"):
        m["choke_s"] = round(r["t_choke"] - r["t_open"], 2)
    else:
        m["t_steady_s"] = round(r["t_steady_end"] - r["t_open"], 2)
        m["t_last_s"] = round(r["t_last"] - r["t_open"], 2)
    c = flow_curve(run_id)
    if c is not None:
        m["t50_s"], m["t90_s"] = t_frac(*c, 0.5), t_frac(*c, 0.9)
    return m


def compare_curve(sim, run_id):
    """RMS (g) of simulated minus measured discharge over the measured record, and the
    times the SIMULATION takes to discharge 50% / 90% of the MEASURED total."""
    c = flow_curve(run_id)
    if c is None:
        return {}
    t, m = c
    ms = np.interp(t, sim["curve_t"], sim["curve_out_g"])
    st, so = np.array(sim["curve_t"]), np.array(sim["curve_out_g"])
    return dict(curve_rms_g=round(float(np.sqrt(np.mean((ms - m) ** 2))), 1),
                t50_s=t_frac(st, so, 0.5, m[-1]), t90_s=t_frac(st, so, 0.9, m[-1]))


def run_one(tag, run_id, params, label):
    r = load_runs()["runs"][run_id]
    p = dict(DEFAULTS)
    p.update(params)
    span = (r["t_choke"] if r.get("choked") else r["t_last"]) - r["t_open"]
    if "discharge_s" not in params:
        p["discharge_s"] = round(span + TAIL_S, 1)
    d = os.path.join(ROOT, "runs", "calib", tag, label, run_id)
    res_path = os.path.join(d, "result.json")
    if os.path.exists(res_path):
        return json.load(open(res_path))
    sc = make(run_id, d, p)
    t0 = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        runner = "dem_clumps.py" if p.get("clump") else "dem_run.py"
        subprocess.run([PY, "-u", os.path.join(ROOT, runner), sc, "--out",
                        os.path.join(d, "out")] + ([] if p["vtk"] else ["--no-vtk"]) + ["--checkpoint-at",
                        f"{p['t_open'] + p['discharge_s'] - 0.1:.3f}"], stdout=log, stderr=subprocess.STDOUT,
                       cwd=ROOT, check=True)
    gm = 4.0 / 3.0 * np.pi * p["radius"] ** 3 * p["density"]
    obs = observables(os.path.join(d, "out", "history.csv"), p["t_open"], gm)
    obs.update(compare_curve(obs, run_id))
    obs.update(pile(os.path.join(d, "out", "checkpoint.npz"), r["drop_mm"] * 1e-3, p["radius"]))
    res = dict(run=run_id, label=label, params=params, sim=obs, meas=measured(r, run_id),
               wall_s=round(time.time() - t0, 1))
    json.dump(res, open(res_path, "w"))
    return res


def pile(ckpt, h, radius):
    """Final pile on the belt: apex height (top of the highest grain within 20 mm of the
    hole axis, below the bucket) and the radius holding 90% of the belt mass."""
    if not os.path.exists(ckpt):
        return {}
    z = np.load(ckpt)
    act = (z["flags"] & 1).astype(bool)
    q = z["q"][act]
    still = np.linalg.norm(z["qd"][act][:, :3], axis=1) < 0.05   # not grains still falling
    hole = json.load(open(os.path.join(HERE, "hole_contour.json")))
    c = np.asarray(hole["contour_mm"]).mean(axis=0) * 1e-3
    on_belt = q[:, 1] < h - 4.0 * radius          # not grains hanging in the hole
    rr = np.hypot(q[:, 0] - c[0], q[:, 2] - c[1])
    core = on_belt & still & (rr < 0.02)
    apex = float(q[core, 1].max() + radius) if core.any() else 0.0
    return dict(pile_apex_mm=round(apex * 1e3, 1),
                pile_r90_mm=round(float(np.percentile(rr[on_belt], 90)) * 1e3, 1)
                if on_belt.any() else 0.0)


def label_of(params):
    return "_".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"
                    for k, v in sorted(params.items())) or "default"


def summary_line(res):
    s, m = res["sim"], res["meas"]
    return (f"{res['run']} {res['label']:<44} m0 {s['m0_g']:6.0f}  residual {s['residual_g']:6.1f} "
            f"(meas {m['residual_g']:6.1f})  t_steady {s['t_steady_s']:5.2f} "
            f"({m.get('t_steady_s', float('nan')):5.2f})  t_last {s['t_last_s']:5.2f} "
            f"({m.get('t_last_s', m.get('choke_s', float('nan'))):5.2f})  t50 {s.get('t50_s', float('nan')):5.2f} "
            f"({m.get('t50_s', float('nan')):5.2f})  t90 {s.get('t90_s', float('nan')):5.2f} "
            f"({m.get('t90_s', float('nan')):5.2f})  rms {s.get('curve_rms_g', float('nan')):5.1f} g  "
            f"rate {s['rate_gs']:5.1f} g/s"
            + ("  STILL FLOWING" if s["still_flowing"] else "") + f"  [{res['wall_s']:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", nargs="+", default=["run1", "run4"])
    ap.add_argument("--set", nargs="*", help="fixed key=value overrides")
    ap.add_argument("--grid", nargs="*", help="key=v1,v2,... (full factorial)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summary", action="store_true",
                    help="re-score every saved run under runs/calib/<tag> with the current metrics")
    a = ap.parse_args()
    if a.summary:
        import glob
        for f in sorted(glob.glob(os.path.join(ROOT, "runs", "calib", a.tag, "*", "run*", "result.json"))):
            res = json.load(open(f))
            d = os.path.dirname(f)
            r = load_runs()["runs"][res["run"]]
            sc = json.load(open(os.path.join(d, "scenario.json")))
            p = sc["notes"]["params"]
            gm = 4.0 / 3.0 * np.pi * p["radius"] ** 3 * p["density"]
            obs = observables(os.path.join(d, "out", "history.csv"), p["t_open"], gm)
            obs.update(compare_curve(obs, res["run"]))
            obs.update(pile(os.path.join(d, "out", "checkpoint.npz"), r["drop_mm"] * 1e-3, p["radius"]))
            res["sim"], res["meas"] = obs, measured(r, res["run"])
            json.dump(res, open(f, "w"))
            print(summary_line(res) + f"  pile {obs.get('pile_apex_mm', 0):.0f} mm r90 "
                  f"{obs.get('pile_r90_mm', 0):.0f} mm")
        sys.exit(0)
    fixed = parse_set(a.set)
    axes = [(g.split("=")[0], [parse_set([f"x={v}"])["x"] for v in g.split("=")[1].split(",")])
            for g in (a.grid or [])]
    points = [dict(fixed, **dict(zip([k for k, _ in axes], combo)))
              for combo in itertools.product(*[v for _, v in axes])] if axes else [fixed]
    jobs = [(pt, rid) for pt in points for rid in a.runs]
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_one, a.tag, rid, pt, label_of(pt)) for pt, rid in jobs]
        for f in futs:
            try:
                print(summary_line(f.result()), flush=True)
            except Exception as e:  # keep the sweep going
                print("FAILED:", e, flush=True)
