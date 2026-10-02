#!/usr/bin/env python3
"""
Discharge curves from the test videos.

Grains cross a band of white wall between the bucket bottom and the pile at nearly the
same speed (free fall from the hole), so the fraction of bean-coloured pixels in the band
is proportional to the mass flow rate.  Integrated from the plug pull and scaled to the
weighed discharge (m0 - m_end) it gives the measured discharged mass vs time.

The band is fixed in the WORLD: 50 mm wide across the bucket axis, from 10 mm below the
bucket bottom down to 40% of the drop (above any pile), in the vertical plane through the
axis.  It is projected through each frame's tracked camera (camera_<run>.json), so it
follows the stream when the handheld camera tilts.  Run2 has no curve: its pile grows
into the band and chokes the hole, which the crop strips time directly (runs.json t_choke).

Needs OpenCV (not in the project venv):
    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/measure_flow.py /path/to/video-runs
Writes flow_curves.json next to this file.
"""
import json
import os
import sys

import cv2
import numpy as np

import camera as cam

HERE = os.path.dirname(os.path.abspath(__file__))
FLOW_RUNS = ("run1", "run3", "run4")
HALF_W = 0.025


RIM = cam.BUCKET[np.abs(cam.BUCKET[:, 1]) < 1e-9]      # the bottom disc


def band_px(drop, R, C):
    """(y0, y1, x0, x1) full-resolution pixels of the world band for one camera pose.
    The top starts 8 px below the lowest projected point of the bucket bottom in the
    band's columns: from below (runs 3, 4) the front rim hangs lower in the image than
    the axis plane does."""
    y_top, y_bot = -0.010, -0.4 * drop
    P = np.array([[x, y, 0.0] for x in (-HALF_W, HALF_W) for y in (y_top, y_bot)])
    uv, _ = cam.project(P, R, C)
    x0, x1 = int(uv[:, 0].min()), int(uv[:, 0].max())
    y0, y1 = int(uv[:, 1].min()), int(uv[:, 1].max())
    rim, z = cam.project(RIM, R, C)
    cols = (z > 0) & (rim[:, 0] >= x0) & (rim[:, 0] <= x1)
    if cols.any():
        y0 = max(y0, int(rim[cols, 1].max()) + 8)
    y1 = max(y1, y0 + 20)
    return (max(y0, 0), min(y1, cam.H), max(x0, 0), min(x1, cam.W))


def camera_track(rid):
    """(times, rotations, camera positions) per video frame from camera_<rid>.json."""
    cj = json.load(open(os.path.join(HERE, f"camera_{rid}.json")))
    Rs = [np.array(R) for R in cj["R"]]
    Cs = [np.array(c) for c in cj["Cs"]] if "Cs" in cj else [np.array(cj["C"])] * len(Rs)
    return np.array(cj["t"]), Rs, Cs


def pose_at(track, t):
    tv, Rs, Cs = track
    i = min(int(np.searchsorted(tv, t)), len(Rs) - 1)
    return Rs[i], Cs[i]


def bean_fraction(img):
    if not img.size:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] > 55) & (hsv[..., 2] > 40)).mean())


def occupancy(path, rid, drop):
    track = camera_track(rid)
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS)
    occ, k = [], 0
    while True:
        ok, im = c.read()
        if not ok:
            break
        y0, y1, x0, x1 = band_px(drop, *pose_at(track, k / fps))
        occ.append(bean_fraction(im[y0:y1, x0:x1]))
        k += 1
    return np.arange(len(occ)) / fps, np.array(occ)


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))["runs"]
    out = {"method": __doc__.strip().splitlines()[0], "runs": {}}
    for rid in FLOW_RUNS:
        r = runs[rid]
        drop = r["drop_mm"] * 1e-3
        t, occ = occupancy(os.path.join(sys.argv[1], r["video"]), rid, drop)
        i0 = int(np.searchsorted(t, r["t_open"]))
        floor = float(np.median(occ[-int(1.0 / np.median(np.diff(t))):]))   # last second: no flow
        q = np.clip(occ[i0:] - floor, 0.0, None)
        # the towel is still leaving the band for a few frames: count flow once it has gone
        clear = int(np.flatnonzero(occ[i0:] < 0.4)[0])
        q[:clear] = q[clear]
        tt = t[i0:] - t[i0]
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(tt))])
        m = (r["m0_g"] - r["m_end_g"]) * cum / cum[-1]
        # resample at 10 Hz
        tg = np.arange(0.0, tt[-1], 0.1)
        mg = np.interp(tg, tt, m)
        out["runs"][rid] = dict(band_world_m=[HALF_W, -0.010, -0.4 * drop], floor=round(floor, 4),
                                t_s=tg.round(2).tolist(), discharged_g=mg.round(1).tolist(),
                                occupancy_t_s=tt[::max(1, int(round(len(tt) / len(tg))))].round(3).tolist(),
                                occupancy=occ[i0:][::max(1, int(round(len(tt) / len(tg))))].round(4).tolist())
        k = lambda f: float(tg[np.searchsorted(mg, f * mg[-1])])
        print(f"{rid}: {mg[-1]:.0f} g; 50% at {k(0.5):.2f} s, 90% at {k(0.9):.2f} s, "
              f"99% at {k(0.99):.2f} s; first-second rate {mg[10]:.0f} g/s")
    json.dump(out, open(os.path.join(HERE, "flow_curves.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
