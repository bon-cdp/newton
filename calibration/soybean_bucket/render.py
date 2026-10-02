#!/usr/bin/env python3
"""
Frame-by-frame comparison of a simulation with its test video.

For every video frame after the plug is pulled, the simulation frame at the same time
since opening is projected through that frame's camera (camera_<run>.json, camera.py) and
drawn as discs.  Both images are then measured on the SAME pixels:

  band occupancy   fraction of a band under the hole covered by grains (flow-rate proxy,
                   now comparable in absolute terms: same band, same camera, same grains)
  pile top         highest grain pixel above the belt near the hole axis, converted to
                   height above the belt using the camera (mm)

and a side-by-side video (left footage, right simulation, outline of the model bucket on
both) is written for a visual check.

    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/render.py <video-runs dir> \
        runs/calib/<tag>/<label>/<run> [--video]
Writes frames.json (per-frame metrics) and, with --video, compare.mp4 into the run dir.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import camera as cam
from measure_flow import BAND          # the same band as the occupancy curves

HERE = os.path.dirname(os.path.abspath(__file__))
BEAN_BGR = (90, 170, 215)


def read_points(path):
    raw = open(path, "rb").read()
    k = raw.index(b"POINTS")
    e = raw.index(b"\n", k)
    n = int(raw[k:e].split()[1])
    return np.frombuffer(raw, dtype=">f4", count=n * 3, offset=e + 1).reshape(n, 3).astype(np.float64)


def render(P, R, C, radius):
    """Mask of grains (uint8) for world points P, painter's algorithm (no colour needed)."""
    uv, z = cam.project(P, R, C)
    ok = (z > 0.02) & (uv[:, 0] > -50) & (uv[:, 0] < cam.W + 50) & (uv[:, 1] > -50) & (uv[:, 1] < cam.H + 50)
    uv, z = uv[ok], z[ok]
    rad = np.maximum(1, np.round(cam.F_PX * radius / z)).astype(int)
    m = np.zeros((cam.H, cam.W), np.uint8)
    for (u, v), r in zip(np.round(uv).astype(int), rad):
        cv2.circle(m, (int(u), int(v)), int(r), 255, -1)
    return m


def bean_mask(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 1] > 55) & (hsv[..., 2] > 40)) * 255).astype(np.uint8)


def pile_top_mm(mask, R, C, drop, cols, y_floor):
    """Height above the belt of the highest grain pixel in `cols`, looking from the belt
    upwards until the first gap: the ray through that pixel is intersected with the
    vertical plane through the bucket axis facing the camera."""
    sub = mask[:y_floor, cols[0]:cols[1]] > 0
    rows = np.flatnonzero(sub.mean(axis=1) > 0.35)        # rows mostly covered
    if not len(rows):
        return 0.0
    # contiguous block touching the bottom of the window
    r = rows[::-1]
    top = r[0]
    for a in r[1:]:
        if top - a > 3:
            break
        top = a
    u = 0.5 * (cols[0] + cols[1])
    ray_c = np.linalg.inv(cam.K) @ np.array([u, top, 1.0])
    ray_w = R.T @ ray_c
    # plane z = 0 (through the axis, normal to world z, the camera looks along +z)
    s = -C[2] / ray_w[2]
    y = C[1] + s * ray_w[1]
    return float((y + drop) * 1e3)


RADII = np.arange(0.0, 0.32, 0.02)


def belt_coverage(mask, R, C, drop, step=4):
    """Fraction of the belt covered by grains in rings around the hole axis: every
    (step-th) pixel below the horizon is cast onto the belt plane y = -drop."""
    vs, us = np.mgrid[0:cam.H:step, 0:cam.W:step]
    pix = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], 0)
    rays = (R.T @ np.linalg.inv(cam.K) @ pix).T
    down = rays[:, 1] < -1e-6
    s = (-drop - C[1]) / np.where(down, rays[:, 1], -1.0)
    X = C[None, :] + s[:, None] * rays
    rr = np.hypot(X[:, 0], X[:, 2])
    hit = (mask[vs.ravel(), us.ravel()] > 0)
    out = []
    for a, b in zip(RADII[:-1], RADII[1:]):
        sel = down & (rr >= a) & (rr < b)
        out.append(round(float(hit[sel].mean()), 4) if sel.sum() > 50 else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("rundir")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--step", type=int, default=1, help="use every n-th video frame")
    a = ap.parse_args()
    run_id = os.path.basename(os.path.normpath(a.rundir))
    meas = json.load(open(os.path.join(HERE, "runs.json")))["runs"][run_id]
    camj = json.load(open(os.path.join(HERE, f"camera_{run_id}.json")))
    sc = json.load(open(os.path.join(a.rundir, "scenario.json")))
    p = sc["notes"]["params"]
    radius, t_open_sim, fps_sim = p["radius"], p["t_open"], p["fps"]
    h = meas["drop_mm"] * 1e-3
    frames = sorted(glob.glob(os.path.join(a.rundir, "out", "frame_*_particles.vtk")))
    C = np.array(camj["C"])
    Rs = [np.array(R) for R in camj["R"]]
    tv = np.array(camj["t"])
    y0, y1, x0, x1 = [4 * v for v in BAND[run_id]] if run_id in BAND else (0, 0, 0, 0)

    cap = cv2.VideoCapture(os.path.join(a.videos, meas["video"]))
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    step = a.step if fps_v < 60 else a.step * int(round(fps_v / 30))
    writer = None
    if a.video:
        writer = cv2.VideoWriter(os.path.join(a.rundir, "compare.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_v / step, (1920, 540))
    outline_pts = cam.BUCKET[::9]
    rows = []
    k = -1
    while True:
        ok, im = cap.read()
        if not ok:
            break
        k += 1
        if k % step:
            continue
        t = k / fps_v
        ts = t - meas["t_open"] + t_open_sim
        fi = int(round(ts * fps_sim))
        if ts < t_open_sim - 0.5 or fi >= len(frames):
            continue
        i = min(int(np.searchsorted(tv, t)), len(Rs) - 1)
        R = Rs[i]
        P = read_points(frames[fi])
        P[:, 1] -= h                                   # sim (belt at 0) -> camera world
        inside = (np.hypot(P[:, 0], P[:, 2]) < cam.R_B + 0.002) & (P[:, 1] > -0.002)
        sim = render(P[~inside], R, C, radius)
        vid = bean_mask(im)
        # the hole axis column in this frame, for the pile-top window
        uva, _ = cam.project(np.array([[0.0, -h, 0.0]]), R, C)
        ua, va = int(uva[0, 0]), int(min(uva[0, 1], cam.H - 1))
        cols = (max(ua - 40, 0), min(ua + 40, cam.W))
        row = dict(t_video=round(t, 3), t_since_open=round(t - meas["t_open"], 3))
        if y1 > y0:
            row["band_video"] = round(float((vid[y0:y1, x0:x1] > 0).mean()), 4)
            row["band_sim"] = round(float((sim[y0:y1, x0:x1] > 0).mean()), 4)
        row["pile_video_mm"] = round(pile_top_mm(vid, R, C, h, cols, va), 1)
        row["pile_sim_mm"] = round(pile_top_mm(sim, R, C, h, cols, va), 1)
        if len(rows) % 15 == 0:
            row["cover_video"] = belt_coverage(vid, R, C, h)
            row["cover_sim"] = belt_coverage(sim, R, C, h)
        rows.append(row)
        if writer is not None:
            left = im.copy()
            right = np.full_like(im, 235)
            right[sim > 0] = BEAN_BGR
            uv, z = cam.project(outline_pts, R, C)
            for img in (left, right):
                for u, v in uv[z > 0].astype(int):
                    if 0 <= u < cam.W and 0 <= v < cam.H:
                        img[v, u] = (0, 0, 255)
                if y1 > y0:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.putText(left, f"video {run_id}  t={t - meas['t_open']:5.2f} s", (30, 70), 0, 2, (0, 0, 255), 4)
            cv2.putText(right, f"DEM  t={ts - t_open_sim:5.2f} s", (30, 70), 0, 2, (0, 0, 255), 4)
            writer.write(np.hstack([cv2.resize(left, (960, 540)), cv2.resize(right, (960, 540))]))
    if writer is not None:
        writer.release()
    json.dump(rows, open(os.path.join(a.rundir, "frames.json"), "w"))
    print(f"{run_id}: {len(rows)} frames compared -> {a.rundir}/frames.json")


if __name__ == "__main__":
    main()
