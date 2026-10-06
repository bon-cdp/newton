#!/usr/bin/env python3
"""
Where the conveyor belt ends, from the footage: the belt's far edge is the boundary between
the dark belt and the white wall behind it.  Detected per column in each run's reference
frame (camera.REF_T) and cast onto the belt plane (y = -drop) through the fitted camera.

    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/measure_belt.py <video-runs dir>
Writes belt_extent.json next to this file and belt_<run>_check.jpg (git-ignored).
"""
import json
import os
import sys

import cv2
import numpy as np

import camera as cam
from measure_flow import camera_track, pose_at

HERE = os.path.dirname(os.path.abspath(__file__))


def far_edge(im, top):
    """(u, v) of the dark-below-bright transition per column, searched below row `top`."""
    V = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    V = cv2.GaussianBlur(V, (9, 9), 0)
    pts = []
    for u in range(20, cam.W - 20, 10):
        col = V[top:, u]
        # the TOPMOST wall-bright -> dark transition (beans lie below the edge)
        for k in range(5, len(col) - 5):
            if col[k - 5] > 150 and col[k + 4] < 100:
                pts.append((u, top + k))
                break
    pts = np.array(pts, float)
    # the far edge is a straight line in the image (a straight belt edge, no distortion):
    # RANSAC a line, keep its inliers
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        if abs(b[0] - a[0]) < 50:
            continue
        slope = (b[1] - a[1]) / (b[0] - a[0])
        res = np.abs(pts[:, 1] - (a[1] + slope * (pts[:, 0] - a[0])))
        inl = res < 5
        if best is None or inl.sum() > best.sum():
            best = inl
    return pts[best]


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))["runs"]
    out = {}
    for rid in ("run1", "run2", "run3", "run4"):
        r = runs[rid]
        drop = r["drop_mm"] * 1e-3
        tr = camera_track(rid)
        t = cam.REF_T[rid]
        R, C = pose_at(tr, t)
        cap = cv2.VideoCapture(os.path.join(sys.argv[1], r["video"]))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, im = cap.read()
        # start below the projected bucket bottom
        uvk, zk = cam.project(cam.BUCKET, R, C)
        top = int(min(cam.H - 50, uvk[zk > 0, 1].max() + 10))
        pts = far_edge(im, top)
        # cast onto the belt plane
        rays = (R.T @ np.linalg.inv(cam.K) @ np.c_[pts, np.ones(len(pts))].T).T
        s = (-drop - C[1]) / rays[:, 1]
        X = C[None, :] + s[:, None] * rays
        good = (s > 0) & (rays[:, 1] < 0)
        X = X[good]
        z_far = float(np.median(X[:, 2]))
        out[rid] = dict(far_edge_z_m=round(z_far, 4), z_spread_m=round(float(np.std(X[:, 2])), 4),
                        x_range_seen_m=[round(float(X[:, 0].min()), 3), round(float(X[:, 0].max()), 3)],
                        n_points=int(len(X)))
        print(rid, out[rid])
        for (u, v) in pts.astype(int):
            cv2.circle(im, (u, v), 6, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(HERE, f"belt_{rid}_check.jpg"), cv2.resize(im, (960, 540)))
    json.dump(out, open(os.path.join(HERE, "belt_extent.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
