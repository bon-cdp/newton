#!/usr/bin/env python3
"""
Camera model for the bucket videos: one pose per video frame, so simulated grains can be
projected into the footage and compared pixel for pixel.

World frame (metres): origin at the centre of the bucket bottom, y up, belt at y = -drop.
Camera: pinhole, focal F_PX pixels (an iPhone 1x lens at 1920x1080: the bucket's 190 mm
silhouette at the stated 210 mm stand-off gives 1550 px), principal point at the image
centre, no distortion.

1. Reference pose (fit_reference): segment the bucket in one frame (grey, unsaturated,
   against the white wall), take its left/right silhouette edges and the lower outline
   of its bottom, and fit the 6 pose parameters so the projected cylinder matches them.
2. Frame-to-frame motion (track): a handheld camera mostly rotates, which maps to a
   homography of the whole image.  Corners on static things (bucket, wall, belt edge;
   bean-coloured pixels masked out) are tracked with pyramidal Lucas-Kanade from frame to
   frame, a homography is fitted with RANSAC, and the rotation closest to it is
   accumulated: R_t = R_ref * dR_t.  Camera translation is ignored.

Needs OpenCV + scipy (the throwaway venv):
    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/camera.py <video-runs dir> [run ...]
writes camera_<run>.json (per-frame R, C) and camera_<run>_check.jpg next to this file.
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
F_PX = 1550.0
W, H = 1920, 1080
R_B, H_B = 0.095, 0.240
K = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1.0]])

# frame used for the reference pose (video seconds): bucket well in view, flow running
REF_T = {"run1": 4.0, "run2": 4.0, "run3": 8.0, "run4": 5.0}


def rot(yaw, pitch, roll):
    """World->camera rotation.  Camera axes: x right, y DOWN, z forward (OpenCV).  At zero
    angles the camera looks along world +z with world y up."""
    base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1.0]])           # world y up -> cam y down
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ base @ Ry


def project(P, R, C):
    """World points (n,3) -> pixels (n,2) and depth (n,)."""
    Xc = (P - C) @ R.T
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def bucket_points(n_th=360, n_h=60, n_r=20):
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    hh = np.linspace(0, H_B, n_h)
    side = np.stack([R_B * np.cos(th)[None, :].repeat(n_h, 0), hh[:, None].repeat(n_th, 1),
                     R_B * np.sin(th)[None, :].repeat(n_h, 0)], -1).reshape(-1, 3)
    rr = np.linspace(0, R_B, n_r)
    disk = np.stack([rr[:, None] * np.cos(th)[None, :], np.zeros((n_r, n_th)),
                     rr[:, None] * np.sin(th)[None, :]], -1).reshape(-1, 3)
    return np.vstack([side, disk])


BUCKET = bucket_points()


def segment_bucket(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    wallv = np.percentile(V[:, :200], 60)
    # "not wall": everything darker or more coloured than the white wall.  The bucket is
    # the only such thing touching the top of the frame (the stream hangs below it and is
    # rejected as an outlier in outline()).
    # Union with "grey metal" (unsaturated, a little darker than the wall): brightly lit
    # metal on the bucket side can be nearly as bright as the wall.
    m = ((V < wallv - 30) | (S > 60) | ((S < 70) & (V < wallv - 25))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = max(range(1, n), key=lambda i: st[i, 4] * (st[i, 1] == 0))
    return (lab == best).astype(np.uint8)


def outline(mask):
    """Silhouette features: bottom y per column (bucket span, stream columns removed) and
    left/right x per row (rows above the bottom)."""
    # bucket span: the longest run of columns occupied in the top rows (the stream/belt
    # leak is lower down; a wall corner can touch the top too)
    occ = np.r_[0, mask[:60].any(axis=0).astype(int), 0]
    d = np.diff(occ)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(ends - starts))
    xl, xr = starts[k], ends[k] - 1
    xs = np.arange(xl + 15, xr - 15, 8)
    yb = np.array([np.argmin(mask[:, x]) if mask[0, x] else 0 for x in xs], float)
    # the stream keeps the mask running down: drop columns far below a robust smooth fit
    ok = yb > 0
    for _ in range(3):
        c = np.polyfit(xs[ok], yb[ok], 2)
        res = yb - np.polyval(c, xs)
        ok = ok & (np.abs(res) < max(6.0, 2.5 * np.median(np.abs(res[ok]))))
    bottom = np.stack([xs[ok], yb[ok]], 1)
    ylo = int(np.percentile(yb[ok], 5)) - 25
    rows = np.arange(0, max(ylo, 10), 8)
    lr = []
    for y in rows:
        r = np.flatnonzero(mask[y])
        r = r[(r > xl - 40) & (r < xr + 40)]
        if len(r) and r.min() > 2 and r.max() < W - 3:
            lr.append((y, r.min(), r.max()))
    return bottom, np.array(lr, float)


def model_outline(R, C, bottom_x, rows):
    uv, z = project(BUCKET, R, C)
    uv = uv[z > 0]
    # lowest projected point per column (bins of 8 px), extreme x per row band
    yb = np.full(len(bottom_x), np.nan)
    for i, x in enumerate(bottom_x):
        s = np.abs(uv[:, 0] - x) < 4
        if s.any():
            yb[i] = uv[s, 1].max()
    xl = np.full(len(rows), np.nan)
    xr = np.full(len(rows), np.nan)
    for i, y in enumerate(rows):
        s = np.abs(uv[:, 1] - y) < 4
        if s.any():
            xl[i], xr[i] = uv[s, 0].min(), uv[s, 0].max()
    return yb, xl, xr


def pose_of(p):
    yaw, pitch, roll, cx, cy, cz = p
    return rot(yaw, pitch, roll), np.array([cx, cy, cz])


def fit_reference(im, init_dist=0.30):
    mask = segment_bucket(im)
    bottom, lr = outline(mask)

    def resid(p):
        R, C = pose_of(p)
        yb, xl, xr = model_outline(R, C, bottom[:, 0], lr[:, 0])
        r = np.concatenate([yb - bottom[:, 1], xl - lr[:, 1], xr - lr[:, 2]])
        return np.nan_to_num(r, nan=200.0)

    # start: camera on -z looking +z, at bucket-bottom height, centred on the bucket
    cxp = 0.5 * (lr[:, 1].mean() + lr[:, 2].mean())
    yaw0 = -np.arctan((cxp - W / 2) / F_PX)
    best = None
    for cy0 in (-0.03, 0.0, 0.03):
        for pitch0 in (-0.15, 0.0, 0.15):
            p0 = [yaw0, pitch0, 0.0, 0.0, cy0, -init_dist]
            s = least_squares(resid, p0, x_scale=[0.05, 0.05, 0.02, 0.01, 0.01, 0.02],
                              loss="soft_l1", f_scale=5.0)
            if best is None or s.cost < best.cost:
                best = s
    r = resid(best.x)
    return best.x, float(np.sqrt(np.mean(r ** 2))), mask, bottom, lr


def static_mask(im):
    """Where to track: not bean-coloured (falling or piled grains move), dilated."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    bean = ((hsv[..., 1] > 50) & (hsv[..., 2] > 50)).astype(np.uint8)
    bean = cv2.dilate(bean, np.ones((31, 31), np.uint8))
    return (1 - bean).astype(np.uint8)


def rotation_from_h(Hm):
    """Closest rotation to K^-1 H K (pure-rotation homography)."""
    M = np.linalg.inv(K) @ Hm @ K
    M /= np.cbrt(np.linalg.det(M))
    U, _, Vt = np.linalg.svd(M)
    Rr = U @ Vt
    if np.linalg.det(Rr) < 0:
        Rr = -Rr
    return Rr


def track(path, step=1, scale=0.5):
    """Per-frame camera rotation relative to frame 0: list of 3x3 (camera-frame) dR."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    S = np.diag([scale, scale, 1.0])
    Ks = S @ K
    prev, prev_pts = None, None
    acc = np.eye(3)
    out, k = [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k % step:
            k += 1
            continue
        small = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None and prev_pts is not None and len(prev_pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, prev_pts, None, winSize=(21, 21), maxLevel=3)
            a, b = prev_pts[st[:, 0] == 1], nxt[st[:, 0] == 1]
            if len(a) >= 12:
                Hm, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
                if Hm is not None and inl.sum() >= 10:
                    Hf = np.linalg.inv(S) @ Hm @ S          # full-res homography
                    acc = rotation_from_h(Hf) @ acc
        out.append((k / fps, acc.copy()))
        m = static_mask(small)
        prev_pts = cv2.goodFeaturesToTrack(g, 400, 0.01, 12, mask=m)
        prev = g
        k += 1
    return out, fps


def main():
    vdir = sys.argv[1]
    want = sys.argv[2:] or ["run1", "run2", "run3", "run4"]
    runs = json.load(open(os.path.join(HERE, "runs.json")))["runs"]
    for rid in want:
        r = runs[rid]
        path = os.path.join(vdir, r["video"])
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, REF_T[rid] * 1000)
        ok, im = cap.read()
        p, rms, mask, bottom, lr = fit_reference(im)
        R0, C0 = pose_of(p)
        print(f"{rid}: reference pose at {REF_T[rid]} s: rms {rms:.1f} px, camera at "
              f"{np.round(C0 * 1e3, 1).tolist()} mm, yaw/pitch/roll "
              f"{np.round(np.degrees(p[:3]), 2).tolist()} deg", flush=True)
        frames, fps = track(path)
        # rotation at the reference frame, so R_t = dR_t dR_ref^-1 R0
        tt = np.array([t for t, _ in frames])
        iref = int(np.argmin(np.abs(tt - REF_T[rid])))
        dref = frames[iref][1]
        poses = [(t, (dR @ dref.T @ R0)) for t, dR in frames]
        json.dump(dict(run=rid, f_px=F_PX, size=[W, H], ref_t=REF_T[rid], ref_rms_px=rms,
                       ref_params=p.tolist(), C=C0.tolist(), fps=fps,
                       t=[round(t, 4) for t, _ in poses],
                       R=[np.round(Rt, 6).tolist() for _, Rt in poses]),
                  open(os.path.join(HERE, f"camera_{rid}.json"), "w"))
        # check image: bucket outline drawn at several times
        tiles = []
        for tq in np.linspace(tt[0] + 0.5, tt[-1] - 0.5, 6):
            i = int(np.argmin(np.abs(tt - tq)))
            cap.set(cv2.CAP_PROP_POS_MSEC, tt[i] * 1000)
            ok, fr = cap.read()
            uv, z = project(BUCKET, poses[i][1], C0)
            for u, v in uv[z > 0][::7].astype(int):
                if 0 <= u < W and 0 <= v < H:
                    fr[v, u] = (0, 0, 255)
            belt = np.array([[x, -r["drop_mm"] * 1e-3, zz] for x in np.linspace(-.3, .3, 13)
                             for zz in np.linspace(-.1, .3, 9)])
            uvb, zb = project(belt, poses[i][1], C0)
            for u, v in uvb[zb > 0].astype(int):
                cv2.circle(fr, (u, v), 4, (255, 0, 0), -1)
            cv2.putText(fr, f"{rid} t={tt[i]:.1f}", (30, 60), 0, 1.5, (0, 0, 255), 3)
            tiles.append(cv2.resize(fr, (640, 360)))
        cv2.imwrite(os.path.join(HERE, f"camera_{rid}_check.jpg"),
                    np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))


if __name__ == "__main__":
    main()
