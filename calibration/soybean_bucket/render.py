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
from measure_flow import FLOW_RUNS, band_px, camera_track, pose_at   # same band and poses

HERE = os.path.dirname(os.path.abspath(__file__))
BEAN_BGR = (125, 190, 228)


def read_points(path):
    raw = open(path, "rb").read()
    k = raw.index(b"POINTS")
    e = raw.index(b"\n", k)
    n = int(raw[k:e].split()[1])
    return np.frombuffer(raw, dtype=">f4", count=n * 3, offset=e + 1).reshape(n, 3).astype(np.float64)


def quat_to_R(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def projected_ellipses(P, Q, axes, R, C):
    """Image ellipses of ellipsoids (centres P, orientations Q x,y,z,w, semi-axes `axes`)
    at their own depth (weak perspective per grain): (uv, z, semi-axes px (n,2), angle deg)."""
    Xc = (P - C) @ R.T
    z = Xc[:, 2]
    zz = np.maximum(z, 1e-6)
    J = np.zeros((len(P), 2, 3))
    J[:, 0, 0] = J[:, 1, 1] = cam.F_PX / zz
    J[:, 0, 2] = -cam.F_PX * Xc[:, 0] / zz ** 2
    J[:, 1, 2] = -cam.F_PX * Xc[:, 1] / zz ** 2
    Rb = R[None] @ quat_to_R(Q)                       # body -> camera
    Mc = Rb @ np.diag(np.asarray(axes) ** 2)[None] @ np.transpose(Rb, (0, 2, 1))
    S = J @ Mc @ np.transpose(J, (0, 2, 1))           # (n,2,2) silhouette covariance
    ev, evec = np.linalg.eigh(S)
    semi = np.sqrt(np.maximum(ev, 1e-12))[:, ::-1]    # major, minor
    ang = np.degrees(np.arctan2(evec[:, 1, 1], evec[:, 0, 1]))
    uv = Xc[:, :2] / zz[:, None] * cam.F_PX + np.array([cam.W / 2, cam.H / 2])
    return uv, z, semi, ang


def _visible(uv, z):
    return (z > 0.02) & (uv[:, 0] > -50) & (uv[:, 0] < cam.W + 50) & (uv[:, 1] > -50) & (uv[:, 1] < cam.H + 50)


def render(P, R, C, radius, Q=None, axes=None):
    """Mask of grains (uint8) for world points P, painter's algorithm (no colour needed).
    With orientations Q and semi-axes, grains are the projected ellipses."""
    m = np.zeros((cam.H, cam.W), np.uint8)
    if Q is not None:
        uv, z, semi, ang = projected_ellipses(P, Q, axes, R, C)
        ok = _visible(uv, z)
        for (u, v), (sa, sb), a in zip(np.round(uv[ok]).astype(int), semi[ok], ang[ok]):
            cv2.ellipse(m, (int(u), int(v)), (max(1, int(round(sa))), max(1, int(round(sb)))),
                        float(a), 0, 360, 255, -1)
        return m
    uv, z = cam.project(P, R, C)
    ok = _visible(uv, z)
    uv, z = uv[ok], z[ok]
    rad = np.maximum(1, np.round(cam.F_PX * radius / z)).astype(int)
    for (u, v), r in zip(np.round(uv).astype(int), rad):
        cv2.circle(m, (int(u), int(v)), int(r), 255, -1)
    return m


WALL_BGR = (226, 228, 228)
BELT_BGR = (58, 58, 60)
METAL_BGR = (150, 152, 155)
LIGHT = np.array([-0.45, -0.65, 0.62])          # image frame (x right, y down, z to camera)
LIGHT = LIGHT / np.linalg.norm(LIGHT)
_SPRITES = {}


def _sprite(r):
    """Shaded sphere of pixel radius r: (BGR float image, alpha) -- Lambert + ambient + a
    small specular highlight, light from the upper left."""
    if r not in _SPRITES:
        y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64) / max(r, 1)
        rr = x * x + y * y
        a = np.clip((1.0 - rr) * r, 0.0, 1.0)            # anti-aliased edge
        z = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
        lam = np.clip(x * LIGHT[0] + y * LIGHT[1] + z * LIGHT[2], 0.0, 1.0)
        spec = lam ** 24 * 0.35
        base = np.array(BEAN_BGR, dtype=np.float64)
        img = base[None, None, :] * (0.38 + 0.72 * lam)[..., None] + 255.0 * spec[..., None]
        _SPRITES[r] = (np.clip(img, 0, 255), a)
    return _SPRITES[r]


def render_shaded(P, R, C, radius, drop, Q=None, axes=None):
    """Display image of the simulation through the camera: wall, belt plane, bucket, soft
    shadows of the grains on the belt, then the grains as shaded spheres (far to near)."""
    img = np.empty((cam.H, cam.W, 3), np.float64)
    img[:] = WALL_BGR
    # belt: a 1.2 m square at y = -drop, clipped to the part in front of the camera
    s = np.linspace(-0.6, 0.6, 25)
    belt = np.array([[x, -drop, zz] for x in s for zz in s])
    uvb, zb = cam.project(belt, R, C)
    belt_mask = np.zeros((cam.H, cam.W), np.uint8)
    ok = zb > 0.02
    if ok.sum() > 3:
        hull = cv2.convexHull(uvb[ok].astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(belt_mask, hull, 1)
    img[belt_mask > 0] = BELT_BGR
    # shadows: each grain darkens a soft disc straight below it on the belt
    sh = np.zeros((cam.H, cam.W), np.float32)
    below = P.copy()
    below[:, 1] = -drop
    uvs, zs = cam.project(below, R, C)
    for (u, v), z, h in zip(uvs, zs, P[:, 1] + drop):
        if z <= 0.02:
            continue
        rp = cam.F_PX * radius / z * (1.0 + 2.0 * min(h, 0.05) / 0.05)   # softer when higher
        cv2.ellipse(sh, (int(u), int(v)), (max(1, int(rp)), max(1, int(rp * 0.45))), 0, 0, 360,
                    0.55 / (1.0 + 20.0 * h), -1)
    sh = cv2.GaussianBlur(np.minimum(sh, 0.6), (0, 0), 3)
    img *= (1.0 - sh * belt_mask)[..., None]
    # bucket silhouette
    uvk, zk = cam.project(cam.BUCKET, R, C)
    okk = zk > 0.02
    if okk.sum() > 3:
        hull = cv2.convexHull(uvk[okk].astype(np.float32)).astype(np.int32)
        bm = np.zeros((cam.H, cam.W), np.uint8)
        cv2.fillConvexPoly(bm, hull, 1)
        img[bm > 0] = METAL_BGR
    # grains, far to near
    if Q is not None:
        # ellipsoids: the shaded sphere sprite mapped affinely onto each projected ellipse
        uv, z, semi, ang = projected_ellipses(P, Q, axes, R, C)
        keep = _visible(uv, z)
        uv, z, semi, ang = uv[keep], z[keep], semi[keep], ang[keep]
        r0 = 24
        spr0, a0 = _sprite(r0)
        spr0 = spr0.astype(np.float32)
        a0 = a0.astype(np.float32)
        for k in np.argsort(-z):
            sa, sb = semi[k]
            hs = int(np.ceil(sa)) + 1
            th = np.radians(ang[k])
            Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            A = Rot @ np.diag([sa / r0, sb / r0])
            Mx = np.c_[A, np.array([hs, hs]) - A @ np.array([r0, r0])].astype(np.float32)
            size = (2 * hs + 1, 2 * hs + 1)
            pc = cv2.warpAffine(spr0, Mx, size, flags=cv2.INTER_LINEAR, borderValue=0)
            pa = cv2.warpAffine(a0, Mx, size, flags=cv2.INTER_LINEAR, borderValue=0)
            u, v = int(round(uv[k, 0])), int(round(uv[k, 1]))
            y0, x0 = v - hs, u - hs
            ys, xs = max(0, -y0), max(0, -x0)
            ye, xe = min(2 * hs + 1, cam.H - y0), min(2 * hs + 1, cam.W - x0)
            if ye <= ys or xe <= xs:
                continue
            reg = img[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
            aa = pa[ys:ye, xs:xe, None]
            reg[:] = reg * (1.0 - aa) + pc[ys:ye, xs:xe] * aa
        return np.clip(img, 0, 255).astype(np.uint8)
    uv, z = cam.project(P, R, C)
    keep = _visible(uv, z)
    uv, z = uv[keep], z[keep]
    order = np.argsort(-z)
    for (u, v), zz in zip(np.round(uv[order]).astype(int), z[order]):
        r = max(1, int(round(cam.F_PX * radius / zz)))
        spr, a = _sprite(r)
        y0, x0 = v - r, u - r
        ys, xs = max(0, -y0), max(0, -x0)
        ye, xe = min(2 * r + 1, cam.H - y0), min(2 * r + 1, cam.W - x0)
        if ye <= ys or xe <= xs:
            continue
        reg = img[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
        aa = a[ys:ye, xs:xe, None]
        reg[:] = reg * (1.0 - aa) + spr[ys:ye, xs:xe] * aa
    return np.clip(img, 0, 255).astype(np.uint8)


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


HEAP_X_MM = (-60.0, -40.0, -25.0, 25.0, 40.0, 60.0)


def heap_profile_mm(mask, R, C, drop, xs_mm=HEAP_X_MM, half_px=6):
    """Heap height (mm) beside the stream: at each lateral offset x (in the vertical plane
    through the axis, facing the camera) the contiguous stack of grain pixels that RESTS ON
    THE BELT -- a block that does not reach down to the projected belt line is a falling
    stream or airborne grains and counts as 0.  Same pixels for footage and simulation."""
    out = []
    for xm in xs_mm:
        uvb, zb = cam.project(np.array([[xm * 1e-3, -drop, 0.0]]), R, C)
        u, vb = int(round(uvb[0, 0])), int(round(uvb[0, 1]))
        if zb[0] <= 0 or not (half_px <= u < cam.W - half_px) or not (0 < vb < cam.H):
            out.append(None)
            continue
        col = mask[:vb + 1, u - half_px:u + half_px + 1] > 0
        cov = col.mean(axis=1) > 0.5
        top = vb
        gap = 0
        found = False
        for v in range(vb, -1, -1):          # climb from the belt line
            if cov[v]:
                top, gap, found = v, 0, True
            else:
                gap += 1
                if gap > 3:
                    break
            if not found and vb - v > 4:     # nothing on the belt here
                break
        if not found:
            out.append(0.0)
            continue
        ray_w = R.T @ (np.linalg.inv(cam.K) @ np.array([u, top, 1.0]))
        s = -C[2] / ray_w[2]
        out.append(round(float((C[1] + s * ray_w[1] + drop) * 1e3), 1))
    return out


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
    ell = sc["notes"].get("ellipsoid")
    axes = np.asarray(ell["axes_mm"]) * 1e-3 if ell else None
    if ell:
        radius = float((axes.prod()) ** (1.0 / 3.0))  # shadows: volume-equivalent size
    h = meas["drop_mm"] * 1e-3
    frames = sorted(glob.glob(os.path.join(a.rundir, "out", "frame_*_particles.vtk")))
    track = camera_track(run_id)

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
        R, C = pose_at(track, t)
        P = read_points(frames[fi])
        P[:, 1] -= h                                   # sim (belt at 0) -> camera world
        inside = (np.hypot(P[:, 0], P[:, 2]) < cam.R_B + 0.002) & (P[:, 1] > -0.002)
        Q = None
        if ell:
            Q = np.load(frames[fi].replace("_particles.vtk", "_quat.npy")).astype(np.float64)[~inside]
        sim = render(P[~inside], R, C, radius, Q, axes)
        vid = bean_mask(im)
        # the hole axis column in this frame, for the pile-top window
        uva, _ = cam.project(np.array([[0.0, -h, 0.0]]), R, C)
        ua, va = int(uva[0, 0]), int(min(uva[0, 1], cam.H - 1))
        cols = (max(ua - 40, 0), min(ua + 40, cam.W))
        y0, y1, x0, x1 = band_px(h, R, C) if run_id in FLOW_RUNS else (0, 0, 0, 0)
        row = dict(t_video=round(t, 3), t_since_open=round(t - meas["t_open"], 3))
        if y1 > y0:
            row["band_video"] = round(float((vid[y0:y1, x0:x1] > 0).mean()), 4)
            row["band_sim"] = round(float((sim[y0:y1, x0:x1] > 0).mean()), 4)
        row["pile_video_mm"] = round(pile_top_mm(vid, R, C, h, cols, va), 1)
        row["pile_sim_mm"] = round(pile_top_mm(sim, R, C, h, cols, va), 1)
        row["heap_video_mm"] = heap_profile_mm(vid, R, C, h)
        row["heap_sim_mm"] = heap_profile_mm(sim, R, C, h)
        if len(rows) % 15 == 0:
            row["cover_video"] = belt_coverage(vid, R, C, h)
            row["cover_sim"] = belt_coverage(sim, R, C, h)
        rows.append(row)
        if writer is not None:
            left = im.copy()
            right = render_shaded(P[~inside], R, C, radius, h, Q, axes)
            uv, z = cam.project(outline_pts, R, C)
            for u, v in uv[z > 0].astype(int):           # model bucket outline on the footage
                if 0 <= u < cam.W and 0 <= v < cam.H:
                    left[v, u] = (0, 0, 255)
            for img in (left, right):
                if y1 > y0:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.putText(left, f"video {run_id}  t={t - meas['t_open']:5.2f} s", (30, 70), 0, 2, (0, 0, 255), 4)
            cv2.putText(right, f"DEM  t={ts - t_open_sim:5.2f} s", (30, 70), 0, 2, (0, 0, 255), 4)
            writer.write(np.hstack([cv2.resize(left, (960, 540)), cv2.resize(right, (960, 540))]))
    if writer is not None:
        writer.release()
    json.dump(rows, open(os.path.join(a.rundir, "frames.json"), "w"))
    print(f"{run_id}: {len(rows)} frames compared -> {a.rundir}/frames.json")
    print(summarize(rows))


def r_half(cov):
    """Radius (mm) where belt coverage first falls below 50%."""
    for k, c in enumerate(cov):
        if c is not None and c < 0.5:
            return float(RADII[k] * 1e3)
    return float(RADII[-1] * 1e3)


def summarize(rows):
    cov = [r for r in rows if "cover_video" in r]
    last = rows[int(0.75 * len(rows)):]
    out = dict(
        r50_video_mm=r_half(cov[-1]["cover_video"]), r50_sim_mm=r_half(cov[-1]["cover_sim"]),
        pile_video_mm=float(np.median([r["pile_video_mm"] for r in last])),
        pile_sim_mm=float(np.median([r["pile_sim_mm"] for r in last])))
    b = [(r["band_video"], r["band_sim"]) for r in rows if "band_video" in r and r["t_since_open"] > 1.0]
    if b:
        b = np.array(b)
        out["band_rms"] = float(np.sqrt(np.mean((b[:, 0] - b[:, 1]) ** 2)))
        out["band_mean_video"], out["band_mean_sim"] = float(b[:, 0].mean()), float(b[:, 1].mean())
    return {k: round(v, 3) for k, v in out.items()}


if __name__ == "__main__":
    main()
