#!/usr/bin/env python3
"""
Trace the drilled hole in the bucket bottom from the photo and write it in millimetres,
relative to the bucket axis, to hole_contour.json.

The rim of the bottom plate is fitted with an ellipse (the photo is not exactly square to
the plate) and mapped to a 190 mm circle; the hole is the dark blob near the middle.
Needs OpenCV (not in the project venv):

    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/measure_hole.py \
        /path/to/video-runs/bucket-hole.jpg
"""
import json
import os
import sys

import cv2
import numpy as np

BUCKET_D_MM = 190.0
HERE = os.path.dirname(os.path.abspath(__file__))

im = cv2.imread(sys.argv[1])
g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (7, 7), 0)

# rim: Canny edges in an annulus around the image centre, least-squares ellipse
e = cv2.Canny(g, 30, 90)
ys, xs = np.nonzero(e)
cx0, cy0 = im.shape[1] * 0.49, im.shape[0] * 0.5
r = np.hypot(xs - cx0, ys - cy0)
sel = (r > 0.26 * im.shape[1]) & (r < 0.33 * im.shape[1])
(ecx, ecy), (ea, eb), eang = cv2.fitEllipse(np.stack([xs[sel], ys[sel]], 1).astype(np.float32))

# hole: largest dark blob
_, th = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY_INV)
n, lab, st, _ = cv2.connectedComponentsWithStats(th)
i = 1 + int(np.argmax(st[1:, 4]))
cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)

# ellipse -> circle of the bucket diameter: rotate into the ellipse axes, scale each axis
t = np.deg2rad(eang)
R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
p = (c - [ecx, ecy]) @ R.T
p[:, 0] *= BUCKET_D_MM / ea
p[:, 1] *= BUCKET_D_MM / eb
p = p @ R                                     # back to image orientation, mm, y down
p[:, 1] *= -1.0                               # y up

# resample to a fixed number of points evenly spaced along the outline
seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T))]
s = np.linspace(0.0, seg[-1], 97)[:-1]
closed = np.vstack([p, p[:1]])
q = np.stack([np.interp(s, seg, closed[:, 0]), np.interp(s, seg, closed[:, 1])], 1)

area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
cen = q.mean(axis=0)
out = dict(
    source=os.path.basename(sys.argv[1]),
    bucket_diameter_mm=BUCKET_D_MM,
    rim_ellipse_px=[ecx, ecy, ea, eb, eang],
    area_mm2=round(float(area), 1),
    equivalent_diameter_mm=round(float(2 * np.sqrt(area / np.pi)), 2),
    bbox_mm=[round(float(v), 1) for v in np.ptp(q, axis=0)],
    centroid_offset_mm=[round(float(v), 1) for v in cen],
    contour_mm=np.round(q, 2).tolist(),
)
json.dump(out, open(os.path.join(HERE, "hole_contour.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "contour_mm"})
