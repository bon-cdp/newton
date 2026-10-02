#!/usr/bin/env python3
"""
Discharge curves from the test videos.

Grains cross a band of white wall between the bucket bottom and the pile at nearly the
same speed (free fall from the hole), so the fraction of bean-coloured pixels in the band
is proportional to the mass flow rate.  Integrated from the plug pull and scaled to the
weighed discharge (m0 - m_end) it gives the measured discharged mass vs time.

The band is a fixed box in a 480x270 copy of each frame (the camera is handheld; the box
has margin for the drift).  Run2 has no curve: its pile grows into the band and chokes the
hole, which the crop strips time directly (runs.json t_choke).

Needs OpenCV (not in the project venv):
    runs/perf/vtkcheck/bin/python calibration/soybean_bucket/measure_flow.py /path/to/video-runs
Writes flow_curves.json next to this file.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BAND = {                       # y0, y1, x0, x1 in the 480x270 frame
    "run1": (128, 150, 240, 340),
    "run3": (62, 100, 190, 310),
    "run4": (40, 80, 180, 300),
}


def occupancy(path, box):
    y0, y1, x0, x1 = box
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS)
    occ = []
    while True:
        ok, im = c.read()
        if not ok:
            break
        s = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)[y0:y1, x0:x1]
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        occ.append(float(((hsv[..., 1] > 55) & (hsv[..., 2] > 40)).mean()))
    return np.arange(len(occ)) / fps, np.array(occ)


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))["runs"]
    out = {"method": __doc__.strip().splitlines()[0], "runs": {}}
    for rid, box in BAND.items():
        r = runs[rid]
        t, occ = occupancy(os.path.join(sys.argv[1], r["video"]), box)
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
        out["runs"][rid] = dict(band=box, floor=round(floor, 4), t_s=tg.round(2).tolist(),
                                discharged_g=mg.round(1).tolist())
        k = lambda f: float(tg[np.searchsorted(mg, f * mg[-1])])
        print(f"{rid}: {mg[-1]:.0f} g; 50% at {k(0.5):.2f} s, 90% at {k(0.9):.2f} s, "
              f"99% at {k(0.99):.2f} s; first-second rate {mg[10]:.0f} g/s")
    json.dump(out, open(os.path.join(HERE, "flow_curves.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
