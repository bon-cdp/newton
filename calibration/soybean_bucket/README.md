# Soybean calibration: bucket discharge onto a belt

Four bench tests (2026-10-02) of 5 mm soybeans draining through a drilled hole in the
bottom of a 190 x 240 mm galvanised bucket onto a stopped rubber conveyor belt, filmed
from the side at about 210 mm.  Used to calibrate the soybean material in EMS DEM.

| run | fill g | left g | drop mm | notes |
|-----|-------:|-------:|--------:|-------|
| 1 | 916  | 604   | 75  | fit |
| 2 | 1560 | 850   | 55  | test: the pile grew up to the bucket and choked the hole |
| 3 | 1502 | 641.6 | 115 | test: filmed at 120 fps (checked real time: g = 10.5 m/s2 from falling grains) |
| 4 | 1326 | 588.4 | 123 | fit |

The videos (about 200 MB) are not in the repository; they live in `video-runs/`
at the repository root (git-ignored).  Everything measured from them is in this folder.

## Files

| file | what |
|------|------|
| `runs.json` | masses, drop heights, video times (plug out, end of steady flow, last grain, choke), fit/test split |
| `hole_contour.json` | the hole traced from the photo: 32.9 mm equivalent diameter (851.6 mm2, box 36.9 x 36.2 mm), centroid 8 mm off the axis |
| `flow_curves.json` | measured discharge curves (g vs s after opening) for runs 1, 3, 4 |
| `camera_<run>.json` | camera pose for every video frame (reference fit + tracked rotation) |
| `measure_hole.py` | photo -> `hole_contour.json` |
| `measure_flow.py` | videos -> `flow_curves.json` (stream-band occupancy, integrated, scaled to the weighed discharge) |
| `camera.py` | videos -> `camera_<run>.json` + `camera_<run>_check.jpg` |
| `make_bucket.py` | one scenario (bucket, holed bottom, timed plug, belt) for a run and parameter set |
| `calibrate.py` | runs parameter points through `dem_run.py`, extracts the observables, compares |
| `render.py` | projects a simulation through the tracked camera: per-frame metrics + side-by-side video |
| `plot_compare.py` | discharge curves, simulation vs video |

`measure_*.py`, `camera.py` and `render.py` need OpenCV (and scipy, matplotlib); the
project venv has neither, so they run in the throwaway venv `runs/perf/vtkcheck`
(`pip install opencv-python-headless scipy matplotlib`).

## Measurements

- **Hole.**  The bucket rim is fitted with an ellipse and mapped to a 190 mm circle; the
  hole is the dark blob.  Not 1 inch: the drilling tore it to ~33 mm equivalent.  It is
  6.6 grain diameters, near the clogging range, so its size matters to the flow rate.
- **Grain size.**  5.0-5.4 mm from isolated falling grains in run 3 (0.208 mm/px).
- **Discharge curves.**  Grains cross a band of white wall under the hole at nearly the
  same speed, so the bean-coloured fraction of the band tracks the mass flow rate.
- **Camera.**  Pinhole, f = 1550 px (the 190 mm silhouette at the stated stand-off).  A
  reference pose per run is fitted to the bucket's silhouette edges and bottom outline
  (0.7-7 px rms); the camera height it implies puts the lens 49-70 mm above the belt in
  every run, which was not used in the fit.  Frame-to-frame motion is tracked as a rotation
  (Lucas-Kanade on non-bean pixels, RANSAC homography).  Translation is not modelled, so the
  pose drifts while the camera is being repositioned before the plug pull; it holds during
  the discharge.

## Observables

| observable | from | sensitive to |
|------------|------|--------------|
| residual mass in the bucket | scale | grain-grain and grain-wall friction, rolling friction |
| discharge curve, t50 / t90 | video band | hole, friction |
| band occupancy per frame | video vs rendered simulation, same pixels | flow rate (absolute) |
| belt coverage vs radius | video vs rendered simulation, pixels cast onto the belt | belt rolling friction, restitution |
| pile top | same | grain rolling friction, belt friction |
| choke (run 2) | video | pile shape |

The first second after opening is not a fair comparison: the towel takes ~1 s to clear
(the band occupancy ramps up), while the simulated plug vanishes at once.

## Running

```bash
# simulations (project venv), reference preset; results in runs/calib/<tag>/<label>/<run>/
.venv/bin/python calibration/soybean_bucket/calibrate.py --tag sens --runs run1 \
    --set rolling_friction=0.02 wall_rolling_friction=0.02
.venv/bin/python calibration/soybean_bucket/calibrate.py --tag final --runs run1 run2 run3 run4 \
    --set vtk=1 <fitted values>

# frame-by-frame comparison (needs vtk=1 frames)
runs/perf/vtkcheck/bin/python calibration/soybean_bucket/render.py video-runs \
    runs/calib/final/<label>/run3 --video
```

Simulations use the REFERENCE preset (true E, dt = 0.35 Rayleigh time, no neighbour list):
the fast preset was seen to throw stray grains.  A run costs ~50 s per simulated second on
the Quadro P5000 (12-20 k grains); running two at once does not help, the GPU is saturated.
