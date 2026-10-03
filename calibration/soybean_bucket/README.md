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

## Results (2026-10-02/03, reference preset)

1. **`rot_damp` must be 0 for small grains.** The default 0.2 (BFA corn's RotatingR) is about
   1400× the rolling-friction torque for a 5 mm grain rolling at 0.5 m/s. Grains could not
   roll at all, so rolling friction did nothing and beans stopped dead on the belt.
2. **Spheres cannot fit two fill levels.**
   - The real residual barely depends on the fill: 604 / 642 / 588 g.
   - Spheres need grain rolling friction ≈ 0.2 for run 1 (916 g) but ≈ 0.14 for run 4
     (1326 g).
   - Spheres also creep out of the crater after the real flow has stopped.
3. **2-sphere clumps (`soy2`) fit both fill levels with one parameter set.**
   - Settings: grain rolling friction 0.02, wall rolling friction 0.02, restitution 0.7,
     sliding friction 0.35.
   - The flow stops like the real one does.

| grain | run | residual g (meas) | t90 s (meas) | curve rms g | notes |
|---|---|---|---|---|---|
| sphere μr 0.17 | 1 fit | 577 (604) | 3.9 (4.3) | 26 | creeps |
| sphere μr 0.17 | 4 fit | 632 (588) | 11.8 (7.0) | 67 | creeps |
| **soy2 μr 0.02** | 1 fit | **603 (604)** | 4.1 (4.3) | 21 | |
| **soy2 μr 0.02** | 4 fit | **599 (588)** | 8.2 (7.0) | 27 | |
| **soy2 μr 0.02** | 3 test | 594 (642) | 8.3 (9.2) | 52 | pile 60 vs 60 mm |
| **soy2 μr 0.02** | 2 test | 921 (850) | – | – | chokes at ~6.0 s vs 5.6 s; 630 vs 710 g out |

The real residuals scatter ±4% between nominally similar runs, and run 3 is the high one.

4. **Spread on the belt comes from bounciness, not rolling.**
   - Clumps barely roll on the belt, so belt rolling friction 0.02 → 0.005 changed nothing.
   - Restitution 0.85 widened the carpet (r50 120 → 160 mm on run 1) and lowered the heap
     to the video's height (34 vs 35 mm).
   - The **final set** is in `soybean_material.json` (`soy2`, restitution 0.85, grain rolling
     0.025, wall rolling 0.01):

| run | residual g (meas) | t90 s (meas) | curve rms g | pile mm (video) | r50 mm (video) |
|---|---|---|---|---|---|
| 1 fit | 601 (604) | 4.3 (4.3) | 17 | 31 (35) | 160 (220) |
| 4 fit | 615 (588) | 8.5 (7.0) | 33 | 32 (44) | 220 (260) |
| 3 test | 604 (642) | 8.3 (9.2) | 48 | 52 (60) | 200 (300) |
| 2 test | 896 (850), chokes at ~6 s vs 5.6 s | – | – | 55 (51) | 140 (260) |

Every residual is within 6%. The heap and spread differences are a few mm and tens of mm.
Side-by-side videos (footage | simulation, same camera) are in
`runs/calib/final/<point>/<run>/compare.mp4`.

**Open items:**

- **Belt carpet is too narrow.** r50 is 120–160 mm against 220–300 mm in the video.
- **The heap is taller than in the video** in runs 1 and 4.
- **The opening is too fast.** The simulated plug vanishes at once, but the towel takes
  ~1 s to clear, so the first second runs fast.

Clumps: `granular_clumps.py` (solver), `dem_clumps.py` (fill-once runner) and
`test_clumps.py` at the repository root. `make_bucket.py` `CLUMPS` holds the shapes, which are
selected with `--set clump=soy2`.

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
