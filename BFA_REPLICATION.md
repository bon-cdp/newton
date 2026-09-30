# Newton DEM — BulkFlowAnalyst replication

Reproducing a BulkFlowAnalyst (BFA) DEM run of the 23087 quarter-scale shiploader spout
(corn, 6 mm grains, 10 s, 8.61 kg/s) in NVIDIA Newton, to establish that Newton can match
a commercial DEM code on a real machine — and then run cases BFA cannot afford.

## Layout

| path | what |
|---|---|
| `granular_dem.py` | **`SolverGranularDEM`** — the soft-sphere DEM. Linear spring-dashpot or Hertz–Mindlin, Coulomb friction, rolling friction, rotation, Cundall–Strack tangential history, open-shell mesh colliders. |
| `dem_scenario.py` | the scenario schema (JSON): parts, material, injectors, domain, regions, solver, output. |
| `dem_run.py` | the generic runner: any scenario → history.csv, run.json, VTK. Injection schedule, recycling, CUDA-graph stepping, per-part time windows. |
| `bfa_import.py` | BFA project (.prj + .lin) → scenario, with every interpretation listed. |
| `dem_analyze.py` | measurements from a finished run's frames: wall loads and maps, flows, regions. |
| `dem_ui/` | the operator screen: FastAPI backend (`server.py`) and a no-build web frontend (three.js). |
| `bfa_dem.py` | the corn replication: its flags and presets → scenario → `dem_run`. |
| `bfa_replication_mpm.py` | the earlier MPM runner. Still imported by the DEM tooling for `COLLIDER_PARTS` and `load_part`. |
| `compare_bfa_dem.py` | scores a run against BFA (hold-up, KE, chute, cascade, discharge, rms). |
| `compare_bfa_mpm.py` | readers for BFA's undocumented binary `.por` / `.his` output. |
| `bfa_params_audit.py` | re-derives every BFA input from the `.prj` / `.est` / `.lin`. |
| `export_bfa_vtk.py` | writes the BFA reference to `bfa_reference_vtk/` in our VTK format. |
| `angle_of_repose.py` | pours a heap and measures the angle our contact law actually produces. |
| `test_rotation.py` | analytic checks: rolling incline `(5/7)g sinθ`, rolling resistance, static friction. |
| `mesh_simplify.py` | optional collapse of sub-grain CAD detail (`--simplify-mm`); off by default — it moves the cascade. |
| `tools/` | diagnostics. `pileavg.py` is the primary visual-match criterion; `perf_table.py` one-row scoring; `dem_bench.py` kernel profiling from a checkpoint; `wall_grid_check.py` wall-distance validation vs float64. |
| `runs/` | all output, git-ignored (~33 GB): `dem/`, `mpm/`, `repose/`, `logs/`. |
| `_backup_fork/` | upstream copies of the two MPM solver files this fork modifies. |

**External input, not in git:** `23087-25sim/` (the BFA project, 1.4 GB, customer data).
Everything reads it from there. `bfa_reference_vtk/` regenerates from it via
`python export_bfa_vtk.py`.

## Running

VTK is on by default (`--no-vtk` opts out); frames are binary legacy VTK, which ParaView
opens directly. Use `python -u` or progress looks stalled.

```bash
# DEFAULT: the "fast" preset.  ~6 s per simulated second on a Quadro P5000 (~97x BFA).
.venv/bin/python -u bfa_dem.py --out runs/dem/my_run

# BFA's own timestep and stiffness, same calibration.  ~32 s per simulated second.
.venv/bin/python -u bfa_dem.py --preset reference --out runs/dem/my_run

# raw BFA project inputs, nothing fitted
.venv/bin/python -u bfa_dem.py --preset bfa --out runs/dem/my_run

# score it
.venv/bin/python tools/perf_table.py runs/dem/my_run    # one row: cost, observables, rms, pile
.venv/bin/python compare_bfa_dem.py runs/dem/my_run     # full breakdown
```

Any explicit flag overrides the preset (e.g. `--mu 0.13`).

## Scenario files (any machine)

A run is defined by a JSON scenario (`dem_scenario.py` documents every field): parts
(STL, sidedness, friction, active time window, surface motion), material, injectors,
domain, reporting regions, solver and output settings.

```bash
# import a BulkFlowAnalyst project (reads .prj + .lin; writes <project>/scenario.json)
.venv/bin/python bfa_import.py <project>               # reference preset (default): BFA's dt, true E
.venv/bin/python bfa_import.py <project> --preset fast --out <project>/scenario_fast.json
# run any scenario
.venv/bin/python -u dem_run.py <project>/scenario.json --out runs/dem/<name>
```

The importer lists every interpretation and every unsupported feature it met; read them.
`bfa_dem.py` is now a thin wrapper that builds the corn scenario from its flags, verified
bitwise-identical to the pre-scenario runner (positions, velocities, spins, contact history).

## Measurements (after the run)

Measurements are chosen after a run and computed from its frames (`dem_analyze.py`), so
the solver pays nothing and any probe can be added later.  Frames carry position,
velocity, spin and a grain id.

```bash
.venv/bin/python dem_analyze.py runs/dem/<run> loads --window 20 25   # part forces/torques + wall maps
.venv/bin/python dem_analyze.py runs/dem/<run> flows                  # scenario flow planes, or --plane ...
.venv/bin/python dem_analyze.py runs/dem/<run> regions
```

Outputs go to `<run>/analysis/`: `part_loads.csv`, `wall_maps.vtk` (window-mean pressure,
shear, wear rate, contact count per triangle), `flows.csv`, `regions.csv`.  Checks: a
settled pile's wall loads equal its weight to 0.2%; frame-based portal flows equal the
live counters exactly; a head chute's horizontal load equals the stream's momentum flux.
Limits: stuck contacts' tangential force uses the sliding law (the history spring is not
in frames); corner second contacts (#13) are not yet reconstructed; impacts shorter than
the frame interval are sampled, not integrated.

## Operator screen: EMS DEM (web UI)

```bash
.venv/bin/python -m pip install -r dem_ui/requirements.txt    # once
.venv/bin/python -m dem_ui.server                            # http://127.0.0.1:8765
```

- **Setup:** projects (BFA folders, `scenario*.json`, or new ones in the git-ignored
  `projects/`); one-click BFA import; STL upload with unit conversion; 3D view of parts,
  injection, flow planes and domain; editable solver preset, material, parts (type wall /
  conveyor belt / rotating, direction picked on an edge in 3D, friction, active window,
  corners) and injection: an STL face, or a box on a belt or a plane (position, lateral
  offset, clearance, length, width, height; live capacity check).  Drafts save and list
  what is missing before they can run.  A live estimate under Run gives grains per second,
  grains held at once, GPU memory and time per simulated second (grain size and mass rate
  set the cost: 10,000 t/h of 12 mm grains is 3 M grains/s and does not fit a 16 GB GPU).
- **Particle pool:** sized from the mass rate x residence time (longest belt at its speed
  + 3 s, else 5 s, at most the run's duration) unless `solver.expected_holdup_kg` is set,
  and capped at what the GPU holds (~960 bytes per grain).  The domain is stretched to hold
  every injection site if it does not (grains outside it are recycled at once).
- **Run:** launches `dem_run.py` as a subprocess; live progress, mass, region and flow charts; stop.
- **Results:** frame playback coloured by speed; post-run measurements (`dem_analyze.py`) for
  a chosen window: part loads (table and chart), flows (rates and shares), regions; wall
  pressure / shear / wear / contact maps on the geometry.
- Deep links: `#scenario=<path>` or `#run=<id>&tab=results&frame=<n>&colour=pressure_Pa`.
- Binds to localhost: the server reads and writes files in the workspace, so do not expose
  it without authentication.  three.js is vendored (MIT) so it works offline.

## Performance

All measured on one Quadro P5000 (Pascal) against the reference config; details and
dead ends in `granular_dem.py` comments. Profile with `tools/dem_bench.py` from a
checkpoint (`bfa_dem.py --checkpoint-at 3 --out runs/perf/ckpt`).

| stage | ms / step | s per sim-s | note |
|---|---|---|---|
| start (BVH walls, per-step hash grid) | 1.73 | 71 | wall contact was 73% of the step |
| baked wall grid | 0.95 | 39 | CSR candidate lists, records inlined |
| + idle grains parked out of reach, fused kernels | 0.71 | 29 | |
| + Verlet neighbour list (N = 8) | 0.47 | 19 | grid rebuild ~180 µs → amortised |
| **fast preset** (E/10, dt × 4, N = 4) | 0.52 | **6.0** | 4× fewer steps |

What did *not* help: sorting the wall launch spatially, BVH builder choice (±20%), CUDA
graphs (launches were already hidden behind device time — kept for determinism), a
smaller hash table (less memset, more aliasing).

**Accuracy along the way** (steady state, t > 3 s; BFA pile top −1.401):

| run | s/sim-s | rms | pile top |
|---|---|---|---|
| old reference code | 71.2 | 0.053 | −1.428 |
| reference preset | 31.6 | 0.057 | −1.403 |
| E/10, dt × 3 | 12.0 | 0.056 | −1.427 |
| **fast preset** | **6.0** | **0.056** | −1.440 |
| E/100, dt × 6 | 7.3 | 0.058 | −1.472 — pile compacts |
| fast + 2 mm mesh simplification | 4.9 | 0.067 | −1.464 — deflector rims matter |

Known limits of the fast paths: a grain that has outrun its neighbour list is searched
directly, but two grains that have *both* outrun theirs can miss each other (needs two
grains above ~12 m/s in the fast preset; the flow tops out near 8). Grains that leave the
domain keep querying until the once-per-frame recycle and may scan the idle-grain bucket
meanwhile — a cost, not an error. `--no-rotation` runs the original solver path (BVH
walls, no neighbour list, no Hertz).

**Wall-contact fix found by this work:** the old BVH wall path computed Spout distances in
float32 against 3 m sliver panels (aspect up to 337): errors up to 0.19 mm against ~50 µm
Hertz overlaps, and 12% of Spout contacts silently dropped (zero sign normal). The grid
path uses per-triangle frames; `tools/wall_grid_check.py` holds it to < 1 µm vs float64.

## Verified configs

BFA targets: hold-up 17.69 kg, KE 154.0 J, chute tube (y 0..2.5) 6.45 kg @ 4.33 m/s,
cascade 4.59 kg @ 2.39 m/s, discharge 8.61 kg/s, settled top-of-slow −1.401 ± 0.020 m.
BFA cost: 5796 s on 24 CPU cores.

| run | μ_pp | wall μ | contact | k_t/k_n | rms | top-of-slow | s/sim-s |
|---|---|---|---|---|---|---|---|
| `hz_mu011_wall053mindlin` | 0.11 | 0.53 | Hertz | 1.0 | 0.054 | −1.417 ± 0.008 | 69.9 |
| `hz_mu011_wall054mindlin` | 0.11 | 0.54 | Hertz | 1.0 | **0.049** | −1.418 ± 0.013 | 69.7 |
| `hz_mu011_wall06mindlin` | 0.11 | 0.60 | Hertz | 1.0 | 0.044 | −1.448 ± 0.014 | 73.1 |
| `hz_mu009` | 0.09 | 0.60 | Hertz | 0 | 0.046 | −1.519 ± 0.006 | 68.1 |
| `lin_fix` | 0.09 | 0.60 | linear | 0 | 0.055 | −1.529 ± 0.003 | 63.6 |
| BFA | 0.09 | 0.50 | Hertzian | — | — | −1.401 ± 0.020 | 579.6 (24 cores) |

`rms` is the rms log error over 7 observables. **It ranked the wrong run first three
times** — it scored configs well that left the pile visibly low. For the visual match use
`tools/pileavg.py` (`top-of-slow`, `slow` fraction), and treat rms as secondary.

Run directories live under `runs/dem/`. Runs made before 2026-09-11 (`ppsh0_*`, `ts00_*`,
`dem_*`) are **not reproducible**: the wall contact damped only on approach back then, so a
wall asked for e = 0.20 delivered 0.58.

Both table entries below were re-run on 2026-09-28 (`runs/dem/verify_best`,
`runs/dem/verify_fast`) and reproduce the originals to within **0.34 % on every physical
observable** — cascade mass 0.21 %, chute mass 0.01 %, KE 0.04 %. Wall clock varies 1–2 %.

## Fitted vs measured

Measured from the BFA project and used as-is: grain 6 mm / 994.05 kg/m³, rolling friction
0.30, restitution 0.20, cohesion 0, mass flow 8.6111 kg/s, injection 3.1321 m/s, dt
24.316 µs, Young's 1.422e8 Pa, contact mode Hertzian.

Fitted: **μ_pp 0.11** (BFA states 0.090), **wall μ 0.53** (BFA states 0.50 on all 8 pairs),
wall rolling 0.50 (BFA states 0.30). Poisson ratio 0.30 is assumed — it is not in the `.prj`.

Wall friction is independently pinned at **≈ 0.57** by chute mass (0.572), chute speed
(0.566) and KE (0.561); 0.53 was chosen instead because it matches settled pile height.
Height responds ~4× more to μ_pp than to wall μ, so the two decouple: set wall from the
chute, set μ_pp from the height. `μ 0.135 / wall 0.57` is the untested best-of-both.

## Open residuals

1. **Cascade mass is 1.09–1.16× in every run that matches height.** Cascade mass and speed
   lie on one line, `speed = −0.572·mass + 5.395` (r = −0.987 over 9 runs), and BFA sits
   0.38 m/s off it. Five mechanisms were tested for whether they leave that line — μ_pp,
   wall μ, restitution calibration, and the tangential spring twice. None does. Whatever
   BFA has is not *more* dissipation but dissipation of a different kind. Untested
   candidate: grain shape (BFA's material defines 7 sizes, 6 marked "Particle Type:
   Cluster", though this run used `Min_Rad = Max_Rad`).
2. **~0.7 kg missing from the spout above the tube window.** At wall 0.60 the chute is
   +0.22 kg and the cascade +0.44 kg over BFA, yet total hold-up lands 0.05 kg under.
3. **Restitution delivers ~0.33 for a requested 0.20** in both contact modes — the `fn ≥ 0`
   clamp ends contact early and truncates the rebound, so the analytic ζ→e formula does not
   apply. `--calibrate-restitution` corrects it numerically (accurate for Hertz, ~17% off
   for linear). It changes results without moving off the locus, so it is default-off.
4. **No tangential contact law can matter in this flow** — the Coulomb cap always binds
   (6 N viscous vs 0.006 N cap) and the Mindlin spring yields in under 1/60 of a timestep.
   It only bites where slip approaches zero, i.e. a static heap. Do not re-test it here.
5. **The original goal (b), the quarter-grain run, is not started.** At ¼ grain size that is
   64× the particles and ~4× the steps — roughly 28 h for 10 s.
