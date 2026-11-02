# EMS DEM: handover

This repository is a fork of NVIDIA's [Newton](https://github.com/newton-physics/newton)
physics engine. On top of it sits **EMS DEM**: a GPU discrete-element simulator for bulk
material handling (conveyor transfers, chutes, spouts), with a web operator screen. It
replaces day-to-day BulkFlowAnalyst (BFA) runs; BFA stays the sign-off tool.

This page is the starting point for whoever takes it over. It covers what is where, how to
run and update it, how changes have been made so far (with real commits as examples), how
to check a change, and what is still open. The user guide for the app itself is in
[`dem_ui/docs/`](dem_ui/docs/README.md). The physics validation history is in
[`BFA_REPLICATION.md`](BFA_REPLICATION.md).

### What is where, in one table

| Group | Files | Status |
|---|---|---|
| **The product** | `dem_ui/` (web app), `dem_run.py`, `dem_scenario.py`, `granular_dem.py`, `clumps.py`, `dem_analyze.py`, `dem_air.py`, `bfa_import.py`, `mesh_simplify.py` | Active: this is what to maintain |
| Old files the product still imports | `compare_bfa_mpm.py` (`read_frame`, the VTK frame reader) and `bfa_replication_mpm.py` (`write_particles_vtk`, `write_geometry_vtk`) | Active dependency inside legacy files: **do not delete** them (see section 7) |
| Tests | `test_cohesion.py`, `test_consolidation.py`, `test_moving.py`, `test_rotation.py`, `test_granular_dem.py` | Section 5 |
| Tools | `bfa_dem.py` (corn-spout benchmark CLI), `compare_bfa_dem.py`, `export_bfa_vtk.py`, `bfa_params_audit.py`, `angle_of_repose.py`, `profile_dem.py`, `tools/` | Useful utilities, mostly tied to the 23087 corn project data |
| Early MPM work | `shiploader_mpm_test*.py`, `mpm_bifurcation.py`, `fit_wall_friction.py`, `build_gasket.py`, `analyze_vtk.py`, `check_contact_normals.py`, `repro_mesh_passthrough.py`, `compression_fix.md`, `_backup_fork/`, the `.stl`/`.jpg` assets at the top level | Legacy, kept for the record (appendix) |
| NVIDIA Newton (upstream) | `newton/`, `docs/`, `asv/`, `pyproject.toml`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/`, … | Upstream code. `newton/` carries a few fork patches to the MPM solver from the early work; the DEM does not depend on them |
| Not in git | `.venv/`, `runs/` (all output, tens of GB), `projects/` (UI projects), customer BFA folders | Section 6 |

The appendix lists every top-level file.

## 1. Set up a machine

Needs Linux, an NVIDIA GPU with a recent driver, and Python 3.11. It was developed on a
Quadro P5000 (16 GB, CUDA 12).

```bash
git clone https://github.com/bon-cdp/newton.git newton-dem
cd newton-dem
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .                       # Newton itself (editable)
.venv/bin/python -m pip install "warp-lang==1.11.1"        # the Warp version everything was tested on
.venv/bin/python -m pip install -r dem_ui/requirements.txt # fastapi, uvicorn, python-multipart
.venv/bin/python -m pip install trimesh scipy              # trimesh: runner, importer, server; scipy: some tools
.venv/bin/python -m dem_ui.server                          # then open http://127.0.0.1:8765
```

The first run compiles the GPU kernels (a minute or two); after that Warp caches them in
`~/.cache/warp/`.

Customer projects and all simulation output are **not** in git (see section 6). Copy
project folders across by hand.

## 2. Update the app

Everything is on **`main`**. To update a working copy:

```bash
cd newton-dem
git pull                         # on main
# restart the server (Ctrl+C, then .venv/bin/python -m dem_ui.server)
# and hard-refresh the browser (Ctrl+Shift+R) so it loads the new JavaScript
```

Until late October 2026 new work lived on a branch (`ui-units-cohesion`) that had to be
pulled specially. It was merged into `main` with pull request #17 (November 2026);
nothing needs a special branch any more.

Other branches still on GitHub, **not** merged, kept for reference:

| Branch | What it holds | Status |
|---|---|---|
| `worktree-soybean-calibration` | Soybean bucket-discharge calibration (issue #16): video measurements, camera tracking, the first clump runner (`dem_clumps.py`, `granular_clumps.py`), ellipsoidal grains, a Bayesian tuner. | Research. Its clump idea was re-done inside the main solver (`clumps.py`), so the branch is a record of the calibration, not something to merge as is. |
| `worktree-qc-dust-fixes` | One commit to the dust model (`dem_air.py`): dust slides along walls instead of freezing. | Small and unreviewed. Worth reviewing and merging if dust work resumes. |

## 3. How a run works

```
 browser (dem_ui/static: app.js, viewer.js)
   │  edits scenario.json, presses Run
   ▼
 dem_ui/server.py  (FastAPI, port 8765)
   │  saves <project>/scenario.json, starts:  python dem_run.py <scenario> --out runs/dem/<name>_<time>
   ▼
 dem_run.py         reads the scenario (dem_scenario.py), builds the model and walls,
   │                 injects grains, steps SolverGranularDEM (granular_dem.py) on the GPU
   ▼
 runs/dem/<run>/    run.json (inputs + derived values), history.csv (one row per frame),
   │                 log.txt, frame_XXXX_particles.vtk, geometry/, analysis/
   ▼
 dem_ui/server.py  reads them back: live charts, playback (frame endpoint), and post-run
                    measurements by starting dem_analyze.py (loads, flows, regions, dust)
```

Everything is plain files. The server holds no state of its own, so the command line
(`dem_run.py`, `dem_analyze.py`, see `dem_ui/docs/05-command-line.md`) and the browser
always see the same thing.

### The pieces that matter

| File | Role |
|---|---|
| `dem_scenario.py` | The scenario format: dataclasses `Scenario`, `Material`, `Part`, `Injector`, `Solver`, `Output`, `Region`, `FlowPlane`. **Every user-facing setting starts here**: a field with a default and a comment. Old scenario files keep working because new fields have defaults. |
| `dem_run.py` | The runner. `build()` turns a scenario into a model, walls, injection lattices and a solver, and logs every derived value; `run()` is the time loop (CUDA graphs, injection, recycling, history, VTK, mass audit). Also `estimate()` (the UI's cost line), `resolve_dt()`, the motion manager (`motion_pose`, `motion_tables`) and the inlet helpers. |
| `granular_dem.py` | The physics: `SolverGranularDEM` and its Warp kernels. Hertz-Mindlin contact with rolling friction, the baked wall grid (`WallGrid`), conveyor and rotating surfaces, cohesion, consolidation strength, moving parts, clumps. |
| `clumps.py` | Random multi-sphere clump shapes for the size distribution. |
| `dem_analyze.py` | Post-run measurements from the frames: wall loads and wear maps, flows through planes, regions. `dem_air.py` adds the experimental dust and air model. |
| `bfa_import.py` | Turns a BFA project folder (`.prj`, `.lin`, STLs) into a `scenario.json`, and lists every interpretation it made. |
| `dem_ui/` | The web app: `server.py` (FastAPI endpoints), `static/app.js` (all forms and tabs), `static/viewer.js` (three.js 3D view), `static/chart.js`, `docs/` (the user guide). No build step: edit the JS and refresh. |

## 4. How to make a change

Every feature so far followed the same path through the layers. Read one of these
commits (`git show <commit>`) next to the recipe below; they are small and complete.

| Commit | Feature | Touches |
|---|---|---|
| `dbdd09fa` | Cohesion: a new material property and per-part override, with its own GPU pass | scenario → solver kernel → runner → UI field → docs → test |
| `b74a80b2` | A new column in the Parts table | UI only |
| `81f4e35f` | Gravity direction, and "up" following it everywhere | scenario, runner, UI, viewer |
| `2ca82250`, `42ada5b7` | Size distribution, then one injection stream per size | scenario, runner, analysis, UI, viewer |
| `6c709606` | Motion manager: parts that translate and rotate | scenario, runner (pose tables), solver (moving-part kernel), UI step editor, viewer, test |
| `dae5d0a1` | Consolidation strength (contacts harden under compaction) | scenario → solver (per-contact memory) → runner → UI → docs → test |
| `124601f5` | Typed conveyor direction | UI only, 13 lines |
| `b898d487` | Camera box in the 3D view | viewer, CSS, docs |

### Recipe: a new physics setting (e.g. cohesion, consolidation)

1. **`dem_scenario.py`**: add the field to `Material` (or `Part`), with a default that
   means "off" and a comment saying what it is in plain words and units.
2. **`granular_dem.py`**: give `SolverGranularDEM.__init__` a parameter, and do the work in
   a **separate kernel that is launched only when the setting is on**. Do not add code to
   the main contact kernels (`eval_particle_contact_*`, `eval_wall_grid_rot`): they sit at
   the GPU's register limit, and even a branch that never runs changes their speed and
   their rounding (see the GPU notes in section 7). Cohesion and consolidation both live in
   `add_pair_cohesion` / `add_wall_cohesion` for this reason.
3. **`dem_run.py`**: pass the scenario value into the solver in `build()`, and add a `log()`
   line so the run log shows the value used.
4. **UI**: material fields are one line in `MATERIAL_FIELDS` in `dem_ui/static/app.js`. A
   Parts-table column needs a `<th>` in `dem_ui/static/index.html` and a `<td>` plus an
   `onchange` in the parts loop of `renderSetup()` in `app.js`. The server needs no change:
   it saves whatever the scenario dict holds.
5. **Docs**: the user guide section in `dem_ui/docs/02-setup.md`.
6. **Test**: a `test_<feature>.py` that checks the new physics against a number worked
   out by hand (an equilibrium overlap, a hold/fall threshold, a rolling speed), plus the
   regression check in section 5.

### Recipe: a UI-only change

Edit `dem_ui/static/app.js` / `viewer.js` / `style.css`, save, and hard-refresh the browser.
No server restart is needed for static files, only for `server.py`. Check syntax quickly
with `node --check` on a copy named `.mjs` (the files are ES modules).

### Conventions worth keeping

- **Off means bit-identical.** A new feature switched off must reproduce the previous
  results exactly (section 5). That is what makes regressions visible.
- **Everything the run decides goes in the log** (`log()` in `build()`): derived timestep,
  pool size, inlet capacity, lattice sizes, clump shapes. Users debug from the log.
- **Units are SI internally.** STL units are converted on upload or by `scenario.units`.
- **The scenario is the contract.** The UI, the command line and the importer all
  produce the same `scenario.json`; never put logic in the UI that the runner does not see.

## 5. Check a change

Tests are standalone scripts (no pytest); each prints PASS/FAIL lines and ends with
`ALL PASS`. They need the GPU and take seconds to a couple of minutes.

```bash
.venv/bin/python test_cohesion.py        # cohesion against closed forms
.venv/bin/python test_consolidation.py   # consolidation hold/fall thresholds
.venv/bin/python test_moving.py          # moving parts: kinematics, riding, rolling at 2/7
```

| Script | Checks | Notes |
|---|---|---|
| `test_rotation.py` | Rolling down an incline at (5/7) g sin θ, rolling resistance, static friction | Standalone geometry |
| `test_moving.py` | Motion manager: pose maths, a grain riding a lifted plate, rolling at 2/7 on a sliding plate and a turntable | |
| `test_cohesion.py` | Cohesion: equilibrium overlaps against closed forms, hangs under a ceiling iff cohesion > 1 | |
| `test_consolidation.py` | Consolidation: a grain pressed at k × its weight stays stuck iff λ·k > 1 | |
| `test_granular_dem.py` | Stock Newton DEM against this solver on the BFA spout | Needs the `23087-25sim/` customer folder |
| `tools/belt_test.py`, `tools/trough_test.py`, `tools/hertz_check.py`, `tools/wall_grid_check.py` | Conveyor surfaces (#11), concave corners (#13), Hertz law, wall-distance precision | `wall_grid_check` needs a checkpoint |
| `tools/dem_bench.py` | Per-kernel GPU timing from a checkpoint (`dem_run.py --checkpoint-at T`) | For performance work |

**Regression against the previous version.** For anything that touches physics, run the
same scenario on the old and new code and compare `history.csv` column by column. With
the new feature off, every column must match exactly except `kinetic_energy_J` (summed
with GPU atomics, so it varies by ~0.3 J in 20,000 between runs) and `wallclock_s`.

```bash
# old code in a second worktree:  git worktree add ../dem-old <old commit>
.venv/bin/python dem_run.py projects/<p>/scenario.json --out runs/dem/regr_new --duration 1.5 --no-vtk
(cd ../dem-old && ../newton-dem/.venv/bin/python dem_run.py ...  --out .../regr_old ...)
# then compare the two history.csv files (any CSV diff, ignoring the two columns above)
```

For performance, compare **interleaved** runs (old, new, old, new): the GPU clock drifts
±5% between runs.

## 6. Data and the public repository

`bon-cdp/newton` is a **public** repository.

- `.gitignore` covers `runs/`, `projects/`, `.venv/` and two customer folders by name
  (`20060-CM-552/`, `23087-25sim/`). **Any other folder at the top level is not ignored.**
  Keep customer work under `projects/`, or add the folder to `.gitignore` before anything
  else, and never `git add -A` / `git add .` at the top level: check `git status` first.
- In commit messages and GitHub issues, report agreement with BFA (ratios, percentages),
  never a customer's absolute results or design details.
- `gh` commands need `-R bon-cdp/newton`; without it they go to NVIDIA's upstream.
- Files that already exist upstream (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/`)
  belong to Newton. Put project documentation in new files (like this one).

## 7. Things learned the hard way

- **GPU register pressure.** `eval_wall_grid_rot` sits at 127 registers on Pascal; over 128
  halves occupancy and costs 30–40%. New per-grain work goes in a separate kernel.
- **Float32 on long sliver triangles.** BFA meshes have triangles 3 m long and millimetres
  wide. Naive closest-point math lost contacts on them; `_tri_closest` uses per-triangle
  frames for this reason. Keep using it.
- **`wp.set_module_options({"enable_backward": False})` is required**: the adjoint of the
  wall kernel exceeds CUDA's parameter limit.
- **Determinism.** Injection is a precomputed schedule and the free list is sorted every
  frame, so runs are reproducible. Keep new random choices seeded from `solver.seed`.
- **Rolling grains.** A free sphere set rolling by a moving surface ends at 2/7 of the
  surface speed. That is physics, not a bug (`test_moving.py` checks it).
- **Mixtures and the inlet.** Size classes are placed independently, so a mixture's inlet
  passes about half of what one size does (`MIXTURE_INLET_FACTOR` in `dem_run.py`).
- **Units in the UI.** Most "the run exploded / nothing flows" reports so far were STL units
  (mm geometry with the scenario set to m) or an inlet that cannot pass the mass rate.
  Check the run log's geometry sizes and the inlet capacity line first.

## 8. Open work

GitHub issues on `bon-cdp/newton`:

| Issue | Topic | State |
|---|---|---|
| #10 | Clumps and size distributions | Implemented (`clumps.py`, size table). Clumps were pushed for testing on 2026-10-21 and not run end to end before that; worth a test (mass audit, rigid members) before relying on them. |
| #12 | End-to-end app: scenario files, operator screen, probes, cloud | Scenario, runner, UI and post-run probes done. Cloud runs not started. |
| #14 | Rigid part motion | Implemented (motion manager). Part loads leave moving parts out; `corners` is ignored on them. |
| #15 | Dust and air (one-way coupling) | Experimental (`dem_air.py`); see the unmerged fix branch. |
| #16 | Material calibration from video (soybean) | Done on its branch; results in its calibration README. |

Known gaps and ideas, roughly in order of value:

- **Consolidation strength** (`dae5d0a1`) was added to reproduce the hang-up of limestone
  with fines in a spout; it was not yet calibrated against the site observation (the
  project folder and notes are with the EMS team, not in git).
- **Clump end-to-end check** (see #10 above).
- **Loads on moving parts** in `dem_analyze.py` (needs the part pose per frame).
- **Mixture inlet efficiency:** placing sizes jointly (pre-packed mixed slabs) would recover
  the full inlet capacity.
- **Cloud runs** (#12 phase 3).

## Appendix: every top-level file

**Product** (section 3): `dem_ui/`, `dem_run.py`, `dem_scenario.py`, `granular_dem.py`,
`clumps.py`, `dem_analyze.py`, `dem_air.py`, `bfa_import.py`, `mesh_simplify.py`.

**Imported by the product, otherwise legacy:**

| File | What the product uses | The rest |
|---|---|---|
| `compare_bfa_mpm.py` | `read_frame` (VTK frames, used by the server and `dem_analyze.py`) | Readers for BFA's binary `.por`/`.his` output; assumes `./23087-25sim/` |
| `bfa_replication_mpm.py` | `write_particles_vtk`, `write_geometry_vtk` (used by `dem_run.py`) | The original MPM runner for the 23087 spout. Importing it also loads Newton's MPM solver |

**Tools:**

| File | Purpose |
|---|---|
| `bfa_dem.py` | The corn-spout replication as a CLI with presets (`fast`, `reference`, `bfa`); builds a scenario and calls `dem_run` |
| `compare_bfa_dem.py` | Scores a DEM run of the corn spout against BFA (hold-up, KE, chute, discharge) |
| `export_bfa_vtk.py` | Converts BFA's reference output into `bfa_reference_vtk/` for ParaView |
| `bfa_params_audit.py` | Decodes and cross-checks every BFA input of the 23087 project |
| `angle_of_repose.py` | Pours a heap and measures its repose angle (calibration) |
| `profile_dem.py` | Per-kernel timing (older; `tools/dem_bench.py` supersedes it) |
| `tools/` | ~20 diagnostics from the replication work (pile height, leaks, clusters, wall-grid checks, benchmarks). Many hard-code `/home/s/Documents/newton-dem` and need the 23087 data |

**Legacy (early MPM work, kept for the record):** `shiploader_mpm_test.py`,
`shiploader_mpm_test_prev_dec_25.py` (hard-codes another user's home folder),
`mpm_bifurcation.py`, `fit_wall_friction.py`, `build_gasket.py`, `analyze_vtk.py`,
`check_contact_normals.py` and `repro_mesh_passthrough.py` (they show why stock Newton's
DEM could not be used), `compression_fix.md` (stale plan), `_backup_fork/` (upstream copies
of the MPM files the fork patched), and the assets `ACS.stl`, `bifurcation_test.stl`,
`23087midslopemesh.stl`, `richardson-dem.jpg`.

**Upstream Newton:** `newton/`, `docs/`, `asv/`, `asv.conf.json`, `pyproject.toml`,
`uv.lock`, `.python-version`, `README.md`, `CHANGELOG.md`, `CITATION.md`,
`CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `LICENSE.md`, `SECURITY.md`, `.github/`,
`.gitlab-ci.yml`, `.coderabbit.yml`, `.licenserc.yaml`, `.pre-commit-config.yaml`.

**Docs:** this file; `BFA_REPLICATION.md` (validation against BFA, performance history,
verified configurations); `dem_ui/docs/` (the user guide).

### Cleanup candidates (deliberately not done)

None of these was changed for the handover, because the code works and was to stay as is.
They are reasonable first tasks for whoever continues:

1. Move `read_frame` and the two VTK writers into a small `dem_io.py`, so the product stops
   importing the legacy MPM files. Then the legacy files can move into a `legacy/` folder.
2. Make the `tools/` scripts find the repository from their own location instead of the
   hard-coded `/home/s/Documents/newton-dem`.
3. `viewer.js` has a JavaScript copy of `dem_run.motion_pose` (for playback). Any change to
   how motion steps compose must be made in both.
4. Stop tracking `23087midslopemesh.stl` (customer geometry, unused). It stays in git history
   regardless, since the repository is public.
5. Presets differ by entry point: the app, `bfa_import.py` and `dem_scenario.Solver` default
   to **reference** (true stiffness, BFA's timestep); `bfa_dem.py` (the corn benchmark CLI)
   defaults to **fast**. `BFA_REPLICATION.md` describes the latter. Worth unifying.
