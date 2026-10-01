# 5. Command line

Everything the app does is a command-line tool underneath; the app only edits files and
starts these.  Useful for batches, overnight sweeps and scripting.  Run from the repository
root.

## Import a BFA project

```bash
.venv/bin/python bfa_import.py <project>                    # -> <project>/scenario.json (reference preset)
.venv/bin/python bfa_import.py <project> --preset fast --out <project>/scenario_fast.json
```

The importer prints every interpretation and every unsupported feature it met.

## Run a scenario

```bash
.venv/bin/python -u dem_run.py <project>/scenario.json                      # runs/dem/<name>_<time>/
.venv/bin/python -u dem_run.py <project>/scenario.json --out runs/dem/my_run --duration 16
.venv/bin/python -u dem_run.py <project>/scenario.json --no-vtk             # no frames: quick check only
```

`scenario.json` is plain JSON; `dem_scenario.py` documents every field.  Editing it by hand
and running is equivalent to Setup → Run.

## Measure

```bash
.venv/bin/python dem_analyze.py runs/dem/<run> loads   --window 20 25   # wall loads and maps
.venv/bin/python dem_analyze.py runs/dem/<run> flows                    # scenario flow planes
.venv/bin/python dem_analyze.py runs/dem/<run> flows --plane left 0 40.5 40 -2 -2 41 0 2
.venv/bin/python dem_analyze.py runs/dem/<run> regions
.venv/bin/python dem_analyze.py runs/dem/<run> dust    --window 6 16    # dust & air, default settings
```

`--plane NAME AXIS VALUE LOx LOy LOz HIx HIy HIz` adds a plane normal to axis 0/1/2 at
VALUE, bounded by the box LO–HI.  Runs made before runs recorded their project take
`--scenario <project>/scenario.json`.

### Dust & air options

```bash
.venv/bin/python dem_air.py runs/dem/<run> --window 6 16 \
    --cell 0.08          # air cell size (m); smaller = finer walls and gaps, slower (cells ~ 1/cell^3)
    --nu-t 0.002         # eddy viscosity (m2/s): turbulent mixing of air and dust
    --sizes 5 10 30 75   # dust sizes (um)
    --parcels 5000       # dust parcels released per simulated second (statistics, not mass)
    --emission 1.0       # g of dust per kJ dissipated: scales the illustrative dust mass only
```

## ParaView

Open in ParaView (File → Open):

- `runs/dem/<run>/frame_*_particles.vtk` (a time series) — glyph by `radius`, colour by
  speed;
- `geometry.vtk` — all parts (`part_id`);
- `analysis/wall_maps.vtk` — pressure, shear, wear per triangle;
- `analysis/air_mean.vtk` — the air field: Slice, Stream Tracer, or Threshold on `solid`.

## Benchmark a GPU

```bash
.venv/bin/python -u bfa_dem.py --duration 10 --no-vtk     # corn spout, fast preset: prints s per simulated second
```

Compare the reported wall clock per simulated second between machines (the Quadro P5000
does ~6 s per simulated second on this case).
