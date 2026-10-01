# 3. Run

Pressing **Run** in Setup starts the simulation as a separate process on the GPU
(`dem_run.py`) and opens the Run tab.  The browser can be closed and reopened at any time;
the run continues, and the Runs list shows its progress.

## What the tab shows

- **Status and progress** — running / done / stopped / failed, and simulated time out of
  the duration (also in the header while a run is going).
- **Mass (kg)** — mass in the domain (*held*), *injected* and *discharged* over time.
  Injected should follow mass rate × time; held levels off once the system is full.
- **Region mass (kg)** — mass in each reporting region, if the scenario defines any.
- **Flow through planes (kg/s)** — 1-second mean flow through each flow plane (BFA
  *portals*), e.g. the left/right split under a head chute.
- **Log** — the solver's own output, refreshed live.

## Reading the log

The header lists the derived settings: grain mass, grains per second, timestep and steps
per frame, the particle pool (how many grains the GPU holds), the estimate per step, the
injection lattice and batch, the wall grid.  Then one line per frame: time, grains, mass,
kinetic energy, discharged mass and wall-clock seconds.

Warnings start with `!!`:

| Warning | Meaning | What to do |
|---|---|---|
| `injection backlog … the inlet is choked` | Spawn points stay occupied; grains queue and are delivered late. | Make the injection box wider or taller, or raise the injection speed. |
| `particle pool exhausted` | More grains in the domain than the pool holds; the excess is not injected. | Usually a mass rate or grain size far larger than intended. The pool is sized from mass rate × residence time and capped by GPU memory. |
| `domain … extended to …` | The injection box stuck out of the domain; the run enlarged it. | Nothing (press *Fit domain* in Setup to save it). |
| `neighbour list overflow` | Extremely dense packing dropped contacts. | Report it; use the reference preset meanwhile. |

At the end the **mass audit** compares injected with the target and accounts for every
kilogram (held + discharged + escaped), and the wall-clock time per simulated second.

## Stopping

**Stop** ends the run cleanly (like Ctrl+C): frames already written stay, and the run can
be analysed in Results up to where it stopped.

## Where a run is stored

`runs/dem/<scenario name>_<date>_<time>/`:

| File | Content |
|---|---|
| `frame_NNNN_particles.vtk` | Each frame: grain positions, velocities, spins, ids (binary VTK, opens in ParaView). |
| `history.csv` | One row per frame: grains, mass, energy, regions, flows, injected, discharged, escaped. |
| `run.json` | The full scenario as run, plus derived values (timestep, pool, stiffness). |
| `geometry/` | A copy of the STLs, so the run stays viewable if its project changes or is deleted. |
| `geometry.vtk` | All parts in one file for ParaView. |
| `log.txt` | The log shown in the tab. |
| `analysis/` | Measurements made later in [Results](04-results.md). |

Runs are git-ignored; delete old run folders to free disk (a 35 s iron-ore run with frames
is ~1.6 GB).
