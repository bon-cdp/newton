# 4. Results

Open a finished (or stopped) run from the Runs list.  Measurements are chosen **after** the
run and computed from its frames, so nothing has to be decided before running and any
measurement can be repeated with another window.

## Playback

The frame slider and ▶ under the 3D view step through the frames; grains are drawn at their
true size and coloured by speed (legend top right).  **particles** hides them, e.g. to see
wall maps or the air underneath.

## Measurements

Set the **window** (start – end, in simulated seconds) to the part of the run you want —
usually after the system has filled, and within one deflector position if parts switch.
Then press a measurement.  Each runs in the background (seconds to a minute) and its
results appear in the panel; the files go to `<run>/analysis/`.

### Wall loads

Forces and wear on every part, recomputed from the frames with the solver's own contact
law.

- Table: window-mean force on each part (|F| and components, N).
- Chart: force on each part over time.
- 3D maps — choose with **colour**: *walls: pressure* (Pa), *shear* (Pa), *wear rate*
  (W/m², sliding power per area — where liners wear), *contacts* (per frame).
- Checks: a settled pile's loads equal its weight within 0.2 %; a head chute's horizontal
  load equals the stream's momentum flux.

### Flows

Mass through each flow plane (portal), matched grain by grain between frames.

- Table: mass, mean rate (kg/s) and **share** — e.g. the left/right split under the chute.
- Chart: flow rate over time (1 s mean).

### Regions

Mass and mean speed inside each reporting region (window means).

### Dust & air (prototype)

The air the grains drag along, and where the dust it carries goes (one-way: computed from
the frames; the grains are not affected).

- **Air drawn in** (m³/s and m³/h) and a table of air in and out through each side of the
  air box — how much air the stream induces and where it leaves.
- **Dust fate** per size (10, 30, 75 µm): still airborne, settled on each part, escaped
  through each side.
- **colour → air: speed slice**: a slice of the time-mean air speed with arrows for the
  flow.  Choose the slice normal (side, end or plan view) and move it with the slider.
- **dust**: dust parcels drawn with the frames (red 10 µm, orange 30 µm, brown 75 µm).

How to read it: dust is released where grains dissipate energy (impacts, sliding) in
proportion to it, so it shows *where* dust comes from and *where it goes*.  Use it to
compare design options.  Absolute numbers are not final:

- air flows are an upper bound (each grain drags air as if alone; no shielding inside
  the stream, no slowing of the grains);
- dust **shares** are the result — dust **mass** needs an emission factor measured for the
  material;
- the air box has open sides: model what the stream lands on (receiving chute, belt),
  or dust leaving with the stream through the bottom is over-counted;
- 10 cm air cells: walls are about a cell thick and gaps narrower than that close.

Air cell size and dust sizes can be changed from the command line (see
[Command line](05-command-line.md)).

## Colouring

| colour | Shows | Needs |
|---|---|---|
| particles: speed | grain speed | — |
| walls: pressure / shear / wear rate / contacts | window-mean maps on the geometry | Wall loads |
| air: speed slice | time-mean air speed and flow arrows | Dust & air |

## Files

`<run>/analysis/`:

| File | From | Content |
|---|---|---|
| `part_loads.csv` | Wall loads | force and torque per part per frame |
| `wall_maps.vtk`, `wall_maps.npz` | Wall loads | per-triangle pressure, shear, wear, contacts (ParaView / app) |
| `flows.csv` | Flows | cumulative mass per plane per frame |
| `regions.csv` | Regions | mass and speed per region per frame |
| `air_dust.json` | Dust & air | summary: air per side, dust fate, settings, caveats |
| `air_dust.npz`, `air_mean.vtk` | Dust & air | air field (app / ParaView: slices, streamlines) and dust per frame |
| `<measurement>.log` | each | the tool's output, errors included |

A failed measurement says so next to the buttons, with the last line of its log.
