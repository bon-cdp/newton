# 2. Setup

The Setup tab edits a **scenario**: the geometry, material, moving parts, injection and
solver settings of one simulation.  Edits stay in the browser until **Save** (or **Run**,
which saves first).  A scenario that is not complete yet still saves, as a draft, and the
message under the buttons lists what is missing.

## Projects

Three ways to start:

| | How | Result |
|---|---|---|
| **Import a BFA project** | Copy the BulkFlowAnalyst project folder (with its `.prj`, `.lin` and STLs) into the repository root; it appears under Projects. Press **import BFA project** (or **re-import BFA** to refresh it). | `scenario.json` in that folder: parts, material, belts, deflector timings, injection, portals (flow planes). The *Importer notes* section lists every interpretation and anything unsupported — read it. |
| **New project** | **+ new project**, give a name. | An empty scenario in `projects/<name>/`. Then upload geometry. |
| **Open an existing scenario** | Click it under its project. | — |

`projects/` and customer project folders are git-ignored: they never go into the
repository.

## Geometry (upload STL)

Open the *Geometry* section, choose one or more STL files, their **units** (m, mm, cm, in,
ft — converted to metres on upload) and what they are:

- **part (wall / conveyor)** — anything grains touch.  Each file becomes one part.
- **injection face** — a flat surface grains are spawned on (alternatively use an
  injection box, below).

**Fit domain** resizes the simulation box around the geometry and the injection box.  Grains
leaving the domain are removed (counted as discharged or escaped).  A run also stretches
the domain itself if the injection box sticks out of it.

## Solver

| Field | Meaning |
|---|---|
| **preset** | **reference** (default): true Young's modulus and BFA's own timestep — closest to BFA. **fast**: Young's modulus / 10, larger timestep, neighbour lists — about 3× faster; on the iron-ore conveyor it kept flows within 1–3 % but over-steered the left/right split by up to 0.04 at the steepest deflectors. |
| **output fps** | Frames written per simulated second (15 is BFA's default). More frames: smoother playback and finer measurements, more disk. |
| **write VTK frames** | Needed for playback and every measurement; turn off only for quick checks. |
| **duration** | Next to Run: simulated seconds. |

## Material

Grain radius and density, Young's modulus, Poisson ratio, restitution, grain–grain sliding
and rolling friction, wall rolling friction, contact law (Hertz–Mindlin or linear) and the
tangential/normal stiffness ratio.  New projects start from placeholder values (6 mm,
1000 kg/m³) — **set the real material**: grain size and density decide how many grains the
run needs and therefore its cost (see *Run estimate*).

## Parts

One row per part:

| Column | Meaning |
|---|---|
| show | Hide or show it in the 3D view (does not affect the run). |
| type | **wall**, **conveyor belt** or **rotating**. |
| 2-sided | Grains bounce off both faces (open sheets, chutes). One-sided parts collide on their outward face only. |
| friction | Grain–wall sliding friction. |
| on / off (s) | Active window: the part exists only in this time span.  This is how BFA's staged deflectors are modelled — one part per position, each active for its window. Empty *off* = until the end. |
| corners | Second contact in concave corners and creases (troughed belts, chute corners). Slightly slower; belts get it automatically. |
| × | Remove the part from the scenario (the STL file stays). |

### Conveyors and rotating parts

Choosing **conveyor belt** opens a row under the part:

- **belt speed** (m/s);
- **direction**: press **pick edge** and click an edge of the part in the 3D view that runs
  along the belt (as in BFA).  The direction snaps to an axis when within 5°; inclined belts
  keep their slope.  **flip** reverses it.  Red arrows on the belt show the running
  direction.

The belt does not move: its surface carries the given velocity, exactly like a real belt
under the material.  **Rotating** parts take a speed (rpm) and an axis (picked on an edge),
through the part's centre; purple arrows show the axis.  Rigid motion of whole parts (gates, swinging chutes) is not available yet
(issue #14); staged parts with on/off windows cover deflector positions.

## Injection

**source** chooses how grains enter:

- **STL face** — grains appear on a lattice over a flat face, with the velocity you set.
- **box on belt** — a box of spawn points above a conveyor (BFA's *injection volume from
  belt reference*): position from the belt tail, lateral offset, clearance above the belt
  surface, and the box's length, width and height.  With **move at belt velocity**, grains start
  at the belt's velocity.
- **box on plane** — the same above a horizontal plane at a given height, centred at (x, z).

Common fields: **mass rate** in kg/s or t/h (the two stay in sync), start and stop times.
The box is drawn in green in the 3D view as you type.

Under the fields, a **capacity check** says whether the box can deliver the mass rate at
that speed (width × height × packing × density × speed).  If not, grains queue up — the
inlet chokes, as a real undersized one would.

## Run estimate

The line under **Run** updates as you edit:

> ≈ 12.1 k grains/s (229 g each) · ~60.6 k grains held at once · ~3.3 ms/step → ~4 s per simulated second, ~2 min in all

It comes from the mass rate, grain size, timestep and the residence time along the belt.
It turns **red** when the run would not fit in GPU memory and amber when it would take more
than ~10 minutes per simulated second.  10,000 t/h of 12 mm grains is 3 million grains per
second — check grain size and mass rate before pressing Run.  Timings are calibrated on
the fast preset on a Quadro P5000; read them as order of magnitude.

## Save and Run

**Save** writes `scenario.json`.  **Run** saves, checks the scenario (a run refuses to start
while anything is missing) and launches it — continue in [Run](03-run.md).
