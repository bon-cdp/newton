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
| **gravity** | Its direction in the geometry's own coordinates: −Y (Y up, the default) or −Z for Z-up CAD exports, any other axis, or a custom vector in m/s². "Up" follows it everywhere: the 3D view, box injectors placed above a belt or plane, belt running directions, and which side of the domain counts as discharge. |
| **duration** | Next to Run: simulated seconds. |

## Material

Grain radius and density, Young's modulus, Poisson ratio, restitution, grain–grain sliding
and rolling friction, wall rolling friction, contact law (Hertz–Mindlin or linear) and the
tangential/normal stiffness ratio.  New projects start from placeholder values (6 mm,
650 kg/m³ bulk) — **set the real material**: grain size and density decide how many grains the
run needs and therefore its cost (see *Run estimate*).

Density is entered as in BFA: the **bulk density** of the material and a **packing ratio**
(particle density ÷ bulk density, 1.5 by default). The grains get the **particle density** =
bulk × packing ratio, shown read-only under the two. A BFA import takes both from the
project. Scenarios saved before this field existed open with bulk = particle density ÷ 1.5,
which leaves the grains unchanged.

**Size distribution** mixes sphere sizes. Press **+ size**: a row with the current radius at
100 % appears, then a smaller size at 0 %. Each row takes a radius (m) and a **mass %**, the
way a sieve curve is given; percentages need not add up to 100, they are normalised. The
*grains %* column shows the share of each size by count. While the table has rows, the single
**grain radius** above is not used. Remove every row to go back to one size.

- The timestep comes from the smallest grain, and injection spacing and grids from the largest,
  so a wide spread costs time.
- Mass rates, held/discharged masses, flows and regions count each grain at its own mass.
- Playback draws each size at its own diameter.

**Cohesion** makes grains stick to each other and to walls, for moist or sticky material.
As in BFA it is a **fraction of grain weight**: while two grains (or a grain and a wall)
touch, they also pull together with cohesion × the grain's weight (mass × 9.81 m/s²,
whatever the gravity setting). For two grains of different sizes the mass is
2·m₁m₂/(m₁+m₂), which is one grain's mass when they are equal. A BFA import carries
the project's *Cohesion_Coefficient* over unchanged.

- **0** (the default) means cohesionless, and costs nothing.
- **Wall cohesion** left blank means the same value as grain–grain.  A single part can
  override it in the **cohesion** column of the Parts table (blank = the material's value),
  for example a sticky liner.

Calibrate the value against a repose or hold-up test. As a scale: above **1** a single
grain hangs under a ceiling by cohesion alone; below 1 it falls (`test_cohesion.py`).
Cohesion adds one extra pass per contact type when switched on.

## Parts

One row per part:

| Column | Meaning |
|---|---|
| show | Hide or show it in the 3D view (does not affect the run). |
| type | **wall**, **conveyor belt** or **rotating**. |
| 2-sided | Grains bounce off both faces (open sheets, chutes). One-sided parts collide on their outward face only. |
| friction | Grain–wall sliding friction. |
| cohesion | Grain–wall cohesion for this part (fraction of grain weight); blank = the material's wall cohesion. |
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

- **STL face**: grains appear on a lattice in the volume swept by a flat face along its
  normal, with the velocity you set.
  - **extrude depth** sets how far the volume reaches. A new face starts at 5 grain
    diameters; 0 puts the sites on the face itself, the original layout.
  - **flip** puts the volume on the other side; the green arrow in the 3D view shows which.
  - **offset from face** starts the volume that far from the face, along the same normal,
    for example to keep spawned grains clear of a wall the face lies on. A negative value
    moves it back through the face.
  - The face may be tilted.
  - As for boxes, the panel shows what the inlet can pass: its cross-section seen along the
    injection velocity × lattice packing × density × speed. If that is less than the mass
    rate, enlarge the face or the depth, or raise the speed. The run log reports the same
    check for every injector.
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
