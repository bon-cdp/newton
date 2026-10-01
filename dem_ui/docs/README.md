# EMS DEM — user guide

EMS DEM is the operator screen for the GPU DEM solver in this repository: set up a transfer
(geometry, material, conveyors, injection), run it, and measure the results — wall loads,
flows, dust and air — in a web browser.

| Guide | What it covers |
|---|---|
| [1. Starting the app](01-start.md) | install, start and stop the server, open it, remote access, troubleshooting |
| [2. Setup](02-setup.md) | projects, BFA import, STL upload, solver preset, material, parts and conveyors, injection, the run estimate |
| [3. Run](03-run.md) | launching a run, live charts, the log and its warnings, stopping |
| [4. Results](04-results.md) | playback, measurements (wall loads, flows, regions, dust & air), colouring, output files |
| [5. Command line](05-command-line.md) | the same workflow without the browser, and ParaView files |

The physics, validation against BulkFlowAnalyst and performance history are in
[`BFA_REPLICATION.md`](../../BFA_REPLICATION.md).

## The workflow at a glance

```
 Setup ──────────────► Run ──────────────► Results
 geometry, material,   GPU simulation,     pick a time window, then
 conveyors, injection  live mass / flow    Wall loads · Flows · Regions · Dust & air
 (or import BFA)       charts              3D: speed, pressure, wear, air, dust
```

Everything is stored as plain files: a project is a folder with STL files and a
`scenario.json`; a run is a folder under `runs/dem/` with frames, `history.csv`, `run.json`
and an `analysis/` folder.  The app never holds state of its own, so the command-line tools
and the browser always see the same thing.
