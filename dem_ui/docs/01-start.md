# 1. Starting the app

## Requirements

- An NVIDIA GPU with a recent driver (developed on a Quadro P5000, 16 GB).  The browser
  only needs WebGL; it can be any machine that reaches the server.
- The repository's Python environment, `.venv/`, with Warp and Newton installed (see the
  repository README).

## Install (once)

From the repository root:

```bash
.venv/bin/python -m pip install -r dem_ui/requirements.txt     # fastapi, uvicorn, python-multipart
```

three.js is bundled in `dem_ui/static/vendor/`, so the app works without internet access.

## Start

```bash
.venv/bin/python -m dem_ui.server                  # http://127.0.0.1:8765
.venv/bin/python -m dem_ui.server --port 9000      # another port
```

Open **http://127.0.0.1:8765** in a browser.  The server prints nothing while it works;
errors go to the terminal.

To keep it running after closing the terminal:

```bash
nohup .venv/bin/python -m dem_ui.server > runs/logs/ui_server.log 2>&1 &
```

## Stop

`Ctrl+C` in its terminal, or, if it was started in the background:

```bash
pkill -f '^[^ ]*python[^ ]* -m dem_ui\.server'
```

Stopping the server does **not** stop simulations already launched — they run as their
own processes and keep writing to `runs/dem/<run>/`.  Restart the server and they show up
again in the Runs list.

## Using it from another computer

The server only listens on the machine it runs on (`127.0.0.1`), on purpose: it can read
and write files in the workspace and has no login.  To use it from a laptop, forward the
port over SSH instead of opening it to the network:

```bash
ssh -L 8765:127.0.0.1:8765 user@gpu-workstation     # then open http://127.0.0.1:8765 on the laptop
```

`--host 0.0.0.0` would expose it to everyone on the network — do not, unless it sits
behind something that authenticates users.

## The screen

```
┌ EMS DEM ─ Setup │ Run │ Results ─────────────────────────────────────────────────┐
│ Projects        │                                       │ panel for the current   │
│  · scenarios    │            3D view                    │ tab: setup form, run    │
│ Runs            │   (drag: rotate, right-drag: pan,     │ charts, or results and  │
│  · status, time │    wheel: zoom)                       │ measurements            │
│                 │  [legend]          [frame slider ▶]   │                         │
└─────────────────┴───────────────────────────────────────┴─────────────────────────┘
```

- **Projects** (left): every BFA project folder and every project under `projects/`, with
  its scenario files.  Click a scenario to open it in Setup.
- **Runs** (left): every run, newest first, with its status (running, done, stopped,
  failed) and how far it got.  Click one to open it in Run (if running) or Results.
- **Tabs** (top): [Setup](02-setup.md), [Run](03-run.md), [Results](04-results.md).

Links can point straight at a view, e.g. to send to a colleague on the same machine:

- `http://127.0.0.1:8765/#scenario=projects/my-transfer/scenario.json`
- `http://127.0.0.1:8765/#run=<run id>&tab=results&frame=120&colour=pressure_Pa`

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Page blank, or an error like `$(...) is null` after an update | The browser kept old scripts. Reload with **Ctrl+Shift+R** once (the server now tells browsers to re-check, so this should not recur). |
| `address already in use` on start | A server is already running (open the page), or use `--port`. |
| "no longer exists (was its project deleted?)" | The scenario or project was deleted on disk; pick another. Runs made since the run-keeps-its-geometry change stay viewable after their project is deleted. |
| 3D view stays grey | WebGL is off in the browser (check `about:support` / `chrome://gpu`). |
| A run fails at once | Open it in the Run tab: the log's last lines say why (for example a box injector with no belt under it). |
