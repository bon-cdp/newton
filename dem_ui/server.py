#!/usr/bin/env python3
"""
Operator screen backend: a thin HTTP API over scenarios, the BFA importer, dem_run and
dem_analyze.  Heavy work (simulation, analysis) runs in subprocesses, so this server never
touches the GPU and stays responsive while runs go.

    .venv/bin/python -m dem_ui.server [--port 8765] [--host 127.0.0.1]

then open http://127.0.0.1:8765.  Binds to localhost by default: it can read and write
files anywhere under the workspace, so do not expose it without authentication.
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # the workspace (repo root)
sys.path.insert(0, ROOT)

from compare_bfa_mpm import read_frame  # noqa: E402
from dem_scenario import INF, Scenario  # noqa: E402

PY = sys.executable
RUNS = os.path.join(ROOT, "runs", "dem")
SKIP = {"runs", ".venv", "newton", "dem_ui", "tools", "docs", ".git", "_backup_fork",
        "bfa_reference_vtk", "asv", "__pycache__"}

app = FastAPI(title="Newton DEM operator screen")
JOBS: dict[str, subprocess.Popen] = {}           # run id -> simulation process
ANALYSES: dict[str, subprocess.Popen] = {}       # run id -> analysis process


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _inside(path: str) -> str:
    """Absolute path, refused unless it lies inside the workspace."""
    p = os.path.realpath(path if os.path.isabs(path) else os.path.join(ROOT, path))
    if not (p == ROOT or p.startswith(ROOT + os.sep)):
        raise HTTPException(403, "outside the workspace")
    return p


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _run_dir(run_id: str) -> str:
    d = _inside(os.path.join(RUNS, run_id))
    if not os.path.isdir(d):
        raise HTTPException(404, f"no run {run_id}")
    return d


def _read_history(d: str):
    path = os.path.join(d, "history.csv")
    if not os.path.exists(path):
        return {}
    rows = list(csv.reader(open(path)))
    if len(rows) < 2:
        return {h: [] for h in (rows[0] if rows else [])}
    head = rows[0]
    cols = {h: [] for h in head}
    for r in rows[1:]:
        if len(r) != len(head):
            continue                         # a row still being written
        for h, x in zip(head, r):
            try:
                cols[h].append(float(x))
            except ValueError:
                cols[h].append(None)
    return cols


def _last_time(d: str) -> float:
    """Last time_s in history.csv, reading only the file's tail (histories get long)."""
    path = os.path.join(d, "history.csv")
    if not os.path.exists(path):
        return 0.0
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 4096))
        lines = fh.read().decode(errors="replace").strip().splitlines()
    for ln in reversed(lines):
        try:
            return float(ln.split(",")[0])
        except ValueError:
            continue
    return 0.0


def _status(run_id: str, d: str) -> str:
    p = JOBS.get(run_id)
    if p is not None and p.poll() is None:
        return "running"
    log = os.path.join(d, "log.txt")
    if os.path.exists(log):
        tail = open(log, errors="replace").read()[-4000:]
        if "interrupted" in tail:
            return "stopped"
        if "wall clock" in tail:
            return "done"
        if "Traceback" in tail or "Error" in tail:
            return "failed"
        return "stopped" if p is not None else "unknown"
    return "done" if os.path.exists(os.path.join(d, "history.csv")) else "unknown"


# ---------------------------------------------------------------------------
# projects and scenarios
# ---------------------------------------------------------------------------

@app.get("/api/projects")
def projects():
    out = []
    for d in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, d)
        if d in SKIP or d.startswith(".") or not os.path.isdir(p):
            continue
        prj = glob.glob(os.path.join(p, "*.prj"))
        scen = sorted(glob.glob(os.path.join(p, "scenario*.json")))
        if prj or scen:
            out.append(dict(name=d, path=_rel(p), bfa=bool(prj),
                            scenarios=[_rel(s) for s in scen]))
    return out


@app.post("/api/projects/{name}/import")
def import_project(name: str, preset: str = "fast"):
    p = _inside(os.path.join(ROOT, name))
    out = os.path.join(p, "scenario.json" if preset == "fast" else f"scenario_{preset}.json")
    r = subprocess.run([PY, os.path.join(ROOT, "bfa_import.py"), p, "--preset", preset,
                        "--out", out], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise HTTPException(500, r.stderr[-2000:])
    report = [ln.strip() for ln in r.stdout.splitlines()
              if ln.strip().startswith(("note", "WARNING", "wrote"))]
    return dict(scenario=_rel(out), report=report)


@app.get("/api/scenario")
def get_scenario(path: str):
    p = _inside(path)
    return json.load(open(p))


@app.put("/api/scenario")
def put_scenario(path: str, body: dict = Body(...)):
    p = _inside(path)
    try:
        Scenario.from_dict(body, base_dir=os.path.dirname(p))      # validates
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, str(e))
    with open(p, "w") as fh:
        json.dump(body, fh, indent=2)
        fh.write("\n")
    return dict(saved=_rel(p))


def _geometry(sc: Scenario):
    import dem_run
    parts = []
    for p in sc.parts:
        v, f = dem_run.load_stl(sc.path(p.stl), p.fix_normals, p.flip, sc.unit_scale)
        parts.append(dict(name=p.name, two_sided=p.two_sided, friction=p.friction,
                          active=[p.active[0], None if p.active[1] >= INF else p.active[1]],
                          motion=p.motion, corners=p.corners,
                          vertices=_b64(v, np.float32), faces=_b64(np.asarray(f).reshape(-1), np.uint32),
                          n_tri=int(len(f))))
    inj = []
    for i in sc.injectors:
        tris, area = dem_run.face_triangles(sc.path(i.face_stl), sc.unit_scale)
        inj.append(dict(name=i.name, triangles=_b64(tris.reshape(-1), np.float32),
                        mass_rate=i.mass_rate, velocity=i.velocity, area=area))
    return dict(parts=parts, injectors=inj, domain=dict(lo=sc.domain_lo, hi=sc.domain_hi),
                flow_planes=[dict(name=f.name, axis=f.axis, value=f.value, lo=f.lo, hi=f.hi)
                             for f in sc.flow_planes],
                regions=[dict(name=r.name, lo=r.lo, hi=r.hi) for r in sc.regions])


@app.get("/api/geometry")
def geometry(scenario: str):
    """Parts as the SOLVER loads them (merged vertices, same triangle order), so
    per-triangle results line up with what is drawn."""
    return _geometry(Scenario.load(_inside(scenario)))


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*")), key=os.path.getmtime, reverse=True):
        meta_p = os.path.join(d, "run.json")
        if not os.path.exists(meta_p):
            continue
        rid = os.path.basename(d)
        try:
            meta = json.load(open(meta_p))
        except json.JSONDecodeError:
            continue
        t_last = _last_time(d)
        out.append(dict(id=rid, name=meta.get("name", rid), status=_status(rid, d),
                        scenario_based="parts" in meta, t=t_last,
                        duration=(meta.get("output") or {}).get("duration"),
                        frames=len(glob.glob(os.path.join(d, "frame_*_particles.vtk"))),
                        modified=os.path.getmtime(d)))
    return out[:200]


@app.post("/api/runs")
def start_run(body: dict = Body(...)):
    scen = _inside(body["scenario"])
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", body.get("name") or
                  f"{os.path.basename(os.path.dirname(scen))}_{time.strftime('%Y%m%d_%H%M%S')}")
    d = os.path.join(RUNS, name)
    if os.path.exists(d):
        raise HTTPException(409, f"run {name} exists")
    os.makedirs(d)
    cmd = [PY, "-u", os.path.join(ROOT, "dem_run.py"), scen, "--out", d]
    if body.get("duration"):
        cmd += ["--duration", str(float(body["duration"]))]
    if body.get("no_vtk"):
        cmd.append("--no-vtk")
    log = open(os.path.join(d, "log.txt"), "w")
    JOBS[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                  start_new_session=True)
    return dict(id=name)


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    p = JOBS.get(run_id)
    if p is None or p.poll() is not None:
        raise HTTPException(409, "not running from this server")
    os.killpg(p.pid, signal.SIGINT)                 # dem_run finishes the frame and closes files
    return dict(stopping=run_id)


@app.get("/api/runs/{run_id}")
def run_info(run_id: str):
    d = _run_dir(run_id)
    meta = json.load(open(os.path.join(d, "run.json")))
    log = os.path.join(d, "log.txt")
    tail = open(log, errors="replace").read()[-3000:] if os.path.exists(log) else ""
    frames = sorted(int(re.search(r"frame_(\d+)_", f).group(1))
                    for f in glob.glob(os.path.join(d, "frame_*_particles.vtk")))
    fps = (meta.get("output") or {}).get("fps", 15.0)
    return dict(id=run_id, status=_status(run_id, d), meta=meta, log=tail,
                frames=frames, fps=fps, history=_read_history(d),
                analysis=sorted(os.listdir(os.path.join(d, "analysis")))
                if os.path.isdir(os.path.join(d, "analysis")) else [])


@app.get("/api/runs/{run_id}/geometry")
def run_geometry(run_id: str):
    d = _run_dir(run_id)
    meta = json.load(open(os.path.join(d, "run.json")))
    if "parts" not in meta:
        raise HTTPException(422, "run predates scenario files")
    known = {"name", "material", "parts", "injectors", "domain", "regions", "flow_planes",
             "output", "gravity", "units", "notes"}
    sc = Scenario.from_dict({k: v for k, v in meta.items() if k in known},
                            base_dir=meta.get("base_dir", d))
    return _geometry(sc)


@app.get("/api/runs/{run_id}/frame/{k}")
def frame(run_id: str, k: int):
    """Binary: uint32 n, then n*3 float32 positions, then n float32 speeds."""
    d = _run_dir(run_id)
    f = os.path.join(d, f"frame_{k:04d}_particles.vtk")
    if not os.path.exists(f):
        raise HTTPException(404, "no such frame")
    fr = read_frame(f)
    pos = fr["pos"].astype(np.float32)
    spd = np.linalg.norm(fr["vel"], axis=1).astype(np.float32)
    body = np.array([len(pos)], dtype=np.uint32).tobytes() + pos.tobytes() + spd.tobytes()
    return Response(body, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

@app.post("/api/runs/{run_id}/analyze")
def analyze(run_id: str, body: dict = Body(...)):
    d = _run_dir(run_id)
    p = ANALYSES.get(run_id)
    if p is not None and p.poll() is None:
        raise HTTPException(409, "an analysis is already running for this run")
    what = body.get("what", "loads")
    if what not in ("loads", "flows", "regions"):
        raise HTTPException(422, "what must be loads, flows or regions")
    cmd = [PY, os.path.join(ROOT, "dem_analyze.py"), d, what]
    if body.get("window"):
        cmd += ["--window", str(float(body["window"][0])), str(float(body["window"][1]))]
    for pl in body.get("planes") or []:
        cmd += ["--plane", pl["name"], str(int(pl["axis"])), str(float(pl["value"])),
                *[str(float(x)) for x in pl["lo"]], *[str(float(x)) for x in pl["hi"]]]
    os.makedirs(os.path.join(d, "analysis"), exist_ok=True)
    log = open(os.path.join(d, "analysis", f"{what}.log"), "w")
    ANALYSES[run_id] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return dict(started=what)


@app.get("/api/runs/{run_id}/analysis")
def analysis(run_id: str):
    d = _run_dir(run_id)
    a = os.path.join(d, "analysis")
    p = ANALYSES.get(run_id)
    out = dict(running=p is not None and p.poll() is None, files=[], logs={})
    if not os.path.isdir(a):
        return out
    out["files"] = sorted(os.listdir(a))
    for f in out["files"]:
        if f.endswith(".log"):
            out["logs"][f] = open(os.path.join(a, f), errors="replace").read()[-3000:]
    for f in ("part_loads.csv", "flows.csv", "regions.csv"):
        path = os.path.join(a, f)
        if os.path.exists(path):
            rows = list(csv.reader(open(path)))
            out[f.split(".")[0]] = dict(header=rows[0], rows=rows[1:]) if rows else None
    maps = os.path.join(a, "wall_maps.npz")
    if os.path.exists(maps):
        z = np.load(maps)
        out["wall_maps"] = {k: _b64(z[k], np.float32) for k in z.files if k != "window"}
        out["wall_maps_window"] = z["window"].tolist() if "window" in z.files else None
    return out


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
