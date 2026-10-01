#!/usr/bin/env python3
"""
EMS DEM operator screen backend: a thin HTTP API over scenarios, the BFA importer, dem_run and
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
# Projects created here go in a git-ignored folder: uploaded geometry is often customer data.
PROJECTS = os.path.join(ROOT, "projects")
SKIP = {"runs", ".venv", "newton", "dem_ui", "tools", "docs", ".git", "_backup_fork",
        "bfa_reference_vtk", "asv", "__pycache__"}

app = FastAPI(title="EMS DEM")
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


def _existing(path: str) -> str:
    """_inside, and 404 (not a 500) if the file is gone, e.g. its project was deleted."""
    p = _inside(path)
    if not os.path.isfile(p):
        raise HTTPException(404, f"{_rel(p)} no longer exists (was its project deleted?)")
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
    cands = [os.path.join(ROOT, d) for d in sorted(os.listdir(ROOT))]
    if os.path.isdir(PROJECTS):
        cands += [os.path.join(PROJECTS, d) for d in sorted(os.listdir(PROJECTS))]
    for p in cands:
        d = os.path.basename(p)
        if d in SKIP or d == "projects" or d.startswith(".") or not os.path.isdir(p):
            continue
        prj = glob.glob(os.path.join(p, "*.prj"))
        scen = sorted(glob.glob(os.path.join(p, "scenario*.json")))
        if prj or scen:
            out.append(dict(name=_rel(p), path=_rel(p), bfa=bool(prj),
                            scenarios=[_rel(s) for s in scen]))
    return out


@app.post("/api/projects/import")
def import_project(project: str, preset: str = "reference"):
    p = _inside(project)
    out = os.path.join(p, "scenario.json" if preset == "reference" else f"scenario_{preset}.json")
    r = subprocess.run([PY, os.path.join(ROOT, "bfa_import.py"), p, "--preset", preset,
                        "--out", out], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise HTTPException(500, r.stderr[-2000:])
    report = [ln.strip() for ln in r.stdout.splitlines()
              if ln.strip().startswith(("note", "WARNING", "wrote"))]
    return dict(scenario=_rel(out), report=report)


DEFAULT_SCENARIO = {
    "name": "", "units": "m", "gravity": [0.0, -9.81, 0.0],
    "material": {"name": "material", "radius": 0.006, "density": 1000.0, "youngs": 1.0e8,
                 "poisson": 0.3, "restitution": 0.2, "friction": 0.3, "rolling_friction": 0.3,
                 "wall_rolling_friction": 0.3, "tangential_ratio": 1.0, "contact": "hertz"},
    "parts": [], "injectors": [], "domain": {"lo": [-1.0, -1.0, -1.0], "hi": [1.0, 1.0, 1.0]},
    "regions": [], "flow_planes": [], "output": {"duration": 5.0, "fps": 15.0, "vtk": True},
    "notes": {"source": "operator screen"},
}


@app.post("/api/projects")
def new_project(body: dict = Body(...)):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", (body.get("name") or "").strip())
    if not name or name in SKIP or name.startswith("."):
        raise HTTPException(422, "choose a project name (letters, digits, _ . -)")
    d = os.path.join(PROJECTS, name)
    if os.path.exists(d):
        raise HTTPException(409, f"{name} already exists")
    os.makedirs(d)
    sc = json.loads(json.dumps(DEFAULT_SCENARIO))
    sc["name"] = name
    with open(os.path.join(d, "scenario.json"), "w") as fh:
        json.dump(sc, fh, indent=2)
    return dict(project=name, scenario=_rel(os.path.join(d, "scenario.json")))


UNITS_TO_M = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254, "ft": 0.3048}


def _auto_domain(sc: dict, base: str, margin_frac: float = 0.15, margin_min: float = 0.3):
    import trimesh
    los, his = [], []
    for p in sc["parts"]:
        m = trimesh.load(os.path.join(base, p["stl"]) if not os.path.isabs(p["stl"]) else p["stl"],
                         force="mesh")
        los.append(m.bounds[0])
        his.append(m.bounds[1])
    for i in sc.get("injectors", []):
        if i.get("box"):
            try:
                import dem_run
                s = Scenario.from_dict(sc, base_dir=base, validate=False)
                c = np.asarray(dem_run.injector_box_corners(s, s.injectors[0]))
            except (ValueError, KeyError, IndexError, StopIteration):
                continue                    # box not placeable yet; the run reports why
            los.append(c.min(axis=0))
            his.append(c.max(axis=0))
        elif i.get("face_stl") and os.path.isfile(os.path.join(base, i["face_stl"])):
            m = trimesh.load(os.path.join(base, i["face_stl"]), force="mesh")
            los.append(m.bounds[0])
            his.append(m.bounds[1])
    if not los:
        return
    lo, hi = np.min(los, axis=0), np.max(his, axis=0)
    pad = np.maximum((hi - lo) * margin_frac, margin_min)
    sc["domain"] = {"lo": (lo - pad).round(4).tolist(), "hi": (hi + pad).round(4).tolist()}


@app.post("/api/projects/upload")
async def upload(request: __import__("fastapi").Request):
    """multipart: files (STL, several), units (m/mm/cm/in/ft), role (part | injection),
    scenario (path).  STLs go next to the scenario, converted to metres."""
    import trimesh
    form = await request.form()
    scen_path = _existing(form.get("scenario") or "")
    d = os.path.dirname(scen_path)
    scale = UNITS_TO_M.get(form.get("units", "m"))
    if scale is None:
        raise HTTPException(422, "units must be one of " + ", ".join(UNITS_TO_M))
    role = form.get("role", "part")
    sc = json.load(open(scen_path))
    added = []
    for up in form.getlist("files"):
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(up.filename)[0]) or "part"
        raw = await up.read()
        try:
            mesh = trimesh.load(trimesh.util.wrap_as_stream(raw), file_type="stl", force="mesh")
        except Exception as e:  # noqa: BLE001 -- report any parser failure to the user
            raise HTTPException(422, f"{up.filename}: not a readable STL ({e})")
        if len(mesh.faces) == 0:
            raise HTTPException(422, f"{up.filename}: no triangles")
        if scale != 1.0:
            mesh.apply_scale(scale)
        fname = f"{stem}.stl"
        mesh.export(os.path.join(d, fname))
        if role == "injection":
            inj = (sc.get("injectors") or [None])[0] or {
                "name": stem, "face_stl": fname, "mass_rate": 10.0, "velocity": [0.0, -1.0, 0.0]}
            inj["face_stl"] = fname
            sc["injectors"] = [inj]
        else:
            names = {p["name"] for p in sc["parts"]}
            pname = stem
            k = 2
            while pname in names:
                pname = f"{stem}_{k}"
                k += 1
            sc["parts"].append({"name": pname, "stl": fname, "two_sided": True, "friction": 0.5,
                                "active": [0.0, INF]})
        ext = mesh.bounds[1] - mesh.bounds[0]
        added.append(dict(file=fname, triangles=int(len(mesh.faces)),
                          size_m=[round(float(x), 4) for x in ext]))
    _auto_domain(sc, os.path.dirname(scen_path))
    with open(scen_path, "w") as fh:
        json.dump(sc, fh, indent=2)
    return dict(added=added, scenario=sc, problems=_problems(sc, os.path.dirname(scen_path)))


@app.post("/api/scenario/auto-domain")
def auto_domain(path: str):
    p = _existing(path)
    sc = json.load(open(p))
    _auto_domain(sc, os.path.dirname(p))
    with open(p, "w") as fh:
        json.dump(sc, fh, indent=2)
    return dict(domain=sc["domain"])


@app.post("/api/geometry-preview")
def geometry_preview(scenario: str, body: dict = Body(...)):
    """Geometry for an UNSAVED scenario (live preview while editing, e.g. an injection box)."""
    p = _existing(scenario)
    return _geometry(Scenario.from_dict(body, base_dir=os.path.dirname(p), validate=False))


_GPU_GB = None


def _gpu_gb() -> float:
    """Total memory of the first GPU, from nvidia-smi (the server itself never touches CUDA)."""
    global _GPU_GB
    if _GPU_GB is None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            _GPU_GB = float(out.split()[0]) / 1024.0
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            _GPU_GB = 16.0
    return _GPU_GB


@app.post("/api/estimate")
def estimate(scenario: str, body: dict = Body(...)):
    """Rough size and cost of an (unsaved) scenario before it runs: grains per second, grains
    held at once, GPU memory, time per simulated second (dem_run.estimate)."""
    import dem_run
    p = _existing(scenario)
    try:
        sc = Scenario.from_dict(body, base_dir=os.path.dirname(p), validate=False)
        if not sc.injectors:
            return dict(error="no injector yet")
        e = dem_run.estimate(sc)
    except (ValueError, KeyError, TypeError, IndexError, OSError, ZeroDivisionError) as ex:
        return dict(error=str(ex))
    gpu = _gpu_gb()
    e["gpu_gb"] = gpu
    e["too_big"] = e["memory_gb"] > gpu - 1.5
    return e


@app.get("/api/stl-files")
def stl_files(scenario: str):
    d = os.path.dirname(_existing(scenario))
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".stl"))


@app.get("/api/scenario")
def get_scenario(path: str):
    p = _existing(path)
    return json.load(open(p))


@app.put("/api/scenario")
def put_scenario(path: str, body: dict = Body(...)):
    p = _existing(path)
    try:
        Scenario.from_dict(body, base_dir=os.path.dirname(p), validate=False)   # structure
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, str(e))                  # malformed: refuse
    with open(p, "w") as fh:
        json.dump(body, fh, indent=2)
        fh.write("\n")
    return dict(saved=_rel(p), problems=_problems(body, os.path.dirname(p)))


def _problems(body: dict, base: str) -> list[str]:
    """Why a (well-formed) scenario cannot run yet -- drafts are saved anyway."""
    try:
        Scenario.from_dict(body, base_dir=base)
        return []
    except (ValueError, TypeError, KeyError) as e:
        return [ln.strip() for ln in str(e).splitlines()[1:]] or [str(e)]


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
        entry = dict(name=i.name, mass_rate=i.mass_rate, velocity=i.velocity)
        try:
            if i.box:
                entry["box_corners"] = dem_run.injector_box_corners(sc, i)
            elif i.face_stl and os.path.exists(sc.path(i.face_stl)):
                tris, area = dem_run.face_triangles(sc.path(i.face_stl), sc.unit_scale)
                entry.update(triangles=_b64(tris.reshape(-1), np.float32), area=area)
        except (ValueError, KeyError, IndexError) as e:
            entry["error"] = str(e)                # shown in the UI, e.g. no belt under the box
        inj.append(entry)
    return dict(parts=parts, injectors=inj, domain=dict(lo=sc.domain_lo, hi=sc.domain_hi),
                flow_planes=[dict(name=f.name, axis=f.axis, value=f.value, lo=f.lo, hi=f.hi)
                             for f in sc.flow_planes],
                regions=[dict(name=r.name, lo=r.lo, hi=r.hi) for r in sc.regions])


@app.get("/api/geometry")
def geometry(scenario: str):
    """Parts as the SOLVER loads them (merged vertices, same triangle order), so
    per-triangle results line up with what is drawn."""
    p = _existing(scenario)
    return _geometry(Scenario.from_dict(json.load(open(p)), base_dir=os.path.dirname(p),
                                        validate=False))


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*")), key=os.path.getmtime, reverse=True):
        meta_p = os.path.join(d, "run.json")
        rid = os.path.basename(d)
        if not os.path.exists(meta_p):
            if os.path.exists(os.path.join(d, "log.txt")):
                # launched from here but died before writing run.json (setup error): show
                # it, so the failure is visible rather than the run silently missing
                out.append(dict(id=rid, name=rid, status=_status(rid, d), scenario_based=True,
                                t=0.0, duration=None, frames=0, modified=os.path.getmtime(d)))
            continue
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
    scen = _existing(body["scenario"])
    probs = _problems(json.load(open(scen)), os.path.dirname(scen))
    if probs:
        raise HTTPException(422, "cannot run yet: " + "; ".join(probs))
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
    mp = os.path.join(d, "run.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
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
    mp = os.path.join(d, "run.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "this run stopped before it started (see its log)")
    meta = json.load(open(mp))
    if "parts" not in meta:
        raise HTTPException(422, "run predates scenario files")
    known = {"name", "material", "parts", "injectors", "domain", "regions", "flow_planes",
             "output", "gravity", "units", "notes"}
    try:
        sc = Scenario.from_dict({k: v for k, v in meta.items() if k in known},
                                base_dir=_run_base_dir(meta, d))
    except ValueError as e:
        # runs made before they kept their own STL copy, whose project was since deleted
        raise HTTPException(404, "this run's geometry is gone (its project was deleted): "
                                 + "; ".join(ln.strip() for ln in str(e).splitlines()[1:3]))
    return _geometry(sc)


def _run_base_dir(meta: dict, d: str) -> str:
    """Where a run's relative STL paths resolve.  Runs made before run.json recorded it
    ("base_dir") were imported scenarios named after their project folder."""
    if meta.get("base_dir"):
        return meta["base_dir"]
    for cand in (os.path.join(ROOT, meta.get("name", "")), d):
        if all(os.path.exists(os.path.join(cand, p["stl"])) or os.path.isabs(p["stl"])
               for p in meta.get("parts", [])):
            return cand
    return d


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
    if what not in ("loads", "flows", "regions", "dust"):
        raise HTTPException(422, "what must be loads, flows, regions or dust")
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
    ad = os.path.join(a, "air_dust.json")
    if os.path.exists(ad):
        out["air_dust"] = json.load(open(ad))
    return out


_AIR = {}


def _air(run_id: str):
    """The run's air_dust.npz, loaded once per file version."""
    f = os.path.join(_run_dir(run_id), "analysis", "air_dust.npz")
    if not os.path.exists(f):
        raise HTTPException(404, "run Dust & air first")
    key = (f, os.path.getmtime(f))
    if _AIR.get(run_id, (None,))[0] != key:
        z = np.load(f)
        _AIR[run_id] = (key, {k: z[k] for k in z.files})
    return _AIR[run_id][1]


@app.get("/api/runs/{run_id}/dust/{k}")
def dust_frame(run_id: str, k: int):
    """Binary: uint32 n, n*3 float32 positions, n uint8 size classes (airborne parcels at
    frame k; empty outside the analysed window)."""
    z = _air(run_id)
    hit = np.flatnonzero(z["frame_ids"] == k)
    if not len(hit):
        pos, cls = np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)
    else:
        a, b = z["dust_off"][hit[0]], z["dust_off"][hit[0] + 1]
        pos, cls = z["dust_pos"][a:b], z["dust_cls"][a:b]
    return Response(np.array([len(pos)], dtype=np.uint32).tobytes() + pos.astype(np.float32).tobytes()
                    + cls.astype(np.uint8).tobytes(), media_type="application/octet-stream")


@app.get("/api/runs/{run_id}/air-slice")
def air_slice(run_id: str, axis: int = 2, frac: float = 0.5):
    """One plane of the time-mean air field normal to `axis` at fraction `frac` of the box:
    speed, the two in-plane components and the solid mask, row-major with u fastest."""
    if axis not in (0, 1, 2):
        raise HTTPException(422, "axis must be 0, 1 or 2")
    z = _air(run_id)
    air, solid, o, h = z["air"].astype(np.float32), z["solid"], z["origin"], float(z["cell"])
    n = air.shape[:3]
    idx = int(min(max(frac, 0.0), 1.0) * (n[axis] - 1))
    au, av = [a for a in range(3) if a != axis]
    sl = [slice(None)] * 3
    sl[axis] = idx
    a2, s2 = air[tuple(sl)], solid[tuple(sl)]            # shape (n[au], n[av], 3)
    # rows over v, u fastest
    a2, s2 = np.swapaxes(a2, 0, 1), np.swapaxes(s2, 0, 1)
    corner = o.astype(np.float64).copy()
    corner[axis] += (idx + 0.5) * h
    du, dv = np.zeros(3), np.zeros(3)
    du[au], dv[av] = n[au] * h, n[av] * h
    sp = np.linalg.norm(a2, axis=-1)
    return dict(axis=axis, index=idx, count=int(n[axis]), position=float(corner[axis]),
                nu=int(n[au]), nv=int(n[av]), corner=corner.tolist(), du=du.tolist(), dv=dv.tolist(),
                speed=_b64(sp.ravel(), np.float32), vu=_b64(a2[..., au].ravel(), np.float32),
                vv=_b64(a2[..., av].ravel(), np.float32), solid=_b64(s2.ravel(), np.uint8),
                max_speed=float(sp.max()), p98=float(np.percentile(sp[s2 == 0], 98)) if (s2 == 0).any() else 1.0)


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.middleware("http")
async def _revalidate(request, call_next):
    """The page and its scripts are revalidated on every load (cheap: 304 if unchanged).
    Without this a browser mixed a new index.html with a cached app.js after an update
    and failed on an element the new page no longer has."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
