#!/usr/bin/env python3
"""
Measurements from a finished DEM run -- chosen after the run, computed from its frames.

    python dem_analyze.py RUN loads   [--window T0 T1] [--scenario S]
    python dem_analyze.py RUN flows   [--window T0 T1] [--plane NAME AXIS VALUE LO0 LO1 LO2 HI0 HI1 HI2]
    python dem_analyze.py RUN regions [--window T0 T1]

Nothing here runs during the simulation, so it costs the solver nothing, and any
measurement can be added or changed afterwards and re-run in seconds.

loads    Wall loads recomputed from each frame: every grain's contact with every part
         (the solver's own wall grid, closest-point and sign rules, surface motion) and
         the run's contact law.  Writes
           analysis/part_loads.csv       force and torque on each part, per frame
           analysis/wall_maps.vtk        geometry with window-mean pressure, shear, wear
                                         rate and contact count per triangle (ParaView)
         Normal loads are exact at each frame.  The tangential force of a STUCK contact
         lives in the solver's history spring, which frames do not store, so tangential
         loads use the sliding law (exact for sliding grains).  Frames are snapshots
         (default 15 per second): time means of dense, sustained loads converge; peak
         impact loads are sampled, not integrated.
flows    Mass through planes (the scenario's flow planes, or --plane), per frame.
regions  Mass and mean speed in the scenario's regions, per frame.

Frames written since spin was added to the output carry it; older frames analyse with
zero spin (affects sliding speed, i.e. wear, only).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
import warp as wp

import granular_dem as G
from compare_bfa_mpm import read_frame
from dem_scenario import INF, FlowPlane, Scenario

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------

class Run:
    """A finished dem_run output directory: its scenario, derived values and frames."""

    def __init__(self, run_dir, scenario_path=None):
        self.dir = run_dir
        meta = json.load(open(os.path.join(run_dir, "run.json")))
        self.derived = meta.get("derived", {})
        if scenario_path:
            self.sc = Scenario.load(scenario_path)
        else:
            known = {"name", "material", "parts", "injectors", "domain", "regions",
                     "flow_planes", "solver", "output", "gravity", "units", "notes"}
            d = {k: v for k, v in meta.items() if k in known}
            if not isinstance(d.get("solver"), dict):
                # run.json written before the fix that stopped a flat "solver" name from
                # overwriting the scenario's solver block: fall back to defaults (the
                # effective stiffness is taken from "derived", which these files keep)
                d.pop("solver", None)
            if "parts" not in d:
                raise ValueError(f"{run_dir}/run.json predates scenario files; pass --scenario")
            base = meta.get("base_dir", run_dir)
            self.sc = Scenario.from_dict(d, base_dir=base)
        fps = self.sc.output.fps
        self.frames = []
        for f in sorted(glob.glob(os.path.join(run_dir, "frame_*_particles.vtk"))):
            k = int(re.search(r"frame_(\d+)_particles", f).group(1))
            self.frames.append((k / fps, f))

    def window(self, t0=None, t1=None):
        t0 = -INF if t0 is None else t0
        t1 = INF if t1 is None else t1
        return [(t, f) for t, f in self.frames if t0 <= t <= t1]


# ---------------------------------------------------------------------------
# wall loads
# ---------------------------------------------------------------------------

@wp.func
def _avg_n(g: G.WallGrid, k0: int, k1: int, cp: wp.vec3):
    acc = wp.vec3(0.0)
    for k in range(k0, k1):
        r = g.rec[k]
        sq, _q = G._tri_closest(r.v0, r.u, r.w, r.n, r.t2, cp)
        if sq < 1.0e-6:
            acc += r.n
    return wp.normalize(acc)


@wp.kernel
def _frame_wall_loads(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    w_ang: wp.array(dtype=wp.vec3),
    radius: float,
    mass: float,
    collider: G.ShellCollider,
    g: G.WallGrid,
    hertz: int,
    e_star: float,
    beta: float,
    force: wp.array(dtype=wp.vec3),
    moment: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=float),
    shear: wp.array(dtype=float),
    wear: wp.array(dtype=float),
    count: wp.array(dtype=float),
):
    """Each grain's primary contact per part (as the solver's main wall kernel finds it),
    its force by the run's contact law, accumulated on the triangle it touches."""
    i = wp.tid()
    x = q[i]
    cc = G._cell_of(g, x)
    if cc[0] < 0 or cc[1] < 0 or cc[2] < 0 or cc[0] >= g.nx or cc[1] >= g.ny or cc[2] >= g.nz:
        return
    cell = (cc[2] * g.ny + cc[1]) * g.nx + cc[0]
    k0 = g.cell_start[cell]
    k1 = g.cell_start[cell + 1]
    v = qd[i]
    w = w_ang[i]
    cur = int(-1)
    seg0 = k0
    best_sq = float(0.0)
    best_k = int(-1)
    for k in range(k0, k1 + 1):
        p = int(-1)
        if k < k1:
            p = g.rec[k].part
        if p != cur:
            if best_k >= 0:
                m = cur
                r = g.rec[best_k]
                _sq, cp = G._tri_closest(r.v0, r.u, r.w, r.n, r.t2, x)
                thick = collider.thickness[m]
                offset = x - cp
                d = wp.sqrt(best_sq)
                face_n = r.n
                if collider.two_sided[m] == 0:
                    face_n = _avg_n(g, seg0, k, cp)
                sdf, n = G._shell_sdf(collider.two_sided[m], thick, offset, d, face_n)
                c = sdf - radius
                if c < 0.0 and c >= -radius:
                    v_rel = v + wp.cross(w, -n * radius)
                    if collider.motion_type[m] != 0:
                        vs, _ws = G._surface_velocity(collider, m, n, cp)
                        v_rel = v_rel - vs
                    vn = wp.dot(v_rel, n)
                    vt = v_rel - n * vn
                    fn = float(0.0)
                    if hertz == 1:
                        delta = -c
                        sq = wp.sqrt(radius * delta)
                        kd_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(2.0 * e_star * sq * mass)
                        fn = (4.0 / 3.0) * e_star * wp.sqrt(radius) * delta * wp.sqrt(delta) - kd_eff * vn
                    else:
                        fn = -c * collider.ke[m] - vn * collider.kd[m]
                    fn = wp.max(fn, 0.0)
                    vtl = wp.length(vt)
                    ft = wp.vec3(0.0)
                    if vtl > 1.0e-3:
                        ft = -(vt / vtl) * (collider.friction[m] * fn)   # sliding law
                    f_wall = -(n * fn + ft)
                    tri = g.entries[best_k]
                    wp.atomic_add(force, tri, f_wall)
                    wp.atomic_add(moment, tri, wp.cross(cp, f_wall))
                    wp.atomic_add(normal, tri, fn)
                    wp.atomic_add(shear, tri, wp.length(ft))
                    wp.atomic_add(wear, tri, fn * vtl)
                    wp.atomic_add(count, tri, 1.0)
            cur = p
            seg0 = k
            best_k = -1
            rr = radius + collider.thickness[wp.max(p, 0)]
            best_sq = rr * rr
            if p >= 0:
                if collider.active[p] == 0:
                    best_sq = 0.0
        if k < k1:
            r2 = g.rec[k]
            h = wp.dot(x - r2.v0, r2.n)
            if h * h < best_sq:
                sq2, _c2 = G._tri_closest(r2.v0, r2.u, r2.w, r2.n, r2.t2, x)
                if sq2 < best_sq:
                    best_sq = sq2
                    best_k = k


def _collider_for(run, device):
    """Parts, collider and baked wall grid exactly as dem_run built them."""
    import dem_run
    sc = run.sc
    m = sc.material
    unit = sc.unit_scale
    parts = [(p.name, *dem_run.load_stl(sc.path(p.stl), p.fix_normals, p.flip, unit)) for p in sc.parts]
    if sc.solver.simplify_mm > 0:
        from mesh_simplify import collapse_short_edges
        parts = [(n, *collapse_short_edges(v, f, sc.solver.simplify_mm * 1e-3,
                                           tol=None if sc.solver.simplify_tol_mm is None
                                           else sc.solver.simplify_tol_mm * 1e-3))
                 for n, v, f in parts]
    kd_wall = run.derived.get("kd_wall")
    if kd_wall is None:
        e = max(min(m.restitution, 0.999), 1e-4)
        zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
        kd_wall = zeta * 2.0 * math.sqrt(m.mass * m.ke)
    collider, meshes = G.build_collider(
        parts, two_sided=[p.two_sided for p in sc.parts], friction=[p.friction for p in sc.parts],
        ke=[m.ke] * len(parts), kd=[kd_wall] * len(parts), kf=[m.kf] * len(parts),
        thickness=[p.thickness if p.two_sided else 0.0 for p in sc.parts], max_dist=0.03,
        device=device, corners=[p.corners for p in sc.parts],
        motions=[dem_run.surface_motion(p.motion, sc.gravity) for p in sc.parts])
    max_thick = max([p.thickness if p.two_sided else 0.0 for p in sc.parts] + [0.0])
    grid, _info = G.build_wall_grid(parts, meshes, collider.lower, collider.upper,
                                    m.radius + max_thick + 1.0e-3 + 1.0e-4,
                                    sc.solver.wall_grid_cell, device)
    return parts, collider, meshes, grid


def wall_loads(run, t0=None, t1=None, device="cuda:0", out_dir=None, quiet=False):
    """Per-frame part loads and window-mean per-triangle maps (see module docstring)."""
    wp.init()
    sc = run.sc
    m = sc.material
    frames = run.window(t0, t1)
    if not frames:
        raise ValueError("no frames in the window")
    parts, collider, meshes, grid = _collider_for(run, device)
    ntri = sum(len(f) for _n, _v, f in parts)
    offsets = np.cumsum([0] + [len(f) for _n, _v, f in parts])
    youngs_eff = run.derived.get("youngs_effective", m.youngs / sc.solver.youngs_divisor)
    e_star = youngs_eff / (1.0 - m.poisson ** 2)
    e = max(min(m.restitution, 0.999), 1e-4)
    beta = abs(math.log(e)) / math.sqrt(math.log(e) ** 2 + math.pi ** 2)
    hertz = 1 if m.contact == "hertz" else 0
    # part reference points: centroid of each part's vertices
    cents = [np.asarray(v).mean(axis=0) for _n, v, _f in parts]

    acc = {k: wp.zeros(ntri, dtype=wp.vec3 if k in ("force", "moment") else float, device=device)
           for k in ("force", "moment", "normal", "shear", "wear", "count")}
    rows = []
    prev = {k: np.zeros((ntri, 3) if k in ("force", "moment") else ntri) for k in acc}
    active_prev = None
    for t, f in frames:
        fr = read_frame(f)
        n = len(fr["pos"])
        if n == 0:
            continue
        qd = fr["vel"]
        spin = fr.get("spin", np.zeros_like(qd))
        # part activity at this time (staged deflectors)
        act = [1 if p.active[0] <= t < p.active[1] else 0 for p in sc.parts]
        if act != active_prev:
            collider.active.assign(np.array(act, dtype=np.int32))
            active_prev = act
        wp.launch(_frame_wall_loads, dim=n, device=device, inputs=[
            wp.array(fr["pos"].astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(qd.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(spin.astype(np.float32), dtype=wp.vec3, device=device),
            float(m.radius), float(m.mass), collider, grid, hertz, float(e_star), float(beta),
            acc["force"], acc["moment"], acc["normal"], acc["shear"], acc["wear"], acc["count"]])
        cur = {k: v.numpy().copy() for k, v in acc.items()}
        d = {k: cur[k] - prev[k] for k in acc}
        prev = cur
        for pi, p in enumerate(sc.parts):
            sl = slice(offsets[pi], offsets[pi + 1])
            F = d["force"][sl].sum(axis=0)
            M_o = d["moment"][sl].sum(axis=0)
            M_c = M_o - np.cross(cents[pi], F)          # torque about the part centroid
            rows.append([t, p.name, *F, np.linalg.norm(F), *M_c, d["normal"][sl].sum(),
                         int(d["count"][sl].sum())])
    nfr = len(frames)
    out_dir = out_dir or os.path.join(run.dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "part_loads.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_s", "part", "Fx_N", "Fy_N", "Fz_N", "F_N", "Mx_Nm", "My_Nm", "Mz_Nm",
                     "normal_sum_N", "contacts"])
        for r in rows:
            wr.writerow([f"{r[0]:.4f}", r[1]] + [f"{x:.6g}" for x in r[2:10]] + [r[10]])
    # window-mean maps per unit area
    areas = np.concatenate([0.5 * np.linalg.norm(np.cross(
        np.asarray(v)[np.asarray(fc).reshape(-1, 3)[:, 1]] - np.asarray(v)[np.asarray(fc).reshape(-1, 3)[:, 0]],
        np.asarray(v)[np.asarray(fc).reshape(-1, 3)[:, 2]] - np.asarray(v)[np.asarray(fc).reshape(-1, 3)[:, 0]]),
        axis=1) for _n, v, fc in parts])
    areas = np.maximum(areas, 1e-12)
    maps = dict(pressure_Pa=prev["normal"] / nfr / areas, shear_Pa=prev["shear"] / nfr / areas,
                wear_rate_W_m2=prev["wear"] / nfr / areas, contacts_per_frame=prev["count"] / nfr,
                part_id=np.concatenate([np.full(len(fc), k) for k, (_n, _v, fc) in enumerate(parts)]))
    _write_maps_vtk(os.path.join(out_dir, "wall_maps.vtk"), parts, maps,
                    title=f"wall loads, mean over {nfr} frames t={frames[0][0]:.2f}-{frames[-1][0]:.2f} s")
    # the same maps for the operator screen (triangle order = the solver's load order)
    np.savez(os.path.join(out_dir, "wall_maps.npz"), window=np.array([frames[0][0], frames[-1][0]]),
             **{k: np.asarray(v, dtype=np.float32) for k, v in maps.items()})
    if not quiet:
        print(f"wall loads: {nfr} frames, t = {frames[0][0]:.2f}-{frames[-1][0]:.2f} s -> {out_dir}")
        print(f"  {'part':<22} {'mean |F| (N)':>12} {'mean Fx':>10} {'mean Fy':>10} {'mean Fz':>10}")
        for p in sc.parts:
            pr = np.array([r[2:6] for r in rows if r[1] == p.name])
            if len(pr):
                mF = pr.mean(axis=0)
                print(f"  {p.name:<22} {np.linalg.norm(mF[:3]):12.2f} {mF[0]:10.2f} {mF[1]:10.2f} {mF[2]:10.2f}")
    return rows, maps


def _write_maps_vtk(path, parts, maps, title):
    verts, faces, base = [], [], 0
    for _n, v, f in parts:
        verts.append(np.asarray(v, dtype=np.float64))
        faces.append(np.asarray(f).reshape(-1, 3) + base)
        base += len(v)
    V, F = np.vstack(verts), np.vstack(faces)
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 3.0\n{title}\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(V)} float\n")
        np.savetxt(fh, V, fmt="%.6f")
        fh.write(f"POLYGONS {len(F)} {4 * len(F)}\n")
        np.savetxt(fh, np.hstack([np.full((len(F), 1), 3), F]), fmt="%d")
        fh.write(f"CELL_DATA {len(F)}\n")
        for name, arr in maps.items():
            fh.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, np.asarray(arr, dtype=np.float64), fmt="%.6g")


# ---------------------------------------------------------------------------
# flows and regions (host side)
# ---------------------------------------------------------------------------

def flows(run, planes=None, t0=None, t1=None, out_dir=None, quiet=False):
    """Mass crossing each plane between consecutive frames (either direction), matching
    grains by the ids stored in the frames.  Frames must be close enough that no grain
    crosses a plane and leaves the domain in between (true at 15 fps for these flows)."""
    planes = planes if planes is not None else run.sc.flow_planes
    gm = run.sc.material.mass
    frames = run.window(t0, t1)
    rows, prev = [], None
    dtf = 1.0 / run.sc.output.fps
    jump = run.sc.solver.max_velocity * dtf        # farther than this = recycled, not moved
    total = np.zeros(len(planes))
    for t, f in frames:
        fr = read_frame(f)
        if "id" not in fr:
            raise ValueError(f"{f} has no grain ids (written before ids were added); "
                             "use the run's own flow_* columns in history.csv instead")
        pos, ids = fr["pos"], fr["id"].astype(np.int64)
        if prev is not None and len(pos):
            ppos, pids = prev
            lookup = np.full(max(ids.max(), pids.max()) + 1, -1, dtype=np.int64)
            lookup[pids] = np.arange(len(pids))
            j = lookup[ids]
            ok = j >= 0
            p0, p1 = ppos[j[ok]], pos[ok]
            moved = np.linalg.norm(p1 - p0, axis=1) <= jump
            p0, p1 = p0[moved], p1[moved]
            for k, fp in enumerate(planes):
                a = fp.axis
                s0, s1 = p0[:, a] - fp.value, p1[:, a] - fp.value
                cr = s0 * s1 < 0.0
                if cr.any():
                    frac = (s0[cr] / (s0[cr] - s1[cr]))[:, None]
                    pt = p0[cr] + frac * (p1[cr] - p0[cr])
                    inside = np.ones(len(pt), dtype=bool)
                    for b in range(3):
                        if b != a:
                            inside &= (pt[:, b] >= fp.lo[b]) & (pt[:, b] <= fp.hi[b])
                    total[k] += inside.sum() * gm
        rows.append([t] + total.tolist())
        prev = (pos, ids)
    out_dir = out_dir or os.path.join(run.dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "flows.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_s"] + [f"{fp.name}_kg" for fp in planes])
        for r in rows:
            wr.writerow([f"{r[0]:.4f}"] + [f"{x:.6f}" for x in r[1:]])
    if not quiet and len(rows) > 1:
        span = rows[-1][0] - rows[0][0]
        print(f"flows over t = {rows[0][0]:.2f}-{rows[-1][0]:.2f} s (cumulative kg, mean kg/s):")
        for k, fp in enumerate(planes):
            print(f"  {fp.name:<24} {rows[-1][k+1]:12.2f} kg  {rows[-1][k+1] / max(span, 1e-9):10.2f} kg/s")
    return rows


def regions(run, t0=None, t1=None, out_dir=None, quiet=False):
    """Mass and mean speed in each scenario region, per frame."""
    regs = run.sc.regions
    gm = run.sc.material.mass
    rows = []
    for t, f in run.window(t0, t1):
        fr = read_frame(f)
        pos, sp = fr["pos"], np.linalg.norm(fr["vel"], axis=1)
        row = [t]
        for r in regs:
            m = np.all((pos > np.array(r.lo)) & (pos < np.array(r.hi)), axis=1)
            row += [m.sum() * gm, sp[m].mean() if m.any() else 0.0]
        rows.append(row)
    out_dir = out_dir or os.path.join(run.dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "regions.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_s"] + [x for r in regs for x in (f"{r.name}_kg", f"{r.name}_speed_ms")])
        for r in rows:
            wr.writerow([f"{r[0]:.4f}"] + [f"{x:.6f}" for x in r[1:]])
    if not quiet and rows:
        a = np.array([r[1:] for r in rows])
        print(f"regions, mean over t = {rows[0][0]:.2f}-{rows[-1][0]:.2f} s:")
        for k, r in enumerate(regs):
            print(f"  {r.name:<16} {a[:, 2*k].mean():10.2f} kg  {a[:, 2*k+1].mean():6.2f} m/s")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("what", choices=["loads", "flows", "regions"])
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--scenario", default=None, help="scenario JSON if run.json lacks one")
    ap.add_argument("--plane", nargs=9, action="append", default=None,
                    metavar=("NAME", "AXIS", "VALUE", "LO0", "LO1", "LO2", "HI0", "HI1", "HI2"))
    args = ap.parse_args()
    run = Run(args.run, args.scenario)
    t0, t1 = (args.window or (None, None))
    if args.what == "loads":
        wall_loads(run, t0, t1)
    elif args.what == "flows":
        planes = None
        if args.plane:
            planes = [FlowPlane(p[0], int(p[1]), float(p[2]), [float(x) for x in p[3:6]],
                                [float(x) for x in p[6:9]]) for p in args.plane]
        flows(run, planes, t0, t1)
    else:
        regions(run, t0, t1)


if __name__ == "__main__":
    sys.exit(main())
