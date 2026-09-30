#!/usr/bin/env python3
"""
Import a BulkFlowAnalyst project into a DEM scenario file.

    python bfa_import.py 20060-CM-552 [--out 20060-CM-552/scenario.json] [--preset fast]

Reads the project's .prj (components, material, interactions, injection box, conveyors,
chutes, portals, simulation settings) and .lin (component STL names and on/off windows,
timestep, Young's modulus), and writes a scenario for dem_run.py.

Every value that needed interpretation is listed under "notes.interpretations" in the
scenario, and anything the solver cannot represent yet under "notes.warnings" -- read them.
Unit handling: the .prj stores lengths in inches, densities in lb/ft3, flow in short tons
per hour and velocities in in/s; the STLs of the projects seen so far are in metres.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import trimesh

from dem_scenario import (INF, FlowPlane, Injector, Material, Output, Part, Region, Scenario,
                          Solver, rayleigh_time)

IN = 0.0254                 # m per inch
LB_FT3 = 16.018463          # kg/m3 per lb/ft3
SHORT_TON_H = 907.18474 / 3600.0   # kg/s per short ton/h
G = 9.81


def bfa_float(tok: str) -> float:
    """BFA stores floats as an integer mantissa and exponent: '11811024E-005' = 118.11024."""
    tok = tok.strip().rstrip(",")
    m = re.fullmatch(r"(-?\d+)E([+-]\d+)", tok)
    if m:
        return int(m.group(1)) * 10.0 ** int(m.group(2))
    return float(tok)


def _vec(s: str):
    """'(10629921E-004, 20838937E-006, 26226030E-006)' -> [x, y, z] (raw units)."""
    return [bfa_float(t) for t in re.findall(r"-?\d+E[+-]\d+", s)[:3]]


def read_lines(path):
    with open(path, encoding="latin1", errors="replace") as fh:
        return [ln.rstrip("\r\n") for ln in fh]


def kv(line):
    k, _, v = line.partition(":")
    return k.strip(), v.strip()


def section(lines, name):
    """Lines of a //~~Save<name> section."""
    start = next((i for i, ln in enumerate(lines) if ln.strip() == f"//~~Save{name}"), None)
    if start is None:
        return []
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("//~~Save")), len(lines))
    return lines[start + 1:end]


def blocks(lines, first_key):
    """Split a section into records, each starting at a line whose key is first_key."""
    out, cur = [], None
    for ln in lines:
        k, v = kv(ln)
        if k == first_key:
            cur = {}
            out.append(cur)
        if cur is not None and ":" in ln and not ln.startswith("3, "):
            cur.setdefault(k, v)
    return out


def read_lin(path):
    """Sequence of (key, value) pairs from a .lin ('# Key:' line, value on the next)."""
    lines = read_lines(path)
    pairs, i = [], 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("#"):
            key = ln.lstrip("# ").rstrip(":")
            vals = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("#"):
                if lines[i].strip():
                    vals.append(lines[i].strip())
                i += 1
            pairs.append((key, vals))
        else:
            i += 1
    return pairs


def safe_name(name: str) -> str:
    """BFA's file-safe component name: 'Conveyor 3m/s' -> 'Conveyor3mXs'."""
    return re.sub(r"[^A-Za-z0-9]", "", name.replace("/", "X"))


def component_surfaces(lines):
    """{component name: array of triangle vertices (inches)} from the component section."""
    out, cur = {}, None
    for ln in section(lines, "ComponentData"):
        if ln.startswith("Component Name:"):
            cur = kv(ln)[1]
            out[cur] = []
        elif ln.startswith("3, ") and cur is not None:
            pts = re.findall(r"\(([^)]*)\)", ln)
            if len(pts) >= 3:
                out[cur].append([_vec(p) for p in pts[:3]])
    return {k: np.array(v) for k, v in out.items() if v}


def import_project(proj_dir: str, preset: str = "reference") -> Scenario:
    prj = sorted(glob.glob(os.path.join(proj_dir, "*.prj")))
    lin = sorted(glob.glob(os.path.join(proj_dir, "*.lin")))
    if len(prj) != 1 or len(lin) != 1:
        raise FileNotFoundError(f"expected one .prj and one .lin in {proj_dir}")
    L = read_lines(prj[0])
    interp, warn = [], []

    # ---------------------------------------------------------------- components
    comps = blocks(section(L, "ComponentData"), "Component Name")
    by_type = {}
    for c in comps:
        by_type.setdefault(c.get("Component Type"), []).append(c)
    guid_name = {c["Component GUID"]: c["Component Name"] for c in comps if "Component GUID" in c}
    surf = component_surfaces(L)

    # ---------------------------------------------------------------- .lin
    lin_pairs = read_lin(lin[0])
    dt_bfa = bfa_float(dict(lin_pairs)["Timestep"][0])
    youngs = bfa_float(dict(lin_pairs)["Young's modulus (Pa)"][0])
    windows, cur = {}, None
    for k, v in lin_pairs:
        if k == "SafeName":
            cur = v[0]
        elif k == "On time" and cur:
            windows.setdefault(cur, [0.0, INF])[0] = bfa_float(v[0])
        elif k == "Off time" and cur:
            windows.setdefault(cur, [0.0, INF])[1] = bfa_float(v[0])

    # ---------------------------------------------------------------- material
    mat_lines = section(L, "MaterialData")
    M = {}
    for ln in mat_lines:
        k, v = kv(ln)
        M.setdefault(k, v)
    rmin, rmax = bfa_float(M["Material Min_Rad"]) * IN, bfa_float(M["Material Max_Rad"]) * IN
    if abs(rmin - rmax) > 1e-9:
        warn.append(f"size distribution {rmin*1e3:.1f}-{rmax*1e3:.1f} mm radius: monodisperse "
                    f"{0.5*(rmin+rmax)*1e3:.1f} mm used (size distributions: issue #10)")
    radius = 0.5 * (rmin + rmax)
    density = bfa_float(M["Material Intr Density"]) * LB_FT3
    coh = bfa_float(M.get("Material Cohesion_Coefficient", "0E+000"))
    if coh:
        warn.append(f"cohesion coefficient {coh:g} ignored (no cohesion model yet)")
    mode = next((kv(ln)[1] for ln in L if ln.startswith("Contact Mode:")), "?")
    interp.append(f"Contact Mode '{mode}' -> Hertz-Mindlin with the .lin Young's modulus "
                  f"({youngs:.4g} Pa); Poisson's ratio is not in the project, 0.30 assumed")

    # ---------------------------------------------------------------- interactions
    inter = blocks(section(L, "InteractionsData"), "ComponentOne GUID")
    mat_guid = next((c["Component GUID"] for c in by_type.get("MaterialCondition", [])), None)
    pair = {}
    for b in inter:
        a, c = b.get("ComponentOne GUID"), b.get("ComponentTwo GUID")
        other = c if a == mat_guid else a if c == mat_guid else None
        if other is None:
            continue
        pair[other] = {k: bfa_float(b[k]) for k in ("Friction", "RotatingMu", "RotatingR", "ContactR")
                       if k in b}
    pp = pair.get(mat_guid, {})
    mu_pp = pp.get("Friction", bfa_float(M["Material Inter-particle Friction"]))
    roll_pp = pp.get("RotatingMu", bfa_float(M["Material Rotating Friction"]))
    rest = pp.get("ContactR", bfa_float(M["Material Coefficient_Restitution"]))
    rotr = pp.get("RotatingR", 0.2)

    # ---------------------------------------------------------------- conveyors
    belts = {}
    for b in blocks(section(L, "ConveyorData"), "Conveyor GUID"):
        speed = bfa_float(b["Conveyor Vel_ips"]) * IN
        edge = re.findall(r"\(([^)]*)\)", b.get("Conveyor Direction Edge", ""))
        if len(edge) >= 2:
            p0, p1 = np.array(_vec(edge[0])), np.array(_vec(edge[1]))
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-12)
        else:
            d = np.array([1.0, 0.0, 0.0])
            warn.append("conveyor direction edge missing: +x assumed")
        belts[b["Conveyor GUID"]] = dict(type="belt", velocity=[float(c) for c in speed * d],
                                         speed=speed, friction=bfa_float(b["Conveyor Friction"]))
        interp.append(f"conveyor '{guid_name.get(b['Conveyor GUID'])}': {speed:.3f} m/s along "
                      f"{np.round(d, 4).tolist()} (from its direction edge); the surface moves "
                      f"along the belt's path, so the pulley wrap and return strand follow it")

    # ---------------------------------------------------------------- chutes (motion)
    for b in blocks(section(L, "ChuteData"), "Chute GUID"):
        moving = (bfa_float(b.get("Chute Moving Boundary", "0E+000")) != 0
                  or bfa_float(b.get("Chute Translation Velocity", "0E+000")) != 0
                  or bfa_float(b.get("Chute dRotationVelocity", "0E+000")) != 0)
        if moving:
            warn.append(f"chute '{guid_name.get(b['Chute GUID'])}' has a moving boundary: "
                        "not simulated (rigid part motion is a follow-up to #11)")

    # ---------------------------------------------------------------- parts
    parts, rolls, rots, rests = [], [], [], []
    for c in comps:
        ctype = c.get("Component Type")
        if ctype not in ("Boundary", "BoundaryWT"):
            continue
        name, guid = c["Component Name"], c["Component GUID"]
        sn = safe_name(name)
        stl = os.path.join(proj_dir, f"{sn}.stl")
        if not os.path.exists(stl):
            warn.append(f"component '{name}': {sn}.stl not found -- skipped")
            continue
        ip = pair.get(guid, {})
        belt = belts.get(guid)
        fric = ip.get("Friction", belt["friction"] if belt else 0.5)
        rolls.append(ip.get("RotatingMu", 0.5))
        rots.append(ip.get("RotatingR", 0.2))
        rests.append(ip.get("ContactR", rest))
        motion = None
        if belt:
            motion = {k: belt[k] for k in ("type", "velocity")}
        parts.append(Part(name=name, stl=os.path.relpath(stl, proj_dir), two_sided=True,
                          thickness=0.0, friction=fric,
                          active=list(windows.get(sn, [0.0, INF])), motion=motion,
                          corners=belt is not None))
    interp.append("every boundary is a zero-thickness two-sided shell (BFA boundaries are "
                  "surfaces; grains cannot cross them from either side)")
    interp.append("concave-crease contacts (corners, #13) enabled on conveyor parts: a troughed "
                  "belt's bottom/wing creases otherwise make settled grains chatter")
    if len(set(rolls)) > 1 or len(set(rots)) > 1 or len(set(rests)) > 1:
        warn.append(f"per-component rolling friction {sorted(set(rolls))} / rotational damping "
                    f"{sorted(set(rots))} / restitution {sorted(set(rests))} differ; the solver "
                    "takes one wall value each -- the most common is used")

    def common(xs, default):
        return max(set(xs), key=xs.count) if xs else default

    # ---------------------------------------------------------------- injection
    ib = blocks(section(L, "InjectionBoxData"), "Injection Box GUID")
    if len(ib) != 1:
        raise NotImplementedError(f"{len(ib)} injection boxes; exactly one supported")
    ib = ib[0]
    inj_comp = by_type["InjectionBox"][0]["Component Name"]
    face = os.path.join(proj_dir, f"{safe_name(inj_comp)}InjFace.stl")
    rate = bfa_float(ib["Injection Box FlowRate"]) * SHORT_TON_H
    ext_len = bfa_float(ib["Injection Extrusion Length"]) * IN
    ref = ib.get("Injection Box Discharge Reference", "")
    if ref in belts:
        vel = belts[ref]["velocity"]
        interp.append(f"injection discharges onto conveyor '{guid_name.get(ref)}': grains "
                      f"extruded at belt velocity (the extrusion length {ext_len:.3f} m equals "
                      f"the belt speed x 1 s)")
    else:
        vel = [0.0, -math.sqrt(2.0 * G * ext_len), 0.0]
        interp.append(f"injection extruded downward at sqrt(2 g L) = {-vel[1]:.4f} m/s "
                      f"(L = extrusion length {ext_len:.4f} m; matched BFA exactly on the corn run)")
    injector = Injector(name=inj_comp, face_stl=os.path.relpath(face, proj_dir), mass_rate=rate,
                        velocity=vel, start=bfa_float(ib.get("Injection Box Injection Start", "0E+000")),
                        stop=bfa_float(ib.get("Injection Box Injection Stop", "0E+000")) or INF,
                        offset=[0.0, -1.5 * radius, 0.0])

    # ---------------------------------------------------------------- domain, regions
    los, his = [], []
    for p in parts:
        m = trimesh.load(os.path.join(proj_dir, p.stl), force="mesh")
        los.append(m.bounds[0])
        his.append(m.bounds[1])
    portals = {}
    for c in by_type.get("Portal", []):
        tri = surf.get(c["Component Name"])
        if tri is not None:
            pts = tri.reshape(-1, 3) * IN
            portals[c["Component Name"]] = dict(lo=pts.min(0).round(4).tolist(),
                                                hi=pts.max(0).round(4).tolist())
            los.append(pts.min(0))
            his.append(pts.max(0))
    lo, hi = np.min(los, axis=0) - 0.5, np.max(his, axis=0) + 0.5
    regions = []
    for p in parts:
        if p.motion:
            m = trimesh.load(os.path.join(proj_dir, p.stl), force="mesh")
            regions.append(Region("belt", lo=(m.bounds[0] - [0, 0, 0]).tolist(),
                                  hi=(m.bounds[1] + [0, 0.5, 0]).tolist()))
    head = next((p for p in parts if "chute" in p.name.lower()), None)
    if head:
        m = trimesh.load(os.path.join(proj_dir, head.stl), force="mesh")
        regions.append(Region("chute", lo=m.bounds[0].tolist(), hi=m.bounds[1].tolist(), spin=True))
        regions.append(Region("below", hi=[INF, float(m.bounds[0][1]), INF]))

    # ---------------------------------------------------------------- solver sizing
    sim = {kv(ln)[0]: kv(ln)[1] for ln in section(L, "SimulationData") if ":" in ln}
    duration = bfa_float(sim.get("Simulation Time", "10000000E-006"))
    fps = float(sim.get("Simulation FramesPerSecond", "15"))
    gmass = 4.0 / 3.0 * math.pi * radius ** 3 * density
    belt_parts = [p for p in parts if p.motion]
    if belt_parts:
        m = trimesh.load(os.path.join(proj_dir, belt_parts[0].stl), force="mesh")
        length = float(np.max(m.bounds[1] - m.bounds[0]))
        spd = max(np.linalg.norm(belt_parts[0].motion["velocity"]), 0.1)
        holdup = rate * (length / spd + 3.0)
    else:
        holdup = rate * 5.0
    wall_cell = max(0.006, 2.0 * radius)
    solver = Solver(expected_holdup_kg=round(holdup, 1), wall_grid_cell=wall_cell,
                    max_velocity=30.0)
    if preset == "fast":
        solver.dt, solver.youngs_divisor, solver.neighbor_every = "auto", 10.0, 4
        solver.skin_speed = 6.0
        interp.append("preset fast: Young's modulus / 10, dt = 0.35 Rayleigh time, neighbour "
                      "lists.  On the iron-ore conveyor project: 3x faster than reference; "
                      "portal split within 0.005 of BFA up to moderate deflection, over-steering "
                      "by up to ~0.04 at the steepest deflectors (reference: within 0.016)")
    else:
        solver.dt = dt_bfa
        solver.youngs_divisor = 1.0
        solver.neighbor_every = 0
        interp.append("preset reference: BFA's own timestep and the true Young's modulus")
    # hash table z must exceed the domain's z-cell span so idle grains can be hidden
    t_r = rayleigh_time(radius, density, youngs / solver.youngs_divisor, 0.30)
    dt_est = dt_bfa if solver.dt != "auto" else solver.rayleigh_fraction * t_r
    skin = 2.0 * solver.skin_speed * max(solver.neighbor_every, 1) * dt_est
    gcell = 2.0 * radius + (max(2.0 * skin, 0.006) if solver.neighbor_every else 0.0)
    zspan = int((hi[2] - lo[2]) / gcell) + 6
    if zspan + 2 >= 128:
        solver.hash_dims = [128, 128, int(2 ** math.ceil(math.log2(zspan + 8)))]
    interp.append(f"BFA timestep {dt_bfa*1e6:.2f} us = {dt_bfa / rayleigh_time(radius, density, youngs, 0.30):.2f} "
                  f"Rayleigh time at the true modulus")

    mat = Material(name=next((c["Component Name"] for c in by_type.get("MaterialCondition", [])), "material"),
                   radius=radius, density=density, youngs=youngs, poisson=0.30, restitution=rest,
                   friction=mu_pp, rolling_friction=roll_pp,
                   wall_rolling_friction=common(rolls, 0.5), rot_damp=rotr,
                   rot_damp_wall=common(rots, 0.2), tangential_ratio=1.0, contact="hertz")
    flow_planes = []
    for n, pr in portals.items():
        lo_p, hi_p = np.array(pr["lo"]), np.array(pr["hi"])
        ax = int(np.argmin(hi_p - lo_p))
        flow_planes.append(FlowPlane(name=safe_name(n), axis=ax, value=float(lo_p[ax]),
                                     lo=lo_p.tolist(), hi=hi_p.tolist()))
        interp.append(f"portal '{n}' {pr['lo']} .. {pr['hi']} -> flow plane "
                      f"flow_{safe_name(n)}_kg (grains pass through it, as in BFA)")

    sc = Scenario(
        name=os.path.basename(os.path.normpath(proj_dir)), material=mat, parts=parts,
        injectors=[injector], domain_lo=lo.round(3).tolist(), domain_hi=hi.round(3).tolist(),
        regions=regions, flow_planes=flow_planes, solver=solver,
        output=Output(duration=duration, fps=fps),
        base_dir=proj_dir,
        notes={"source": os.path.basename(prj[0]), "importer": "bfa_import.py",
               "preset": preset, "bfa_timestep": dt_bfa, "portals": portals,
               "interpretations": interp, "warnings": warn})
    sc.validate()
    return sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="BFA project directory (contains the .prj, .lin and STLs)")
    ap.add_argument("--out", default=None, help="scenario JSON (default <project>/scenario.json)")
    ap.add_argument("--preset", choices=["reference", "fast"], default="reference",
                    help="fast (default) = E/10, dt 0.35 Rayleigh time, neighbour lists, ~3x "
                         "faster; reference = BFA's own timestep and modulus")
    args = ap.parse_args()
    sc = import_project(args.project, args.preset)
    out = args.out or os.path.join(args.project, "scenario.json")
    sc.save(out)
    print(f"wrote {out}")
    m = sc.material
    print(f"  material   {m.name}: r {m.radius*1e3:.1f} mm, rho {m.density:.0f}, E {m.youngs:.3g}, "
          f"mu {m.friction:g} / roll {m.rolling_friction:g}, e {m.restitution:g}")
    for p in sc.parts:
        win = "" if p.active == [0.0, INF] else f"  active {p.active[0]:g}-{p.active[1]:g} s"
        print(f"  part       {p.name:22s} mu {p.friction:g}{win}"
              + (f"  belt {np.round(p.motion['velocity'], 3).tolist()} m/s" if p.motion else ""))
    i = sc.injectors[0]
    print(f"  injector   {i.name}: {i.mass_rate:.1f} kg/s at {np.round(i.velocity, 3).tolist()} m/s")
    print(f"  solver     dt {sc.solver.dt}, holdup ~{sc.solver.expected_holdup_kg:g} kg, "
          f"wall cell {sc.solver.wall_grid_cell*1e3:.0f} mm, hash {sc.solver.hash_dims}")
    for s in sc.notes["interpretations"]:
        print(f"  note       {s}")
    for s in sc.notes["warnings"]:
        print(f"  WARNING    {s}")


if __name__ == "__main__":
    sys.exit(main())
