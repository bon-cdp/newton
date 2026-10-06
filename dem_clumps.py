#!/usr/bin/env python3
"""
Fill-once DEM runs with multi-sphere clumps (granular_clumps.py, issue #10).

A scenario as for dem_run.py, plus in "notes":

    "clump": {"offsets_mm": [[x, y, z], ...], "radii_mm": [r, ...]}
    "fill":  {"shape": "cylinder", "center": [x, z], "radius": m, "y0": m, "mass": kg,
              "clearance": m}

At t = 0 the fill mass is placed as clumps on a lattice inside the cylinder (from y0
upwards, random orientations) and left to fall and settle; the scenario's injectors are
ignored.  Bodies whose centre falls below domain_lo[1] are retired (no recycling).
Output matches dem_run.py: history.csv (region masses by clump centre of mass, so
calibrate.py reads it unchanged), frame_*_particles.vtk (sub-spheres), checkpoint.npz.

    .venv/bin/python dem_clumps.py <scenario.json> --out <dir> [--no-vtk] [--checkpoint-at T]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

import newton
from bfa_replication_mpm import write_particles_vtk
from dem_run import (clear_of_walls, load_stl, resolve_dt, surface_motion, wall_grid_cell)
from dem_scenario import INF, Scenario, rayleigh_time
from granular_clumps import ClumpTemplate, SolverGranularClumps, random_quats
from granular_ellipsoids import EllipsoidShape, SolverGranularEllipsoids
from granular_dem import build_collider, build_wall_grid


def fill_sites(tpl, fill, parts, device, rng):
    """Clump centres on a cubic lattice inside the fill cylinder, lowest first."""
    s = 2.0 * tpl.bound_radius * 1.04
    cx, cz = fill["center"]
    R = fill["radius"] - tpl.bound_radius - fill.get("clearance", 0.001)
    n_want = int(round(fill["mass"] / tpl.mass))
    g = np.arange(-R, R + 1e-12, s)
    X, Z = np.meshgrid(g, g, indexing="ij")
    ring = (X ** 2 + Z ** 2) <= R * R
    layer = np.stack([X[ring] + cx, Z[ring] + cz], 1)
    n_layers = int(np.ceil(n_want / len(layer))) + 2
    pts = []
    for k in range(n_layers):
        y = fill["y0"] + tpl.bound_radius + fill.get("clearance", 0.001) + k * s
        off = (0.5 * s if k % 2 else 0.0)                  # stagger alternate layers
        pts.append(np.stack([layer[:, 0] + off, np.full(len(layer), y), layer[:, 1] + off], 1))
    pts = np.concatenate(pts)
    pts = pts[(pts[:, 0] - cx) ** 2 + (pts[:, 2] - cz) ** 2 <= R * R]
    pts = pts[clear_of_walls(pts, parts, tpl.bound_radius * 1.02, device, mask=True)]
    if len(pts) < n_want:
        raise ValueError(f"fill: room for {len(pts)} clumps, {n_want} wanted")
    return pts[:n_want], n_want


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-vtk", action="store_true")
    ap.add_argument("--checkpoint-at", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()
    sc = Scenario.load(a.scenario)
    if a.duration is not None:
        sc.output.duration = a.duration
    out = sc.output
    m, s = sc.material, sc.solver
    device = s.device
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    wp.set_module_options({"enable_backward": False})

    fill = sc.notes["fill"]
    ell = sc.notes.get("ellipsoid")
    if ell:
        # one rigid ellipsoid per particle (granular_ellipsoids.py)
        tpl = EllipsoidShape(*(np.asarray(ell["axes_mm"]) * 1e-3), m.density)
        m.radius = float(tpl.axes.min())               # dt from the smallest semi-axis
    else:
        cl = sc.notes["clump"]
        tpl = ClumpTemplate(np.asarray(cl["offsets_mm"]) * 1e-3, np.asarray(cl["radii_mm"]) * 1e-3,
                            m.density)
        m.radius = float(tpl.radii.min())              # dt follows the smallest sphere
    dt, substeps, youngs_eff = resolve_dt(sc)
    hertz = m.contact == "hertz"
    e = max(min(m.restitution, 0.999), 1e-4)
    zeta = -np.log(e) / np.sqrt(np.pi ** 2 + np.log(e) ** 2)
    kd_pp = zeta * 2.0 * np.sqrt(0.5 * tpl.mass * m.ke)
    kd_wall = zeta * 2.0 * np.sqrt(tpl.mass * m.ke)
    print(f"clump template        {tpl.describe()}")
    print(f"dt                    {dt*1e6:.3f} us ({substeps} steps per frame), "
          f"E {youngs_eff:.3g} Pa, {int(round(out.duration / dt)):,} steps")

    unit = sc.unit_scale
    parts = [(p.name, *load_stl(sc.path(p.stl), p.fix_normals, p.flip, unit)) for p in sc.parts]
    rng = np.random.default_rng(s.seed)
    com, n_cl = fill_sites(tpl, fill, parts, device, rng)
    quats = random_quats(n_cl, rng)
    n_part = n_cl * tpl.n
    print(f"fill                  {n_cl:,} clumps ({n_cl*tpl.mass:.3f} kg), {n_part:,} spheres, "
          f"top layer at {com[:, 1].max()*1e3:.0f} mm")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    for p, (name, v, f) in zip(sc.parts, parts):
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()),
                               cfg=newton.ModelBuilder.ShapeConfig(mu=p.friction), key=name)
    park = (float(com[:, 0].mean()), 50.0, float(com[:, 2].mean()))
    for _ in range(n_part):
        builder.add_particle(pos=wp.vec3(*park), vel=wp.vec3(0.0), mass=tpl.mass,
                             radius=float(tpl.radii.max()), flags=0)
    model = builder.finalize(device=device)
    model.particle_ke, model.particle_kd, model.particle_kf = m.ke, kd_pp, m.kf
    model.particle_mu, model.particle_cohesion, model.particle_adhesion = m.friction, 0.0, 0.0
    model.particle_max_velocity = s.max_velocity
    model.set_gravity(tuple(sc.gravity))

    collider, meshes = build_collider(
        parts, two_sided=[p.two_sided for p in sc.parts], friction=[p.friction for p in sc.parts],
        ke=[m.ke] * len(parts), kd=[kd_wall] * len(parts), kf=[m.kf] * len(parts),
        thickness=[p.thickness if p.two_sided else 0.0 for p in sc.parts],
        max_dist=0.03, device=device, corners=[p.corners for p in sc.parts],
        motions=[surface_motion(p.motion, sc.gravity) for p in sc.parts])
    rmax = float(tpl.radii.max())
    max_thick = max([p.thickness if p.two_sided else 0.0 for p in sc.parts] + [0.0])
    reach = rmax + max_thick + 1.1e-3
    cell, _ = wall_grid_cell(parts, rmax, reach, s.wall_grid_cell)
    wall_grid, winfo = build_wall_grid(parts, meshes, collider.lower, collider.upper, reach, cell, device)
    print(f"wall grid             {winfo['dims']} cells of {cell*1e3:.0f} mm")

    Solver = SolverGranularEllipsoids if ell else SolverGranularClumps
    solver = Solver(model, collider, grid_cell=2.0 * rmax, keepalive=meshes,
                                  wall_grid=wall_grid,
                                  hash_dims=tuple(s.hash_dims) if s.hash_dims else None,
                                  wall_cache=s.wall_cache, rotation=True,
                                  mu_roll=m.rolling_friction, mu_roll_wall=m.wall_rolling_friction,
                                  rot_damp=m.rot_damp, rot_damp_wall=m.rot_damp_wall,
                                  tangential_ratio=m.tangential_ratio, hertz=hertz,
                                  youngs=youngs_eff, poisson=m.poisson, restitution=e)
    st = model.state()
    solver.bind_state(st)
    if ell:
        solver.set_ellipsoids(tpl, com, quats, kill_y=sc.domain_lo[1], park=park)
    else:
        solver.set_clumps(tpl, com, quats, kill_y=sc.domain_lo[1], park=park)

    # graph capture of K steps (in place)
    K = max(1, min(s.graph_steps, substeps))
    graph = graph_rem = None
    if device.startswith("cuda"):
        model.particle_grid.reserve(model.particle_count)
        import granular_clumps as _gc
        import granular_ellipsoids as _ge
        import granular_dem as _gd
        import newton._src.solvers.semi_implicit.kernels_contact as _kc
        for _mod in (_gd, _gc, _ge, _kc):
            wp.load_module(_mod, device=device)

        def capture(n):
            with wp.ScopedCapture(device=device, force_module_load=False) as cap:
                for _ in range(n):
                    solver.step(st, st, None, None, dt)
            return cap.graph
        graph = capture(K)
        rem = substeps % K
        graph_rem = capture(rem) if rem else None

    windows = [(p.active[0], p.active[1]) for p in sc.parts]
    regs = sc.regions
    cols = ["time_s", "n_grains", "mass_kg", "kinetic_energy_J", "max_speed_ms"]
    for r in regs:
        cols += [f"{r.name}_mass_kg", f"{r.name}_speed_ms"]
    cols += ["discharged_kg", "injected_kg", "escaped_kg", "wallclock_s"]
    hist = open(os.path.join(a.out, "history.csv"), "w")
    hist.write(",".join(cols) + "\n")
    json.dump(dict(scenario=sc.to_dict(), template=dict(offsets=tpl.offsets.tolist(),
                   radii=tpl.radii.tolist(), mass=tpl.mass, inertia=tpl.inertia.tolist(),
                   equiv_diameter=tpl.equiv_diameter), n_clumps=n_cl, dt=dt),
              open(os.path.join(a.out, "run.json"), "w"), indent=1, default=lambda o: None)

    n_frames = int(round(out.duration * out.fps))
    t0 = time.time()
    sim_t = 0.0
    saved = False
    k = tpl.n
    for frame in range(n_frames + 1):
        if frame > 0:
            solver.set_part_active([1 if lo <= sim_t < hi else 0 for lo, hi in windows])
            if graph is not None:
                for _ in range(substeps // K):
                    wp.capture_launch(graph)
                if graph_rem is not None:
                    wp.capture_launch(graph_rem)
            else:
                for _ in range(substeps):
                    solver.step(st, st, None, None, dt)
            sim_t += substeps * dt
        xc = solver.clump_x.numpy()
        vc = solver.clump_v.numpy()
        alive = (model.particle_flags.numpy()[::k] & int(newton.ParticleFlags.ACTIVE)) > 0
        xa, va = xc[alive], vc[alive]
        sp = np.linalg.norm(va, axis=1)
        row = [f"{frame / out.fps:.4f}", str(int(alive.sum())), f"{alive.sum() * tpl.mass:.6f}",
               f"{0.5 * tpl.mass * (sp ** 2).sum():.6f}", f"{sp.max() if len(sp) else 0.0:.4f}"]
        for r in regs:
            inside = np.all((xa > np.asarray(r.lo)) & (xa < np.asarray(r.hi)), axis=1)
            row += [f"{inside.sum() * tpl.mass:.6f}", f"{sp[inside].mean() if inside.any() else 0.0:.4f}"]
        row += [f"{(~alive).sum() * tpl.mass:.6f}", f"{n_cl * tpl.mass:.6f}", "0.000000",
                f"{time.time() - t0:.2f}"]
        hist.write(",".join(row) + "\n")
        hist.flush()
        if frame % int(out.fps) == 0:
            print(f"{frame / out.fps:7.3f}  " + "  ".join(
                f"{c} {v}" for c, v in zip(cols[5:], row[5:]) if c.endswith("_mass_kg"))
                + f"  wall {time.time() - t0:.0f} s", flush=True)
        if out.vtk:
            fl = model.particle_flags.numpy()
            act = np.flatnonzero(fl & int(newton.ParticleFlags.ACTIVE))
            if len(act):
                write_particles_vtk(os.path.join(a.out, f"frame_{frame:04d}_particles.vtk"), frame,
                                    st.particle_q.numpy()[act], st.particle_qd.numpy()[act], rmax,
                                    spin=solver.particle_w.numpy()[act], ids=act)
                if ell:
                    # orientation of each written body (same order as the VTK points), so
                    # render.py can draw the projected ellipses
                    np.save(os.path.join(a.out, f"frame_{frame:04d}_quat.npy"),
                            solver.clump_q.numpy()[act].astype(np.float32))
        if a.checkpoint_at is not None and not saved and frame / out.fps >= a.checkpoint_at:
            saved = True
            np.savez(os.path.join(a.out, "checkpoint.npz"), q=st.particle_q.numpy(),
                     qd=st.particle_qd.numpy(), flags=model.particle_flags.numpy(),
                     w=solver.particle_w.numpy(), clump_x=xc, clump_v=vc,
                     clump_q=solver.clump_q.numpy(), sim_t=sim_t, frame=frame)
    hist.close()
    wall = time.time() - t0
    print(f"\nwall clock {wall:.1f} s = {wall / max(sim_t, 1e-9):.1f} s per simulated second")


if __name__ == "__main__":
    main()
