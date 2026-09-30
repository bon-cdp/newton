#!/usr/bin/env python3
"""
Run a DEM scenario file with SolverGranularDEM.

    python dem_run.py scenario.json [--out DIR] [--duration S] [--no-vtk] [--checkpoint-at T]

Everything machine-specific comes from the scenario (see dem_scenario.py): geometry,
material, injectors, domain, reporting regions, solver settings.  bfa_dem.py builds the
corn-spout scenario from its command line and runs it through here; bfa_import.py writes
scenarios from BulkFlowAnalyst projects.

Output (in --out): history.csv (per-frame totals and per-region mass/speed), run.json
(the scenario plus derived values), geometry.vtk and frame_XXXX_particles.vtk for
ParaView, optionally checkpoint.npz.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time

import numpy as np
import trimesh
import warp as wp

import newton
from bfa_replication_mpm import write_geometry_vtk, write_particles_vtk
from dem_scenario import INF, Scenario, rayleigh_time
from granular_dem import SolverGranularDEM, build_collider, build_wall_grid, lattice_sites

# Forward kernels only.  Nothing here is differentiated, and the adjoint of the wall kernel
# needs a parameter block over CUDA's 4 KB limit (it failed to compile once the collider
# gained per-part active flags).  Also roughly halves compile time.
wp.set_module_options({"enable_backward": False})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

@wp.kernel
def spawn_scheduled(
    sites: wp.array(dtype=wp.vec3),
    picks: wp.array2d(dtype=wp.int32),
    event_of_step: wp.array(dtype=wp.int32),
    step_arr: wp.array(dtype=int),
    step_offset: int,
    spawn_vel: wp.vec3,
    free_idx: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    wall_slack: wp.array(dtype=float),
):
    """Inject one batch of grains at lattice sites, driven by a precomputed schedule
    rather than the host.  Launched every step; fires only on steps the schedule marks.
    Reading the step from the solver's device counter is what lets injection live inside
    a captured CUDA graph."""
    tid = wp.tid()
    g = step_arr[0] - step_offset
    if g < 0 or g >= event_of_step.shape[0]:
        return
    e = event_of_step[g]
    if e < 0:
        return
    # Deterministic pop: thread tid takes the tid-th entry from the top.  An atomic
    # decrement hands out indices in whatever order threads arrive, which changes which
    # pool slot a grain gets, hence the summation order inside hash cells -- and DEM
    # chaos turns that rounding into visibly different runs.  _consume_free lowers the
    # count afterwards, in its own launch, so no thread here sees a partial update.
    slot = free_count[0] - 1 - tid
    if slot < 0:
        return
    idx = free_idx[slot]
    particle_q[idx] = sites[picks[e, tid]]
    particle_qd[idx] = spawn_vel
    particle_flags[idx] = wp.int32(newton.ParticleFlags.ACTIVE)
    wall_slack[idx] = 0.0


@wp.kernel
def _consume_free(event_of_step: wp.array(dtype=wp.int32), step_arr: wp.array(dtype=int),
                  step_offset: int, n: int, free_count: wp.array(dtype=wp.int32),
                  starved: wp.array(dtype=int)):
    g = step_arr[0] - step_offset
    if g < 0 or g >= event_of_step.shape[0]:
        return
    if event_of_step[g] >= 0:
        if free_count[0] < n:
            starved[0] = starved[0] + (n - free_count[0])   # grains the pool could not supply
        free_count[0] = wp.max(free_count[0] - n, 0)


@wp.kernel
def recycle(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    lo: wp.vec3,
    hi: wp.vec3,
    park: wp.vec3,
    free_idx: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    discharged: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    p = particle_q[i]
    if p[0] >= lo[0] and p[0] <= hi[0] and p[1] >= lo[1] and p[1] <= hi[1] and p[2] >= lo[2] and p[2] <= hi[2]:
        return
    particle_flags[i] = wp.int32(0)
    particle_q[i] = park          # see the note on parking in build()
    particle_qd[i] = wp.vec3(0.0)
    slot = wp.atomic_add(free_count, 0, 1)
    free_idx[slot] = i
    if p[1] < lo[1]:
        wp.atomic_add(discharged, 0, 1)


@wp.kernel
def region_stats(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_w: wp.array(dtype=wp.vec3),
    mass: float,
    reg_lo: wp.array(dtype=wp.vec3),
    reg_hi: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """out = [n, KE, sum|w|, max speed, then per region r: n, sum speed, sum |w|].
    Region membership is strict (lo < p < hi on every axis)."""
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    x = particle_q[i]
    v = particle_qd[i]
    speed = wp.length(v)
    spin = wp.length(particle_w[i])
    wp.atomic_add(out, 0, 1.0)
    wp.atomic_add(out, 1, 0.5 * mass * wp.dot(v, v))
    wp.atomic_add(out, 2, spin)
    wp.atomic_max(out, 3, speed)
    for r in range(reg_lo.shape[0]):
        lo = reg_lo[r]
        hi = reg_hi[r]
        if (x[0] > lo[0] and x[0] < hi[0] and x[1] > lo[1] and x[1] < hi[1]
                and x[2] > lo[2] and x[2] < hi[2]):
            wp.atomic_add(out, 4 + 3 * r, 1.0)
            wp.atomic_add(out, 5 + 3 * r, speed)
            wp.atomic_add(out, 6 + 3 * r, spin)


@wp.kernel
def _min_wall_distance(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float,
                       out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if r.result:
        d = wp.length(pts[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v))
        wp.atomic_min(out, i, d)


def clear_of_walls(sites, parts, clearance, device):
    """Keep only injection sites at least ``clearance`` from every collider surface."""
    p = wp.array(sites.astype(np.float32), dtype=wp.vec3, device=device)
    d = wp.full(len(sites), 1.0e9, dtype=float, device=device)
    for _name, v, f in parts:
        m = wp.Mesh(wp.array(np.asarray(v, dtype=np.float32), dtype=wp.vec3, device=device),
                    wp.array(np.asarray(f, dtype=np.int32).flatten(), dtype=int, device=device))
        wp.launch(_min_wall_distance, dim=len(sites), inputs=[m.id, p, clearance * 6.0, d],
                  device=device)
    return sites[d.numpy() >= clearance]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def load_stl(path, fix_normals=False, flip=False, scale=1.0):
    """(vertices float64, faces int) exactly as the replication runner always loaded them."""
    mesh = trimesh.load(path, force="mesh")
    if fix_normals:
        trimesh.repair.fix_normals(mesh, multibody=True)
    faces = np.asarray(mesh.faces)
    if flip:
        faces = faces[:, ::-1]
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if scale != 1.0:
        v = v * scale
    return v, np.ascontiguousarray(faces)


def face_triangles(path, scale=1.0):
    """Triangles of a planar injection face, through the same float32 path the original
    corn runner used (lattice sites depend on it bit-for-bit)."""
    mesh = trimesh.load(path, force="mesh")
    tris = np.asarray(mesh.triangles, dtype=np.float32)
    if scale != 1.0:
        tris = (tris * scale).astype(np.float32)
    v0 = tris[:, 0]
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    return np.stack([v0, v0 + e1, v0 + e2], axis=1).astype(np.float64), float(area.sum())


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def resolve_dt(sc: Scenario):
    """(dt, substeps, effective Young's modulus).  dt divides the frame interval."""
    m, s = sc.material, sc.solver
    youngs_eff = m.youngs / s.youngs_divisor
    if s.dt == "auto":
        dt_req = s.rayleigh_fraction * rayleigh_time(m.radius, m.density, youngs_eff, m.poisson)
    else:
        dt_req = float(s.dt)
    frame_dt = 1.0 / sc.output.fps
    substeps = max(1, int(round(frame_dt / dt_req)))
    return frame_dt / substeps, substeps, youngs_eff


def build(sc: Scenario, out_dir: str | None = None, quiet: bool = False):
    """Everything up to the time loop.  Returns a namespace (model, solver, state, pools,
    injection sites...) shared by run() and tools/dem_bench.py."""
    log = (lambda *a, **k: None) if quiet else print
    m, s = sc.material, sc.solver
    radius, gmass = m.radius, m.mass
    hertz = m.contact == "hertz"
    unit = sc.unit_scale

    # damping from the measured restitution, separately for each reduced mass
    e = max(min(m.restitution, 0.999), 1e-4)
    zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    kd_pp = m.kd if m.kd is not None else zeta * 2.0 * math.sqrt(0.5 * gmass * m.ke)
    kd_wall = m.kd_wall if m.kd_wall is not None else zeta * 2.0 * math.sqrt(gmass * m.ke)

    dt, substeps, youngs_eff = resolve_dt(sc)
    frame_dt = 1.0 / sc.output.fps
    total_rate = sum(i.mass_rate for i in sc.injectors) / gmass       # grains per second
    steady_n = s.expected_holdup_kg / gmass
    pool = int(max(s.min_pool, steady_n * s.pool_factor))

    out_dir = out_dir or os.path.join(SCRIPT_DIR, "runs", "dem",
                                      f"{sc.name}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    t_r = rayleigh_time(radius, m.density, youngs_eff, m.poisson)
    log("=" * 74)
    log(f"DEM scenario: {sc.name}  (SolverGranularDEM)")
    log("=" * 74)
    log(f"  grain                 r {radius*1e3:.1f} mm, m {gmass*1e3:.4f} g, rho {m.density:g}")
    log(f"  mass flow             {total_rate*gmass:g} kg/s -> {total_rate:,.0f} grains/s")
    log(f"  friction              grain-grain slide {m.friction:.3f}  roll {m.rolling_friction:.3f}"
        f";  wall roll {m.wall_rolling_friction:.3f}")
    log(f"  contact               {m.contact}" + (f", E {youngs_eff:.3g} Pa (true {m.youngs:.3g} / "
        f"{s.youngs_divisor:g}), nu {m.poisson}" if hertz else f", ke {m.ke:.3g} N/m")
        + f";  rotation {'ON' if m.rotation else 'OFF'}")
    log(f"  restitution           {e:.3f} -> kd {kd_pp:.3f} grain-grain, {kd_wall:.3f} grain-wall")
    log(f"  dt                    {dt*1e6:.3f} us = {dt/t_r:.2f} Rayleigh time"
        f"   ({substeps} steps per frame)")
    log(f"  steps                 {int(round(sc.output.duration / dt)):,}")
    log(f"  expected holdup       ~{steady_n:,.0f} grains ({s.expected_holdup_kg:g} kg);  pool {pool:,}")
    log(f"  output                {out_dir}\n")

    wp.init()
    device = s.device

    parts = [(p.name, *load_stl(sc.path(p.stl), p.fix_normals, p.flip, unit)) for p in sc.parts]
    if s.simplify_mm > 0:
        from mesh_simplify import collapse_short_edges
        parts = [(n, *collapse_short_edges(
            v, f, s.simplify_mm * 1e-3,
            tol=None if s.simplify_tol_mm is None else s.simplify_tol_mm * 1e-3))
            for n, v, f in parts]
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    for p, (name, v, f) in zip(sc.parts, parts):
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()),
                               cfg=newton.ModelBuilder.ShapeConfig(mu=p.friction), key=name)
        win = "" if p.active == [0.0, INF] else f"  active {p.active[0]:g}-{p.active[1]:g} s"
        log(f"  collider {name:<16} {len(f):6,d} tris  "
            f"{'two-sided' if p.two_sided else 'one-sided'}  mu {p.friction:g}{win}"
            + ("  [surface motion: not simulated yet, #11]" if p.motion else ""))

    # --- injection lattice: never uniform random -------------------------------------
    # Two grains seeded 2 mm apart overlap by 10 mm -- tens of newtons on a gram-scale
    # grain -- so sites are a lattice, and sites touching a wall are dropped (a site 1 mm
    # inside a face boundary puts the grain inside the wall: a kick on the first step).
    if len(sc.injectors) != 1:
        raise NotImplementedError("exactly one injector for now")
    inj = sc.injectors[0]
    tris, face_area = face_triangles(sc.path(inj.face_stl), unit)
    site_spacing = radius * inj.site_spacing
    sites = lattice_sites(tris, site_spacing)
    if inj.set_coordinate is not None:
        sites[:, int(inj.set_coordinate[0])] = inj.set_coordinate[1]
    sites += np.asarray(inj.offset, dtype=np.float64)
    n_raw = len(sites)
    sites = clear_of_walls(sites, parts, radius * inj.wall_clearance, device)
    if not len(sites):
        raise RuntimeError(f"injector {inj.name!r}: every lattice site is inside a wall")
    rate = inj.mass_rate / gmass
    batch = max(1, int(len(sites) * inj.batch_fraction))
    inj_interval = batch / rate
    spawn_speed = float(np.linalg.norm(inj.velocity))
    log(f"\n  injection face        {face_area:.4f} m2 -> {len(sites)} of {n_raw} lattice "
        f"sites at {site_spacing*1e3:.1f} mm ({n_raw-len(sites)} dropped for wall clearance)")
    log(f"  batch                 {batch} grains every {inj_interval*1e3:.1f} ms -> consecutive "
        f"batches {spawn_speed*inj_interval*1e3:.0f} mm apart "
        f"({spawn_speed*inj_interval/(2*radius):.1f} diameters)\n")
    if spawn_speed * inj_interval < 2.0 * radius:
        log("  !! consecutive injection batches overlap: raise the injection speed or lower "
            "batch_fraction")

    # --- neighbour list sizing -------------------------------------------------------
    skin = s.skin if s.skin is not None else 2.0 * s.skin_speed * max(s.neighbor_every, 1) * dt
    skin_max = max(2.0 * skin, 0.006)

    # --- where idle grains wait ------------------------------------------------------
    # They are still points in the hash grid, and the grid buckets cells MODULO its table
    # size, so where they sit decides which buckets they pollute.  History: all at one
    # point put 46,000 grains in a bucket some active grains aliased onto (180x slower
    # contact); a wide lattice fixed that but still cost 17%.  Instead pick one point
    # whose z-cell RESIDUE no active grain or neighbour query can produce: the domain spans
    # only (hi.z - lo.z)/cell + 3 residues of the table's z dimension, the rest are unused.
    hash_dz = s.hash_dims[2] if s.hash_dims else 128
    uses_list = bool(s.neighbor_every) and m.rotation     # as in SolverGranularDEM.step
    gcell = 2.0 * radius + (skin_max if uses_list else 0.0)
    z0, z1 = int(sc.domain_lo[2] / gcell) - 1, int(sc.domain_hi[2] / gcell) + 1
    span = z1 - z0 + 1
    if span + 2 >= hash_dz:
        raise ValueError(f"domain spans {span} z-cells; hash z-dim {hash_dz} is too small to "
                         f"hide idle grains -- set solver.hash_dims with a larger z")
    void_cell = z1 + 1 + (hash_dz - span) // 2          # middle of the unreachable residues
    park_lo = wp.vec3(float(sites[:, 0].mean()), float(sites[:, 1].max()) + 2.0,
                      (void_cell + 0.5) * gcell)
    for _k in range(pool):
        builder.add_particle(pos=park_lo, vel=wp.vec3(0.0), mass=gmass, radius=radius, flags=0)

    model = builder.finalize(device=device)
    model.particle_ke, model.particle_kd, model.particle_kf = m.ke, kd_pp, m.kf
    model.particle_mu, model.particle_cohesion, model.particle_adhesion = m.friction, 0.0, 0.0
    model.particle_max_velocity = s.max_velocity
    model.set_gravity(tuple(sc.gravity))
    n_pool = model.particle_count

    collider, meshes = build_collider(
        parts,
        two_sided=[p.two_sided for p in sc.parts],
        friction=[p.friction for p in sc.parts],
        ke=[m.ke] * len(parts), kd=[kd_wall] * len(parts), kf=[m.kf] * len(parts),
        thickness=[p.thickness if p.two_sided else 0.0 for p in sc.parts],
        max_dist=0.03, device=device)
    if s.neighbor_every:
        log(f"  neighbour list        rebuilt every {s.neighbor_every} steps, skin "
            f"{skin*1e3:.2f} mm (two grains closing at {s.skin_speed:g} m/s)")
    wall_grid = None
    if not s.bvh_walls:
        # reach: contact (radius + shell) plus the 1 mm band the sign test averages over
        max_thick = max([p.thickness if p.two_sided else 0.0 for p in sc.parts] + [0.0])
        reach = radius + max_thick + 1.0e-3 + 1.0e-4
        _t = time.time()
        wall_grid, winfo = build_wall_grid(parts, meshes, collider.lower, collider.upper,
                                           reach, s.wall_grid_cell, device)
        log(f"  wall grid             {winfo['dims']} cells of {s.wall_grid_cell*1e3:.0f} mm, "
            f"{winfo['occupied']:,} occupied, lists mean {winfo['mean_list']:.1f} / max "
            f"{winfo['max_list']}, {winfo['mbytes']:.0f} MB, baked in {time.time()-_t:.2f} s")
    solver = SolverGranularDEM(model, collider, grid_cell=2.0 * radius, keepalive=meshes,
                               wall_grid=wall_grid,
                               hash_dims=tuple(s.hash_dims) if s.hash_dims else None,
                               neighbor_every=s.neighbor_every, skin=skin, skin_max=skin_max,
                               wall_cache=s.wall_cache, rotation=m.rotation,
                               mu_roll=m.rolling_friction, mu_roll_wall=m.wall_rolling_friction,
                               rot_damp=m.rot_damp, rot_damp_wall=m.rot_damp_wall,
                               tangential_ratio=m.tangential_ratio,
                               hertz=hertz, youngs=youngs_eff, poisson=m.poisson, restitution=e,
                               calibrate_restitution=m.calibrate_restitution, cal_dt=dt)

    s0, s1 = model.state(), model.state()
    free_idx = wp.array(np.arange(n_pool, dtype=np.int32), dtype=wp.int32, device=device)
    free_count = wp.array([n_pool], dtype=wp.int32, device=device)
    discharged = wp.zeros(1, dtype=wp.int32, device=device)
    starved = wp.zeros(1, dtype=int, device=device)
    regs = sc.regions
    reg_lo = wp.array(np.array([r.lo for r in regs] or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3,
                      device=device)
    reg_hi = wp.array(np.array([r.hi for r in regs] or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3,
                      device=device)
    stats = wp.zeros(4 + 3 * max(len(regs), 1), dtype=float, device=device)
    spawn_vel = wp.vec3(*[float(c) for c in inj.velocity])
    return argparse.Namespace(
        sc=sc, model=model, solver=solver, collider=collider, meshes=meshes, parts=parts,
        sites=sites, batch=batch, inj=inj, inj_interval=inj_interval, rate=rate,
        substeps=substeps, dt=dt, frame_dt=frame_dt, n_pool=n_pool, s0=s0, s1=s1,
        free_idx=free_idx, free_count=free_count, discharged=discharged, starved=starved,
        reg_lo=reg_lo, reg_hi=reg_hi, stats=stats, spawn_vel=spawn_vel, park_lo=park_lo,
        out_dir=out_dir, device=device, kd_pp=kd_pp, kd_wall=kd_wall, restitution=e, zeta=zeta,
        youngs_eff=youngs_eff, skin=skin, grain_mass=gmass, grain_radius=radius)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(S, extra_meta: dict | None = None, reference: dict | None = None):
    """The time loop.  ``reference`` optionally maps history columns to target values
    printed under the header (e.g. BFA's steady-state numbers)."""
    sc, model, solver, device = S.sc, S.model, S.solver, S.device
    s0, dt, substeps, batch = S.s0, S.dt, S.substeps, S.batch
    gmass, out_dir, n_pool = S.grain_mass, S.out_dir, S.n_pool
    out = sc.output

    if out.vtk:
        write_geometry_vtk(os.path.join(out_dir, "geometry.vtk"), S.parts)
    meta = sc.to_dict()
    meta.update(extra_meta or {})
    meta["derived"] = dict(grain_radius=S.grain_radius, grain_mass=gmass, rate=S.rate, dt=dt,
                           substeps=substeps, youngs_effective=S.youngs_eff, kd_pp=S.kd_pp,
                           kd_wall=S.kd_wall, zeta=S.zeta, pool=n_pool, sites=len(S.sites),
                           batch=batch, skin=S.skin, solver="SolverGranularDEM")
    # flat copies of the fields the analysis tools read (compare_bfa_dem, perf_table)
    meta.update(grain_radius=S.grain_radius, grain_mass=gmass, dt=dt, ke=sc.material.ke,
                mu=sc.material.friction, youngs=S.youngs_eff if sc.material.contact == "hertz" else None,
                wall_mu=sc.parts[0].friction, solver="SolverGranularDEM")
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: None)

    regs = sc.regions
    cols = ["time_s", "n_grains", "mass_kg", "kinetic_energy_J", "max_speed_ms"]
    for r in regs:
        cols += [f"{r.name}_mass_kg", f"{r.name}_speed_ms"]
    cols += ["spin_mean_rads"] + [f"spin_{r.name}_rads" for r in regs if r.spin]
    cols += ["discharged_kg", "wallclock_s"]
    hist = open(os.path.join(out_dir, "history.csv"), "w")
    hist.write(",".join(cols) + "\n")

    n_frames = int(round(out.duration * out.fps))
    rng = np.random.default_rng(sc.solver.seed)
    t0 = time.time()
    sim_t, nstep = 0.0, 0
    saved_ckpt = False
    nbr_fallbacks = 0

    # --- injection schedule, precomputed ---------------------------------------------
    # The host logic the loop used to run (float64 sim_t, the same rng calls in the same
    # order), replayed up front so the device can inject without host input.
    total_steps = n_frames * substeps
    event_of_step = np.full(total_steps, -1, dtype=np.int32)
    picks = []
    _t_sim, _next = 0.0, max(S.inj.start, 0.0)
    for gstep in range(total_steps):
        if _t_sim >= _next and _t_sim < S.inj.stop:
            _next += S.inj_interval
            event_of_step[gstep] = len(picks)
            picks.append(rng.choice(len(S.sites), size=batch, replace=False))
        _t_sim += dt
    w_sites = wp.array(S.sites.astype(np.float32), dtype=wp.vec3, device=device)
    w_picks = wp.array(np.array(picks, dtype=np.int32).reshape(-1, batch) if picks
                       else np.zeros((1, batch), np.int32), dtype=wp.int32, device=device)
    w_event = wp.array(event_of_step, dtype=wp.int32, device=device)
    step_offset = solver.step_count          # device counter value at global step 0

    def launch_steps(n, eager=True):
        if eager and n:
            solver.request_rebuild()   # graph replays do not advance the host's rebuild phase
        for _ in range(n):
            wp.launch(spawn_scheduled, dim=batch, device=device, inputs=[
                w_sites, w_picks, w_event, solver.step_arr, step_offset, S.spawn_vel,
                S.free_idx, S.free_count, s0.particle_q, s0.particle_qd, model.particle_flags,
                solver.wall_slack])
            wp.launch(_consume_free, dim=1, device=device, inputs=[
                w_event, solver.step_arr, step_offset, batch, S.free_count, S.starved])
            solver.step(s0, s0, None, None, dt)     # in place -- see SolverGranularDEM.step

    # --- CUDA graph ------------------------------------------------------------------
    # A frame is floor(substeps/K) replays of a K-step graph plus one replay of a graph
    # for the remainder, so no step runs eagerly.  Each graph starts with a neighbour-list
    # rebuild and rebuilds every N steps inside, so no rebuild interval exceeds N.
    s = sc.solver
    graph, graph_rem, K = None, None, max(0, min(s.graph_steps, substeps))
    if s.neighbor_every:
        K -= K % s.neighbor_every
    if K > 0 and device.startswith("cuda"):
        # capture only RECORDS launches; buffers the hash grid would allocate on first
        # build are reserved, and the kernel modules a step uses are loaded, beforehand
        model.particle_grid.reserve(n_pool)
        # force_module_load=True would compile every Warp module in the process, dozens
        # of unrelated Newton ones (measured: 140 s), so load just ours
        import granular_dem as _gd
        import newton._src.solvers.semi_implicit.kernels_contact as _kc
        for _mod in (_gd, _kc, __import__(__name__)):
            wp.load_module(_mod, device=device)

        def capture(n):
            solver.request_rebuild()
            with wp.ScopedCapture(device=device, force_module_load=False) as cap:
                launch_steps(n, eager=False)
            return cap.graph

        graph = capture(K)
        rem = substeps % K
        graph_rem = capture(rem) if rem else None
        print(f"  cuda graph            {substeps // K} x {K} steps"
              + (f" + {rem}" if rem else "") + " per frame", flush=True)

    # --- per-part active windows -----------------------------------------------------
    windows = [(p.active[0], p.active[1]) for p in sc.parts]
    timed = any(w != (0.0, INF) for w in windows)

    def set_active(t):
        """Friction/contact of parts outside their window is switched off by moving
        nothing: an inactive part is simply skipped by the wall kernels."""
        solver.set_part_active([1 if a <= t < b else 0 for a, b in windows])

    hdr = f"{'t (s)':>7} {'grains':>9} {'kg':>8} {'KE (J)':>10} " + \
        " ".join(f"{r.name[:8] + ' kg':>11} {'m/s':>5}" for r in regs) + f" {'out kg':>9} {'wall':>6}"
    print(hdr)
    if reference:
        print(f"{'ref ->':>7} {'':>9} {reference.get('mass_kg', ''):>8} "
              f"{reference.get('kinetic_energy_J', ''):>10} " +
              " ".join(f"{reference.get(r.name + '_mass_kg', ''):>11} "
                       f"{reference.get(r.name + '_speed_ms', ''):>5}" for r in regs))

    try:
        for frame in range(n_frames + 1):
            if frame > 0:
                if timed:
                    set_active(sim_t)
                if graph is not None:
                    for _ in range(substeps // K):
                        wp.capture_launch(graph)
                    if graph_rem is not None:
                        wp.capture_launch(graph_rem)
                else:
                    launch_steps(substeps)
                sim_t += substeps * dt
                nstep += substeps
                wp.launch(recycle, dim=n_pool, device=device, inputs=[
                    s0.particle_q, s0.particle_qd, model.particle_flags,
                    wp.vec3(*sc.domain_lo), wp.vec3(*sc.domain_hi), S.park_lo,
                    S.free_idx, S.free_count, S.discharged])
                # recycle appends with atomics (arbitrary order): re-sort the free list so
                # the next spawns get the same pool slots on every run.  Once per frame.
                _fc = int(S.free_count.numpy()[0])
                _fi = S.free_idx.numpy()
                _fi[:_fc] = np.sort(_fi[:_fc])[::-1]
                S.free_idx.assign(_fi)
                starved = int(S.starved.numpy()[0])
                if starved:
                    print(f"  !! particle pool exhausted: {starved} grains not injected -- raise "
                          f"solver.expected_holdup_kg or pool_factor", flush=True)
                    S.starved.zero_()

            if s.neighbor_every and frame > 0:
                # fallbacks: grain-steps that outran their list and searched directly
                # instead (only slower).  Overflows drop contacts, so shout.
                nbr_fallbacks += int(solver.nbr_fallbacks.numpy()[0])
                no = int(solver.nbr_overflow.numpy()[0])
                if no:
                    print(f"  !! neighbour list overflow on {no} grain-builds: contacts dropped",
                          flush=True)
                    solver.nbr_overflow.zero_()
                solver.nbr_fallbacks.zero_()

            S.stats.zero_()
            wp.launch(region_stats, dim=n_pool, device=device, inputs=[
                s0.particle_q, s0.particle_qd, model.particle_flags, solver.particle_w,
                gmass, S.reg_lo, S.reg_hi, S.stats])
            st = S.stats.numpy()
            n = st[0]
            out_kg = int(S.discharged.numpy()[0]) * gmass
            wall = time.time() - t0
            row = [f"{frame*S.frame_dt:.4f}", f"{int(n)}", f"{n*gmass:.6f}", f"{st[1]:.6f}",
                   f"{st[3]:.4f}"]
            show = []
            for k, r in enumerate(regs):
                c, sv = st[4 + 3 * k], st[5 + 3 * k]
                v = sv / c if c > 0 else 0.0
                row += [f"{c*gmass:.6f}", f"{v:.4f}"]
                show.append(f"{c*gmass:11.2f} {v:5.2f}")
            row.append(f"{(st[2] / n if n > 0 else 0.0):.3f}")
            for k, r in enumerate(regs):
                if r.spin:
                    c = st[4 + 3 * k]
                    row.append(f"{(st[6 + 3 * k] / c if c > 0 else 0.0):.3f}")
            row += [f"{out_kg:.6f}", f"{wall:.2f}"]
            hist.write(",".join(row) + "\n")
            hist.flush()
            if out.checkpoint_at is not None and not saved_ckpt and frame * S.frame_dt >= out.checkpoint_at:
                saved_ckpt = True
                np.savez(os.path.join(out_dir, "checkpoint.npz"),
                         q=s0.particle_q.numpy(), qd=s0.particle_qd.numpy(),
                         flags=model.particle_flags.numpy(), w=solver.particle_w.numpy(),
                         tang_partner=solver.tang_partner.numpy(),
                         tang_stamp=solver.tang_stamp.numpy(), tang_xi=solver.tang_xi.numpy(),
                         wall_slack=solver.wall_slack.numpy(), free_idx=S.free_idx.numpy(),
                         free_count=S.free_count.numpy(), discharged=S.discharged.numpy(),
                         sim_t=sim_t, nstep=nstep, solver_step=solver.step_count - 1,
                         frame=frame)
                # No rng state: the injection schedule is drawn up front from the seed, so a
                # resume regenerates it and continues at global step nstep.
                print(f"  checkpoint written at t = {sim_t:.4f} s", flush=True)
            print(f"{frame*S.frame_dt:7.3f} {int(n):9,d} {n*gmass:8.2f} {st[1]:10.2f} "
                  + " ".join(show) + f" {out_kg:9.2f} {wall:6.0f}", flush=True)

            if out.vtk:
                flags = model.particle_flags.numpy()
                act = np.flatnonzero(flags & int(newton.ParticleFlags.ACTIVE))
                if len(act):
                    write_particles_vtk(os.path.join(out_dir, f"frame_{frame:04d}_particles.vtk"),
                                        frame, s0.particle_q.numpy()[act],
                                        s0.particle_qd.numpy()[act], S.grain_radius)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        hist.close()
    if s.neighbor_every:
        print(f"neighbour list: {nbr_fallbacks} grain-steps fell back to a direct search")
    wall = time.time() - t0
    print(f"\nwall clock {wall:.1f} s = {wall / max(out.duration, 1e-9):.1f} s per simulated second")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="scenario JSON (see dem_scenario.py)")
    ap.add_argument("--out", default=None, help="output directory (default runs/dem/<name>_<time>)")
    ap.add_argument("--duration", type=float, default=None, help="override output.duration")
    ap.add_argument("--no-vtk", action="store_true")
    ap.add_argument("--checkpoint-at", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sc = Scenario.load(args.scenario)
    if args.duration is not None:
        sc.output.duration = args.duration
    if args.no_vtk:
        sc.output.vtk = False
    if args.checkpoint_at is not None:
        sc.output.checkpoint_at = args.checkpoint_at
    if args.device:
        sc.solver.device = args.device
    run(build(sc, args.out))


if __name__ == "__main__":
    main()
