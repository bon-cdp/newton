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

@wp.func
def _event(event_of_step: wp.array(dtype=wp.int32), step_arr: wp.array(dtype=int), step_offset: int):
    """Injection event index for the step about to run, or -1."""
    g = step_arr[0] - step_offset
    if g < 0 or g >= event_of_step.shape[0]:
        return -1
    return int(event_of_step[g])


@wp.kernel
def mark_occupied(
    event_of_step: wp.array(dtype=wp.int32),
    step_arr: wp.array(dtype=int),
    step_offset: int,
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    lat_origin: wp.vec3,
    lat_u: wp.vec3,
    lat_v: wp.vec3,
    lat_n: wp.vec3,
    lat_spacing: float,
    lat_nu: int,
    lat_nv: int,
    lat_map: wp.array(dtype=int),
    sites: wp.array(dtype=wp.vec3),
    clear_dist: float,
    occupied: wp.array(dtype=int),
):
    """On injection steps, flag every lattice site with a grain closer than clear_dist.
    Sites form a regular grid in the injection plane, so each grain near the plane checks
    only the 3x3 sites around its projection -- O(grains), not O(grains x sites)."""
    i = wp.tid()
    if _event(event_of_step, step_arr, step_offset) < 0:
        return
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return
    rel = particle_q[i] - lat_origin
    if wp.abs(wp.dot(rel, lat_n)) >= clear_dist:
        return
    fu = wp.dot(rel, lat_u) / lat_spacing
    fv = wp.dot(rel, lat_v) / lat_spacing
    iu0 = int(wp.floor(fu + 0.5))
    iv0 = int(wp.floor(fv + 0.5))
    for du in range(-1, 2):
        for dv in range(-1, 2):
            iu = iu0 + du
            iv = iv0 + dv
            if iu >= 0 and iu < lat_nu and iv >= 0 and iv < lat_nv:
                k = lat_map[iu * lat_nv + iv]
                if k >= 0:
                    if wp.length_sq(particle_q[i] - sites[k]) < clear_dist * clear_dist:
                        occupied[k] = 1


@wp.kernel
def inject_event(
    event_of_step: wp.array(dtype=wp.int32),
    step_arr: wp.array(dtype=int),
    step_offset: int,
    cand: wp.array2d(dtype=wp.int32),
    batch: int,
    occupied: wp.array(dtype=int),
    sites: wp.array(dtype=wp.vec3),
    spawn_vel: wp.vec3,
    free_idx: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    backlog: wp.array(dtype=int),
    starved: wp.array(dtype=int),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    wall_slack: wp.array(dtype=float),
    injected: wp.array(dtype=int),
):
    """One thread, launched every step, exits at once unless this step injects.

    Takes the event's candidate sites in order, skipping occupied ones, until
    batch + backlog grains are placed; whatever cannot be placed now is carried to the
    next event, so the delivered mass rate catches up instead of being lost.  Selection,
    placement and the free-list update happen in this single thread: events are rare
    (every ~150 steps), and three separate launches per step cost ~5% of a run.
    Deterministic, and it places exactly the plain schedule's grains whenever the picks
    are free.  Pool slots are popped from the top of the free list in order."""
    e = _event(event_of_step, step_arr, step_offset)
    if e < 0:
        return
    want = batch + backlog[0]
    top = int(free_count[0])
    placed = int(0)
    for k in range(cand.shape[1]):
        s = int(cand[e, k])
        if placed < want and placed < top and occupied[s] == 0:
            idx = free_idx[top - 1 - placed]
            particle_q[idx] = sites[s]
            particle_qd[idx] = spawn_vel
            particle_flags[idx] = wp.int32(newton.ParticleFlags.ACTIVE)
            wall_slack[idx] = 0.0
            placed += 1
        occupied[s] = 0                 # reset for the next event
    if top < want and placed == top:
        starved[0] = starved[0] + (want - placed)
    backlog[0] = want - placed
    free_count[0] = top - placed
    injected[0] = injected[0] + placed   # delivered, not scheduled


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
    escaped: wp.array(dtype=wp.int32),
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
    else:
        wp.atomic_add(escaped, 0, 1)   # left through a side or the top: a leak, not discharge


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


def surface_motion(motion, gravity):
    """Scenario motion dict -> (type, axis, point, rate) for build_collider (issue #11).

      {"type": "belt", "velocity": [vx, vy, vz]}         belt running at |v| along v
      {"type": "rotating", "point": [..], "axis": [..], "omega": rad/s}   (or "rpm")

    A belt's velocity is the running direction on its CARRYING strand; the solver turns
    it into a width axis a = up x v/|v| (up = against gravity) and moves each contact
    along a x n, so pulley wraps and the return strand follow the belt's path."""
    if not motion:
        return None
    kind = motion.get("type")
    if kind == "belt":
        u = np.asarray(motion["velocity"], dtype=np.float64)
        speed = float(np.linalg.norm(u))
        if speed == 0.0:
            return None
        up = -np.asarray(gravity, dtype=np.float64)
        up /= max(np.linalg.norm(up), 1e-12)
        a = np.cross(up, u / speed)
        if np.linalg.norm(a) < 1e-6:
            raise ValueError("belt velocity is vertical: its width axis is undefined")
        return (1, tuple(a / np.linalg.norm(a)), (0.0, 0.0, 0.0), speed)
    if kind == "rotating":
        ax = np.asarray(motion["axis"], dtype=np.float64)
        ax /= np.linalg.norm(ax)
        omega = motion["omega"] if "omega" in motion else motion["rpm"] * 2.0 * math.pi / 60.0
        return (2, tuple(ax), tuple(motion["point"]), float(omega))
    raise ValueError(f"unknown surface motion type {kind!r}")


def describe_motion(motion):
    if motion.get("type") == "belt":
        u = np.asarray(motion["velocity"], dtype=float)
        return f"belt {np.linalg.norm(u):g} m/s along {np.round(u / np.linalg.norm(u), 3).tolist()}"
    om = motion.get("omega", motion.get("rpm", 0.0) * 2.0 * math.pi / 60.0)
    return f"rotating {om:g} rad/s about {motion['axis']} through {motion['point']}"


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
            + ("  corners" if p.corners else "")
            + (f"  {describe_motion(p.motion)}" if p.motion else ""))

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
    # The lattice as a grid in the injection plane, so mark_occupied can find the sites
    # around a grain directly (same construction as lattice_sites).
    fpts = tris.reshape(-1, 3)
    flo, fhi = fpts.min(axis=0), fpts.max(axis=0)
    ax = int(np.argmin(fhi - flo))
    au, av = [k for k in range(3) if k != ax]
    off = np.asarray(inj.offset, dtype=np.float64)
    o_u, o_v = flo[au] + site_spacing * 0.5 + off[au], flo[av] + site_spacing * 0.5 + off[av]
    nu = len(np.arange(flo[au] + site_spacing * 0.5, fhi[au], site_spacing))
    nv = len(np.arange(flo[av] + site_spacing * 0.5, fhi[av], site_spacing))
    fu, fv = (sites[:, au] - o_u) / site_spacing, (sites[:, av] - o_v) / site_spacing
    iu, iv = np.rint(fu).astype(int), np.rint(fv).astype(int)
    assert np.allclose(fu, iu, atol=1e-4) and np.allclose(fv, iv, atol=1e-4) and \
        np.ptp(sites[:, ax]) < 1e-9, "injection sites are not a planar lattice"
    lat_map = np.full(nu * nv, -1, dtype=np.int32)
    lat_map[iu * nv + iv] = np.arange(len(sites), dtype=np.int32)
    e_ax = np.eye(3)
    lat_origin = np.zeros(3)
    lat_origin[au], lat_origin[av], lat_origin[ax] = o_u, o_v, sites[0, ax]
    lattice = dict(origin=wp.vec3(*lat_origin), u=wp.vec3(*e_ax[au]), v=wp.vec3(*e_ax[av]),
                   n=wp.vec3(*e_ax[ax]), spacing=float(site_spacing), nu=int(nu), nv=int(nv),
                   map=wp.array(lat_map, dtype=int, device=device))

    rate = inj.mass_rate / gmass
    # At most batch_fraction of the sites per event, but also no more than max_interval
    # between events: a large face with small grains at a low rate would otherwise inject
    # in big lumps (36,663 sites -> 9,165-grain batches 0.83 s apart delivered 166% of the
    # target in the first second).  Both validated projects already meet the interval
    # (15 and 36 ms), so their batches are unchanged.
    batch = max(1, min(int(len(sites) * inj.batch_fraction), math.ceil(rate * inj.max_interval)))
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
    uses_list = bool(s.neighbor_every) and m.rotation     # as in SolverGranularDEM.step
    gcell = 2.0 * radius + (skin_max if uses_list else 0.0)
    z0, z1 = int(sc.domain_lo[2] / gcell) - 1, int(sc.domain_hi[2] / gcell) + 1
    span = z1 - z0 + 1
    if s.hash_dims is None and span + 2 >= 128:
        # the table's z dimension must exceed the domain's z-cell span, or there is no
        # unreachable residue to park idle grains in: grow it (x, y stay at 128)
        s.hash_dims = [128, 128, int(2 ** math.ceil(math.log2(span + 8)))]
        log(f"  hash grid             z dimension {s.hash_dims[2]} (domain spans {span} z-cells)")
    hash_dz = s.hash_dims[2] if s.hash_dims else 128
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
        max_dist=0.03, device=device, corners=[p.corners for p in sc.parts],
        motions=[surface_motion(p.motion, sc.gravity) for p in sc.parts])
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
    escaped = wp.zeros(1, dtype=wp.int32, device=device)
    injected = wp.zeros(1, dtype=int, device=device)
    starved = wp.zeros(1, dtype=int, device=device)
    occupied = wp.zeros(len(sites), dtype=int, device=device)
    backlog = wp.zeros(1, dtype=int, device=device)
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
        free_idx=free_idx, free_count=free_count, discharged=discharged, escaped=escaped,
        injected=injected, starved=starved, lattice=lattice, occupied=occupied, backlog=backlog,
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
    meta["base_dir"] = sc.base_dir          # where the scenario's relative paths resolve
    meta.update(extra_meta or {})
    meta["derived"] = dict(grain_radius=S.grain_radius, grain_mass=gmass, rate=S.rate, dt=dt,
                           substeps=substeps, youngs_effective=S.youngs_eff, kd_pp=S.kd_pp,
                           kd_wall=S.kd_wall, zeta=S.zeta, pool=n_pool, sites=len(S.sites),
                           batch=batch, skin=S.skin, solver="SolverGranularDEM")
    # flat copies of the fields the analysis tools read (compare_bfa_dem, perf_table)
    meta.update(grain_radius=S.grain_radius, grain_mass=gmass, dt=dt, ke=sc.material.ke,
                mu=sc.material.friction, youngs=S.youngs_eff if sc.material.contact == "hertz" else None,
                wall_mu=sc.parts[0].friction, solver_class="SolverGranularDEM")
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: None)

    regs = sc.regions
    cols = ["time_s", "n_grains", "mass_kg", "kinetic_energy_J", "max_speed_ms"]
    for r in regs:
        cols += [f"{r.name}_mass_kg", f"{r.name}_speed_ms"]
    cols += ["spin_mean_rads"] + [f"spin_{r.name}_rads" for r in regs if r.spin]
    cols += ["discharged_kg", "injected_kg", "escaped_kg"]
    cols += [f"flow_{fp.name}_kg" for fp in sc.flow_planes]
    cols += ["wallclock_s"]
    # flow planes: crossings between consecutive frames, from positions read on the host.
    # A grain moves well under a metre per frame, so none can cross and leave unseen.
    flow_kg = np.zeros(len(sc.flow_planes))
    prev_q = None
    hist = open(os.path.join(out_dir, "history.csv"), "w")
    hist.write(",".join(cols) + "\n")

    n_frames = int(round(out.duration * out.fps))
    rng = np.random.default_rng(sc.solver.seed)
    t0 = time.time()
    sim_t, nstep = 0.0, 0
    saved_ckpt = False
    nbr_fallbacks = 0
    max_backlog = 0

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
    # Candidate order per event: its random picks first (so an unobstructed event places
    # exactly the grains the plain schedule would), then every other site in a fixed
    # rotating order -- the reserve used when picks are occupied or a backlog is owed.
    # The reserve order comes from its own generator so it never perturbs the picks.
    n_sites = len(S.sites)
    order0 = np.random.default_rng(sc.solver.seed + 7919).permutation(n_sites)
    cand = np.empty((max(len(picks), 1), n_sites), dtype=np.int32)
    for ev, pk in enumerate(picks):
        mask = np.ones(n_sites, dtype=bool)
        mask[pk] = False
        rot = np.roll(order0, -((ev * batch) % n_sites))
        cand[ev, :batch] = pk
        cand[ev, batch:] = rot[mask[rot]]
    if not picks:
        cand[0] = order0
    w_sites = wp.array(S.sites.astype(np.float32), dtype=wp.vec3, device=device)
    w_cand = wp.array(cand, dtype=wp.int32, device=device)
    w_event = wp.array(event_of_step, dtype=wp.int32, device=device)
    step_offset = solver.step_count          # device counter value at global step 0
    lat = S.lattice
    clear_dist = 2.0 * S.grain_radius * 1.05  # a grain closer than this would overlap

    def launch_steps(n, eager=True):
        if eager and n:
            solver.request_rebuild()   # graph replays do not advance the host's rebuild phase
        for _ in range(n):
            wp.launch(mark_occupied, dim=n_pool, device=device, inputs=[
                w_event, solver.step_arr, step_offset, s0.particle_q, model.particle_flags,
                lat["origin"], lat["u"], lat["v"], lat["n"], lat["spacing"], lat["nu"], lat["nv"],
                lat["map"], w_sites, clear_dist, S.occupied])
            wp.launch(inject_event, dim=1, device=device, inputs=[
                w_event, solver.step_arr, step_offset, w_cand, batch, S.occupied, w_sites,
                S.spawn_vel, S.free_idx, S.free_count, S.backlog, S.starved,
                s0.particle_q, s0.particle_qd, model.particle_flags, solver.wall_slack,
                S.injected])
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
                if sc.flow_planes:
                    cur_q = s0.particle_q.numpy()
                    if prev_q is not None:
                        for k, fp in enumerate(sc.flow_planes):
                            a = fp.axis
                            p0, p1 = prev_q[:, a] - fp.value, cur_q[:, a] - fp.value
                            cross = (p0 * p1 < 0.0)
                            if cross.any():
                                fr = (p0[cross] / (p0[cross] - p1[cross]))[:, None]
                                pt = prev_q[cross] + fr * (cur_q[cross] - prev_q[cross])
                                inside = np.ones(len(pt), dtype=bool)
                                for b in range(3):
                                    if b != a:
                                        inside &= (pt[:, b] >= fp.lo[b]) & (pt[:, b] <= fp.hi[b])
                                flow_kg[k] += inside.sum() * gmass
                wp.launch(recycle, dim=n_pool, device=device, inputs=[
                    s0.particle_q, s0.particle_qd, model.particle_flags,
                    wp.vec3(*sc.domain_lo), wp.vec3(*sc.domain_hi), S.park_lo,
                    S.free_idx, S.free_count, S.discharged, S.escaped])
                # recycle appends with atomics (arbitrary order): re-sort the free list so
                # the next spawns get the same pool slots on every run.  Once per frame.
                _fc = int(S.free_count.numpy()[0])
                _fi = S.free_idx.numpy()
                _fi[:_fc] = np.sort(_fi[:_fc])[::-1]
                S.free_idx.assign(_fi)
                if sc.flow_planes:
                    prev_q = s0.particle_q.numpy()      # after recycling: parked grains moved
                owed = int(S.backlog.numpy()[0])
                max_backlog = max(max_backlog, owed)
                if owed > batch:
                    print(f"  !! injection backlog {owed} grains (> one batch): the inlet is "
                          f"choked -- sites stay occupied", flush=True)
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
            inj_kg = int(S.injected.numpy()[0]) * gmass
            esc_kg = int(S.escaped.numpy()[0]) * gmass
            row += [f"{out_kg:.6f}", f"{inj_kg:.6f}", f"{esc_kg:.6f}"]
            row += [f"{kg:.6f}" for kg in flow_kg]
            row += [f"{wall:.2f}"]
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
                                        s0.particle_qd.numpy()[act], S.grain_radius,
                                        spin=solver.particle_w.numpy()[act], ids=act)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        hist.close()
    if s.neighbor_every:
        print(f"neighbour list: {nbr_fallbacks} grain-steps fell back to a direct search")
    # mass audit: scheduled vs delivered injection, and where the mass went
    t_end = min(sim_t, S.inj.stop) - max(S.inj.start, 0.0)
    inj_kg = int(S.injected.numpy()[0]) * gmass
    target = S.inj.mass_rate * max(t_end, 0.0)
    held = int(S.stats.numpy()[0]) * gmass
    out_kg = int(S.discharged.numpy()[0]) * gmass
    esc_kg = int(S.escaped.numpy()[0]) * gmass
    print(f"mass audit: injected {inj_kg:.2f} kg vs target {target:.2f} kg "
          f"({S.inj.mass_rate:g} kg/s x {t_end:.3f} s; {100*inj_kg/max(target,1e-12):.2f}%)"
          f";  held {held:.2f} + discharged {out_kg:.2f} + escaped {esc_kg:.2f} = "
          f"{held+out_kg+esc_kg:.2f} kg;  injection backlog max {max_backlog}, "
          f"end {int(S.backlog.numpy()[0])} grains")
    wall = time.time() - t0
    print(f"\nwall clock {wall:.1f} s = {wall / max(sim_t, 1e-9):.1f} s per simulated second"
          f" ({sim_t:.3f} of {out.duration:g} s simulated)")
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
