#!/usr/bin/env python3
"""
DEM replication of the BulkFlowAnalyst run in 23087-25sim/, using SolverGranularDEM.

This simulates the same objects BFA does -- 12 mm spheres with mass, radius, rotation and
pairwise contact -- so the BFA material parameters transfer directly:

    grain radius / mass    6.0 mm / 0.8994 g      (prj Min_Rad, Intr Density 994.05)
    mass flow              8.6111 kg/s            (prj FlowRate 34.171652 short ton/h)
                           -> 9573 grains/s, 95,730 over the 10 s run
    injection velocity     3.1321 m/s down        (= sqrt(2*g*0.5), extrusion length)
    friction               slide 0.09 / roll 0.30 grain-grain, 0.50 / 0.50 grain-wall
    restitution            0.20                   (damping derived from it)

Presets (see PRESETS): `fast` (default, ~6 s per simulated second), `reference` (BFA's
own timestep and stiffness, ~32 s), `bfa` (raw project inputs, nothing fitted).  The
calibrated presets use grain-grain slide 0.11 and wall 0.53; see BFA_REPLICATION.md.

Injection uses a lattice, never uniform random: two grains seeded 2 mm apart overlap by
10 mm, which is 10 N on a 0.9 g mass -- the classic "particles behave like a gas" failure.

Usage:
    python bfa_dem.py                             # fast preset, 10 s, VTK every frame
    python bfa_dem.py --preset reference          # BFA timestep and stiffness
    python bfa_dem.py --duration 1 --no-vtk       # quick shakedown
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time

import numpy as np
import warp as wp

import newton
from bfa_replication_mpm import (
    COLLIDER_PARTS, DOMAIN_HI, DOMAIN_LO, INJECTION_PLANE_Y, INJECTION_SPEED,
    MASS_FLOW, RHO_GRAIN, WALL_MU, injection_face_triangles, load_part,
    write_geometry_vtk, write_particles_vtk,
)
from granular_dem import SolverGranularDEM, build_collider, build_wall_grid, lattice_sites

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GRAIN_RADIUS = 0.006
GRAIN_MASS = 4.0 / 3.0 * math.pi * GRAIN_RADIUS**3 * RHO_GRAIN  # 8.9943e-4 kg
ANGLE_OF_REPOSE = 23.3
TUBE_LO, TUBE_HI = 0.0, 2.5
# The bottom cascade (Top/Mid/Bot/SS/Baf/Def deflector assembly).  BFA holds 4.59 kg here
# at a mass-weighted 2.39 m/s; this is the region still short in our runs.
CASCADE_HI = -1.0


@wp.kernel
def spawn_at_sites(
    sites: wp.array(dtype=wp.vec3),
    spawn_vel: wp.vec3,
    free_idx: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    wall_slack: wp.array(dtype=float),
):
    """Host-driven injection (one launch per event), used by angle_of_repose.py.  The BFA
    runner uses spawn_scheduled instead, which needs no host input and is deterministic."""
    tid = wp.tid()
    prev = wp.atomic_sub(free_count, 0, 1)
    if prev <= 0:
        wp.atomic_add(free_count, 0, 1)
        return
    idx = free_idx[prev - 1]
    particle_q[idx] = sites[tid]
    particle_qd[idx] = spawn_vel
    particle_flags[idx] = wp.int32(newton.ParticleFlags.ACTIVE)
    # A recycled index inherits the previous occupant's wall-distance cache; clear it or
    # the new grain may skip its first queries while sitting next to a wall.
    wall_slack[idx] = 0.0


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
    rather than the host.  Launched every step; fires only on steps the schedule marks.  Reading the step from the
    solver's device counter is what lets injection live inside a captured CUDA graph."""
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
                  step_offset: int, n: int, free_count: wp.array(dtype=wp.int32)):
    g = step_arr[0] - step_offset
    if g < 0 or g >= event_of_step.shape[0]:
        return
    if event_of_step[g] >= 0:
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
    particle_q[i] = park          # see the note on parking at pool construction
    particle_qd[i] = wp.vec3(0.0)
    slot = wp.atomic_add(free_count, 0, 1)
    free_idx[slot] = i
    if p[1] < lo[1]:
        wp.atomic_add(discharged, 0, 1)


@wp.kernel
def stats_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_w: wp.array(dtype=wp.vec3),
    mass: float,
    out: wp.array(dtype=float),
):
    """out = [n, KE, unused, max_speed, n_tube, sum_v_tube, n_cascade, sum_v_cascade,
    sum_|w| all, sum_|w| cascade].  Spin is the diagnostic for whether the rotational
    damping is reaching the grains at all: if |w| is unchanged by a 10x coefficient
    change the term is not being applied; if |w| is already ~0 the contact is
    over-damped and the grains are sliding rather than rolling."""
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    v = particle_qd[i]
    y = particle_q[i][1]
    speed = wp.length(v)
    wp.atomic_add(out, 0, 1.0)
    wp.atomic_add(out, 1, 0.5 * mass * wp.dot(v, v))
    wp.atomic_max(out, 3, speed)
    if y > TUBE_LO and y < TUBE_HI:
        wp.atomic_add(out, 4, 1.0)
        wp.atomic_add(out, 5, speed)
    spin = wp.length(particle_w[i])
    wp.atomic_add(out, 8, spin)
    if y < CASCADE_HI:
        wp.atomic_add(out, 6, 1.0)
        wp.atomic_add(out, 7, speed)
        wp.atomic_add(out, 9, spin)


@wp.kernel
def _min_wall_distance(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float,
                       out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if r.result:
        d = wp.length(pts[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v))
        wp.atomic_min(out, i, d)


def _clear_of_walls(sites, parts, clearance, device):
    """Keep only injection sites at least ``clearance`` from every collider surface."""
    p = wp.array(sites.astype(np.float32), dtype=wp.vec3, device=device)
    d = wp.full(len(sites), 1.0e9, dtype=float, device=device)
    for _name, v, f in parts:
        m = wp.Mesh(wp.array(np.asarray(v, dtype=np.float32), dtype=wp.vec3, device=device),
                    wp.array(np.asarray(f, dtype=np.int32).flatten(), dtype=int, device=device))
        wp.launch(_min_wall_distance, dim=len(sites), inputs=[m.id, p, clearance * 6.0, d],
                  device=device)
    return sites[d.numpy() >= clearance]


# Named configurations.  The argument defaults below are the raw BFA project inputs
# ("bfa"); a preset overrides them, and any flag given explicitly overrides the preset.
PRESETS = {
    # Calibrated replication at BFA's own timestep and stiffness (runs/dem/perf_grid_ref):
    # rms 0.057, pile top -1.403 vs BFA -1.401.  ~32 s per simulated second.
    "reference": dict(mu=0.11, wall_mu=0.53, shell_thickness=0.0, hertz=True,
                      tangential_ratio=1.0),
    # Same physics, 10x softer grains at 4x the timestep, neighbour lists
    # (runs/dem/perf_fast_nosimp): rms 0.056, pile top -1.440.  ~6 s per simulated second,
    # ~97x faster than BFA.  Young's modulus / 10 keeps every observable within 1%;
    # / 100 lowered the pile 7 cm, and 2 mm mesh simplification lowered it 2.4 cm.
    "fast": dict(mu=0.11, wall_mu=0.53, shell_thickness=0.0, hertz=True,
                 tangential_ratio=1.0, youngs=1.4220405e7, dt=9.7253e-5,
                 neighbor_every=4, skin_speed=6.0),
    # The BFA project's measured inputs with nothing fitted (the argument defaults).
    "bfa": dict(),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fast",
                    help="named configuration (default fast); explicit flags override it")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--dt", type=float, default=2.4316429e-5, help="BFA's own DEM timestep")
    ap.add_argument("--ke", type=float, default=2.0e4, help="normal contact stiffness (N/m)")
    ap.add_argument("--restitution", type=float, default=0.20,
                    help="coefficient of restitution (BFA prj Coefficient_Restitution: 0.20).  "
                         "Damping is DERIVED from it: for a linear spring-dashpot "
                         "zeta = -ln(e)/sqrt(pi^2+ln^2 e) and kd = zeta*2*sqrt(m_eff*ke).  The "
                         "reduced mass differs between grain-grain (m/2) and grain-wall (m), so "
                         "a single kd cannot give the same e for both -- the old kd = 3.0 gave "
                         "e = 0.16 grain-grain but 0.30 at the wall.")
    ap.add_argument("--kd", type=float, default=None,
                    help="override grain-grain damping directly (bypasses --restitution)")
    ap.add_argument("--kd-wall", type=float, default=None,
                    help="override grain-wall damping directly")
    ap.add_argument("--kf", type=float, default=3.0, help="tangential viscous coeff, Coulomb-capped")
    ap.add_argument("--mu", type=float, default=0.09,
                    help="grain-grain sliding friction (BFA prj: 0.09)")
    ap.add_argument("--mu-roll", type=float, default=0.30,
                    help="grain-grain rolling friction (BFA prj Rotating Friction: 0.30).  "
                         "Requires rotation; with --no-rotation this has no effect and --mu "
                         "must instead absorb it (tan(23.3 deg) = 0.4307, which then jams).")
    ap.add_argument("--tangential-ratio", type=float, default=0.0,
                    help="tangential spring stiffness as a fraction of ke (Cundall-Strack / "
                         "Mindlin history).  0 keeps the viscous-Coulomb law.  The Mindlin "
                         "ratio is k_t/k_n = 2(1-nu)/(2-nu) = 0.82 at BFA's nu = 0.3; the "
                         "often-quoted 2/7 is an equal-energy-partition figure and is 3x too "
                         "soft -- measured, it left a grain creeping at 25 mm/s on a slope it "
                         "should sit still on, where 0.82-1.0 holds it.  A viscous law only "
                         "resists while sliding, so grains under load always creep; a spring "
                         "stores displacement and pushes back at zero slip rate, which is what "
                         "lets force chains hold a dense slow layer.")
    ap.add_argument("--rot-damp", type=float, default=0.20,
                    help="grain-grain rotational damping (prj RotatingR = 0.20).  Viscous "
                         "torque opposing relative spin, scaled as eta*kd*r_eff^2.  BFA has "
                         "this channel and TwistingR; this solver had neither -- rolling "
                         "friction is Coulomb-like and does not scale with spin rate.")
    ap.add_argument("--rot-damp-wall", type=float, default=0.20,
                    help="grain-wall rotational damping (prj RotatingR = 0.20)")
    ap.add_argument("--mu-roll-wall", type=float, default=0.50,
                    help="grain-wall rolling friction.  The prj gives particle<->boundary "
                         "RotatingMu = 0.50 for all seven components, distinct from the "
                         "particle<->particle 0.30.")
    ap.add_argument("--no-rotation", action="store_true",
                    help="disable angular DOF (the original non-rotating behaviour).  That path "
                         "is the original solver: BVH walls, no neighbour list, no Hertz.")
    ap.add_argument("--wall-mu", type=float, default=WALL_MU,
                    help="Coulomb friction on the Spout, i.e. the long chute")
    ap.add_argument("--cascade-mu", type=float, default=None,
                    help="Coulomb friction on the six cascade components (Top/Mid/Bot/SS/"
                         "Baf/Def), separate from the Spout.  The deficit is 0.58 kg of "
                         "STOPPED material on the deflectors -- BFA has 49%% of the cascade "
                         "below 1.5 m/s against our 31%% -- so friction there is the targeted "
                         "lever, and it leaves the chute (which already matches) alone.  "
                         "Defaults to --wall-mu.")
    ap.add_argument("--calibrate-restitution", action="store_true",
                    help="solve for the damping that actually DELIVERS --restitution.  The "
                         "fn>=0 clamp ends contact early and truncates the rebound, so the "
                         "textbook zeta gives e=0.33 when 0.20 was asked for (measured, and "
                         "dt-independent).  This bisects the damping against the solver's own "
                         "discrete contact instead of trusting the formula.")
    ap.add_argument("--hertz", action=argparse.BooleanOptionalAction, default=False,
                    help="Hertz-Mindlin contact (fn = (4/3)E* sqrt(R*) d^1.5, damping ~ d^0.25) "
                         "instead of the linear spring-dashpot.  BFA's prj says Contact Mode: "
                         "Hertzian, and the linear law cannot match it in both regimes at once: "
                         "matched to the plug it is ~9x too soft at chute impact speeds.")
    ap.add_argument("--youngs", type=float, default=1.4220405e8,
                    help="Young's modulus for --hertz (BFA lin: 1.422e8 Pa)")
    ap.add_argument("--poisson", type=float, default=0.30,
                    help="Poisson ratio for --hertz; NOT in the prj, so this is an assumption")
    ap.add_argument("--shell-thickness", type=float, default=0.002)
    ap.add_argument("--max-velocity", type=float, default=30.0, help="safety clamp (m/s)")
    ap.add_argument("--pool-factor", type=float, default=2.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-vtk", action="store_true")
    ap.add_argument("--no-wall-cache", action="store_true",
                    help="with --bvh-walls: query every wall every step; results must match the "
                         "cached path.  (The baked wall grid has no distance cache.)")
    ap.add_argument("--graph-steps", type=int, default=256,
                    help="steps captured into one CUDA graph and replayed as a single launch "
                         "(0 = launch every kernel from Python each step)")
    ap.add_argument("--wall-grid-cell", type=float, default=0.006,
                    help="cell size (m) of the baked candidate-triangle grid used for wall "
                         "contact.  Smaller = shorter lists, more memory.")
    ap.add_argument("--bvh-walls", action="store_true",
                    help="query the wall BVHs every step instead of the baked grid (the "
                         "reference path; ~10x slower wall contact)")
    ap.add_argument("--neighbor-every", type=int, default=0,
                    help="rebuild a Verlet neighbour list every N steps instead of rebuilding "
                         "and scanning the hash grid every step (0 = off)")
    ap.add_argument("--skin", type=float, default=None,
                    help="neighbour-list skin (m).  Default: two grains closing at --skin-speed "
                         "each for N steps, 2*v*N*dt.  Grains that move more than skin/2 before "
                         "the next rebuild fall back to a direct search; the count is reported.")
    ap.add_argument("--skin-speed", type=float, default=10.0,
                    help="grain speed (m/s) the default skin is sized for; the flow's measured "
                         "maximum is ~8 m/s")
    ap.add_argument("--simplify-mm", type=float, default=0.0,
                    help="collapse collider edges shorter than this (mm) before baking.  2 mm "
                         "removes CAD fillet/chamfer detail no 12 mm grain can resolve (Top "
                         "6372 -> 2285 tris) at <= 1.35 mm surface deviation.  0 = exact STL.")
    ap.add_argument("--simplify-tol-mm", type=float, default=None,
                    help="with --simplify-mm: only collapse where the surface moves less than "
                         "this (mm), preserving sharp rims")
    ap.add_argument("--hash-dims", type=int, nargs=3, default=None,
                    help="grain hash-grid table dims (Newton default 128 128 128)")
    ap.add_argument("--checkpoint-at", type=float, default=None,
                    help="save the full solver state to <out>/checkpoint.npz at this sim time; "
                         "tools/dem_bench.py replays steps from it")
    pre, _ = ap.parse_known_args(argv)
    ap.set_defaults(**PRESETS[pre.preset])
    return ap.parse_args(argv)


def build(args, quiet=False):
    """Everything up to the time loop: model, collider, solver, injector.  Shared by main()
    and tools/dem_bench.py so a benchmark steps exactly the machine the run does."""
    import builtins
    _print = builtins.print
    if quiet:
        def print(*a, **k):  # noqa: A001
            pass
    else:
        print = _print  # noqa: A001

    # derive damping from the measured restitution, separately for each reduced mass
    e = max(min(args.restitution, 0.999), 1e-4)
    zeta = -math.log(e) / math.sqrt(math.pi**2 + math.log(e) ** 2)
    kd_pp = args.kd if args.kd is not None else zeta * 2.0 * math.sqrt(0.5 * GRAIN_MASS * args.ke)
    kd_wall = args.kd_wall if args.kd_wall is not None else zeta * 2.0 * math.sqrt(GRAIN_MASS * args.ke)

    casc_mu = args.wall_mu if args.cascade_mu is None else args.cascade_mu

    rate = MASS_FLOW / GRAIN_MASS  # grains per second
    steady_n = 17.6 / GRAIN_MASS
    pool = int(max(8192, steady_n * args.pool_factor))
    frame_dt = 1.0 / args.fps
    substeps = max(1, int(round(frame_dt / args.dt)))
    dt = frame_dt / substeps

    out_dir = args.out or os.path.join(SCRIPT_DIR, "runs", "dem", f"bfa_dem_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    omega = math.sqrt(args.ke / GRAIN_MASS)
    print("=" * 74)
    print("BFA (DEM) -> Newton SolverGranularDEM")
    print("=" * 74)
    print(f"  grain                 r {GRAIN_RADIUS*1e3:.1f} mm, m {GRAIN_MASS*1e3:.4f} g, rho {RHO_GRAIN}")
    print(f"  mass flow             {MASS_FLOW} kg/s -> {rate:,.0f} grains/s")
    print(f"  injection             {INJECTION_SPEED:.4f} m/s down at y = {INJECTION_PLANE_Y} m")
    print(f"  grain-grain friction  slide {args.mu:.3f}  roll {args.mu_roll:.3f}"
          f"   (BFA prj: 0.09 / 0.30 -> {ANGLE_OF_REPOSE} deg repose)")
    print(f"  rotation              {'ON' if not args.no_rotation else 'OFF'}"
          f"   rot damping {args.rot_damp} / {args.rot_damp_wall} wall  (prj RotatingR 0.20)")
    print(f"  wall friction         Spout {args.wall_mu}, cascade {casc_mu}   (BFA: 0.50 all)")
    print(f"  contact ke/kf         {args.ke:.1e} / {args.kf}")
    print(f"  tangential spring     ratio {args.tangential_ratio:.4f} -> k_t "
          f"{args.tangential_ratio * args.ke:.3g} N/m" if args.tangential_ratio > 0
          else "  tangential spring     OFF (viscous-Coulomb)")
    print(f"  restitution           {e:.3f} -> zeta {zeta:.4f} -> "
          f"kd {kd_pp:.3f} grain-grain, {kd_wall:.3f} grain-wall")
    print(f"  dt                    {dt*1e6:.3f} us   dt*sqrt(ke/m) = {dt*omega:.3f}  (stable < 2)")
    print(f"  steps                 {int(args.duration/dt):,}   (BFA: 411,245)")
    print(f"  expected holdup       ~{steady_n:,.0f} grains (17.6 kg);  pool {pool:,}")
    print(f"  output                {out_dir}\n")

    wp.init()
    device = args.device

    parts = [(n, *load_part(n, fx, fl)) for n, ts, fx, fl in COLLIDER_PARTS]
    if args.simplify_mm > 0:
        from mesh_simplify import collapse_short_edges
        parts = [(n, *collapse_short_edges(
            v, f, args.simplify_mm * 1e-3,
            tol=None if args.simplify_tol_mm is None else args.simplify_tol_mm * 1e-3))
            for n, v, f in parts]
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    for name, v, f in parts:
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()),
                               cfg=newton.ModelBuilder.ShapeConfig(mu=args.wall_mu), key=name)
        print(f"  collider {name:<6} {len(f):6,d} tris  {'two-sided' if dict((p[0], p[1]) for p in COLLIDER_PARTS)[name] else 'one-sided'}")

    # --- injection lattice: never uniform random (see module docstring) -----
    v0, e1, e2, _cdf, face_area = injection_face_triangles()
    tris = np.stack([v0, v0 + e1, v0 + e2], axis=1).astype(np.float64)
    site_spacing = GRAIN_RADIUS * 2.2
    sites = lattice_sites(tris, site_spacing)
    sites[:, 1] = INJECTION_PLANE_Y - GRAIN_RADIUS * 1.5
    # Drop sites that would spawn a grain already overlapping a wall.  A site 1 mm inside
    # the face boundary puts the grain 5 mm into the spout wall: 100 N on a 0.9 g mass,
    # a 2.7 m/s kick on the first step.  Injection overlap -- with geometry or with other
    # grains -- is the most destructive mistake available in a DEM setup.
    n_raw = len(sites)
    sites = _clear_of_walls(sites, parts, GRAIN_RADIUS * 1.25, device)
    # emit a subset each event, spaced so consecutive layers clear each other vertically
    batch = max(1, len(sites) // 4)
    inj_interval = batch / rate
    layer_gap = INJECTION_SPEED * inj_interval
    print(f"\n  injection face        {face_area:.4f} m2 -> {len(sites)} of {n_raw} lattice "
          f"sites at {site_spacing*1e3:.1f} mm ({n_raw-len(sites)} dropped for wall clearance)")
    print(f"  batch                 {batch} grains every {inj_interval*1e3:.1f} ms "
          f"-> layers {layer_gap*1e3:.0f} mm apart ({layer_gap/(2*GRAIN_RADIUS):.1f} diameters)\n")

    # Where inactive grains wait.  They are still points in the hash grid, and the grid
    # buckets cells MODULO its table size, so where they sit decides which buckets they
    # pollute.  History: all at one point put 46,000 grains in one bucket that some active
    # grains aliased onto (particle-particle 0.095 -> 16.9 ms, 180x); a wide lattice fixed
    # that but still spread 30,000 grains over the table, costing 60% more contact time
    # the moment the table shrank.  Instead, pick one point whose z-cell RESIDUE no active
    # grain or its neighbour query can ever produce: the domain spans only
    # (DOMAIN_HI.z - DOMAIN_LO.z)/cell + 3 residues of the table's z dimension, so the
    # rest are unreachable.  One bucket then holds every idle grain and is never read.
    skin = args.skin if args.skin is not None else \
        2.0 * args.skin_speed * max(args.neighbor_every, 1) * dt
    skin_max = max(2.0 * skin, 0.006)
    hash_dz = args.hash_dims[2] if args.hash_dims else 128
    # the cell size the solver will actually build its grid with
    uses_list = bool(args.neighbor_every) and not args.no_rotation   # as in SolverGranularDEM.step
    gcell = 2.0 * GRAIN_RADIUS + (skin_max if uses_list else 0.0)
    z0, z1 = int(DOMAIN_LO[2] / gcell) - 1, int(DOMAIN_HI[2] / gcell) + 1
    span = z1 - z0 + 1
    assert span + 2 < hash_dz, f"domain spans {span} z-cells; hash z-dim {hash_dz} too small to hide idle grains"
    void_cell = z1 + 1 + (hash_dz - span) // 2          # middle of the unreachable residues
    park_lo = wp.vec3(float(sites[:, 0].mean()), INJECTION_PLANE_Y + 2.0, (void_cell + 0.5) * gcell)
    for _k in range(pool):
        builder.add_particle(pos=park_lo, vel=wp.vec3(0.0), mass=GRAIN_MASS,
                             radius=GRAIN_RADIUS, flags=0)

    model = builder.finalize(device=device)
    model.particle_ke, model.particle_kd, model.particle_kf = args.ke, kd_pp, args.kf
    model.particle_mu, model.particle_cohesion, model.particle_adhesion = args.mu, 0.0, 0.0
    model.particle_max_velocity = args.max_velocity
    model.set_gravity((0.0, -9.81, 0.0))
    n_pool = model.particle_count

    collider, meshes = build_collider(
        parts,
        two_sided=[p[1] for p in COLLIDER_PARTS],
        friction=[args.wall_mu if n == "Spout" else casc_mu for n, *_ in parts],
        ke=[args.ke] * len(parts), kd=[kd_wall] * len(parts), kf=[args.kf] * len(parts),
        thickness=[args.shell_thickness if p[1] else 0.0 for p in COLLIDER_PARTS],
        max_dist=0.03, device=device)
    if args.neighbor_every:
        print(f"  neighbour list        rebuilt every {args.neighbor_every} steps, skin "
              f"{skin*1e3:.2f} mm (two grains closing at {args.skin_speed:g} m/s)")
    wall_grid = None
    if not args.bvh_walls:
        # reach: contact (radius + shell) plus the 1 mm band the sign test averages over
        reach = GRAIN_RADIUS + args.shell_thickness + 1.0e-3 + 1.0e-4
        _t = time.time()
        wall_grid, winfo = build_wall_grid(parts, meshes, collider.lower, collider.upper,
                                           reach, args.wall_grid_cell, device)
        print(f"  wall grid             {winfo['dims']} cells of {args.wall_grid_cell*1e3:.0f} mm, "
              f"{winfo['occupied']:,} occupied, lists mean {winfo['mean_list']:.1f} / max "
              f"{winfo['max_list']}, {winfo['mbytes']:.0f} MB, baked in {time.time()-_t:.2f} s")
    solver = SolverGranularDEM(model, collider, grid_cell=2.0 * GRAIN_RADIUS, keepalive=meshes,
                               wall_grid=wall_grid,
                               hash_dims=tuple(args.hash_dims) if args.hash_dims else None,
                               neighbor_every=args.neighbor_every, skin=skin,
                               skin_max=skin_max,
                               wall_cache=not args.no_wall_cache,
                               rotation=not args.no_rotation,
                               mu_roll=args.mu_roll, mu_roll_wall=args.mu_roll_wall,
                               rot_damp=args.rot_damp, rot_damp_wall=args.rot_damp_wall,
                               tangential_ratio=args.tangential_ratio,
                               hertz=args.hertz, youngs=args.youngs,
                               poisson=args.poisson, restitution=e,
                               calibrate_restitution=args.calibrate_restitution,
                               cal_dt=args.dt)

    s0, s1 = model.state(), model.state()
    free_idx = wp.array(np.arange(n_pool, dtype=np.int32), dtype=wp.int32, device=device)
    free_count = wp.array([n_pool], dtype=wp.int32, device=device)
    discharged = wp.zeros(1, dtype=wp.int32, device=device)
    stats = wp.zeros(10, dtype=float, device=device)
    spawn_vel = wp.vec3(0.0, -INJECTION_SPEED, 0.0)
    del print
    return argparse.Namespace(**{k: v for k, v in locals().items() if k not in ("_print", "quiet")})


def main():
    args = parse_args()
    S = build(args)
    (model, solver, collider, parts, sites, batch, inj_interval, substeps, dt, frame_dt, n_pool,
     s0, free_idx, free_count, discharged, stats, spawn_vel, park_lo,
     out_dir, device, rate, casc_mu, kd_pp, kd_wall, e, zeta) = (
        S.model, S.solver, S.collider, S.parts, S.sites, S.batch, S.inj_interval, S.substeps,
        S.dt, S.frame_dt, S.n_pool, S.s0, S.free_idx, S.free_count, S.discharged,
        S.stats, S.spawn_vel, S.park_lo, S.out_dir, S.device,
        S.rate, S.casc_mu, S.kd_pp, S.kd_wall, S.e, S.zeta)

    if not args.no_vtk:
        write_geometry_vtk(os.path.join(out_dir, "geometry.vtk"), parts)
    meta = dict(vars(args))
    meta.update(grain_radius=GRAIN_RADIUS, grain_mass=GRAIN_MASS, rate=rate, dt=dt,
                cascade_mu=casc_mu, hertz=args.hertz,
                calibrate_restitution=args.calibrate_restitution,
                youngs=args.youngs if args.hertz else None,
                poisson=args.poisson if args.hertz else None,
                rotation=not args.no_rotation, kd_pp=kd_pp, kd_wall=kd_wall,
                restitution=e, zeta=zeta,
                pool=n_pool, sites=len(sites), batch=batch, solver="SolverGranularDEM")
    json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=2)

    hist = open(os.path.join(out_dir, "history.csv"), "w")
    hist.write("time_s,n_grains,mass_kg,kinetic_energy_J,max_speed_ms,"
               "tube_mass_kg,tube_speed_ms,cascade_mass_kg,cascade_speed_ms,"
               "spin_mean_rads,spin_cascade_rads,discharged_kg,wallclock_s\n")

    n_frames = int(round(args.duration * args.fps))
    rng = np.random.default_rng(0)
    t0 = time.time()
    sim_t = 0.0
    nstep = 0
    saved_ckpt = False
    nbr_fallbacks = 0

    # --- injection schedule, precomputed ------------------------------------------
    # Replays exactly the host logic the loop used to run (float64 sim_t, the same rng
    # calls in the same order), so the grains and sites are the ones earlier runs used.
    # On the device it becomes a per-step table the scheduled spawn kernel reads.
    total_steps = n_frames * substeps
    event_of_step = np.full(total_steps, -1, dtype=np.int32)
    picks = []
    _t_sim, _next = 0.0, 0.0
    for gstep in range(total_steps):
        if _t_sim >= _next:
            _next += inj_interval
            event_of_step[gstep] = len(picks)
            picks.append(rng.choice(len(sites), size=batch, replace=False))
        _t_sim += dt
    w_sites = wp.array(sites.astype(np.float32), dtype=wp.vec3, device=device)
    w_picks = wp.array(np.array(picks, dtype=np.int32).reshape(-1, batch) if picks
                       else np.zeros((1, batch), np.int32), dtype=wp.int32, device=device)
    w_event = wp.array(event_of_step, dtype=wp.int32, device=device)
    step_offset = solver.step_count          # device counter value at global step 0

    def launch_steps(n, eager=True):
        if eager and n:
            solver.request_rebuild()   # graph replays do not advance the host's rebuild phase
        for _ in range(n):
            wp.launch(spawn_scheduled, dim=batch, device=device, inputs=[
                w_sites, w_picks, w_event, solver.step_arr, step_offset, spawn_vel,
                free_idx, free_count, s0.particle_q, s0.particle_qd, model.particle_flags,
                solver.wall_slack])
            wp.launch(_consume_free, dim=1, device=device, inputs=[
                w_event, solver.step_arr, step_offset, batch, free_count])
            solver.step(s0, s0, None, None, dt)     # in place -- see SolverGranularDEM.step

    # --- CUDA graph ---------------------------------------------------------------
    # A step is ~6 kernel launches; from Python each costs ~15-20 us of host time, which
    # at this point exceeds the device time of the smaller kernels.  Capturing K steps
    # into a graph turns K*6 launches into one.  Everything a step needs lives on the
    # device (positions, free list, schedule, step counter), so the graph replays
    # correctly without any host input.
    # A frame is split into floor(substeps/K) replays of a K-step graph plus one replay of
    # a graph for the remainder, so no step runs eagerly.  Each graph starts with a
    # neighbour-list rebuild and rebuilds every N steps inside, so no interval between
    # rebuilds exceeds N whatever K and the remainder are.
    graph, graph_rem, K = None, None, max(0, min(args.graph_steps, substeps))
    if args.neighbor_every:
        K -= K % args.neighbor_every             # whole rebuild cycles per graph
    if K > 0 and device.startswith("cuda"):
        # Capture only RECORDS launches; nothing runs until capture_launch.  Buffers the
        # hash grid would otherwise allocate on first build are reserved up front, and
        # the kernel modules a step uses are loaded before recording starts.
        solver.model.particle_grid.reserve(n_pool)
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

    print(f"{'t (s)':>7} {'grains':>9} {'kg':>7} {'KE (J)':>9} "
          f"{'tube kg':>8} {'tube m/s':>9} {'casc kg':>8} {'casc m/s':>9} "
          f"{'out kg':>8} {'wall':>7}")
    print(f"{'BFA ->':>7} {'19,667':>9} {'17.69':>7} {'154.0':>9} "
          f"{'6.45':>8} {'4.33':>9} {'4.59':>8} {'2.39':>9}")

    try:
        for frame in range(n_frames + 1):
            if frame > 0:
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
                    DOMAIN_LO, DOMAIN_HI, park_lo,
                    free_idx, free_count, discharged])
                # recycle appends with atomics (arbitrary order): re-sort the free list so
                # the next spawns get the same pool slots on every run.  Once per frame.
                _fc = int(free_count.numpy()[0])
                _fi = free_idx.numpy()
                _fi[:_fc] = np.sort(_fi[:_fc])[::-1]
                free_idx.assign(_fi)

            if args.neighbor_every and frame > 0:
                # fallbacks: grain-steps that outran their list and searched exactly instead
                # (harmless, only slower).  Overflows: a grain had more than MAX_NEIGHBORS
                # within cutoff -- that DROPS contacts, so widen nothing and shout.
                nfb = int(solver.nbr_fallbacks.numpy()[0])
                no = int(solver.nbr_overflow.numpy()[0])
                nbr_fallbacks += nfb
                if no:
                    print(f"  !! neighbour list overflow on {no} grain-builds: contacts dropped",
                          flush=True)
                    solver.nbr_overflow.zero_()
                solver.nbr_fallbacks.zero_()
            stats.zero_()
            wp.launch(stats_kernel, dim=n_pool, device=device, inputs=[
                s0.particle_q, s0.particle_qd, model.particle_flags, solver.particle_w,
                GRAIN_MASS, stats])
            st = stats.numpy()
            out_kg = int(discharged.numpy()[0]) * GRAIN_MASS
            tube_v = st[5] / st[4] if st[4] > 0 else 0.0
            casc_v = st[7] / st[6] if st[6] > 0 else 0.0
            spin_all = st[8] / st[0] if st[0] > 0 else 0.0
            spin_c = st[9] / st[6] if st[6] > 0 else 0.0
            wall = time.time() - t0
            hist.write(f"{frame*frame_dt:.4f},{int(st[0])},{st[0]*GRAIN_MASS:.6f},{st[1]:.6f},"
                       f"{st[3]:.4f},{st[4]*GRAIN_MASS:.6f},{tube_v:.4f},"
                       f"{st[6]*GRAIN_MASS:.6f},{casc_v:.4f},{spin_all:.3f},{spin_c:.3f},"
                       f"{out_kg:.6f},{wall:.2f}\n")
            hist.flush()
            if args.checkpoint_at is not None and not saved_ckpt and frame * frame_dt >= args.checkpoint_at:
                saved_ckpt = True
                np.savez(os.path.join(out_dir, "checkpoint.npz"),
                         q=s0.particle_q.numpy(), qd=s0.particle_qd.numpy(),
                         flags=model.particle_flags.numpy(), w=solver.particle_w.numpy(),
                         tang_partner=solver.tang_partner.numpy(),
                         tang_stamp=solver.tang_stamp.numpy(), tang_xi=solver.tang_xi.numpy(),
                         wall_slack=solver.wall_slack.numpy(), free_idx=free_idx.numpy(),
                         free_count=free_count.numpy(), discharged=discharged.numpy(),
                         sim_t=sim_t, nstep=nstep, solver_step=solver.step_count - 1,
                         frame=frame)
                # No rng state: the whole injection schedule is drawn up front from
                # seed 0, so a resume regenerates it and continues at global step nstep.
                print(f"  checkpoint written at t = {sim_t:.4f} s", flush=True)
            print(f"{frame*frame_dt:7.3f} {int(st[0]):9,d} {st[0]*GRAIN_MASS:7.2f} {st[1]:9.2f} "
                  f"{st[4]*GRAIN_MASS:8.2f} {tube_v:9.2f} {st[6]*GRAIN_MASS:8.2f} {casc_v:9.2f} "
                  f"{out_kg:8.2f} {wall:7.0f}   spin {spin_all:6.1f}/{spin_c:6.1f} rad/s",
                  flush=True)

            if not args.no_vtk:
                flags = model.particle_flags.numpy()
                act = np.flatnonzero(flags & int(newton.ParticleFlags.ACTIVE))
                if len(act):
                    write_particles_vtk(os.path.join(out_dir, f"frame_{frame:04d}_particles.vtk"),
                                        frame, s0.particle_q.numpy()[act],
                                        s0.particle_qd.numpy()[act], GRAIN_RADIUS)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        hist.close()
    if args.neighbor_every:
        print(f"neighbour list: {nbr_fallbacks} grain-steps fell back to an exact search")
    print(f"\nwall clock {time.time()-t0:.1f} s   (BFA DEM: 5796 s on 24 CPU cores)")


if __name__ == "__main__":
    main()
