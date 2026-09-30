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
import math
import os

import numpy as np  # noqa: F401  (kept for tools importing it via this module)
import warp as wp

import newton
from bfa_replication_mpm import (
    BFA_DIR, COLLIDER_PARTS, DOMAIN_HI, DOMAIN_LO, INJECTION_PLANE_Y, INJECTION_SPEED,
    MASS_FLOW, RHO_GRAIN, WALL_MU,
)
import dem_run
from dem_run import recycle  # noqa: F401  (re-exported for angle_of_repose.py)
from dem_scenario import INF, Injector, Material, Output, Part, Region, Scenario, Solver

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


def scenario_from_args(args) -> Scenario:
    """The corn-spout machine as a scenario, from this script's flags and presets.  Every
    value maps to what the pre-scenario runner used, so results are bit-identical."""
    casc_mu = args.wall_mu if args.cascade_mu is None else args.cascade_mu
    mat = Material(
        name="corn (BFA 23087)", radius=GRAIN_RADIUS, density=RHO_GRAIN,
        youngs=args.youngs, poisson=args.poisson, restitution=args.restitution,
        friction=args.mu, rolling_friction=args.mu_roll, wall_rolling_friction=args.mu_roll_wall,
        rot_damp=args.rot_damp, rot_damp_wall=args.rot_damp_wall,
        tangential_ratio=args.tangential_ratio, contact="hertz" if args.hertz else "linear",
        ke=args.ke, kf=args.kf, kd=args.kd, kd_wall=args.kd_wall,
        calibrate_restitution=args.calibrate_restitution, rotation=not args.no_rotation)
    parts = [Part(name=n, stl=os.path.join(BFA_DIR, f"{n}.stl"), two_sided=ts, fix_normals=fx,
                  flip=fl, thickness=args.shell_thickness if ts else 0.0,
                  friction=args.wall_mu if n == "Spout" else casc_mu)
             for n, ts, fx, fl in COLLIDER_PARTS]
    inj = Injector(name="spout inlet", face_stl=os.path.join(BFA_DIR, "InjectionRegionInjFace.stl"),
                   mass_rate=MASS_FLOW, velocity=[0.0, -INJECTION_SPEED, 0.0],
                   set_coordinate=[1, INJECTION_PLANE_Y - GRAIN_RADIUS * 1.5])
    regions = [Region("tube", lo=[-INF, TUBE_LO, -INF], hi=[INF, TUBE_HI, INF]),
               Region("cascade", hi=[INF, CASCADE_HI, INF], spin=True)]
    solver = Solver(
        dt=args.dt, neighbor_every=args.neighbor_every, skin=args.skin,
        skin_speed=args.skin_speed, wall_grid_cell=args.wall_grid_cell,
        bvh_walls=args.bvh_walls, wall_cache=not args.no_wall_cache,
        graph_steps=args.graph_steps, hash_dims=args.hash_dims,
        max_velocity=args.max_velocity, expected_holdup_kg=17.6, pool_factor=args.pool_factor,
        simplify_mm=args.simplify_mm, simplify_tol_mm=args.simplify_tol_mm, device=args.device)
    output = Output(duration=args.duration, fps=args.fps, vtk=not args.no_vtk,
                    checkpoint_at=args.checkpoint_at)
    return Scenario(name="bfa_dem", material=mat, parts=parts, injectors=[inj],
                    domain_lo=[float(c) for c in DOMAIN_LO], domain_hi=[float(c) for c in DOMAIN_HI],
                    regions=regions, solver=solver, output=output,
                    notes={"source": "bfa_dem.py", "preset": args.preset})


def build(args, quiet=False):
    """Scenario from the flags, then dem_run.build.  Kept as the entry point for
    tools/dem_bench.py and tools/wall_grid_check.py."""
    return dem_run.build(scenario_from_args(args), args.out, quiet=quiet)


# BFA's steady-state numbers, printed under the header for comparison
BFA_REFERENCE = {"mass_kg": 17.69, "kinetic_energy_J": 154.0, "tube_mass_kg": 6.45,
                 "tube_speed_ms": 4.33, "cascade_mass_kg": 4.59, "cascade_speed_ms": 2.39}


def main():
    args = parse_args()
    S = build(args)
    dem_run.run(S, extra_meta={"cli": vars(args)}, reference=BFA_REFERENCE)


if __name__ == "__main__":
    main()
