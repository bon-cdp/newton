#!/usr/bin/env python3
"""Measure the angle of repose that OUR contact law produces, for a given (mu, mu_roll).

Why this matters.  BFA's prj lists inter-particle friction 0.090, rolling friction 0.300
AND angle of repose 23.3 deg.  Those first two are not independent material facts -- they
are the numbers that, *in BFA's Hertz-Mindlin contact law*, reproduce the 23.3 deg that was
actually measured in a lab.  Copying 0.090 into a linear spring-dashpot model assumes the
two laws map one-to-one, which they do not: Hertz tangent stiffness scales as sqrt(overlap)
and spans 5.6x across this machine, while a linear ke is one number (see
scratchpad/hertzmap.py).  The transferable quantity is the BULK one, 23.3 deg.

So: pour a heap with our solver and measure its slope.  If our model needs a different mu
to reach 23.3 deg, that is not a fudge -- it is the correct transfer of the same material
through a different contact law.

Protocol: rain grains from a disc onto a rough plate at a constant rate, stop, let settle,
then fit the free surface between 25% and 85% of the heap radius (avoiding the impact
crater at the centre and the single-grain toe at the edge).

Usage:  python angle_of_repose.py --mu 0.09 --mu-roll 0.30
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import warp as wp

import newton
from bfa_dem import GRAIN_MASS, GRAIN_RADIUS, recycle, spawn_at_sites
from granular_dem import SolverGranularDEM, build_collider

BASE_R = 0.20      # circular base; surplus spills off the rim so the heap self-limits
POUR_R = 0.035
POOL = 4000        # live grains; spill is recycled back into it
DROP = 0.10        # release height ABOVE the growing apex, kept small on purpose
# Idle grains all wait at ONE point, so it must hash to a bucket no active query reads.
# The hash grid (12 mm cells, 128 per axis) wraps every 1.536 m; active grains live at
# y in [-0.10, ~0.5] m, cells -9..42, so y-cell 70 (y = 0.846 m) is unreachable.
PARK = wp.vec3(0.0, 70.5 * 0.012, 0.0)


def plate_mesh(radius: float, nseg: int = 128):
    """A single-sided circular base at y = 0 whose normal points up.

    A FIXED base radius is what makes this measurable.  On an unbounded plate the heap
    just keeps spreading and its slope records how much material was poured, not the
    friction -- which is why mu 0.09 and mu 0.22 first gave the same 11 deg.  With a rim,
    surplus falls off and the heap settles at its own angle on a known footprint, so the
    apex height alone gives the angle: tan(theta) = apex / BASE_R."""
    t = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    v = np.vstack([[0.0, 0.0, 0.0],
                   np.stack([radius * np.cos(t), np.zeros(nseg), radius * np.sin(t)], 1)])
    f = np.array([[0, i + 1, 1 + (i + 1) % nseg] for i in range(nseg)], dtype=np.int32)
    n = np.cross(v[f[0, 1]] - v[f[0, 0]], v[f[0, 2]] - v[f[0, 0]])
    if n[1] < 0:
        f = f[:, ::-1]
    return v, np.ascontiguousarray(f)


def pour_sites(radius: float, y: float, spacing: float) -> np.ndarray:
    """A lattice of non-overlapping release points on a disc.

    Random points on a small disc overlap: 14 grains of 12 mm diameter need more area
    than a 35 mm disc has, and a DEM spring turns a 10 mm overlap into a 10 N impulse on
    a 0.9 g grain.  The first version of this file did exactly that and flung grains off
    the plate at 20 m/s.  Sites must be at least one diameter apart, and successive
    releases must be separated by spacing / spawn_speed so a batch clears before the
    next lands on top of it."""
    k = int(math.ceil(radius / spacing))
    g = np.arange(-k, k + 1) * spacing
    xx, zz = np.meshgrid(g, g)
    xx, zz = xx.ravel(), zz.ravel()
    keep = np.hypot(xx, zz) <= radius
    return np.stack([xx[keep], np.full(keep.sum(), y), zz[keep]], axis=1)


def measure(pos: np.ndarray, vel: np.ndarray, radius: float):
    """Return (angle_deg, heap_radius, apex, n_settled) for the heap on the fixed base.

    Two independent estimators, which must agree or the heap is not conical:
      - apex / BASE_R
      - a straight-line fit to the free surface over the middle of the flank
    """
    at_rest = np.linalg.norm(vel, axis=1) < 0.05
    p = pos[at_rest & (pos[:, 1] > -radius) & (pos[:, 1] < 1.0)]
    if len(p) < 200:
        return float("nan"), float("nan"), float("nan"), len(p)
    rr = np.hypot(p[:, 0], p[:, 2])
    # heap radius: 99th percentile, so a few escapees rolling away do not set it
    R = float(np.percentile(rr, 99))
    on_base = rr < BASE_R * 1.02
    core = rr < 0.12 * BASE_R
    h0 = float(np.percentile(p[core][:, 1], 90)) if core.sum() > 10 else float("nan")
    lo, hi = 0.30 * BASE_R, 0.90 * BASE_R
    edges = np.linspace(lo, hi, 13)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (rr >= a) & (rr < b)
        if k.sum() < 8:
            continue
        xs.append(0.5 * (a + b))
        ys.append(np.percentile(p[k][:, 1], 95))   # the surface, not the interior
    if len(xs) < 5:
        return float("nan"), R, h0, len(p)
    slope = np.polyfit(xs, ys, 1)[0]
    return math.degrees(math.atan(-slope)), R, h0, len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.09)
    ap.add_argument("--mu-roll", type=float, default=0.30)
    ap.add_argument("--plate-mu", type=float, default=0.60,
                    help="base friction; must exceed the material's so the heap does not "
                         "slide on the plate instead of shearing internally")
    ap.add_argument("--ke", type=float, default=2.0e4)
    ap.add_argument("--kf", type=float, default=3.0)
    ap.add_argument("--restitution", type=float, default=0.20)
    ap.add_argument("--rot-damp", type=float, default=0.20)
    ap.add_argument("--tangential-ratio", type=float, default=0.0)
    ap.add_argument("--grains", type=int, default=6000,
                    help="total poured; surplus spills off the rim and is recycled")
    ap.add_argument("--rate", type=float, default=800.0, help="grains per second")
    ap.add_argument("--settle", type=float, default=1.0, help="quiet time after the pour (s)")
    ap.add_argument("--dt", type=float, default=2.4316429e-5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    e = max(min(args.restitution, 0.999), 1e-4)
    zeta = -math.log(e) / math.sqrt(math.pi**2 + math.log(e) ** 2)
    kd_pp = zeta * 2.0 * math.sqrt(0.5 * GRAIN_MASS * args.ke)
    kd_wall = zeta * 2.0 * math.sqrt(GRAIN_MASS * args.ke)

    print(f"  mu {args.mu}  mu_roll {args.mu_roll}  plate_mu {args.plate_mu}  "
          f"ke {args.ke:.1e}  e {e}  ->  kd {kd_pp:.3f}/{kd_wall:.3f}")
    print(f"  pouring {args.grains} grains at {args.rate:.0f}/s, then settling {args.settle}s")

    wp.init()
    dev = args.device
    rng = np.random.default_rng(args.seed)

    verts, faces = plate_mesh(BASE_R)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(verts, faces.flatten()),
                           cfg=newton.ModelBuilder.ShapeConfig(mu=args.plate_mu), key="plate")
    # whole pool starts at the unreachable park point (see PARK)
    for _i in range(POOL):
        builder.add_particle(pos=PARK, vel=wp.vec3(0.0), mass=GRAIN_MASS,
                             radius=GRAIN_RADIUS, flags=0)

    model = builder.finalize(device=dev)
    model.particle_ke, model.particle_kd, model.particle_kf = args.ke, kd_pp, args.kf
    model.particle_mu, model.particle_cohesion = args.mu, 0.0
    model.set_gravity((0.0, -9.81, 0.0))

    collider, meshes = build_collider(
        [("plate", verts, faces)], two_sided=[False], friction=[args.plate_mu],
        ke=[args.ke], kd=[kd_wall], kf=[args.kf], thickness=[0.0],
        max_dist=0.03, device=dev)
    solver = SolverGranularDEM(model, collider, grid_cell=2.0 * GRAIN_RADIUS,
                               keepalive=meshes, rotation=True, mu_roll=args.mu_roll,
                               mu_roll_wall=args.mu_roll, rot_damp=args.rot_damp,
                               rot_damp_wall=args.rot_damp,
                               tangential_ratio=args.tangential_ratio)

    free_idx = wp.array(np.arange(POOL, dtype=np.int32), dtype=wp.int32, device=dev)
    free_count = wp.array(np.array([POOL], dtype=np.int32), dtype=wp.int32, device=dev)
    spilled = wp.zeros(1, dtype=int, device=dev)

    s0, s1 = model.state(), model.state()
    spawn_speed = 1.0
    spacing = 2.6 * GRAIN_RADIUS
    sites_np = pour_sites(POUR_R, 0.0, spacing)      # y is set per release, see below
    n_sites = len(sites_np)
    sites = wp.array(sites_np.astype(np.float32), dtype=wp.vec3, device=dev)
    apex = 0.0
    # a batch must have fallen clear of the release plane before the next one appears
    # release interval is the larger of "previous batch has cleared" and the asked rate
    every = max(int(round((spacing / spawn_speed) / args.dt)),
                int(round(n_sites / (args.rate * args.dt))))
    rate = n_sites / (every * args.dt)
    pour_t = args.grains / rate
    total = pour_t + args.settle
    nsteps = int(total / args.dt)
    spawned = 0
    print(f"  {n_sites} release sites at {spacing*1e3:.1f} mm pitch -> {rate:.0f} grains/s, "
          f"pour {pour_t:.2f}s + settle {args.settle}s")

    for step in range(nsteps):
        t = step * args.dt
        if step % every == 0 and spawned < args.grains:
            # Release just above the growing apex.  A fixed high release point turns the
            # pour into a 2.4 m/s jet that craters the top and spreads material, which is
            # what made the first protocol report the same angle for every friction.
            sites_np[:, 1] = apex + DROP
            sites.assign(sites_np.astype(np.float32))
            k = min(n_sites, args.grains - spawned)
            wp.launch(spawn_at_sites, dim=k, device=dev,
                      inputs=[sites, wp.vec3(0.0, -spawn_speed, 0.0), free_idx, free_count,
                              s0.particle_q, s0.particle_qd, model.particle_flags,
                              solver.wall_slack])
            spawned += k
        solver.step(s0, s1, None, None, args.dt)
        s0, s1 = s1, s0
        # Grains that miss the base must be returned to the pool.  Left alone they fall
        # forever (the first version reached y = -77 m), keep costing a hash-grid slot,
        # and exhaust the pool so the pour stops before the heap has converged.
        wp.launch(recycle, dim=model.particle_count, device=dev,
                  inputs=[s0.particle_q, s0.particle_qd, model.particle_flags,
                          wp.vec3(-1.0, -0.10, -1.0), wp.vec3(1.0, 5.0, 1.0),
                          PARK, free_idx, free_count, spilled])
        if step % 2000 == 0:
            qy = s0.particle_q.numpy()
            vy = s0.particle_qd.numpy()
            fl = model.particle_flags.numpy() != 0
            on, ov = qy[fl], vy[fl]
            if len(on):
                still = np.linalg.norm(ov, axis=1) < 0.05
                rr = np.hypot(on[:, 0], on[:, 2])
                core = still & (rr < 0.15 * BASE_R) & (on[:, 1] > -0.01)
                if core.sum() > 10:
                    apex = float(np.percentile(on[core][:, 1], 90))
        if step % 8000 == 0:
            q = s0.particle_q.numpy()
            v = s0.particle_qd.numpy()
            act = model.particle_flags.numpy() != 0
            ang, R, h, n = measure(q[act], v[act], GRAIN_RADIUS)
            pa, va = q[act], v[act]
            sp = np.linalg.norm(va, axis=1) if len(va) else np.zeros(1)
            print(f"    t {t:5.2f}s  spawned {spawned:5d}  active {act.sum():5d}  "
                  f"y {pa[:,1].min():+.3f}..{pa[:,1].max():+.3f}  "
                  f"|v| p50 {np.median(sp):5.2f} p95 {np.percentile(sp,95):5.2f}  "
                  f"rest(<0.05) {(sp<0.05).sum():4d}  rest(<0.2) {(sp<0.2).sum():4d}  "
                  f"angle {ang:6.2f}", flush=True)

    q, v = s0.particle_q.numpy(), s0.particle_qd.numpy()
    act = model.particle_flags.numpy() != 0
    ang, R, h, n = measure(q[act], v[act], GRAIN_RADIUS)
    cone = math.degrees(math.atan(h / BASE_R)) if h == h else float("nan")
    print(f"\n  ANGLE OF REPOSE  {cone:.2f} deg from apex/base   "
          f"({ang:.2f} deg from the flank fit -- these must agree)")
    print(f"  base radius {BASE_R:.3f} m, apex {h:.3f} m ({h/(2*GRAIN_RADIUS):.1f} grain "
          f"diameters), spread to {R:.3f} m, {n} grains settled")
    print(f"  BFA target       23.30 deg")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        # Save the settled heap itself.  The surface fit and the cone-equivalent
        # disagreed by 17 deg on the first run, which means the heap is not a cone and
        # neither single number can be trusted without looking at the radial profile.
        np.savez(os.path.join(args.out, "heap.npz"), pos=q[act], vel=v[act])
        json.dump(dict(mu=args.mu, mu_roll=args.mu_roll, plate_mu=args.plate_mu,
                       ke=args.ke, restitution=e, tangential_ratio=args.tangential_ratio,
                       angle_deg=ang, cone_deg=cone, heap_radius=R, apex=h, settled=n),
                  open(os.path.join(args.out, "repose.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
