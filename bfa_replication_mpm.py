#!/usr/bin/env python3
"""
Newton implicit-MPM replication of the BulkFlowAnalyst (DEM) run in 23087-25sim/.

Everything below is decoded from the BFA project rather than guessed:

  source                                   value
  ---------------------------------------- --------------------------------------
  Material "Small Corn" intrinsic density   62.055188 lb/ft3 = 994.05 kg/m3
  Material bulk density                     44.325134 lb/ft3 = 710.1 kg/m3
  Min_Rad = Max_Rad                         0.23622047 in   = 6.0 mm  (12 mm grain)
  Angle of repose (calibrated)              23.3 deg  -> internal mu = tan = 0.4307
  Inter-particle friction / rolling         0.09 / 0.30   (DEM contact law only)
  Particle-boundary friction (all 7 parts)  0.50
  Cohesion / adhesion / tensile             0.0  (cohesionless)
  Injection FlowRate                        34.171652 short ton/h = 8.6111 kg/s
  Injection face                            InjectionRegionInjFace.stl, y = 3.994 m
  Injection velocity                        (0, -3.132, 0) m/s = -sqrt(2*g*0.5)
  Simulation time / output rate             10.0 s @ 15 fps (150 frames)
  DEM timestep                              2.4316429e-5 s  (411,245 steps)

Verified against the BFA binary output (Local Output/*.por, *.his):
  injection rate  9570.2 particles/s -> 8.6087 kg/s  (0.03% under the requested
                  FlowRate: BFA quantises the stream into whole grains)
  frame-1 translational KE 7.3692 J   (his: 7.3692 J, with rho = 994.05)
  injection velocity  mean -3.1320 m/s   (= -sqrt(2*g*0.5), extrusion length)
  Re-derive all of it with:  python bfa_params_audit.py

MPM mapping notes
-----------------
* Newton's implicit MPM derives the continuum density from
  ``particle_mass / (8 * particle_radius**3)`` -- a material point owns a CUBE
  of side ``2*radius``, not a sphere.  So the material-point mass here is
  ``rho_bulk * spacing**3`` with ``spacing = 2*radius``.  (Using a sphere volume
  understates the density by pi/6 = 0.5236.)
* Internal friction is a Drucker-Prager tan(phi), so the calibrated 23.3 deg
  angle of repose is the right thing to feed it -- NOT the DEM sliding
  coefficient 0.09, which only has meaning together with rolling friction.
* Grain size in MPM is a discretisation, not a physical grain.  ``--grain-scale``
  therefore refines the material-point spacing and the voxel size together.

Usage
-----
  # replication of the BFA run (12 mm grain equivalent)
  python bfa_replication_mpm.py

  # quarter-size grain (4x finer discretisation, 64x the material points)
  python bfa_replication_mpm.py --grain-scale 0.25 --tag quarter

  # quick smoke test
  python bfa_replication_mpm.py --duration 1.0 --substeps 16
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
from newton.solvers import SolverImplicitMPM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BFA_DIR = os.path.join(SCRIPT_DIR, "23087-25sim")

# ---------------------------------------------------------------------------
# BFA-derived constants (SI)
# ---------------------------------------------------------------------------
RHO_BULK = 710.1  # kg/m3, "Material Bulk Density"
RHO_GRAIN = 994.05  # kg/m3, "Material Intr Density" 62.055188 lb/ft3 (reporting only)
GRAIN_DIAMETER = 0.012  # m, 2 * Min_Rad
ANGLE_OF_REPOSE = 23.3  # deg, calibrated in the BFA material
INTERNAL_MU = math.tan(math.radians(ANGLE_OF_REPOSE))  # 0.43067
WALL_MU = 0.5  # particle<->boundary friction, all 7 components
MASS_FLOW = 8.6111  # kg/s = 34.171652 short ton/h (prj FlowRate); the rate BFA
# actually achieved, measured from the .por particle ids, is 8.6087 kg/s -- it
# quantises the stream into whole grains.  The 0.03% gap is not worth chasing.
INJECTION_SPEED = math.sqrt(2.0 * 9.81 * 0.5)  # 3.1321 m/s, matches BFA exactly
INJECTION_PLANE_Y = 3.994  # m, top of Spout / injection face plane
GRAVITY = (0.0, -9.81, 0.0)

# Collider parts.  The mode for each STL was chosen by evaluating the collider
# SDF at the BFA DEM particle cloud -- positions that are known to be free space
# -- and picking the winding/mode with the fewest false "inside" classifications:
#
#   part    one-raw  one-flip  one-fixn  one-fixn+flip   two-sided
#   Spout    0.997     0.004     0.996      0.003          0.354
#   SS       0.451     0.557     0.020      0.980          0.058
#   Def      0.662     0.333     0.001      0.998          0.029
#   Top      0.535     0.475     0.482      0.518          0.217
#   Mid      0.284     0.724     0.786      0.214          0.126
#   Bot      0.404     0.596     0.779      0.221          0.106
#   Baf      0.553     0.454     0.667      0.333          0.086
#
# Spout / SS / Def are cleanly one-sided.  Top / Mid / Bot / Baf are plates with
# material flowing over both faces -- no global winding gets them below 0.21 --
# so they need the two-sided shell.
#   name,   two_sided, fix_normals, flip
COLLIDER_PARTS = [
    ("Spout", False, False, True),
    ("Top", True, False, False),
    ("Mid", True, False, False),
    ("Bot", True, False, False),
    ("SS", False, True, False),
    ("Baf", True, False, False),
    ("Def", False, True, False),
]

# Anything leaving this box is recycled back into the injection pool.
DOMAIN_LO = wp.vec3(-6.30, -2.15, 3.60)
DOMAIN_HI = wp.vec3(-0.90, 5.20, 4.60)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@wp.kernel
def spawn_particles(
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    tri_cdf: wp.array(dtype=float),
    spawn_offset: wp.vec3,
    spawn_vel: wp.vec3,
    seed: int,
    free_idx: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
):
    """Pop a free particle index and place it on the BFA injection face."""
    tid = wp.tid()

    prev = wp.atomic_sub(free_count, 0, 1)
    if prev <= 0:
        # pool exhausted -- undo the decrement and bail out
        wp.atomic_add(free_count, 0, 1)
        return
    idx = free_idx[prev - 1]

    state = wp.rand_init(seed, tid)
    # area-weighted triangle pick
    u = wp.randf(state)
    tri = int(0)
    for t in range(tri_cdf.shape[0]):
        if u > tri_cdf[t]:
            tri = t + 1
    if tri >= tri_v0.shape[0]:
        tri = tri_v0.shape[0] - 1

    # uniform point in the triangle
    a = wp.randf(state)
    b = wp.randf(state)
    if a + b > 1.0:
        a = 1.0 - a
        b = 1.0 - b
    p = tri_v0[tri] + tri_e1[tri] * a + tri_e2[tri] * b

    particle_q[idx] = p + spawn_offset
    particle_qd[idx] = spawn_vel
    particle_flags[idx] = wp.int32(newton.ParticleFlags.ACTIVE)


@wp.kernel
def recycle_particles(
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
    """Deactivate particles that left the domain and return them to the pool."""
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    p = particle_q[i]
    if p[0] >= lo[0] and p[0] <= hi[0] and p[1] >= lo[1] and p[1] <= hi[1] and p[2] >= lo[2] and p[2] <= hi[2]:
        return

    particle_flags[i] = wp.int32(0)
    particle_q[i] = park
    particle_qd[i] = wp.vec3(0.0)
    slot = wp.atomic_add(free_count, 0, 1)
    free_idx[slot] = i
    if p[1] < lo[1]:
        wp.atomic_add(discharged, 0, 1)


# Constant-section part of the inclined chute (0.19 x 0.14 m tube).  DEM runs a
# thin stream through here at ~5 m/s; it is the sharpest single discriminator
# between a correct solution and an over-braked one.
TUBE_LO = 0.0
TUBE_HI = 2.5


@wp.kernel
def accumulate_stats(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: float,
    plane_y: float,
    out: wp.array(dtype=float),
):
    """out = [n, KE, n_below_inlet, max_speed, n_in_tube, speed_sum_in_tube]."""
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    v = particle_qd[i]
    y = particle_q[i][1]
    speed = wp.length(v)
    wp.atomic_add(out, 0, 1.0)
    wp.atomic_add(out, 1, 0.5 * particle_mass * wp.dot(v, v))
    if y < plane_y:
        wp.atomic_add(out, 2, 1.0)
    wp.atomic_max(out, 3, speed)
    if y > TUBE_LO and y < TUBE_HI:
        wp.atomic_add(out, 4, 1.0)
        wp.atomic_add(out, 5, speed)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def load_part(name: str, fix_normals: bool, flip: bool):
    mesh = trimesh.load(os.path.join(BFA_DIR, f"{name}.stl"), force="mesh")
    if fix_normals:
        trimesh.repair.fix_normals(mesh, multibody=True)
    faces = np.asarray(mesh.faces)
    if flip:
        faces = faces[:, ::-1]
    return np.asarray(mesh.vertices, dtype=np.float64), np.ascontiguousarray(faces)


def injection_face_triangles():
    """Triangles of the BFA injection face, plus their area CDF."""
    mesh = trimesh.load(os.path.join(BFA_DIR, "InjectionRegionInjFace.stl"), force="mesh")
    tris = np.asarray(mesh.triangles, dtype=np.float32)
    v0 = tris[:, 0]
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    cdf = np.cumsum(area) / area.sum()
    return v0, e1, e2, cdf.astype(np.float32), float(area.sum())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_geometry_vtk(path, parts):
    """All collider parts as one POLYDATA file with a per-cell part id."""
    verts, faces, part_id, base = [], [], [], 0
    for pid, (_, v, f) in enumerate(parts):
        verts.append(v)
        faces.append(f + base)
        part_id.append(np.full(len(f), pid))
        base += len(v)
    verts = np.vstack(verts)
    faces = np.vstack(faces)
    part_id = np.concatenate(part_id)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nBFA collider geometry\nASCII\nDATASET POLYDATA\n\n")
        fh.write(f"POINTS {len(verts)} float\n")
        fh.writelines(f"{x} {y} {z}\n" for x, y, z in verts)
        fh.write(f"\nPOLYGONS {len(faces)} {len(faces) * 4}\n")
        fh.writelines(f"3 {a} {b} {c}\n" for a, b, c in faces)
        fh.write(f"\nCELL_DATA {len(faces)}\nSCALARS part_id int 1\nLOOKUP_TABLE default\n")
        fh.writelines(f"{p}\n" for p in part_id)


def write_particles_vtk(path, frame, pos, vel, radius, binary=True):
    """Legacy-VTK point cloud (positions, radius, speed, velocity) for ParaView.

    Binary by default.  The ASCII writer formatted ~100k lines through Python per frame,
    ~0.3 s -- once the DEM got fast, that was a large share of a 10 s run.  Binary is a
    few ms and ~3x smaller; read_mpm_frame reads either.
    """
    n = len(pos)
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    speed = np.linalg.norm(vel, axis=1).astype(np.float32)
    if not binary:
        with open(path, "w") as f:
            f.write(f"# vtk DataFile Version 3.0\nMPM particles frame {frame}\nASCII\nDATASET UNSTRUCTURED_GRID\n\n")
            f.write(f"POINTS {n} float\n")
            f.writelines(f"{p[0]} {p[1]} {p[2]}\n" for p in pos)
            f.write(f"\nCELLS {n} {n * 2}\n")
            f.writelines(f"1 {i}\n" for i in range(n))
            f.write(f"\nCELL_TYPES {n}\n")
            f.writelines("1\n" for _ in range(n))
            f.write(f"\nPOINT_DATA {n}\n")
            f.write("SCALARS radius float 1\nLOOKUP_TABLE default\n")
            f.writelines(f"{radius}\n" for _ in range(n))
            f.write("\nSCALARS speed float 1\nLOOKUP_TABLE default\n")
            f.writelines(f"{s}\n" for s in speed)
            f.write("\nVECTORS velocity float\n")
            f.writelines(f"{v[0]} {v[1]} {v[2]}\n" for v in vel)
        return
    be_f = np.dtype(">f4")
    be_i = np.dtype(">i4")
    cells = np.empty((n, 2), dtype=be_i)
    cells[:, 0] = 1
    cells[:, 1] = np.arange(n)
    with open(path, "wb") as f:
        f.write(f"# vtk DataFile Version 3.0\nMPM particles frame {frame}\nBINARY\n"
                f"DATASET UNSTRUCTURED_GRID\nPOINTS {n} float\n".encode())
        f.write(pos.astype(be_f).tobytes())
        f.write(f"\nCELLS {n} {n * 2}\n".encode())
        f.write(cells.tobytes())
        f.write(f"\nCELL_TYPES {n}\n".encode())
        f.write(np.ones(n, dtype=be_i).tobytes())
        f.write(f"\nPOINT_DATA {n}\nSCALARS radius float 1\nLOOKUP_TABLE default\n".encode())
        f.write(np.full(n, radius, dtype=be_f).tobytes())
        f.write(b"\nSCALARS speed float 1\nLOOKUP_TABLE default\n")
        f.write(speed.astype(be_f).tobytes())
        f.write(b"\nVECTORS velocity float\n")
        f.write(vel.astype(be_f).tobytes())
        f.write(b"\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grain-scale", type=float, default=1.0,
                    help="material-point spacing relative to the BFA 12 mm grain (0.25 = quarter size)")
    ap.add_argument("--voxel", type=float, default=None,
                    help="MPM grid voxel size (m); default 0.012 * grain_scale")
    ap.add_argument("--ppc-axis", type=int, default=2,
                    help="material points per voxel per axis (spacing = voxel / ppc_axis)")
    ap.add_argument("--duration", type=float, default=10.0, help="simulated seconds (BFA run is 10 s)")
    ap.add_argument("--fps", type=float, default=15.0, help="output frame rate (BFA run is 15 fps)")
    ap.add_argument("--substeps", type=int, default=None,
                    help="solver substeps per output frame; default sized from a CFL of ~1 voxel at 8 m/s")
    ap.add_argument("--young-modulus", type=float, default=1.0e10)
    ap.add_argument("--poisson-ratio", type=float, default=0.3)
    ap.add_argument("--yield-pressure", type=float, default=1.0e12)
    ap.add_argument("--air-drag", type=float, default=1.0)
    ap.add_argument("--max-iterations", type=int, default=250)
    ap.add_argument("--tolerance", type=float, default=1.0e-5)
    ap.add_argument("--max-velocity-gradient", type=float, default=500.0,
                    help="fork-local APIC C clamp (1/s); <=0 disables it")
    ap.add_argument("--max-grid-velocity-cfl", type=float, default=500.0,
                    help="fork-local grid velocity clamp, in cells per step; <=0 disables it")
    ap.add_argument("--wall-mu", type=float, default=WALL_MU,
                    help="Coulomb friction on the Spout, the long chute whose stream is "
                         "thinner than a grid cell (BFA measures 0.5 for every component, but "
                         "MPM over-brakes a sub-grid stream, so this is an effective value)")
    ap.add_argument("--cascade-mu", type=float, default=None,
                    help="Coulomb friction on the six bottom-cascade components (Top/Mid/Bot/"
                         "SS/Baf/Def).  The material there is dense and grid-resolved, so the "
                         "measured 0.5 needs no correction.  Defaults to --wall-mu.")
    ap.add_argument("--internal-mu", type=float, default=INTERNAL_MU,
                    help="Drucker-Prager tan(phi) (BFA: tan(23.3 deg) = 0.4307)")
    ap.add_argument("--critical-fraction", type=float, default=0.0)
    ap.add_argument("--shell-thickness", type=float, default=0.002,
                    help="physical half-thickness (m) of the two-sided collider plates.  NOT a "
                         "leak fix: applying it to every collider does seal the seams, but a "
                         "3.5 mm offset eats 64%% of the 5.5 mm chute stream and drops the tube "
                         "from 4.27 to 2.60 m/s.  Seal the geometry instead.")
    ap.add_argument("--wall-offset", type=float, default=0.0,
                    help="offset (m) applied to EVERY collider, giving the wall the size the "
                         "material points lack.  MPM collision is point-centre only -- "
                         "particle_radius appears nowhere in rasterized_collisions.py -- so "
                         "points sit at 0 mm from a wall while DEM grain centres are excluded by "
                         "one radius (measured: DEM min 6.0 mm, MPM min 0.0 mm).  A point carries "
                         "a cube of side `spacing`, so the physically correct value is half the "
                         "point spacing, i.e. voxel/(2*ppc_axis).")
    ap.add_argument("--seal-seams", type=float, default=0.008,
                    help="bridge cracks between the separately tessellated STLs up to this "
                         "width (m); 0 disables.  The seven parts do not share vertices and "
                         "leave 2-7 mm gaps that a 12 mm DEM grain cannot pass but a "
                         "dimensionless MPM material point walks straight through.")
    ap.add_argument("--project-max-dist", type=float, default=0.0,
                    help="search radius for project_outside, in voxels; 0 uses the solver default "
                         "of sqrt(3).  Widening it does NOT help -- measured at 4.0 it costs 14%% "
                         "of the chute speed (4.27 -> 3.67 m/s) and worsens the fit 0.073 -> "
                         "0.113, because it exposes more particles to the edge-normal sign test "
                         "in get_average_face_normal, which mis-classifies them near edges and "
                         "projects them spuriously.")
    ap.add_argument("--pool-factor", type=float, default=2.5,
                    help="particle pool size as a multiple of the expected steady-state holdup")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="replication")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-vtk", action="store_true")
    args = ap.parse_args()

    # ---- discretisation -----------------------------------------------------
    voxel = args.voxel if args.voxel else 0.012 * args.grain_scale
    spacing = voxel / args.ppc_axis
    radius = 0.5 * spacing
    point_mass = RHO_BULK * spacing**3

    frame_dt = 1.0 / args.fps
    if args.substeps is None:
        args.substeps = max(1, int(math.ceil(frame_dt / (voxel / 8.0))))
    sim_dt = frame_dt / args.substeps

    points_per_second = MASS_FLOW / point_mass
    # BFA steady-state holdup is ~17.6 kg; size the pool from that.
    steady_points = 17.6 / point_mass
    pool = int(max(4096, steady_points * args.pool_factor))

    out_dir = args.out or os.path.join(
        SCRIPT_DIR, f"bfa_mpm_{args.tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print("BFA (DEM) -> Newton implicit MPM replication")
    print("=" * 74)
    print(f"  grain scale           {args.grain_scale}  (BFA grain {GRAIN_DIAMETER * 1e3:.1f} mm"
          f" -> {GRAIN_DIAMETER * args.grain_scale * 1e3:.2f} mm)")
    print(f"  voxel size            {voxel * 1e3:.2f} mm")
    print(f"  point spacing/radius  {spacing * 1e3:.2f} / {radius * 1e3:.2f} mm"
          f"   ({args.ppc_axis ** 3} points per cell)")
    print(f"  bulk density          {RHO_BULK} kg/m3   point mass {point_mass * 1e3:.4f} g")
    print(f"  internal friction     {args.internal_mu:.4f}"
          f"   (BFA tan({ANGLE_OF_REPOSE} deg) = {INTERNAL_MU:.4f})")
    print(f"  wall friction         Spout {args.wall_mu}, cascade "
          f"{args.wall_mu if args.cascade_mu is None else args.cascade_mu}   (BFA {WALL_MU} for all)")
    print(f"  mass flow             {MASS_FLOW} kg/s -> {points_per_second:,.0f} points/s")
    print(f"  injection velocity    {INJECTION_SPEED:.4f} m/s down, plane y = {INJECTION_PLANE_Y} m")
    print(f"  expected holdup       ~{steady_points:,.0f} points (17.6 kg)")
    print(f"  particle pool         {pool:,}")
    print(f"  dt                    {sim_dt * 1e3:.3f} ms  ({args.substeps} substeps x {args.fps} fps,"
          f" {int(args.duration / sim_dt):,} steps)")
    print(f"  BFA reference dt      0.0243 ms (411,245 steps)  -> {int(args.duration / sim_dt) / 411245:.4f}x")
    print(f"  output                {out_dir}")
    print()

    wp.init()
    device = args.device

    # ---- colliders ----------------------------------------------------------
    parts = []
    for name, two_sided, fix_n, flip in COLLIDER_PARTS:
        v, f = load_part(name, fix_n, flip)
        parts.append((name, v, f))
        print(f"  collider {name:<6} {len(f):6,d} tris  "
              f"{'two-sided, shell %.1f mm' % (args.shell_thickness * 1e3) if two_sided else 'one-sided'}")
    if args.seal_seams > 0.0:
        from build_gasket import build_gasket
        gv, gf = build_gasket({n: (v, f) for n, v, f in parts},
                              max_gap=args.seal_seams, verbose=False)
        if len(gf):
            parts.append(("Gasket", gv, gf))
            print(f"  collider Gasket {len(gf):6,d} tris  two-sided, seam bridge "
                  f"<= {args.seal_seams * 1e3:.1f} mm")

    lo = np.min([v.min(0) for _, v, _ in parts], axis=0)
    hi = np.max([v.max(0) for _, v, _ in parts], axis=0)
    print(f"  geometry bounds       {np.round(lo, 3)} .. {np.round(hi, 3)} m\n")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY[1])
    for name, v, f in parts:
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(v, f.flatten()),
            cfg=newton.ModelBuilder.ShapeConfig(mu=args.wall_mu),
            key=name,
        )

    # ---- particle pool (all inactive; streamed in later) ---------------------
    v0, e1, e2, cdf, face_area = injection_face_triangles()
    park = wp.vec3(float(v0[:, 0].mean()), INJECTION_PLANE_Y + 1.0, float(v0[:, 2].mean()))
    print(f"  injection face area   {face_area:.4f} m2 ({len(v0)} triangles)")
    print(f"  bulk-equivalent curtain thickness at {INJECTION_SPEED:.2f} m/s: "
          f"{MASS_FLOW / RHO_BULK / face_area / INJECTION_SPEED * 1e3:.2f} mm\n")

    dim = int(math.ceil(pool ** (1.0 / 3.0)))
    builder.add_particle_grid(
        pos=wp.vec3(park[0] - 0.5 * dim * spacing, park[1], park[2] - 0.5 * dim * spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim, dim_y=dim, dim_z=dim,
        cell_x=spacing, cell_y=spacing, cell_z=spacing,
        mass=point_mass,
        jitter=0.0,
        radius_mean=radius,
        flags=0,  # inactive: excluded from the grid, the solve and the bounds
    )

    model = builder.finalize(device=device)
    model.particle_mu = args.internal_mu
    model.set_gravity(GRAVITY)
    # Read material parameters from the MPM options, not from the contact model.
    model.particle_ke = None
    model.particle_kd = None
    model.particle_cohesion = None
    model.particle_adhesion = None

    n_pool = model.particle_count
    print(f"  allocated pool        {n_pool:,} particles ({dim}^3)\n")

    # ---- MPM options --------------------------------------------------------
    opts = SolverImplicitMPM.Options()
    opts.voxel_size = voxel
    opts.grid_type = "sparse"
    opts.max_iterations = args.max_iterations
    opts.tolerance = args.tolerance
    opts.young_modulus = args.young_modulus
    opts.poisson_ratio = args.poisson_ratio
    opts.yield_pressure = args.yield_pressure  # no cap on compressive strength
    opts.tensile_yield_ratio = 0.0             # cohesionless (BFA adhesion = 0)
    opts.yield_stress = 0.0                    # cohesionless (BFA cohesion = 0)
    opts.hardening = 0.0
    opts.critical_fraction = args.critical_fraction
    opts.damping = 0.0
    opts.air_drag = args.air_drag
    opts.transfer_scheme = "apic"
    opts.max_velocity_gradient = args.max_velocity_gradient
    opts.max_grid_velocity_cfl = args.max_grid_velocity_cfl

    mpm_model = SolverImplicitMPM.Model(model, opts)

    collider_meshes = []
    for _, v, f in parts:
        pts = wp.array(v, dtype=wp.vec3, device=device)
        collider_meshes.append(
            wp.Mesh(pts, wp.array(f.flatten(), dtype=int, device=device), wp.zeros_like(pts))
        )
    # One-sided parts are true zero-thickness surfaces; two-sided plates get a
    # physical half-thickness rather than a voxel-sized numerical stand-off.
    two_sided = [p[1] for p in COLLIDER_PARTS] + [True] * (len(parts) - len(COLLIDER_PARTS))
    thicknesses = [max(args.wall_offset, args.shell_thickness if ts else 0.0)
                   for ts in two_sided]
    # Per-part friction: the sub-grid correction applies only where the stream is
    # thinner than a cell.  In the cascade the flow is dense and resolved, so the
    # measured coefficient stands.
    cascade_mu = args.wall_mu if args.cascade_mu is None else args.cascade_mu
    frictions = [args.wall_mu if name == "Spout" else cascade_mu for name, *_ in parts]
    mpm_model.setup_collider(
        collider_meshes=collider_meshes,
        collider_thicknesses=thicknesses,
        collider_friction=frictions,
        collider_adhesion=[0.0] * len(parts),
        collider_two_sided=two_sided,
        ground_height=-1.0e9,  # no ground plane; the geometry is the only boundary
    )

    solver = SolverImplicitMPM(mpm_model, opts)
    state_0 = model.state()
    state_1 = model.state()
    solver.enrich_state(state_0)
    solver.enrich_state(state_1)

    # ---- streaming bookkeeping ---------------------------------------------
    free_idx = wp.array(np.arange(n_pool, dtype=np.int32), dtype=wp.int32, device=device)
    free_count = wp.array([n_pool], dtype=wp.int32, device=device)
    discharged = wp.zeros(1, dtype=wp.int32, device=device)
    stats = wp.zeros(6, dtype=float, device=device)

    tri_v0 = wp.array(v0, dtype=wp.vec3, device=device)
    tri_e1 = wp.array(e1, dtype=wp.vec3, device=device)
    tri_e2 = wp.array(e2, dtype=wp.vec3, device=device)
    tri_cdf = wp.array(cdf, dtype=float, device=device)
    # spawn just inside the spout inlet so the point starts on the free side
    spawn_offset = wp.vec3(0.0, -0.5 * voxel, 0.0)
    spawn_vel = wp.vec3(0.0, -INJECTION_SPEED, 0.0)

    if not args.no_vtk:
        write_geometry_vtk(os.path.join(out_dir, "geometry.vtk"), parts)

    meta = dict(vars(args))
    meta.update(voxel=voxel, spacing=spacing, radius=radius, point_mass=point_mass,
                tube_lo=TUBE_LO, tube_hi=TUBE_HI,
                shell_thickness=args.shell_thickness, wall_offset=args.wall_offset,
                sim_dt=sim_dt, pool=n_pool, points_per_second=points_per_second,
                internal_mu=args.internal_mu, wall_mu=args.wall_mu,
                cascade_mu=cascade_mu, rho_bulk=RHO_BULK,
                mass_flow=MASS_FLOW, injection_speed=INJECTION_SPEED,
                injection_face_area=face_area)
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- run ----------------------------------------------------------------
    hist_path = os.path.join(out_dir, "history.csv")
    hist = open(hist_path, "w")
    hist.write("time_s,n_points,mass_kg,mass_below_inlet_kg,kinetic_energy_J,max_speed_ms,"
               "tube_mass_kg,tube_speed_ms,discharged_kg,wallclock_s\n")

    n_frames = int(round(args.duration * args.fps))
    accum = 0.0
    seed = 0
    t0 = time.time()
    total_discharged = 0

    print(f"{'t (s)':>7} {'points':>9} {'kg':>7} {'KE (J)':>9} {'vmax':>6} "
          f"{'tube kg':>8} {'tube m/s':>9} {'out kg':>8} {'wall (s)':>9}")
    try:
        for frame in range(n_frames + 1):
            if frame > 0:
                for sub in range(args.substeps):
                    accum += points_per_second * sim_dt
                    n_spawn = int(accum)
                    if n_spawn > 0:
                        accum -= n_spawn
                        seed += 1
                        wp.launch(spawn_particles, dim=n_spawn, device=device, inputs=[
                            tri_v0, tri_e1, tri_e2, tri_cdf, spawn_offset, spawn_vel, seed,
                            free_idx, free_count,
                            state_0.particle_q, state_0.particle_qd, model.particle_flags,
                        ])

                    solver.step(state_0, state_1, None, None, sim_dt)
                    solver.project_outside(
                        state_1, state_1, sim_dt,
                        max_dist=args.project_max_dist * voxel if args.project_max_dist > 0 else None)
                    state_0, state_1 = state_1, state_0

                    if (sub % 4) == 3 or sub == args.substeps - 1:
                        wp.launch(recycle_particles, dim=n_pool, device=device, inputs=[
                            state_0.particle_q, state_0.particle_qd, model.particle_flags,
                            DOMAIN_LO, DOMAIN_HI, park, free_idx, free_count, discharged,
                        ])

            stats.zero_()
            wp.launch(accumulate_stats, dim=n_pool, device=device, inputs=[
                state_0.particle_q, state_0.particle_qd, model.particle_flags,
                point_mass, INJECTION_PLANE_Y, stats,
            ])
            s = stats.numpy()
            total_discharged = int(discharged.numpy()[0])
            n_free = int(free_count.numpy()[0])
            t = frame * frame_dt
            wall = time.time() - t0
            tube_v = s[5] / s[4] if s[4] > 0 else 0.0
            hist.write(f"{t:.4f},{int(s[0])},{s[0] * point_mass:.6f},{s[2] * point_mass:.6f},"
                       f"{s[1]:.6f},{s[3]:.4f},{s[4] * point_mass:.6f},{tube_v:.4f},"
                       f"{total_discharged * point_mass:.6f},{wall:.2f}\n")
            hist.flush()
            print(f"{t:7.3f} {int(s[0]):9,d} {s[0] * point_mass:7.2f} {s[1]:9.2f} {s[3]:6.2f} "
                  f"{s[4] * point_mass:8.2f} {tube_v:9.2f} {total_discharged * point_mass:8.2f} "
                  f"{wall:9.1f}")

            if n_free == 0:
                print("  !! particle pool exhausted -- increase --pool-factor")

            if not args.no_vtk:
                flags = model.particle_flags.numpy()
                act = np.flatnonzero(flags & int(newton.ParticleFlags.ACTIVE))
                if len(act):
                    write_particles_vtk(
                        os.path.join(out_dir, f"frame_{frame:04d}_particles.vtk"), frame,
                        state_0.particle_q.numpy()[act], state_0.particle_qd.numpy()[act], radius)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        hist.close()

    print(f"\nwall clock {time.time() - t0:.1f} s   (BFA DEM run: 5796 s on 24 CPU cores)")
    print(f"history -> {hist_path}")


if __name__ == "__main__":
    main()
