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
import shutil
import time

import numpy as np
import trimesh
import warp as wp

import newton
from bfa_replication_mpm import write_geometry_vtk, write_particles_vtk
from dem_scenario import INF, Scenario, rayleigh_time
from granular_dem import (SolverGranularDEM, build_collider, build_wall_grid, lattice_sites,
                          set_part_poses)

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
    particle_radius: wp.array(dtype=float),
    site_radius: float,
    lat_origin: wp.vec3,
    lat_u: wp.vec3,
    lat_v: wp.vec3,
    lat_n: wp.vec3,
    lat_spacing: float,
    lat_nu: int,
    lat_nv: int,
    lat_nw: int,
    lat_map: wp.array(dtype=int),
    sites: wp.array(dtype=wp.vec3),
    clear_dist: float,
    reach: int,
    occupied: wp.array(dtype=int),
):
    """On injection steps, stamp every lattice site with a grain closer than clear_dist
    with the event's number + 1 (a stamp, not a flag: nothing has to clear it afterwards).
    Sites form a regular grid (one layer for a face, several for a box), so each grain
    near it checks only the sites within `reach` of it (3x3x3 for one size) -- O(grains),
    not O(grains x sites).  With size classes a small grains' lattice is finer than a big
    grain, so reach grows to ceil(clear_dist / spacing)."""
    i = wp.tid()
    e = _event(event_of_step, step_arr, step_offset)
    if e < 0:
        return
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return
    # with size classes (site_radius > 0) the clearance is this grain's own: a small grain
    # blocks a site only where it would overlap the grain spawned there; clear_dist is the
    # largest such distance and bounds the search
    cd = clear_dist
    if site_radius > 0.0:
        cd = (site_radius + particle_radius[i]) * 1.05
    rel = particle_q[i] - lat_origin
    wn = wp.dot(rel, lat_n)
    if wn <= -clear_dist or wn >= float(lat_nw - 1) * lat_spacing + clear_dist:
        return
    fu = wp.dot(rel, lat_u) / lat_spacing
    fv = wp.dot(rel, lat_v) / lat_spacing
    iu0 = int(wp.floor(fu + 0.5))
    iv0 = int(wp.floor(fv + 0.5))
    iw0 = int(wp.floor(wn / lat_spacing + 0.5))
    for du in range(-reach, reach + 1):
        for dv in range(-reach, reach + 1):
            for dw in range(-reach, reach + 1):
                iu = iu0 + du
                iv = iv0 + dv
                iw = iw0 + dw
                if iu >= 0 and iu < lat_nu and iv >= 0 and iv < lat_nv and iw >= 0 and iw < lat_nw:
                    k = lat_map[(iu * lat_nv + iv) * lat_nw + iw]
                    if k >= 0:
                        if wp.length_sq(particle_q[i] - sites[k]) < cd * cd:
                            occupied[k] = e + 1


INJECT_BLOCK = 256
MIXTURE_INLET_FACTOR = 0.5     # inlet capacity of a size mixture / one size's (see build)


@wp.func
def _spawn(idx: int, site: wp.vec3, spawn_vel: wp.vec3,
           particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3),
           particle_flags: wp.array(dtype=wp.int32), wall_slack: wp.array(dtype=float)):
    particle_q[idx] = site
    particle_qd[idx] = spawn_vel
    particle_flags[idx] = wp.int32(newton.ParticleFlags.ACTIVE)
    wall_slack[idx] = 0.0


@wp.kernel
def inject_event(
    event_of_step: wp.array(dtype=wp.int32),
    step_arr: wp.array(dtype=int),
    step_offset: int,
    picks: wp.array2d(dtype=wp.int32),
    reserve_order: wp.array(dtype=wp.int32),
    reserve_start: wp.array(dtype=wp.int32),
    occupied: wp.array(dtype=int),
    picked: wp.array(dtype=int),
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
    """One block (launch_tiled, INJECT_BLOCK threads), launched every step, exits at once
    unless this step injects.

    Candidates, in order: the event's random picks, then every other site in a fixed
    order rotated per event (the reserve, used when picks are occupied or a backlog is
    owed).  The first batch + backlog free candidates get grains; whatever cannot be
    placed now is carried to the next event, so the delivered mass rate catches up instead
    of being lost.  The block walks the candidates INJECT_BLOCK at a time, ranks the free
    ones with a block scan and stops as soon as enough are placed, so a large box costs
    what it places, not what it holds (a 2.6 M-site box took 1.2 s per event on one
    thread).  Deterministic: the k-th free candidate takes the k-th slot popped from the
    top of the free list, exactly as the one-thread walk did."""
    _blk, lane = wp.tid()
    e = _event(event_of_step, step_arr, step_offset)
    if e < 0:
        return
    stamp = e + 1                        # mark_occupied's stamp for this event
    batch = picks.shape[1]
    n_sites = sites.shape[0]
    want = batch + backlog[0]
    top = int(free_count[0])
    limit = wp.min(want, top)
    placed = int(0)
    base = int(0)
    while base < batch and placed < limit:
        k = base + lane
        s = int(0)
        free = int(0)
        if k < batch:
            s = int(picks[e, k])
            picked[s] = stamp            # the reserve skips these
            if occupied[s] != stamp:
                free = 1
        t = wp.tile(free)
        r = wp.untile(wp.tile_scan_exclusive(t))
        if free == 1 and placed + r < limit:
            _spawn(int(free_idx[top - 1 - placed - r]), sites[s], spawn_vel,
                   particle_q, particle_qd, particle_flags, wall_slack)
        placed = wp.min(placed + int(wp.tile_sum(t)[0]), limit)
        base += INJECT_BLOCK
    # reached only with every pick walked (and stamped in `picked`)
    r0 = int(reserve_start[e])
    base = int(0)
    while base < n_sites and placed < limit:
        j = base + lane
        s = int(0)
        free = int(0)
        if j < n_sites:
            s = int(reserve_order[(r0 + j) % n_sites])
            if picked[s] != stamp and occupied[s] != stamp:
                free = 1
        t = wp.tile(free)
        r = wp.untile(wp.tile_scan_exclusive(t))
        if free == 1 and placed + r < limit:
            _spawn(int(free_idx[top - 1 - placed - r]), sites[s], spawn_vel,
                   particle_q, particle_qd, particle_flags, wall_slack)
        placed = wp.min(placed + int(wp.tile_sum(t)[0]), limit)
        base += INJECT_BLOCK
    if lane == 0:
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
    free_off: wp.array(dtype=wp.int32),
    discharged: wp.array(dtype=wp.int32),
    escaped: wp.array(dtype=wp.int32),
    down_axis: int,
    down_sign: int,
    class_id: wp.array(dtype=wp.int32),
    particle_body: wp.array(dtype=int),
):
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    if particle_body[i] >= 0:
        return                            # clump member: recycle_clumps retires whole bodies
    p = particle_q[i]
    if p[0] >= lo[0] and p[0] <= hi[0] and p[1] >= lo[1] and p[1] <= hi[1] and p[2] >= lo[2] and p[2] <= hi[2]:
        return
    particle_flags[i] = wp.int32(0)
    particle_q[i] = park          # see the note on parking in build()
    particle_qd[i] = wp.vec3(0.0)
    # each size class has its own block of pool slots and its own free list segment
    c = int(class_id[i])                  # counted per size class: mass = count x class mass
    slot = wp.atomic_add(free_count, c, 1)
    free_idx[free_off[c] + slot] = i
    # discharged = left through the domain face on the gravity ("down") side
    out_bottom = p[down_axis] < lo[down_axis]
    if down_sign > 0:
        out_bottom = p[down_axis] > hi[down_axis]
    if out_bottom:
        wp.atomic_add(discharged, c, 1)
    else:
        wp.atomic_add(escaped, c, 1)   # left through a side or the top: a leak, not discharge


@wp.kernel
def spawn_clumps(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_w: wp.array(dtype=wp.vec3),
    wall_slack: wp.array(dtype=float),
    first: wp.array(dtype=int),
    count: wp.array(dtype=int),
    sub_offset: wp.array(dtype=wp.vec3),
    body_x: wp.array(dtype=wp.vec3),
    body_v: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.quat),
    body_L: wp.array(dtype=wp.vec3),
    state: wp.array(dtype=int),
    step_arr: wp.array(dtype=int),
    seed: int,
):
    """Clumps (#10): inject_event spawns a clump as its FIRST member at the site; this
    makes it a body there -- centre at the site, the spawn velocity, a random orientation
    (uniform, Shoemake) -- and places all its members.  Free lists of clump classes hold
    first members only."""
    b = wp.tid()
    if state[b] != 0:
        return
    i0 = first[b]
    if (particle_flags[i0] & newton.ParticleFlags.ACTIVE) == 0:
        return
    x = particle_q[i0]
    v = particle_qd[i0]
    rng = wp.rand_init(seed, b * 7919 + step_arr[0])
    u1 = wp.randf(rng)
    u2 = wp.randf(rng)
    u3 = wp.randf(rng)
    a = wp.sqrt(1.0 - u1)
    c = wp.sqrt(u1)
    q = wp.quat(a * wp.sin(6.2831853 * u2), a * wp.cos(6.2831853 * u2),
                c * wp.sin(6.2831853 * u3), c * wp.cos(6.2831853 * u3))
    R = wp.quat_to_matrix(q)
    body_x[b] = x
    body_v[b] = v
    body_q[b] = q
    body_L[b] = wp.vec3(0.0)
    state[b] = 1
    for k in range(count[b]):
        i = i0 + k
        particle_q[i] = x + R * sub_offset[i]
        particle_qd[i] = v
        particle_w[i] = wp.vec3(0.0)
        particle_flags[i] = wp.int32(newton.ParticleFlags.ACTIVE)
        wall_slack[i] = 0.0


@wp.kernel
def recycle_clumps(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_w: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    hi: wp.vec3,
    park: wp.vec3,
    first: wp.array(dtype=int),
    count: wp.array(dtype=int),
    body_x: wp.array(dtype=wp.vec3),
    state: wp.array(dtype=int),
    class_id: wp.array(dtype=wp.int32),
    free_idx: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    free_off: wp.array(dtype=wp.int32),
    discharged: wp.array(dtype=wp.int32),
    escaped: wp.array(dtype=wp.int32),
    down_axis: int,
    down_sign: int,
):
    """recycle for clumps: a body whose centre has left the domain is retired whole, its
    first member back on its class's free list; counted as its members (each carries its
    share of the clump's mass, so counts x class mass stay exact)."""
    b = wp.tid()
    if state[b] == 0:
        return
    p = body_x[b]
    if p[0] >= lo[0] and p[0] <= hi[0] and p[1] >= lo[1] and p[1] <= hi[1] and p[2] >= lo[2] and p[2] <= hi[2]:
        return
    i0 = first[b]
    for k in range(count[b]):
        i = i0 + k
        particle_flags[i] = wp.int32(0)
        particle_q[i] = park
        particle_qd[i] = wp.vec3(0.0)
        particle_w[i] = wp.vec3(0.0)
    state[b] = 0
    c = int(class_id[i0])
    slot = wp.atomic_add(free_count, c, 1)
    free_idx[free_off[c] + slot] = i0
    out_bottom = p[down_axis] < lo[down_axis]
    if down_sign > 0:
        out_bottom = p[down_axis] > hi[down_axis]
    if out_bottom:
        wp.atomic_add(discharged, c, count[b])
    else:
        wp.atomic_add(escaped, c, count[b])


@wp.kernel
def region_stats(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_w: wp.array(dtype=wp.vec3),
    class_id: wp.array(dtype=wp.int32),
    class_mass: wp.array(dtype=float),
    n_cls: int,
    reg_lo: wp.array(dtype=wp.vec3),
    reg_hi: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """out = [n, KE, sum|w|, max speed, then per region r: n, sum speed, sum |w|, then grains
    per size class in the domain, then per region per class].  Region membership is strict
    (lo < p < hi on every axis)."""
    i = wp.tid()
    if ~particle_flags[i] & wp.int32(newton.ParticleFlags.ACTIVE):
        return
    c = int(class_id[i])
    mass = class_mass[c]
    base = 4 + 3 * reg_lo.shape[0]
    wp.atomic_add(out, base + c, 1.0)
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
            wp.atomic_add(out, base + n_cls * (1 + r) + c, 1.0)


@wp.kernel
def _min_wall_distance(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float,
                       out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if r.result:
        d = wp.length(pts[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v))
        wp.atomic_min(out, i, d)


def clear_of_walls(sites, parts, clearance, device, mask=False):
    """Injection sites at least ``clearance`` from every collider surface (or, with
    mask=True, the boolean mask selecting them)."""
    p = wp.array(sites.astype(np.float32), dtype=wp.vec3, device=device)
    d = wp.full(len(sites), 1.0e9, dtype=float, device=device)
    for _name, v, f in parts:
        m = wp.Mesh(wp.array(np.asarray(v, dtype=np.float32), dtype=wp.vec3, device=device),
                    wp.array(np.asarray(f, dtype=np.int32).flatten(), dtype=int, device=device))
        wp.launch(_min_wall_distance, dim=len(sites), inputs=[m.id, p, clearance * 6.0, d],
                  device=device)
    keep = d.numpy() >= clearance
    return keep if mask else sites[keep]


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
    if kind == "moving":
        return (3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)     # filled per step (set_part_poses)
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


def is_moving(motion) -> bool:
    return bool(motion) and motion.get("type") == "moving"


def _rodrigues(axis, theta):
    """(N,3,3) rotations by angles theta (N,) about one unit axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    st, ct = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3)[None] + st * K[None] + (1.0 - ct) * (K @ K)[None]


def motion_pose(motion, t):
    """Pose of a moving part (#14, BFA's motion manager) at times t (N,): rest geometry x
    maps to R x + T.  motion = {"type": "moving", "steps": [...]}, each step active over
    [start, end] s:

      {"kind": "translate", "start", "end", "velocity": [vx, vy, vz]}     m/s
      {"kind": "rotate", "start", "end", "point": [..], "axis": [..], "deg_per_s": w}

    A rotation turns about a line FIXED IN THE WORLD (as picked on the geometry), at w
    deg/s about the axis (right-hand rule).  Steps compose in list order, each acting on
    the result of the ones before; steps whose windows overlap compose the same way, which
    is exact for translations and close for small steps."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    R = np.repeat(np.eye(3)[None], len(t), axis=0)
    T = np.zeros((len(t), 3))
    for st in motion.get("steps", []):
        t0, t1 = float(st.get("start", 0.0)), float(st.get("end", 0.0))
        tau = np.clip(t - t0, 0.0, max(t1 - t0, 0.0))
        if st.get("kind") == "translate":
            T = T + tau[:, None] * np.asarray(st["velocity"], dtype=np.float64)[None]
        elif st.get("kind") == "rotate":
            Rk = _rodrigues(st["axis"], np.radians(float(st["deg_per_s"])) * tau)
            pt = np.asarray(st["point"], dtype=np.float64)
            R = Rk @ R
            T = np.einsum("nij,nj->ni", Rk, T - pt[None]) + pt[None]
        else:
            raise ValueError(f"unknown motion step kind {st.get('kind')!r}")
    return R, T


def motion_tables(motion, n_steps, dt):
    """Per-step pose (R, T) and wall velocity (v of the point T, angular velocity w) of a
    moving part, for set_part_poses: the motion over each step [t, t + dt] exactly."""
    t = np.arange(n_steps + 1) * dt
    R, T = motion_pose(motion, t)
    dR = np.einsum("nij,nkj->nik", R[1:], R[:-1])           # R(t+dt) R(t)^T
    v = (T[1:] - T[:-1]) / dt
    cos = np.clip((np.trace(dR, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(cos)
    ax = np.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]], 1)
    sn = np.linalg.norm(ax, axis=1)
    w = np.where(sn[:, None] > 1e-15, ax / np.maximum(sn, 1e-30)[:, None] * (ang / dt)[:, None], 0.0)
    return R[:-1], T[:-1], v, w


def describe_motion(motion):
    if is_moving(motion):
        parts = []
        for st in motion.get("steps", []):
            win = f"{st.get('start', 0):g}-{st.get('end', 0):g} s"
            if st.get("kind") == "translate":
                parts.append(f"translate {np.round(st['velocity'], 4).tolist()} m/s {win}")
            else:
                parts.append(f"rotate {st['deg_per_s']:g} deg/s about {np.round(st['axis'], 3).tolist()} "
                             f"through {np.round(st['point'], 3).tolist()} {win}")
        return "moving: " + "; ".join(parts or ["no steps"])
    if motion.get("type") == "belt":
        u = np.asarray(motion["velocity"], dtype=float)
        return f"belt {np.linalg.norm(u):g} m/s along {np.round(u / np.linalg.norm(u), 3).tolist()}"
    om = motion.get("omega", motion.get("rpm", 0.0) * 2.0 * math.pi / 60.0)
    return f"rotating {om:g} rad/s about {motion['axis']} through {motion['point']}"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def injector_box(sc, inj, parts):
    """Frame and size of a box injector (Injector.box), BFA's "injection volume from belt
    reference".  Axes: e1 along the belt's running direction (its horizontal part) or the
    given direction, e2 across (up x e1), e3 up (against gravity).

      belt:  "part" names a belt; "position" is from the belt's upstream end to the box's
             upstream face, "lateral" from the belt's centreline; the belt's surface height
             is found by a ray cast down at the box's centre (troughs, inclines);
             "clearance" is the gap from that surface to the box's bottom.
      plane: a horizontal plane at "plane_height"; "center" [x, y, z] (y ignored) is the
             box centre; "clearance" above the plane.
    Returns dict(origin, e1, e2, e3, length, width, height, velocity or None, describe)."""
    bx = inj.box
    up = -np.asarray(sc.gravity, dtype=np.float64)
    up /= np.linalg.norm(up)
    L, W, H = float(bx["length"]), float(bx["width"]), float(bx["height"])
    clearance = float(bx.get("clearance", 0.0))
    ref = bx.get("reference", "plane")
    velocity = None

    def horiz(d):
        d = np.asarray(d, dtype=np.float64)
        d = d - up * d.dot(up)
        n = np.linalg.norm(d)
        if n < 1e-9:
            raise ValueError(f"injector {inj.name!r}: box direction is vertical")
        return d / n

    if ref == "belt":
        part = next((p for p in sc.parts if p.name == bx.get("part")), None)
        if part is None or not part.motion or part.motion.get("type") != "belt":
            raise ValueError(f"injector {inj.name!r}: box reference {bx.get('part')!r} is not a "
                             f"part with belt motion")
        vel = np.asarray(part.motion["velocity"], dtype=np.float64)
        e1 = horiz(vel)
        e2 = np.cross(up, e1)
        k = [p.name for p in sc.parts].index(part.name)
        v, f = parts[k][1], np.asarray(parts[k][2]).reshape(-1, 3)
        s_up = float((v @ e1).min())
        c_lat = float(0.5 * ((v @ e2).min() + (v @ e2).max()))
        s0 = s_up + float(bx.get("position", 0.0))
        l0 = c_lat + float(bx.get("lateral", 0.0))
        top = float((v @ up).max()) + 1.0
        centre = e1 * (s0 + L / 2) + e2 * l0 + up * top
        hits = _ray_hits(centre, -up, v, f)
        if not len(hits):
            raise ValueError(f"injector {inj.name!r}: no belt surface under the box centre -- "
                             f"check position/lateral against the belt's length and width")
        h_ref = float((hits @ up).max())             # the belt's upper surface there
        origin = e1 * s0 + e2 * (l0 - W / 2) + up * (h_ref + clearance)
        if bx.get("match_belt", True):
            velocity = vel.tolist()
        describe = (f"on belt {part.name!r}, {bx.get('position', 0.0):g} m from its upstream end, "
                    f"{clearance:g} m above its surface")
    else:
        e1 = horiz(bx.get("direction", [1.0, 0.0, 0.0]))
        e2 = np.cross(up, e1)
        c = np.asarray(bx.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        c = c - up * c.dot(up) + up * float(bx.get("plane_height", 0.0))
        origin = c - e1 * L / 2 - e2 * W / 2 + up * clearance
        describe = f"above the plane at height {bx.get('plane_height', 0.0):g} m"
    return dict(origin=origin, e1=e1, e2=e2, e3=up, length=L, width=W, height=H,
                velocity=velocity, describe=describe)


def _ray_hits(o, d, v, f):
    """Points where the ray o + t d (t > 0) crosses the triangles (Moller-Trumbore, numpy;
    one ray against a part, so no acceleration structure or extra dependency is needed)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    w = (q @ d) * inv
    t = (q * e2).sum(1) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0)
    return o + np.outer(t[hit], d)


def face_frame(tris, flip=False):
    """(centroid, e1, e2, n) of a planar injection face: n is the area-weighted face normal
    (reversed with flip), e1 the face's longest in-plane direction, e2 = n x e1."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    nv = np.cross(b - a, c - a).sum(axis=0)
    nl = float(np.linalg.norm(nv))
    if nl < 1e-12:
        raise ValueError("injection face has no area (or its triangles cancel out)")
    n = nv / nl * (-1.0 if flip else 1.0)
    pts = tris.reshape(-1, 3)
    p0 = pts.mean(axis=0)
    q = (pts - p0) - np.outer((pts - p0) @ n, n)
    _u, _s, vt = np.linalg.svd(q, full_matrices=False)
    e1 = vt[0] - (vt[0] @ n) * n
    e1 /= np.linalg.norm(e1)
    return p0, e1, np.cross(n, e1), n


def face_volume_lattice(tris, spacing, depth, flip=False, offset=0.0):
    """Lattice sites filling the volume swept by a planar face along its normal: the face
    extruded by `depth` (m) on its normal side (the other side with flip), starting `offset`
    (m) from the face along the same normal.  Grains need
    room to spawn, which a bare face does not give.  Returns (sites, idx (iu, iv, iw),
    lattice origin, (e1, e2, n), (nu, nv, nw), max off-plane distance of the face)."""
    p0, e1, e2, n = face_frame(tris, flip)
    pts = tris.reshape(-1, 3)
    off_plane = float(np.abs((pts - p0) @ n).max())
    u, v = (pts - p0) @ e1, (pts - p0) @ e2
    gu = np.arange(u.min() + 0.5 * spacing, u.max(), spacing)
    gv = np.arange(v.min() + 0.5 * spacing, v.max(), spacing)
    nw = max(1, int(depth // spacing))
    UU, VV = np.meshgrid(gu, gv, indexing="ij")
    cu, cv = UU.ravel(), VV.ravel()
    inside = np.zeros(cu.size, dtype=bool)
    tu, tv = ((tris - p0) @ e1), ((tris - p0) @ e2)            # (n_tri, 3) in-plane coords
    for k in range(len(tris)):                                   # 2D point-in-triangle
        x0, y0 = tu[k, 0], tv[k, 0]
        d1x, d1y = tu[k, 1] - x0, tv[k, 1] - y0
        d2x, d2y = tu[k, 2] - x0, tv[k, 2] - y0
        den = d1x * d2y - d2x * d1y
        if abs(den) < 1e-18:
            continue
        px, py = cu - x0, cv - y0
        s = (px * d2y - d2x * py) / den
        t = (d1x * py - px * d1y) / den
        inside |= (s >= 0) & (t >= 0) & (s + t <= 1)
    iu, iv = np.unravel_index(np.flatnonzero(inside), UU.shape)
    origin = (p0 + e1 * gu[0] + e2 * gv[0] + n * (offset + 0.5 * spacing)
              if len(gu) and len(gv) else p0)
    idx = np.array([(a, b, w) for w in range(nw) for a, b in zip(iu, iv)], dtype=int).reshape(-1, 3)
    sites = (origin + np.outer(idx[:, 0] * spacing, e1) + np.outer(idx[:, 1] * spacing, e2)
             + np.outer(idx[:, 2] * spacing, n))
    return sites, idx, origin, (e1, e2, n), (len(gu), len(gv), nw), off_plane


def inlet_cross_section(points, velocity):
    """Area (m^2) of an inlet seen along its injection velocity: the points projected onto
    the plane normal to the velocity, convex hull.  A lattice inlet passes at most this area x
    packing x density x speed (for a box moving along its length, this is width x height)."""
    v = np.asarray(velocity, dtype=np.float64)
    sp = float(np.linalg.norm(v))
    if sp < 1e-12:
        return 0.0
    d = v / sp
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ d) * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    q = np.unique(np.round(np.stack([p @ e1, p @ e2], axis=1), 9), axis=0)
    if len(q) < 3:
        return 0.0
    q = q[np.lexsort((q[:, 1], q[:, 0]))]

    def half(pts):                                   # Andrew's monotone chain
        h = []
        for pt in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], pt - h[-2]) <= 0:
                h.pop()
            h.append(pt)
        return h
    hull = np.array(half(q)[:-1] + half(q[::-1])[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def face_volume_triangles(tris, depth, flip=False, offset=0.0):
    """(base, top) triangles of an extruded injection volume, for drawing."""
    _p0, _e1, _e2, n = face_frame(tris, flip)
    return tris + n * offset, tris + n * (offset + depth)


def box_lattice(fr, spacing):
    """Lattice sites filling an injector box: (sites, (iu, iv, iw) per site, lattice origin,
    (u, v, n) axes, (nu, nv, nw))."""
    n = [max(1, int(fr[k] // spacing)) for k in ("length", "width", "height")]
    iu, iv, iw = np.meshgrid(*[np.arange(k) for k in n], indexing="ij")
    idx = np.stack([iu.ravel(), iv.ravel(), iw.ravel()], axis=1)
    # centre the lattice in the box along each axis
    pad = [(fr[k] - (m - 1) * spacing) / 2 for k, m in zip(("length", "width", "height"), n)]
    origin = fr["origin"] + fr["e1"] * pad[0] + fr["e2"] * pad[1] + fr["e3"] * pad[2]
    sites = (origin + np.outer(idx[:, 0] * spacing, fr["e1"]) + np.outer(idx[:, 1] * spacing, fr["e2"])
             + np.outer(idx[:, 2] * spacing, fr["e3"]))
    return sites, idx, origin, (fr["e1"], fr["e2"], fr["e3"]), tuple(n)


def injector_box_corners(sc, inj):
    """8 corners of a box injector, for drawing (loads the parts as build() does)."""
    parts = [(p.name, *load_stl(sc.path(p.stl), p.fix_normals, p.flip, sc.unit_scale))
             for p in sc.parts]
    fr = injector_box(sc, inj, parts)
    o, a, b, c = fr["origin"], fr["e1"] * fr["length"], fr["e2"] * fr["width"], fr["e3"] * fr["height"]
    return [(o + i * a + j * b + k * c).tolist() for i in (0, 1) for j in (0, 1) for k in (0, 1)]


def wall_grid_cell(parts, radius, reach, requested=None, budget=64.0e6):
    """(cell size, note).  The requested size, else the grain radius (>= 6 mm), enlarged
    if the dense cell table would exceed `budget` cells: a 20 m project at 6 mm asked for
    8 billion cells and failed to allocate; larger cells only mean longer candidate lists."""
    cell = requested if requested else max(0.006, radius)
    allv = np.vstack([np.asarray(v) for _n, v, _f in parts])
    ext = allv.max(axis=0) - allv.min(axis=0) + 2.0 * (reach + cell)
    if np.prod(ext / cell) <= budget:
        return float(cell), ""
    new = float((np.prod(ext) / budget) ** (1.0 / 3.0))
    return new, (f"cell {cell*1e3:.0f} mm -> {new*1e3:.0f} mm to stay within "
                 f"{budget/1e6:.0f} M cells ({ext.round(1).tolist()} m geometry)")


def wall_cohesion(m, part) -> float:
    """Grain-wall cohesion (fraction of grain weight) for one part: its own value, else the material's
    wall_cohesion, else the grain-grain cohesion."""
    if part.cohesion is not None:
        return float(part.cohesion)
    return float(m.cohesion if m.wall_cohesion is None else m.wall_cohesion)


def resolve_dt(sc: Scenario):
    """(dt, substeps, effective Young's modulus).  dt divides the frame interval."""
    m, s = sc.material, sc.solver
    youngs_eff = m.youngs / s.youngs_divisor
    if s.dt == "auto":
        # the smallest grains have the shortest contact time
        dt_req = s.rayleigh_fraction * rayleigh_time(m.radius_min, m.density, youngs_eff, m.poisson)
    else:
        dt_req = float(s.dt)
    frame_dt = 1.0 / sc.output.fps
    substeps = max(1, int(round(frame_dt / dt_req)))
    return frame_dt / substeps, substeps, youngs_eff


# Measured on the Quadro P5000 (fast preset): GPU memory per pooled grain (state, neighbour
# list, contact history, hash grid), and the step cost per active grain (32 ns corn, 53 ns
# iron ore) plus a fixed per-step part.  For sizing and warnings, not promises.
BYTES_PER_GRAIN = 960
SEC_PER_GRAIN_STEP = 50e-9
SEC_PER_STEP = 0.25e-3


def _duration(sec):
    return (f"{sec:.0f} s" if sec < 120 else f"{sec/60:.0f} min" if sec < 7200
            else f"{sec/3600:.1f} h")


def holdup_estimate(sc: Scenario):
    """(kg, seconds, how): mass expected in the domain at once.  The scenario's
    solver.expected_holdup_kg if set, else the mass rate times a residence time: along the
    longest belt at its speed plus 3 s for the transfer (as bfa_import does), or 5 s without
    a belt -- never more than the run injects in total."""
    rate = sum(i.mass_rate for i in sc.injectors)
    if sc.solver.expected_holdup_kg is not None:
        return float(sc.solver.expected_holdup_kg), None, "set in the scenario"
    res, how = 5.0, "5 s residence (no belt)"
    for p in sc.parts:
        if p.motion and p.motion.get("type") == "belt":
            vel = np.asarray(p.motion["velocity"], dtype=np.float64)
            spd = float(np.linalg.norm(vel))
            if spd < 0.1:
                continue
            v, _f = load_stl(sc.path(p.stl), p.fix_normals, p.flip, sc.unit_scale)
            length = float(np.ptp(v @ (vel / spd)))
            if length / spd + 3.0 > res or how.startswith("5 s"):
                res, how = length / spd + 3.0, f"{length:.1f} m of belt at {spd:g} m/s + 3 s"
    res = min(res, sc.output.duration)
    return rate * res, res, how


def spheres_per_grain(sc: Scenario) -> float:
    """Mean particles per grain: 1 for spheres, more with clumps (#10)."""
    m = sc.material
    return sum(p * n for (_r, _m, p), n in zip(m.classes(), m.class_spheres()))


def pool_size(sc: Scenario, device=None):
    """(pool, notes): grains to allocate.  Auto-sized from holdup_estimate unless the
    scenario sets expected_holdup_kg, and capped at what the GPU can hold -- an operator-
    screen project at 10,000 t/h of 12 mm grains asked for 3 M grains/s against a 55 k pool
    sized for the old fixed 20 kg default and starved from the first frame."""
    s, gmass = sc.solver, sc.material.mean_mass
    kg, _res, _how = holdup_estimate(sc)
    pool = int(max(s.min_pool, kg / gmass * s.pool_factor))
    notes = []
    if s.expected_holdup_kg is None:
        # the run can never hold more than it injects
        total = sum(i.mass_rate for i in sc.injectors) * sc.output.duration / gmass
        pool = int(max(s.min_pool, min(pool, 1.05 * total + 8192)))
    if device is not None and str(device).startswith("cuda"):
        free = wp.get_device(device).free_memory
        per = BYTES_PER_GRAIN * spheres_per_grain(sc)        # a clump is n particles
        fit = int(max(0.0, free - 1.5e9) / per)
        if pool > fit:
            notes.append(f"pool {pool:,} grains needs {pool*per/1e9:.1f} GB; "
                         f"capped at {fit:,} to fit the GPU's {free/1e9:.1f} GB free")
            pool = max(s.min_pool, fit)
    return pool, notes


def estimate(sc: Scenario):
    """Rough size and cost of a run, from the scenario alone (no GPU): for the operator
    screen's pre-run summary and build()'s log."""
    gmass = sc.material.mean_mass
    dt, _sub, _E = resolve_dt(sc)
    rate = sum(i.mass_rate for i in sc.injectors)
    kg, res, how = holdup_estimate(sc)
    grains = kg / gmass
    step = SEC_PER_STEP + grains * SEC_PER_GRAIN_STEP
    return dict(grains_per_s=rate / gmass, grain_mass=gmass, holdup_kg=kg, holdup_grains=grains,
                residence_s=res, holdup_how=how, dt=dt, steps=int(round(sc.output.duration / dt)),
                memory_gb=pool_size(sc)[0] * BYTES_PER_GRAIN * spheres_per_grain(sc) / 1e9,
                step_ms=step * 1e3, s_per_sim_s=step / dt, run_s=step / dt * sc.output.duration)


def build(sc: Scenario, out_dir: str | None = None, quiet: bool = False):
    """Everything up to the time loop.  Returns a namespace (model, solver, state, pools,
    injection sites...) shared by run() and tools/dem_bench.py."""
    log = (lambda *a, **k: None) if quiet else print
    m, s = sc.material, sc.solver
    # with a size distribution: the largest radius sizes the lattice, grids and clearances,
    # the mean grain mass converts mass rates to grain rates (one size: radius and mass)
    radius, gmass = m.radius_max, m.mean_mass
    classes = m.classes()
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
    est = estimate(sc)
    steady_n = est["holdup_grains"]
    pool, pool_notes = pool_size(sc, s.device)

    out_dir = out_dir or os.path.join(SCRIPT_DIR, "runs", "dem",
                                      f"{sc.name}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    t_r = rayleigh_time(m.radius_min, m.density, youngs_eff, m.poisson)
    log("=" * 74)
    log(f"DEM scenario: {sc.name}  (SolverGranularDEM)")
    log("=" * 74)
    if len(classes) == 1:
        log(f"  grain                 r {radius*1e3:.1f} mm, m {gmass*1e3:.4f} g, rho {m.density:g}")
    else:
        log(f"  grain sizes           {len(classes)} classes, rho {m.density:g}, mean grain "
            f"{gmass*1e3:.4f} g:")
        tot_w = sum(mk * pk for _r, mk, pk in classes)
        for rk, mk, pk in classes:
            log(f"                          r {rk*1e3:6.2f} mm  m {mk*1e3:9.4f} g  "
                f"{100*mk*pk/tot_w:5.1f}% by mass  {100*pk:5.1f}% by number")
    log(f"  mass flow             {total_rate*gmass:g} kg/s -> {total_rate:,.0f} grains/s")
    log(f"  friction              grain-grain slide {m.friction:.3f}  roll {m.rolling_friction:.3f}"
        f";  wall roll {m.wall_rolling_friction:.3f}")
    log(f"  contact               {m.contact}" + (f", E {youngs_eff:.3g} Pa (true {m.youngs:.3g} / "
        f"{s.youngs_divisor:g}), nu {m.poisson}" if hertz else f", ke {m.ke:.3g} N/m")
        + f";  rotation {'ON' if m.rotation else 'OFF'}")
    log(f"  restitution           {e:.3f} -> kd {kd_pp:.3f} grain-grain, {kd_wall:.3f} grain-wall")
    if m.cohesion > 0.0 or any(wall_cohesion(m, p) > 0.0 for p in sc.parts):
        wc = sorted({wall_cohesion(m, p) for p in sc.parts})
        log(f"  cohesion              {m.cohesion:g} x grain weight grain-grain, "
            f"{', '.join(f'{x:g}' for x in wc)} x grain weight grain-wall")
        if s.bvh_walls and any(x > 0.0 for x in wc):
            log("  !! grain-wall cohesion needs the baked wall grid: ignored with bvh_walls")
    log(f"  dt                    {dt*1e6:.3f} us = {dt/t_r:.2f} Rayleigh time"
        f"   ({substeps} steps per frame)")
    log(f"  steps                 {int(round(sc.output.duration / dt)):,}")
    log(f"  expected holdup       ~{steady_n:,.0f} grains ({est['holdup_kg']:.4g} kg, "
        f"{est['holdup_how']});  pool {pool:,}")
    for note in pool_notes:
        log("  !! " + note)
    log(f"  estimate              ~{est['step_ms']:.2f} ms per step at that holdup -> "
        f"~{_duration(est['s_per_sim_s'])} per simulated second")
    if steady_n > 2.0e6:
        log(f"  !! ~{steady_n/1e6:.1f} M grains at once is a very large run: check the mass rate "
            f"and grain size ({est['grains_per_s']:,.0f} grains/s)")
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
    def lattice_for(r_site):
        """Injection sites for grains of radius r_site (one lattice per size class: a
        lattice spaced for the largest grain wastes most of its sites on small ones)."""
        site_spacing = r_site * inj.site_spacing
        spawn_velocity = [float(c) for c in inj.velocity]
        fr = tris = None
        if inj.box:
            # a box of lattice sites above a belt or plane (BFA's "injection volume from belt
            # reference"); see injector_box
            fr = injector_box(sc, inj, parts)
            sites, idx, lat_origin, axes, dims = box_lattice(fr, site_spacing)
            face_area = fr["length"] * fr["width"]
            if fr["velocity"] is not None:
                spawn_velocity = fr["velocity"]
            what = (f"injection box {fr['length']:g} x {fr['width']:g} x {fr['height']:g} m "
                    f"{fr['describe']}")
        elif inj.extrude > 0.0:
            # the face extruded along its normal into a volume, so grains have room to spawn
            tris, face_area = face_triangles(sc.path(inj.face_stl), unit)
            tris = tris + np.asarray(inj.offset, dtype=np.float64)
            sites, idx, lat_origin, axes, dims, off_plane = face_volume_lattice(
                tris, site_spacing, inj.extrude, inj.extrude_flip, inj.extrude_offset)
            if off_plane > 0.5 * site_spacing:
                log(f"  !! injection face is not flat ({off_plane*1e3:.1f} mm off its mean plane): "
                    f"the volume follows the mean plane")
            what = (f"injection face        {face_area:.4f} m2 extruded {inj.extrude:g} m along its "
                    f"{'reversed ' if inj.extrude_flip else ''}normal"
                    + (f" from {inj.extrude_offset:g} m off the face" if inj.extrude_offset else "") + " "
                    f"({', '.join(f'{x:.3f}' for x in axes[2])}), {dims[2]} layer(s)")
        else:
            tris, face_area = face_triangles(sc.path(inj.face_stl), unit)
            sites = lattice_sites(tris, site_spacing)
            if inj.set_coordinate is not None:
                sites[:, int(inj.set_coordinate[0])] = inj.set_coordinate[1]
            sites += np.asarray(inj.offset, dtype=np.float64)
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
            idx = np.stack([iu, iv, np.zeros_like(iu)], axis=1)
            e_ax = np.eye(3)
            lat_origin = np.zeros(3)
            lat_origin[au], lat_origin[av], lat_origin[ax] = o_u, o_v, sites[0, ax]
            axes = (e_ax[au], e_ax[av], e_ax[ax])
            dims = (nu, nv, 1)
            what = f"injection face        {face_area:.4f} m2"
        n_raw = len(sites)
        # Grains outside the domain are recycled as escaped at once, so the domain must hold
        # every site: a box sized after "fit domain" (2 m tall over a belt whose domain top was
        # 0.6 m above it) silently spawned grains only in its bottom 0.6 m.
        pad = 2.0 * site_spacing
        lo_s, hi_s = sites.min(axis=0) - pad, sites.max(axis=0) + pad
        if (lo_s < np.asarray(sc.domain_lo)).any() or (hi_s > np.asarray(sc.domain_hi)).any():
            new_lo = np.minimum(sc.domain_lo, lo_s).round(4).tolist()
            new_hi = np.maximum(sc.domain_hi, hi_s).round(4).tolist()
            log(f"  !! domain {sc.domain_lo} .. {sc.domain_hi} does not hold the injection sites: "
                f"extended to {new_lo} .. {new_hi}")
            sc.domain_lo, sc.domain_hi = new_lo, new_hi
        keep = clear_of_walls(sites, parts, r_site * inj.wall_clearance, device, mask=True)
        sites, idx = sites[keep], idx[keep]
        if not len(sites):
            raise RuntimeError(f"injector {inj.name!r}: every lattice site is inside a wall")
        nu, nv, nw = dims
        lat_map = np.full(nu * nv * nw, -1, dtype=np.int32)
        lat_map[(idx[:, 0] * nv + idx[:, 1]) * nw + idx[:, 2]] = np.arange(len(sites), dtype=np.int32)
        lattice = dict(origin=wp.vec3(*lat_origin), u=wp.vec3(*axes[0]), v=wp.vec3(*axes[1]),
                       n=wp.vec3(*axes[2]), spacing=float(site_spacing), nu=int(nu), nv=int(nv),
                       nw=int(nw), map=wp.array(lat_map, dtype=int, device=device))
        return dict(sites=sites, lattice=lattice, n_raw=n_raw, what=what, spacing=site_spacing,
                    fr=fr, tris=tris, face_area=face_area, spawn_velocity=spawn_velocity)


    rate = inj.mass_rate / gmass
    # clumps (#10): per class None (spheres) or its random shapes; a clump's lattice is
    # spaced for its reach (bounding radius), not its equivalent sphere
    tpl = m.class_templates(s.seed)
    nsph = m.class_spheres()
    any_clump = any(n > 1 for n in nsph)
    if any_clump:
        from clumps import describe as describe_clumps
        for k, t in enumerate(tpl):
            if t is not None:
                log(f"  clumps, class {k}       {describe_clumps(t)}")
    # One injection stream per size class: its own lattice (spaced for its radius), its own
    # rate (its number fraction of the grains) and schedule.  One class: one stream, as before.
    streams = []
    for k, (r_eq, _m_k, p_k) in enumerate(classes):
        r_k = r_eq if tpl[k] is None else max(t.bound_radius for t in tpl[k])
        L = lattice_for(r_k)
        sites, what, n_raw, site_spacing = L["sites"], L["what"], L["n_raw"], L["spacing"]
        fr, tris, face_area, spawn_velocity = L["fr"], L["tris"], L["face_area"], L["spawn_velocity"]
        rate_k = rate * p_k
        # At most batch_fraction of the sites per event, but also no more than max_interval
        # between events: a large face with small grains at a low rate would otherwise inject
        # in big lumps (36,663 sites -> 9,165-grain batches 0.83 s apart delivered 166% of the
        # target in the first second).  Both validated projects already meet the interval
        # (15 and 36 ms), so their batches are unchanged.
        batch = max(1, min(int(len(sites) * inj.batch_fraction), math.ceil(rate_k * inj.max_interval)))
        inj_interval = batch / rate_k
        spawn_speed = float(np.linalg.norm(spawn_velocity))
        if len(classes) > 1:
            log(f"\n  size class {k}          radius {r_eq*1e3:.2f} mm, {rate_k:,.0f} grains/s"
                + ("" if tpl[k] is None else f", clumps of {nsph[k]} reaching {r_k*1e3:.2f} mm"))
        log(f"{'' if len(classes) > 1 else chr(10)}  {what} -> {len(sites)} of {n_raw} lattice "
            f"sites at {site_spacing*1e3:.1f} mm ({n_raw-len(sites)} dropped for wall clearance)")
        log(f"  batch                 {batch} grains every {inj_interval*1e3:.1f} ms -> consecutive "
            f"batches {spawn_speed*inj_interval*1e3:.0f} mm apart "
            f"({spawn_speed*inj_interval/(2*r_k):.1f} diameters)\n")
        if spawn_speed * inj_interval < 2.0 * r_k:
            log("  !! consecutive injection batches overlap: raise the injection speed or lower "
                "batch_fraction")
        streams.append(dict(sites=sites, lattice=L["lattice"], batch=batch, interval=inj_interval,
                            rate=rate_k, radius=r_k))
    sites = np.concatenate([st["sites"] for st in streams])
    # inlet capacity, the same rule for every source (and the UI's): the inlet seen along the
    # injection velocity passes at most cross-section x lattice packing x density x speed
    # With size classes each class has a lattice spaced for its own radius, but the classes
    # are placed independently: the many small grains keep filling the gaps a big one needs,
    # so a mixture passes about half what one size does (measured, 30 + 10 mm 50/50: 100%
    # delivered at 0.4 x the one-size capacity, 60% of the rate at 0.8 x).
    packing = (4.0 / 3.0) * math.pi / inj.site_spacing ** 3
    if len(classes) > 1:
        packing *= MIXTURE_INLET_FACTOR
    if inj.box:
        xs = fr["width"] * fr["height"]
    elif inj.extrude > 0.0:
        xs = inlet_cross_section(np.concatenate(face_volume_triangles(
            tris, inj.extrude, inj.extrude_flip, inj.extrude_offset)), spawn_velocity)
    else:
        xs = inlet_cross_section(tris, spawn_velocity)
    cap = xs * packing * m.density * spawn_speed
    log(f"  inlet capacity        ~{cap:.4g} kg/s through {xs:.4g} m2 at {spawn_speed:g} m/s "
        f"(mass rate {inj.mass_rate:g} kg/s)")
    if cap < inj.mass_rate:
        log("  !! the inlet cannot pass the mass rate: enlarge it (face, extrusion depth or box) "
            "or raise the injection speed -- injection will back up")

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
    # grains per class (contiguous blocks; see below) and the particles they need: a clump
    # of n spheres is n particles
    n_cls = len(classes)
    cls_count = np.array([pool], dtype=np.int64)
    if n_cls > 1:
        cls_count = np.floor(np.array([pk for _r, _m, pk in classes]) * pool).astype(np.int64)
        cls_count[-1] = pool - cls_count[:-1].sum()
    cls_parts = cls_count * np.array(nsph, dtype=np.int64)
    for _k in range(int(cls_parts.sum())):
        builder.add_particle(pos=park_lo, vel=wp.vec3(0.0), mass=gmass, radius=radius, flags=0)

    model = builder.finalize(device=device)
    model.particle_ke, model.particle_kd, model.particle_kf = m.ke, kd_pp, m.kf
    model.particle_mu, model.particle_cohesion, model.particle_adhesion = m.friction, 0.0, 0.0
    model.particle_max_velocity = s.max_velocity
    model.set_gravity(tuple(sc.gravity))
    n_pool = model.particle_count
    # size classes: a contiguous block of pool slots per class, sized by its number
    # fraction, so each class keeps its own free list (its own injection stream pops from
    # it); each slot has its class's radius and mass for good, so the GPU injection,
    # recycling and contact code are unchanged.  A clump class's block holds n particles
    # per clump; its members carry the CLUMP's mass (contact damping sees the body) and
    # its free list holds first members only.  One class of spheres: nothing to set.
    multi = n_cls > 1 or any_clump
    class_host = np.repeat(np.arange(n_cls), cls_parts).astype(np.int32)
    radius_host = np.full(n_pool, radius, dtype=np.float32)
    free_segs = [np.arange(n_pool, dtype=np.int32)]
    count_mass = np.array([mk / nk for (_r, mk, _p), nk in zip(classes, nsph)])  # per counted particle
    clump = None
    if multi:
        rad = np.zeros(n_pool)
        mas = np.zeros(n_pool)
        body = np.full(n_pool, -1, dtype=np.int32)
        sub_off = np.zeros((n_pool, 3))
        first, cnt, b_im, b_ii = [], [], [], []
        free_segs = []
        ps = 0
        for k, ((r_k, m_k, _p), n_k) in enumerate(zip(classes, nsph)):
            nb = int(cls_count[k])
            if tpl[k] is None:
                rad[ps:ps + nb] = r_k
                mas[ps:ps + nb] = m_k
                free_segs.append(np.arange(ps, ps + nb, dtype=np.int32))
            else:
                firsts = ps + n_k * np.arange(nb)
                for b in range(nb):
                    t = tpl[k][b % len(tpl[k])]
                    idx = firsts[b] + np.arange(n_k)
                    rad[idx] = t.radii
                    mas[idx] = t.mass
                    sub_off[idx] = t.offsets
                    body[idx] = len(first)
                    first.append(int(firsts[b]))
                    cnt.append(n_k)
                    b_im.append(1.0 / t.mass)
                    b_ii.append(1.0 / t.inertia)
                free_segs.append(firsts.astype(np.int32))
            ps += int(cls_parts[k])
        model.particle_radius.assign(rad.astype(np.float32))
        model.particle_inv_mass.assign((1.0 / mas).astype(np.float32))
        if getattr(model, "particle_mass", None) is not None:
            model.particle_mass.assign(mas.astype(np.float32))
        radius_host = rad.astype(np.float32)
        if first:
            clump = dict(body=body, first=first, count=cnt, inv_mass=b_im, inv_inertia=b_ii,
                         sub_offset=sub_off)
            log(f"  clumps                {len(first):,} bodies of {int(sum(cnt)):,} spheres in the pool")
    class_id = wp.array(class_host, dtype=wp.int32, device=device)
    class_mass = wp.array(np.asarray(count_mass if multi else [gmass], dtype=np.float32),
                          dtype=float, device=device)

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
    moving_idx = [k for k, p in enumerate(sc.parts) if is_moving(p.motion)]
    moving_grids = []
    if not s.bvh_walls:
        # reach: contact (radius + shell) plus the 1 mm band the sign test averages over
        max_thick = max([p.thickness if p.two_sided else 0.0 for p in sc.parts] + [0.0])
        reach = radius + max_thick + 1.0e-3 + 1.0e-4
        # cell: the requested size, else the grain radius (>= 6 mm); enlarged if the dense
        # cell table would exceed its budget.  A 20 m project at 6 mm asked for 8 billion
        # cells and failed to allocate; larger cells only mean longer candidate lists.
        cell, note = wall_grid_cell(parts, radius, reach, s.wall_grid_cell)
        if note:
            log("  wall grid             " + note)
        s.wall_grid_cell = cell
        _t = time.time()
        if not moving_idx:
            wall_grid, winfo = build_wall_grid(parts, meshes, collider.lower, collider.upper,
                                               reach, cell, device)
        else:
            # moving parts (#14) each get a grid of their own, baked in their rest frame;
            # the main grid holds the static ones
            static_idx = [k for k in range(len(parts)) if k not in moving_idx]
            if not static_idx:
                raise ValueError("every part moves: keep at least one static part (a floor)")
            lo_np, hi_np = collider.lower.numpy(), collider.upper.numpy()

            def subgrid(idx, cell_k):
                return build_wall_grid(
                    [parts[k] for k in idx], [meshes[k] for k in idx],
                    wp.array(lo_np[idx], dtype=wp.vec3, device=device),
                    wp.array(hi_np[idx], dtype=wp.vec3, device=device),
                    reach, cell_k, device, part_ids=idx)
            wall_grid, winfo = subgrid(static_idx, cell)
            for k in moving_idx:
                cell_k, _n = wall_grid_cell([parts[k]], radius, reach, None)
                g_k, info_k = subgrid([k], cell_k)
                moving_grids.append(g_k)
                log(f"  moving part grid      {parts[k][0]}: {info_k['dims']} cells of "
                    f"{cell_k*1e3:.0f} mm in its own frame, {info_k['mbytes']:.0f} MB")
        log(f"  wall grid             {winfo['dims']} cells of {s.wall_grid_cell*1e3:.0f} mm, "
            f"{winfo['occupied']:,} occupied, lists mean {winfo['mean_list']:.1f} / max "
            f"{winfo['max_list']}, {winfo['mbytes']:.0f} MB, baked in {time.time()-_t:.2f} s")
    elif moving_idx:
        raise ValueError("moving parts need the baked wall grid: unset solver.bvh_walls")
    solver = SolverGranularDEM(model, collider, grid_cell=2.0 * radius, keepalive=meshes,
                               wall_grid=wall_grid,
                               hash_dims=tuple(s.hash_dims) if s.hash_dims else None,
                               neighbor_every=s.neighbor_every, skin=skin, skin_max=skin_max,
                               wall_cache=s.wall_cache, rotation=m.rotation,
                               mu_roll=m.rolling_friction, mu_roll_wall=m.wall_rolling_friction,
                               rot_damp=m.rot_damp, rot_damp_wall=m.rot_damp_wall,
                               tangential_ratio=m.tangential_ratio,
                               hertz=hertz, youngs=youngs_eff, poisson=m.poisson, restitution=e,
                               calibrate_restitution=m.calibrate_restitution, cal_dt=dt,
                               cohesion=m.cohesion,
                               wall_cohesion=[wall_cohesion(m, p) for p in sc.parts])
    if clump is not None:
        solver.set_clumps(clump["body"], clump["first"], clump["count"], clump["inv_mass"],
                          clump["inv_inertia"], clump["sub_offset"])
    if moving_idx:
        solver.set_moving(moving_grids,
                          wp.array(np.repeat(np.eye(3, dtype=np.float32)[None], len(moving_idx), 0),
                                   dtype=wp.mat33, device=device),
                          wp.zeros(len(moving_idx), dtype=wp.vec3, device=device))

    s0, s1 = model.state(), model.state()
    free_len = np.array([len(fs) for fs in free_segs], dtype=np.int64)
    free_idx = wp.array(np.concatenate(free_segs).astype(np.int32), dtype=wp.int32, device=device)
    free_count = wp.array(free_len.astype(np.int32), dtype=wp.int32, device=device)   # per class
    free_off_host = np.concatenate([[0], np.cumsum(free_len)[:-1]]).astype(np.int32)
    free_off = wp.array(free_off_host, dtype=wp.int32, device=device)
    discharged = wp.zeros(n_cls, dtype=wp.int32, device=device)    # per size class
    escaped = wp.zeros(n_cls, dtype=wp.int32, device=device)
    injected = wp.zeros(1, dtype=int, device=device)
    starved = wp.zeros(1, dtype=int, device=device)
    for st in streams:
        st["occupied"] = wp.zeros(len(st["sites"]), dtype=int, device=device)
    backlog = wp.zeros(n_cls, dtype=int, device=device)           # per stream
    regs = sc.regions
    reg_lo = wp.array(np.array([r.lo for r in regs] or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3,
                      device=device)
    reg_hi = wp.array(np.array([r.hi for r in regs] or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3,
                      device=device)
    # [n, KE, sum|w|, max speed, 3 per region, then grains per size class: in the domain,
    # then per region] -- counts, so the host turns them into kg exactly
    stats = wp.zeros(4 + 3 * max(len(regs), 1) + n_cls * (1 + len(regs)), dtype=float, device=device)
    spawn_vel = wp.vec3(*[float(c) for c in spawn_velocity])
    return argparse.Namespace(
        sc=sc, model=model, solver=solver, collider=collider, meshes=meshes, parts=parts,
        sites=sites, batch=streams[0]["batch"], inj=inj, streams=streams, rate=rate,
        substeps=substeps, dt=dt, frame_dt=frame_dt, n_pool=n_pool, s0=s0, s1=s1,
        free_idx=free_idx, free_count=free_count, discharged=discharged, escaped=escaped,
        injected=injected, starved=starved, backlog=backlog, free_off=free_off,
        free_off_host=free_off_host,
        reg_lo=reg_lo, reg_hi=reg_hi, stats=stats, spawn_vel=spawn_vel, park_lo=park_lo,
        out_dir=out_dir, device=device, kd_pp=kd_pp, kd_wall=kd_wall, restitution=e, zeta=zeta,
        youngs_eff=youngs_eff, skin=skin, grain_mass=gmass, grain_radius=radius,
        classes=classes, n_cls=n_cls, class_id=class_id, class_mass=class_mass,
        moving_idx=moving_idx, any_clump=clump is not None, multi=multi, nsph=nsph,
        radius_host=radius_host, count_mass=count_mass, free_total=int(free_len.sum()),
        class_host=class_host)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(S, extra_meta: dict | None = None, reference: dict | None = None):
    """The time loop.  ``reference`` optionally maps history columns to target values
    printed under the header (e.g. BFA's steady-state numbers)."""
    sc, model, solver, device = S.sc, S.model, S.solver, S.device
    s0, dt, substeps, batch = S.s0, S.dt, S.substeps, S.batch
    gmass, out_dir, n_pool = S.grain_mass, S.out_dir, S.n_pool
    # mass from per-size-class grain counts (exact; one class: count x grain mass as before)
    n_cls = S.n_cls
    multi = S.multi                     # size classes or clumps: counts x class mass
    cls_m = np.asarray(S.count_mass, dtype=np.float64)     # per counted particle
    cls_r_host = S.radius_host if multi else None
    cls_mass_host = cls_m[S.class_host].astype(np.float32) if multi else None
    nsph = np.asarray(S.nsph, dtype=np.float64)

    def kg(counts):
        counts = np.asarray(counts, dtype=np.float64).reshape(-1)
        return float(counts[0] * gmass) if not multi else float(counts @ cls_m)
    out = sc.output

    if out.vtk:
        write_geometry_vtk(os.path.join(out_dir, "geometry.vtk"), S.parts)
    meta = sc.to_dict()
    # The run keeps its own copy of the STLs (geometry/), so it stays viewable and
    # analysable after its project folder is edited or deleted; run.json points there.
    gdir = os.path.join(out_dir, "geometry")
    os.makedirs(gdir, exist_ok=True)
    copied = {}

    def keep(rel):
        if not rel:
            return rel
        src = sc.path(rel)
        if src not in copied:
            stem, ext = os.path.splitext(os.path.basename(src))
            name, k = stem + ext, 2
            while name in copied.values():
                name, k = f"{stem}_{k}{ext}", k + 1
            shutil.copy2(src, os.path.join(gdir, name))
            copied[src] = name
        return os.path.join("geometry", copied[src])

    for p in meta["parts"]:
        p["stl"] = keep(p["stl"])
    for i in meta["injectors"]:
        i["face_stl"] = keep(i.get("face_stl", ""))
    meta["base_dir"] = os.path.abspath(out_dir)   # where the relative paths above resolve
    meta["source_dir"] = sc.base_dir        # the project the run was made from
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

    # gravity's dominant axis and sign: grains leaving the domain on that side are
    # "discharged", anywhere else "escaped" (default [0, -9.81, 0]: below the domain in y)
    _g = np.asarray(sc.gravity, dtype=np.float64)
    down_axis = int(np.argmax(np.abs(_g)))
    down_sign = 1 if _g[down_axis] > 0.0 else -1
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
    order_rng = np.random.default_rng(sc.solver.seed + 7919)
    step_offset = solver.step_count          # device counter value at global step 0
    for k, st in enumerate(S.streams):
        event_of_step = np.full(total_steps, -1, dtype=np.int32)
        picks = []
        sbatch = st["batch"]
        _t_sim, _next = 0.0, max(S.inj.start, 0.0)
        for gstep in range(total_steps):
            if _t_sim >= _next and _t_sim < S.inj.stop:
                _next += st["interval"]
                event_of_step[gstep] = len(picks)
                picks.append(rng.choice(len(st["sites"]), size=sbatch, replace=False))
            _t_sim += dt
        # Candidate order per event: its random picks first (so an unobstructed event places
        # exactly the grains the plain schedule would), then every other site in a fixed
        # rotating order -- the reserve used when picks are occupied or a backlog is owed.
        # The reserve order comes from its own generator so it never perturbs the picks.
        # Only the picks are stored per event; inject_event rotates the reserve itself (a
        # full candidate table was events x sites: 1.3 GB for a 2.6 M-site box over 5 s).
        n_sites = len(st["sites"])
        order0 = order_rng.permutation(n_sites)
        n_ev = max(len(picks), 1)
        st["w_picks"] = wp.array(np.asarray(picks, dtype=np.int32).reshape(len(picks), sbatch) if picks
                                 else np.zeros((1, sbatch), dtype=np.int32), dtype=wp.int32, device=device)
        st["w_order"] = wp.array(order0.astype(np.int32), dtype=wp.int32, device=device)
        st["w_rstart"] = wp.array((np.arange(n_ev, dtype=np.int64) * sbatch % n_sites).astype(np.int32),
                                  dtype=wp.int32, device=device)
        st["w_sites"] = wp.array(st["sites"].astype(np.float32), dtype=wp.vec3, device=device)
        st["occupied"].zero_()        # stamps are event numbers: start clean on every run
        st["picked"] = wp.zeros(n_sites, dtype=int, device=device)
        st["w_event"] = wp.array(event_of_step, dtype=wp.int32, device=device)
        # a grain closer than this would overlap one spawned here (any class against the
        # largest), searched over enough neighbouring sites to cover that distance
        st["clear"] = (st["radius"] + S.grain_radius) * 1.05
        st["reach"] = max(1, math.ceil(st["clear"] / st["lattice"]["spacing"]))
        off, cnt = int(S.free_off_host[k]), int(S.free_off_host[k + 1]) if k + 1 < n_cls else S.free_total
        st["free_idx"] = S.free_idx[off:cnt]
        st["free_count"] = S.free_count[k:k + 1]
        st["backlog"] = S.backlog[k:k + 1]

    # moving parts (#14): pose and wall velocity per global step, read on the device
    if S.moving_idx:
        tabs = [motion_tables(sc.parts[k].motion, total_steps, dt) for k in S.moving_idx]
        mv_R, mv_T, mv_v, mv_w = (
            wp.array(np.stack([tb[j] for tb in tabs], axis=1).astype(np.float32),
                     dtype=wp.mat33 if j == 0 else wp.vec3, device=device) for j in range(4))
        mv_part = wp.array(np.array(S.moving_idx, dtype=np.int32), dtype=int, device=device)
        for k, tb in zip(S.moving_idx, tabs):
            c0 = np.asarray(S.parts[k][1], dtype=np.float64).mean(axis=0)     # part centroid
            c1 = tb[0][-1] @ c0 + tb[1][-1] if len(tb[1]) else c0
            print(f"  moving part {sc.parts[k].name}: centroid {np.round(c0, 4).tolist()} -> "
                  f"{np.round(c1, 4).tolist()} m by the end", flush=True)

    def launch_steps(n, eager=True):
        if eager and n:
            solver.request_rebuild()   # graph replays do not advance the host's rebuild phase
        for _ in range(n):
            if S.moving_idx:
                wp.launch(set_part_poses, dim=len(S.moving_idx), device=device, inputs=[
                    solver.step_arr, step_offset, mv_R, mv_T, mv_v, mv_w, mv_part,
                    solver.pose_R, solver.pose_T, solver.collider])
            for st in S.streams:
                lat = st["lattice"]
                wp.launch(mark_occupied, dim=n_pool, device=device, inputs=[
                    st["w_event"], solver.step_arr, step_offset, s0.particle_q, model.particle_flags,
                    model.particle_radius, st["radius"] if multi else 0.0, lat["origin"], lat["u"], lat["v"], lat["n"], lat["spacing"], lat["nu"], lat["nv"],
                    lat["nw"], lat["map"], st["w_sites"], st["clear"], st["reach"], st["occupied"]])
                wp.launch_tiled(inject_event, dim=[1], block_dim=INJECT_BLOCK, device=device, inputs=[
                    st["w_event"], solver.step_arr, step_offset, st["w_picks"], st["w_order"],
                    st["w_rstart"], st["occupied"], st["picked"], st["w_sites"],
                    S.spawn_vel, st["free_idx"], st["free_count"], st["backlog"], S.starved,
                    s0.particle_q, s0.particle_qd, model.particle_flags, solver.wall_slack,
                    S.injected])
                if solver.clumps is not None:
                    cl = solver.clumps
                    wp.launch(spawn_clumps, dim=cl["n"], device=device, inputs=[
                        s0.particle_q, s0.particle_qd, model.particle_flags, solver.particle_w,
                        solver.wall_slack, cl["first"], cl["count"], cl["sub_offset"], cl["x"],
                        cl["v"], cl["q"], cl["L"], cl["state"], solver.step_arr,
                        int(sc.solver.seed)])
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
                                flow_kg[k] += (inside.sum() * gmass if not multi else
                                               kg(np.bincount(S.class_host[cross][inside],
                                                              minlength=n_cls)))
                wp.launch(recycle, dim=n_pool, device=device, inputs=[
                    s0.particle_q, s0.particle_qd, model.particle_flags,
                    wp.vec3(*sc.domain_lo), wp.vec3(*sc.domain_hi), S.park_lo,
                    S.free_idx, S.free_count, S.free_off, S.discharged, S.escaped, down_axis,
                    down_sign, S.class_id, solver.body])
                if solver.clumps is not None:
                    cl = solver.clumps
                    wp.launch(recycle_clumps, dim=cl["n"], device=device, inputs=[
                        s0.particle_q, s0.particle_qd, model.particle_flags, solver.particle_w,
                        wp.vec3(*sc.domain_lo), wp.vec3(*sc.domain_hi), S.park_lo,
                        cl["first"], cl["count"], cl["x"], cl["state"], S.class_id,
                        S.free_idx, S.free_count, S.free_off, S.discharged, S.escaped,
                        down_axis, down_sign])
                # recycle appends with atomics (arbitrary order): re-sort the free list so
                # the next spawns get the same pool slots on every run.  Once per frame.
                _fc = S.free_count.numpy()
                _fi = S.free_idx.numpy()
                for _k in range(n_cls):
                    _o = int(S.free_off_host[_k])
                    _fi[_o:_o + _fc[_k]] = np.sort(_fi[_o:_o + _fc[_k]])[::-1]
                S.free_idx.assign(_fi)
                if sc.flow_planes:
                    prev_q = s0.particle_q.numpy()      # after recycling: parked grains moved
                owed_k = S.backlog.numpy()
                owed = int(owed_k.sum())
                max_backlog = max(max_backlog, owed)
                if any(int(o) > st["batch"] for o, st in zip(owed_k, S.streams)):
                    print(f"  !! injection backlog {owed} grains (> one batch): the inlet is "
                          f"choked -- sites stay occupied", flush=True)
                starved = int(S.starved.numpy()[0])
                if starved:
                    print(f"  !! particle pool exhausted: {starved} grains not injected -- raise "
                          f"solver.expected_holdup_kg or pool_factor (or lower the mass rate)", flush=True)
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
                S.class_id, S.class_mass, n_cls, S.reg_lo, S.reg_hi, S.stats])
            st = S.stats.numpy()
            n = st[0]
            cb = 4 + 3 * max(len(regs), 1)            # start of the per-class counts
            held_kg = n * gmass if not multi else kg(st[cb:cb + n_cls])
            if S.any_clump:
                n = float((st[cb:cb + n_cls] / nsph).sum())     # grains, not member spheres
            out_kg = kg(S.discharged.numpy()) if multi else int(S.discharged.numpy()[0]) * gmass
            wall = time.time() - t0
            row = [f"{frame*S.frame_dt:.4f}", f"{int(n)}", f"{held_kg:.6f}", f"{st[1]:.6f}",
                   f"{st[3]:.4f}"]
            show = []
            for k, r in enumerate(regs):
                c, sv = st[4 + 3 * k], st[5 + 3 * k]
                v = sv / c if c > 0 else 0.0
                rk = c * gmass if not multi else kg(st[cb + n_cls * (1 + k):cb + n_cls * (2 + k)])
                row += [f"{rk:.6f}", f"{v:.4f}"]
                show.append(f"{rk:11.2f} {v:5.2f}")
            row.append(f"{(st[2] / n if n > 0 else 0.0):.3f}")
            for k, r in enumerate(regs):
                if r.spin:
                    c = st[4 + 3 * k]
                    row.append(f"{(st[6 + 3 * k] / c if c > 0 else 0.0):.3f}")
            esc_kg = kg(S.escaped.numpy()) if multi else int(S.escaped.numpy()[0]) * gmass
            # grains only enter by injection: with several sizes, injected = held + out + escaped
            inj_kg = (int(S.injected.numpy()[0]) * gmass if not multi
                      else held_kg + out_kg + esc_kg)
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
            print(f"{frame*S.frame_dt:7.3f} {int(n):9,d} {held_kg:8.2f} {st[1]:10.2f} "
                  + " ".join(show) + f" {out_kg:9.2f} {wall:6.0f}", flush=True)

            if out.vtk:
                flags = model.particle_flags.numpy()
                act = np.flatnonzero(flags & int(newton.ParticleFlags.ACTIVE))
                if len(act):
                    write_particles_vtk(os.path.join(out_dir, f"frame_{frame:04d}_particles.vtk"),
                                        frame, s0.particle_q.numpy()[act],
                                        s0.particle_qd.numpy()[act],
                                        S.grain_radius if not multi else cls_r_host[act],
                                        spin=solver.particle_w.numpy()[act], ids=act,
                                        mass=cls_mass_host[act] if S.any_clump else None,
                                        body=solver.body.numpy()[act] if S.any_clump else None)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        hist.close()
    if s.neighbor_every:
        print(f"neighbour list: {nbr_fallbacks} grain-steps fell back to a direct search")
    # mass audit: scheduled vs delivered injection, and where the mass went
    t_end = min(sim_t, S.inj.stop) - max(S.inj.start, 0.0)
    target = S.inj.mass_rate * max(t_end, 0.0)
    if not multi:
        inj_kg = int(S.injected.numpy()[0]) * gmass
        held = int(S.stats.numpy()[0]) * gmass
        out_kg = int(S.discharged.numpy()[0]) * gmass
        esc_kg = int(S.escaped.numpy()[0]) * gmass
    else:
        cb = 4 + 3 * max(len(regs), 1)
        held = kg(S.stats.numpy()[cb:cb + n_cls])
        out_kg, esc_kg = kg(S.discharged.numpy()), kg(S.escaped.numpy())
        inj_kg = held + out_kg + esc_kg
    print(f"mass audit: injected {inj_kg:.2f} kg vs target {target:.2f} kg "
          f"({S.inj.mass_rate:g} kg/s x {t_end:.3f} s; {100*inj_kg/max(target,1e-12):.2f}%)"
          f";  held {held:.2f} + discharged {out_kg:.2f} + escaped {esc_kg:.2f} = "
          f"{held+out_kg+esc_kg:.2f} kg;  injection backlog max {max_backlog}, "
          f"end {int(S.backlog.numpy().sum())} grains")
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
