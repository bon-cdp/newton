#!/usr/bin/env python3
"""
Air entrainment and dust transport from a finished DEM run (one-way coupling).

    python dem_air.py runs/dem/<run> [--window T0 T1] [--cell 0.1] [--scenario S.json]
    python dem_analyze.py runs/dem/<run> dust        # the same, from the analysis tool

The grains drive the air; the air does not act back on them (the DEM already ran).  It
answers "how much air does the falling stream pull along, where does it come in and go
out, and where does the dust it carries end up" -- for comparing designs, not (yet) for
absolute dust mass, which needs an emission factor measured for the material.

Air: incompressible flow on a uniform staggered (MAC) grid around the grains and parts.
  * drag from every grain each step, implicitly: a grain of radius r moving at w relative
    to the air pushes it with 1/2 rho_air Cd(Re) pi r^2 |w| w (Schiller-Naumann Cd, single
    sphere: no shielding inside a dense stream, so the stream's pull is an upper bound);
  * semi-Lagrangian advection, a constant eddy viscosity, pressure projection by red-black
    SOR (warm-started; it is a quasi-steady flow);
  * walls are the scenario's parts voxelised into solid cells (each part's own cells, so
    staged deflectors switch on and off with their active windows); belt cells carry the
    belt's velocity;
  * the box's six sides are open (pressure 0): air is drawn in and pushed out wherever the
    stream needs, and the summary reports how much crosses each side.

Dust: Lagrangian parcels.
  * released where grains dissipate energy -- impacts and sliding -- in proportion to it:
    for each grain matched by id between frames, D = KE0 + PE0 - KE1 - PE1 (zero in free
    fall and on the belt);
  * three sizes (default 10, 30 and 75 um); each relaxes to the local air velocity plus its
    settling velocity (Stokes time, Schiller-Naumann corrected) and random-walks with the
    eddy diffusivity;
  * a parcel entering a solid cell deposits on that part; one leaving the box escapes
    through that side.
  Parcels carry a mass only through --emission (g of dust per kJ dissipated), which is a
  material property nobody has measured here: fractions are the result, masses are
  illustrative.

Outputs (run/analysis/): air_dust.json (summary), air_dust.npz (time-mean air field,
solid mask, dust parcels per frame for the operator screen), air_mean.vtk (ParaView).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import warp as wp

from compare_bfa_mpm import read_frame

wp.set_module_options({"enable_backward": False})

RHO_AIR = 1.2          # kg/m3
NU_AIR = 1.5e-5        # m2/s, molecular
MU_AIR = RHO_AIR * NU_AIR

FACES = ["-x", "+x", "-y", "+y", "-z", "+z"]


# ---------------------------------------------------------------------------
# grid sampling
# ---------------------------------------------------------------------------

@wp.func
def _trilerp(a: wp.array3d(dtype=float), fx: float, fy: float, fz: float):
    """Trilinear sample of a in index coordinates, clamped to the array."""
    nx, ny, nz = a.shape[0], a.shape[1], a.shape[2]
    fx = wp.clamp(fx, 0.0, float(nx - 1))
    fy = wp.clamp(fy, 0.0, float(ny - 1))
    fz = wp.clamp(fz, 0.0, float(nz - 1))
    i0 = wp.min(int(fx), nx - 2)
    j0 = wp.min(int(fy), ny - 2)
    k0 = wp.min(int(fz), nz - 2)
    i0 = wp.max(i0, 0)
    j0 = wp.max(j0, 0)
    k0 = wp.max(k0, 0)
    tx = wp.clamp(fx - float(i0), 0.0, 1.0)
    ty = wp.clamp(fy - float(j0), 0.0, 1.0)
    tz = wp.clamp(fz - float(k0), 0.0, 1.0)
    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)
    c00 = a[i0, j0, k0] * (1.0 - tx) + a[i1, j0, k0] * tx
    c10 = a[i0, j1, k0] * (1.0 - tx) + a[i1, j1, k0] * tx
    c01 = a[i0, j0, k1] * (1.0 - tx) + a[i1, j0, k1] * tx
    c11 = a[i0, j1, k1] * (1.0 - tx) + a[i1, j1, k1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.func
def _air_at(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
            origin: wp.vec3, h: float, p: wp.vec3):
    """Air velocity at a point (each staggered component sampled at its own faces)."""
    g = (p - origin) / h
    return wp.vec3(_trilerp(u, g[0], g[1] - 0.5, g[2] - 0.5),
                   _trilerp(v, g[0] - 0.5, g[1], g[2] - 0.5),
                   _trilerp(w, g[0] - 0.5, g[1] - 0.5, g[2]))


@wp.func
def _cell_of(origin: wp.vec3, h: float, p: wp.vec3):
    g = (p - origin) / h
    return wp.vec3i(int(wp.floor(g[0])), int(wp.floor(g[1])), int(wp.floor(g[2])))


@wp.func
def _cd(re: float):
    """Single-sphere drag coefficient (Schiller-Naumann, Newton regime above Re 1000)."""
    if re < 1.0e-6:
        return 0.0
    if re < 1000.0:
        return 24.0 / re * (1.0 + 0.15 * wp.pow(re, 0.687))
    return 0.44


# ---------------------------------------------------------------------------
# air kernels
# ---------------------------------------------------------------------------

@wp.kernel
def voxelise(mesh: wp.uint64, origin: wp.vec3, h: float, reach: float, bit: int,
             mask: wp.array3d(dtype=wp.int32)):
    """Flag cells within `reach` of the part's surface (thin shells become solid slabs;
    reach >= sqrt(3)/2 h leaves no 6-connected leak through a wall)."""
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * h
    q = wp.mesh_query_point_no_sign(mesh, p, reach)
    if q.result:
        mask[i, j, k] = mask[i, j, k] | bit


@wp.kernel
def solid_from_mask(mask: wp.array3d(dtype=wp.int32), active_bits: int,
                    part_vel: wp.array(dtype=wp.vec3), n_parts: int,
                    solid: wp.array3d(dtype=wp.int32), wall_vel: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    m = mask[i, j, k] & active_bits
    solid[i, j, k] = 0
    wall_vel[i, j, k] = wp.vec3(0.0)
    if m != 0:
        solid[i, j, k] = 1
        for b in range(n_parts):
            if (m >> b) & 1:
                wall_vel[i, j, k] = part_vel[b]


@wp.kernel
def deposit_drag(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), radius: float,
                 u: wp.array3d(dtype=float), v: wp.array3d(dtype=float),
                 w: wp.array3d(dtype=float), origin: wp.vec3, h: float,
                 kc: wp.array3d(dtype=float), kuc: wp.array3d(dtype=wp.vec3)):
    """Each grain's drag rate K = 1/2 Cd pi r^2 |w| / V_cell (1/s, per unit air density)
    and K * v_grain, splatted trilinearly onto cell centres."""
    t = wp.tid()
    p = pos[t]
    ua = _air_at(u, v, w, origin, h, p)
    rel = vel[t] - ua
    s = wp.length(rel)
    re = s * 2.0 * radius / 1.5e-5
    k = 0.5 * _cd(re) * 3.14159265 * radius * radius * s / (h * h * h)
    g = (p - origin) / h - wp.vec3(0.5)
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    tx = g[0] - float(i0)
    ty = g[1] - float(j0)
    tz = g[2] - float(k0)
    nx, ny, nz = kc.shape[0], kc.shape[1], kc.shape[2]
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di
                jj = j0 + dj
                kk = k0 + dk
                if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                    wt = wp.where(di == 1, tx, 1.0 - tx) * wp.where(dj == 1, ty, 1.0 - ty) * \
                        wp.where(dk == 1, tz, 1.0 - tz)
                    wp.atomic_add(kc, ii, jj, kk, wt * k)
                    wp.atomic_add(kuc, ii, jj, kk, wt * k * vel[t])


@wp.func
def _face_pos(axis: int, i: int, j: int, k: int, origin: wp.vec3, h: float):
    f = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    f[axis] = f[axis] - 0.5
    return origin + f * h


@wp.kernel
def advect(axis: int, src: wp.array3d(dtype=float), u: wp.array3d(dtype=float),
           v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), origin: wp.vec3, h: float,
           dt: float, dst: wp.array3d(dtype=float)):
    """Semi-Lagrangian: the value at the face is the value one step upstream."""
    i, j, k = wp.tid()
    p = _face_pos(axis, i, j, k, origin, h)
    back = p - dt * _air_at(u, v, w, origin, h, p)
    g = (back - origin) / h
    if axis == 0:
        dst[i, j, k] = _trilerp(src, g[0], g[1] - 0.5, g[2] - 0.5)
    elif axis == 1:
        dst[i, j, k] = _trilerp(src, g[0] - 0.5, g[1], g[2] - 0.5)
    else:
        dst[i, j, k] = _trilerp(src, g[0] - 0.5, g[1] - 0.5, g[2])


@wp.func
def _cell_val(kc: wp.array3d(dtype=float), i: int, j: int, k: int):
    if i < 0 or j < 0 or k < 0 or i >= kc.shape[0] or j >= kc.shape[1] or k >= kc.shape[2]:
        return 0.0
    return kc[i, j, k]


@wp.func
def _cell_vec(kuc: wp.array3d(dtype=wp.vec3), i: int, j: int, k: int):
    if i < 0 or j < 0 or k < 0 or i >= kuc.shape[0] or j >= kuc.shape[1] or k >= kuc.shape[2]:
        return wp.vec3(0.0)
    return kuc[i, j, k]


@wp.func
def _solid(solid: wp.array3d(dtype=wp.int32), i: int, j: int, k: int):
    """Solid flag; cells outside the box are open air (0)."""
    if i < 0 or j < 0 or k < 0 or i >= solid.shape[0] or j >= solid.shape[1] or k >= solid.shape[2]:
        return 0
    return solid[i, j, k]


@wp.kernel
def drag_diffuse_walls(axis: int, src: wp.array3d(dtype=float), kc: wp.array3d(dtype=float),
                       kuc: wp.array3d(dtype=wp.vec3), solid: wp.array3d(dtype=wp.int32),
                       wall_vel: wp.array3d(dtype=wp.vec3), nu: float, h: float, dt: float,
                       dst: wp.array3d(dtype=float)):
    """Implicit grain drag, explicit eddy viscosity, then walls: a face touching a solid
    cell takes that wall's velocity (zero, or the belt's)."""
    i, j, k = wp.tid()
    # the two cells this face separates
    ia, ja, ka = i, j, k
    if axis == 0:
        ia = i - 1
    elif axis == 1:
        ja = j - 1
    else:
        ka = k - 1
    sa = _solid(solid, ia, ja, ka)
    sb = _solid(solid, i, j, k)
    if sa != 0 or sb != 0:
        vw = wp.vec3(0.0)
        if sa != 0:
            vw = wall_vel[ia, ja, ka]
        if sb != 0:
            vw = wall_vel[i, j, k]
        dst[i, j, k] = vw[axis]
        return
    x = src[i, j, k]
    # eddy viscosity (neighbours clamped at the box edge: zero gradient)
    nx, ny, nz = src.shape[0], src.shape[1], src.shape[2]
    lap = src[wp.max(i - 1, 0), j, k] + src[wp.min(i + 1, nx - 1), j, k] + \
        src[i, wp.max(j - 1, 0), k] + src[i, wp.min(j + 1, ny - 1), k] + \
        src[i, j, wp.max(k - 1, 0)] + src[i, j, wp.min(k + 1, nz - 1)] - 6.0 * x
    x = x + dt * nu * lap / (h * h)
    kf = 0.5 * (_cell_val(kc, ia, ja, ka) + _cell_val(kc, i, j, k))
    kuf = 0.5 * (_cell_vec(kuc, ia, ja, ka)[axis] + _cell_vec(kuc, i, j, k)[axis])
    dst[i, j, k] = (x + dt * kuf) / (1.0 + dt * kf)


@wp.kernel
def divergence(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float),
               w: wp.array3d(dtype=float), h: float, div: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    div[i, j, k] = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k]
                    + w[i, j, k + 1] - w[i, j, k]) / h


@wp.func
def _phi_nb(phi: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32), i: int, j: int,
            k: int, acc: wp.vec2):
    """Accumulate a neighbour into (sum, count): solid = Neumann (skipped), outside the box
    = open air at phi 0 (counted)."""
    if i < 0 or j < 0 or k < 0 or i >= phi.shape[0] or j >= phi.shape[1] or k >= phi.shape[2]:
        return wp.vec2(acc[0], acc[1] + 1.0)
    if solid[i, j, k] != 0:
        return acc
    return wp.vec2(acc[0] + phi[i, j, k], acc[1] + 1.0)


@wp.kernel
def sor_sweep(phi: wp.array3d(dtype=float), div: wp.array3d(dtype=float),
              solid: wp.array3d(dtype=wp.int32), h: float, omega: float, colour: int):
    i, j, k = wp.tid()
    if (i + j + k) % 2 != colour:
        return
    if solid[i, j, k] != 0:
        phi[i, j, k] = 0.0
        return
    acc = wp.vec2(0.0, 0.0)
    acc = _phi_nb(phi, solid, i - 1, j, k, acc)
    acc = _phi_nb(phi, solid, i + 1, j, k, acc)
    acc = _phi_nb(phi, solid, i, j - 1, k, acc)
    acc = _phi_nb(phi, solid, i, j + 1, k, acc)
    acc = _phi_nb(phi, solid, i, j, k - 1, acc)
    acc = _phi_nb(phi, solid, i, j, k + 1, acc)
    if acc[1] < 0.5:
        phi[i, j, k] = 0.0
        return
    gs = (acc[0] - h * h * div[i, j, k]) / acc[1]
    phi[i, j, k] = (1.0 - omega) * phi[i, j, k] + omega * gs


@wp.func
def _phi_at(phi: wp.array3d(dtype=float), i: int, j: int, k: int):
    if i < 0 or j < 0 or k < 0 or i >= phi.shape[0] or j >= phi.shape[1] or k >= phi.shape[2]:
        return 0.0
    return phi[i, j, k]


@wp.kernel
def project(axis: int, vel: wp.array3d(dtype=float), phi: wp.array3d(dtype=float),
            solid: wp.array3d(dtype=wp.int32), h: float):
    i, j, k = wp.tid()
    ia, ja, ka = i, j, k
    if axis == 0:
        ia = i - 1
    elif axis == 1:
        ja = j - 1
    else:
        ka = k - 1
    if _solid(solid, ia, ja, ka) != 0 or _solid(solid, i, j, k) != 0:
        return
    vel[i, j, k] = vel[i, j, k] - (_phi_at(phi, i, j, k) - _phi_at(phi, ia, ja, ka)) / h


# ---------------------------------------------------------------------------
# dust kernels
# ---------------------------------------------------------------------------

@wp.kernel
def move_dust(x: wp.array(dtype=wp.vec3), vp: wp.array(dtype=wp.vec3),
              state: wp.array(dtype=wp.int32), diam: wp.array(dtype=float), rho_p: float,
              u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
              solid_part: wp.array3d(dtype=wp.int32), origin: wp.vec3, h: float,
              gravity: wp.vec3, down_axis: int, down_sign: int, diffusivity: float, dt: float,
              seed: int):
    """state: 0 airborne, 1 + part index deposited, -1 - face escaped (faces as FACES).
    A parcel deposits only when it settles onto a surface from above (it enters a solid
    cell moving down the gravity axis); against a side or an underside it stays where it
    was and moves on with the air (fine dust does not stick to every wall it brushes)."""
    t = wp.tid()
    if state[t] != 0:
        return
    p = x[t]
    ua = _air_at(u, v, w, origin, h, p)
    d = diam[t]
    tau0 = rho_p * d * d / (18.0 * 1.8e-5)
    # Schiller-Naumann correction at the current slip
    re = wp.length(vp[t] - ua) * d / 1.5e-5
    tau = tau0 / (1.0 + 0.15 * wp.pow(re, 0.687))
    target = ua + gravity * tau                   # air velocity plus settling velocity
    a = wp.exp(-dt / tau)
    vn = target + (vp[t] - target) * a
    # displacement: exact integral of the relaxation, plus the turbulent random walk
    rng = wp.rand_init(seed, t)
    walk = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * wp.sqrt(2.0 * diffusivity * dt)
    pn = p + target * dt + (vp[t] - target) * tau * (1.0 - a) + walk
    c0 = _cell_of(origin, h, p)
    c = _cell_of(origin, h, pn)
    nx, ny, nz = solid_part.shape[0], solid_part.shape[1], solid_part.shape[2]
    if c[0] < 0:
        state[t] = -1
    elif c[0] >= nx:
        state[t] = -2
    elif c[1] < 0:
        state[t] = -3
    elif c[1] >= ny:
        state[t] = -4
    elif c[2] < 0:
        state[t] = -5
    elif c[2] >= nz:
        state[t] = -6
    else:
        sp = solid_part[c[0], c[1], c[2]]
        if sp > 0:
            if (c[down_axis] - c0[down_axis]) * down_sign > 0:
                state[t] = sp                      # settled onto part sp - 1
            else:
                vp[t] = ua                         # bounced: stay, carried on by the air
            return
    vp[t] = vn
    x[t] = pn


@wp.kernel
def solid_part_index(mask: wp.array3d(dtype=wp.int32), active_bits: int, n_parts: int,
                     out: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    m = mask[i, j, k] & active_bits
    out[i, j, k] = 0
    for b in range(n_parts):
        if (m >> b) & 1:
            out[i, j, k] = b + 1


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _open_air(p, d, r0, solid, origin, h, rng, tries=8, reach=0.4):
    """Release points: p + d r0, moved out of solid cells.  Walls are voxelised ~0.9 cell
    thick, so a grain touching one usually sits in a wall cell; walk outward along d (then
    random directions) until a cell of open air, up to `reach` metres."""
    x = p + d * r0
    dims = np.array(solid.shape)

    def in_solid(q):
        c = np.floor((q - origin) / h).astype(int)
        ok = np.all((c >= 0) & (c < dims), axis=1)
        s = np.zeros(len(q), dtype=bool)
        s[ok] = solid[c[ok, 0], c[ok, 1], c[ok, 2]] != 0
        return s

    bad = in_solid(x)
    for k in range(tries):
        if not bad.any():
            break
        todo = np.flatnonzero(bad)
        dd = d[todo] if k == 0 else rng.normal(size=(len(todo), 3))
        dd /= np.linalg.norm(dd, axis=1, keepdims=True)
        for s in np.linspace(r0, reach, 6):
            left = bad[todo]                            # still unplaced, among todo
            if not left.any():
                break
            q = p[todo[left]] + dd[left] * s
            ok = ~in_solid(q)
            idx = todo[left][ok]
            x[idx] = q[ok]
            bad[idx] = False
    return x


def _box(run, frames, parts, cell, margin, budget):
    """Air box: where the grains go over the window, plus the parts around them, plus a
    margin; clipped to the DEM domain.  Cell enlarged if the box would exceed `budget`."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for _t, f in frames[:: max(1, len(frames) // 20)]:
        q = read_frame(f)["pos"]
        if len(q):
            lo, hi = np.minimum(lo, q.min(0)), np.maximum(hi, q.max(0))
    for _n, v, _f in parts:
        vlo, vhi = np.asarray(v).min(0), np.asarray(v).max(0)
        # only parts near the grains matter to the air
        if np.all(vhi > lo - 1.0) and np.all(vlo < hi + 1.0):
            lo, hi = np.minimum(lo, vlo), np.maximum(hi, vhi)
    lo = np.maximum(lo - margin, run.sc.domain_lo)
    hi = np.minimum(hi + margin, run.sc.domain_hi)
    ext = hi - lo
    if np.prod(ext / cell) > budget:
        cell = float((np.prod(ext) / budget) ** (1.0 / 3.0))
    dims = np.maximum(np.ceil(ext / cell).astype(int), 2)
    return lo, cell, dims


def air_dust(run, t0=None, t1=None, cell=0.1, margin=0.5, nu_t=0.002, sizes_um=(10, 30, 75),
             rho_dust=None, parcels_per_s=3000, emission_g_per_kJ=1.0, sor_iters=60,
             max_dt=0.02, budget=4.0e6, device="cuda:0", out_dir=None, quiet=False, seed=1):
    import dem_run
    wp.init()
    log = (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, **k, flush=True))
    sc = run.sc
    m = sc.material
    frames = run.window(t0, t1)
    if len(frames) < 2:
        raise ValueError("need at least two frames in the window")
    has_ids = "id" in read_frame(frames[-1][1])
    parts = [(p.name, *dem_run.load_stl(sc.path(p.stl), p.fix_normals, p.flip, sc.unit_scale))
             for p in sc.parts]
    if len(parts) > 31:
        raise ValueError("at most 31 parts (part masks are bit fields)")
    origin, h, dims = _box(run, frames, parts, cell, margin, budget)
    nx, ny, nz = (int(d) for d in dims)
    org = wp.vec3(*origin)
    gvec = np.asarray(sc.gravity, dtype=np.float64)
    rho_p = float(rho_dust or m.density)
    down_axis = int(np.argmax(np.abs(gvec)))
    down_sign = -1 if gvec[down_axis] < 0 else 1       # cell index change when moving down
    fdt = 1.0 / sc.output.fps
    nsub = max(1, int(math.ceil(fdt / max_dt)))
    dt = fdt / nsub
    log(f"air box {np.round(origin, 2).tolist()} + {np.round(dims * h, 2).tolist()} m: "
        f"{nx} x {ny} x {nz} = {nx*ny*nz:,} cells of {h*100:.1f} cm;  {len(frames)} frames, "
        f"{nsub} air steps each (dt {dt*1e3:.1f} ms);  eddy viscosity {nu_t:g} m2/s")
    if not has_ids:
        log("  !! frames carry no grain ids: no impact-based dust release (air only)")

    # --- walls: one bit per part, so active windows switch parts on and off ----------
    mask = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    meshes = []
    for b, (_n, v, f) in enumerate(parts):
        mesh = wp.Mesh(points=wp.array(np.asarray(v, dtype=np.float32), dtype=wp.vec3, device=device),
                       indices=wp.array(np.asarray(f, dtype=np.int32).ravel(), dtype=int, device=device))
        meshes.append(mesh)
        wp.launch(voxelise, dim=(nx, ny, nz), device=device,
                  inputs=[mesh.id, org, h, 0.87 * h, 1 << b, mask])
    part_vel = []
    for p in sc.parts:
        mo = p.motion or {}
        part_vel.append(mo.get("velocity", [0.0, 0.0, 0.0]) if mo.get("type") == "belt" else [0.0] * 3)
    part_vel = wp.array(np.asarray(part_vel, dtype=np.float32), dtype=wp.vec3, device=device)
    solid = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    solid_part = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    wall_vel = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=device)

    # --- air state -------------------------------------------------------------------
    u = wp.zeros((nx + 1, ny, nz), dtype=float, device=device)
    v = wp.zeros((nx, ny + 1, nz), dtype=float, device=device)
    w = wp.zeros((nx, ny, nz + 1), dtype=float, device=device)
    tmp = [wp.zeros_like(a) for a in (u, v, w)]
    tmp2 = [wp.zeros_like(a) for a in (u, v, w)]
    phi = wp.zeros((nx, ny, nz), dtype=float, device=device)
    div = wp.zeros((nx, ny, nz), dtype=float, device=device)
    kc = wp.zeros((nx, ny, nz), dtype=float, device=device)
    kuc = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=device)
    mean = [np.zeros(a.shape, dtype=np.float64) for a in (u, v, w)]
    n_mean = 0
    t_spin = frames[0][0] + min(2.0, 0.3 * (frames[-1][0] - frames[0][0]))  # mean after spin-up

    # --- dust state ------------------------------------------------------------------
    cap = int(parcels_per_s * (frames[-1][0] - frames[0][0]) * 1.2) + 1024
    dx = wp.zeros(cap, dtype=wp.vec3, device=device)
    dv = wp.zeros(cap, dtype=wp.vec3, device=device)
    dstate = wp.full(cap, value=-99, dtype=wp.int32, device=device)   # -99 = unused slot
    ddiam = wp.zeros(cap, dtype=float, device=device)
    n_dust = 0
    rng = np.random.default_rng(seed)
    released_J = 0.0
    dust_frames = []      # per frame: (positions of airborne parcels, their size class)
    sizes = np.asarray(sizes_um, dtype=np.float64) * 1e-6
    face_flow = np.zeros((6, 2))      # m3 in, m3 out through each side (after spin-up)

    active_prev = None
    prev = None
    t_start = time.time()
    for fi, (t, f) in enumerate(frames):
        fr = read_frame(f)
        act = [1 if p.active[0] <= t < p.active[1] else 0 for p in sc.parts]
        if act != active_prev:
            bits = sum(1 << b for b, a in enumerate(act) if a)
            wp.launch(solid_from_mask, dim=(nx, ny, nz), device=device,
                      inputs=[mask, bits, part_vel, len(parts), solid, wall_vel])
            wp.launch(solid_part_index, dim=(nx, ny, nz), device=device,
                      inputs=[mask, bits, len(parts), solid_part])
            solid_np = solid.numpy()
            active_prev = act
        pos = wp.array(fr["pos"].astype(np.float32), dtype=wp.vec3, device=device)
        vel = wp.array(fr["vel"].astype(np.float32), dtype=wp.vec3, device=device)

        # dust released by the energy grains dissipated since the previous frame
        if has_ids and prev is not None and len(fr["pos"]):
            ppos, pvel, pids = prev
            ids = fr["id"].astype(np.int64)
            lookup = np.full(max(ids.max(), pids.max()) + 1, -1, dtype=np.int64)
            lookup[pids] = np.arange(len(pids))
            j = lookup[ids]
            ok = j >= 0
            p0, v0 = ppos[j[ok]], pvel[j[ok]]
            p1, v1 = fr["pos"][ok], fr["vel"][ok]
            moved = np.linalg.norm(p1 - p0, axis=1) <= sc.solver.max_velocity * fdt
            p0, v0, p1, v1 = p0[moved], v0[moved], p1[moved], v1[moved]
            diss = 0.5 * m.mass * ((v0 ** 2).sum(1) - (v1 ** 2).sum(1)) + m.mass * ((p1 - p0) @ gvec)
            diss = np.where(diss > 1e-3 * m.mass * 9.81, diss, 0.0)   # below ~1 mm of fall: noise
            tot = diss.sum()
            n_new = min(int(rng.poisson(parcels_per_s * fdt)) if tot > 0 else 0, cap - n_dust)
            if n_new > 0:
                released_J += tot
                src = rng.choice(len(diss), size=n_new, p=diss / tot)
                d = rng.normal(size=(n_new, 3))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                xs = _open_air(p1[src], d, m.radius * 1.1, solid_np, origin, h, rng)
                cls = np.arange(n_dust, n_dust + n_new) % len(sizes)
                sl = slice(n_dust, n_dust + n_new)
                dx[sl].assign(xs.astype(np.float32))
                dv[sl].assign(v1[src].astype(np.float32))
                ddiam[sl].assign(sizes[cls].astype(np.float32))
                dstate[sl].assign(np.zeros(n_new, dtype=np.int32))
                n_dust += n_new
        if has_ids:
            prev = (fr["pos"], fr["vel"], fr["id"].astype(np.int64))

        # air: nsub steps with this frame's grains
        for s in range(nsub):
            kc.zero_()
            kuc.zero_()
            if len(fr["pos"]):
                wp.launch(deposit_drag, dim=len(fr["pos"]), device=device,
                          inputs=[pos, vel, float(m.radius), u, v, w, org, h, kc, kuc])
            for ax, (src, dst) in enumerate(zip((u, v, w), tmp)):
                wp.launch(advect, dim=src.shape, device=device,
                          inputs=[ax, src, u, v, w, org, h, dt, dst])
            for ax, (src, dst) in enumerate(zip(tmp, tmp2)):
                wp.launch(drag_diffuse_walls, dim=src.shape, device=device,
                          inputs=[ax, src, kc, kuc, solid, wall_vel, nu_t, h, dt, dst])
            for a, b in zip((u, v, w), tmp2):
                wp.copy(a, b)
            wp.launch(divergence, dim=(nx, ny, nz), device=device, inputs=[u, v, w, h, div])
            for _it in range(sor_iters):
                for col in (0, 1):
                    wp.launch(sor_sweep, dim=(nx, ny, nz), device=device,
                              inputs=[phi, div, solid, h, 1.85, col])
            for ax, a in enumerate((u, v, w)):
                wp.launch(project, dim=a.shape, device=device, inputs=[ax, a, phi, solid, h])
            if n_dust:
                wp.launch(move_dust, dim=n_dust, device=device, inputs=[
                    dx, dv, dstate, ddiam, rho_p, u, v, w, solid_part, org, h,
                    wp.vec3(*gvec), down_axis, down_sign, nu_t, dt, seed * 1000003 + fi * 31 + s])
            if t >= t_spin:
                un, vn, wn = u.numpy(), v.numpy(), w.numpy()
                for k, arr in enumerate((un, vn, wn)):
                    mean[k] += arr
                n_mean += 1
                # flow through the six sides this step (m3)
                A = h * h
                for k, (arr, ax) in enumerate(((un, 0), (un, 0), (vn, 1), (vn, 1), (wn, 2), (wn, 2))):
                    side = [slice(None)] * 3
                    side[ax] = 0 if k % 2 == 0 else -1
                    q = arr[tuple(side)] * A * dt
                    inflow = q[q > 0].sum() if k % 2 == 0 else -q[q < 0].sum()
                    outflow = -q[q < 0].sum() if k % 2 == 0 else q[q > 0].sum()
                    face_flow[k] += (inflow, outflow)
        if n_dust:
            st = dstate.numpy()[:n_dust]
            air = st == 0
            dust_frames.append((dx.numpy()[:n_dust][air].astype(np.float32),
                                (np.arange(n_dust)[air] % len(sizes)).astype(np.uint8)))
        else:
            dust_frames.append((np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)))
        if not quiet and (fi % 30 == 0 or fi == len(frames) - 1):
            wp.synchronize()
            dres = float(np.abs(div.numpy()).mean())
            sp = max(float(np.abs(u.numpy()).max()), float(np.abs(v.numpy()).max()),
                     float(np.abs(w.numpy()).max()))
            log(f"  t {t:6.2f} s  max air {sp:5.2f} m/s  mean |div| {dres:.3g} 1/s  "
                f"dust parcels {n_dust:,} ({int((dstate.numpy()[:n_dust] == 0).sum()):,} airborne)  "
                f"{time.time()-t_start:5.0f} s")

    # --- results ---------------------------------------------------------------------
    span = n_mean * dt
    mu_, mv_, mw_ = (a / max(n_mean, 1) for a in mean)
    cu = 0.5 * (mu_[:-1] + mu_[1:])
    cv = 0.5 * (mv_[:, :-1] + mv_[:, 1:])
    cw = 0.5 * (mw_[:, :, :-1] + mw_[:, :, 1:])
    centre = np.stack([cu, cv, cw], axis=-1).astype(np.float32)
    solid_np = solid.numpy().astype(np.uint8)
    st = dstate.numpy()[:n_dust]
    cls = np.arange(n_dust) % len(sizes)
    fate = {}
    g_per_parcel = emission_g_per_kJ * released_J / 1e3 / max(n_dust, 1)
    for c, d in enumerate(sizes_um):
        sc_ = st[cls == c]
        n = max(len(sc_), 1)
        fate[f"{d:g}um"] = dict(
            parcels=int(len(sc_)), airborne=round(float((sc_ == 0).sum()) / n, 4),
            deposited={sc.parts[k].name: round(float((sc_ == k + 1).sum()) / n, 4)
                       for k in range(len(parts)) if (sc_ == k + 1).any()},
            escaped={FACES[k]: round(float((sc_ == -1 - k).sum()) / n, 4)
                     for k in range(6) if (sc_ == -1 - k).any()})
    summary = dict(
        window_s=[frames[0][0], frames[-1][0]], mean_from_s=t_spin, cell_m=h,
        box_lo=np.round(origin, 4).tolist(), box_size_m=np.round(dims * h, 4).tolist(),
        dims=[nx, ny, nz], eddy_viscosity_m2_s=nu_t, air_steps_per_frame=nsub,
        air_in_m3_s={FACES[k]: round(face_flow[k, 0] / max(span, 1e-9), 4) for k in range(6)},
        air_out_m3_s={FACES[k]: round(face_flow[k, 1] / max(span, 1e-9), 4) for k in range(6)},
        max_mean_air_speed_m_s=round(float(np.linalg.norm(centre, axis=-1).max()), 3),
        dissipated_kJ=round(released_J / 1e3, 3), dust_parcels=int(n_dust),
        emission_g_per_kJ=emission_g_per_kJ, dust_g_per_parcel=g_per_parcel,
        dust_sizes_um=list(sizes_um), dust_density=rho_p, dust_fate=fate,
        caveats=["one-way: grains are not slowed by the air",
                 "single-sphere drag without shielding: the stream's pull on the air is an upper bound",
                 "constant eddy viscosity; coarse grid (no boundary layers)",
                 "dust masses scale with an emission factor that has not been measured for this material"])
    out_dir = out_dir or os.path.join(run.dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "air_dust.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    offs = np.cumsum([0] + [len(p) for p, _c in dust_frames])
    np.savez_compressed(
        os.path.join(out_dir, "air_dust.npz"), origin=origin.astype(np.float32), cell=np.float32(h),
        air=centre.astype(np.float16), solid=solid_np,
        frame_times=np.array([t for t, _f in frames], dtype=np.float32),
        frame_ids=np.array([int(round(t * sc.output.fps)) for t, _f in frames], dtype=np.int32),
        dust_pos=np.concatenate([p for p, _c in dust_frames]).astype(np.float32),
        dust_cls=np.concatenate([c for _p, c in dust_frames]), dust_off=offs.astype(np.int64),
        sizes_um=np.asarray(sizes_um, dtype=np.float32))
    _write_air_vtk(os.path.join(out_dir, "air_mean.vtk"), origin, h, centre, solid_np)
    if not quiet:
        tot_in = sum(summary["air_in_m3_s"].values())
        log(f"air & dust: window {frames[0][0]:.2f}-{frames[-1][0]:.2f} s (mean from {t_spin:.2f} s) "
            f"-> {out_dir}")
        log(f"  air drawn in {tot_in:.2f} m3/s, out {sum(summary['air_out_m3_s'].values()):.2f} m3/s "
            f"(max mean speed {summary['max_mean_air_speed_m_s']:.2f} m/s)")
        for k in range(6):
            log(f"    {FACES[k]:>3}: in {summary['air_in_m3_s'][FACES[k]]:7.3f}  "
                f"out {summary['air_out_m3_s'][FACES[k]]:7.3f} m3/s")
        for name, fa in fate.items():
            log(f"  dust {name:>5}: airborne {fa['airborne']:.1%}  deposited "
                f"{sum(fa['deposited'].values()):.1%}  escaped {sum(fa['escaped'].values()):.1%} "
                f"{fa['escaped']}")
    return summary


def _write_air_vtk(path, origin, h, centre, solid):
    nx, ny, nz = solid.shape
    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\ntime-mean air velocity\nBINARY\nDATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\nORIGIN {origin[0]+h/2} {origin[1]+h/2} {origin[2]+h/2}\n"
                 f"SPACING {h} {h} {h}\nPOINT_DATA {nx*ny*nz}\n".encode())
        # VTK point order: x fastest
        fh.write(b"VECTORS air_velocity float\n")
        fh.write(np.ascontiguousarray(centre.transpose(2, 1, 0, 3)).astype(">f4").tobytes())
        fh.write(b"\nSCALARS solid int 1\nLOOKUP_TABLE default\n")
        fh.write(np.ascontiguousarray(solid.transpose(2, 1, 0)).astype(">i4").tobytes())


def main():
    from dem_analyze import Run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--scenario", default=None, help="scenario JSON if run.json lacks one")
    ap.add_argument("--cell", type=float, default=0.1, help="air cell size (m)")
    ap.add_argument("--nu-t", type=float, default=0.002, help="eddy viscosity (m2/s)")
    ap.add_argument("--sizes", type=float, nargs="+", default=[10, 30, 75], help="dust sizes (um)")
    ap.add_argument("--parcels", type=float, default=3000, help="dust parcels released per second")
    ap.add_argument("--emission", type=float, default=1.0, help="g of dust per kJ dissipated (illustrative)")
    args = ap.parse_args()
    run = Run(args.run, args.scenario)
    t0, t1 = (args.window or (None, None))
    air_dust(run, t0, t1, cell=args.cell, nu_t=args.nu_t, sizes_um=tuple(args.sizes),
             parcels_per_s=args.parcels, emission_g_per_kJ=args.emission)


if __name__ == "__main__":
    sys.exit(main())
