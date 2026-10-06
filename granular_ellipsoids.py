#!/usr/bin/env python3
"""
Ellipsoidal grains: one rigid body per particle with semi-axes (a, b, c) -- a parametric
shape, not an assembly of spheres.

Contact geometry uses the SUPPORT FUNCTION of an ellipsoid, h(n) = sqrt(n^T M n) with
M = R diag(a^2, b^2, c^2) R^T (its extent along unit direction n), and the support point
s(n) = M n / h(n) (the surface point whose outward normal is n).

  grain-wall: the wall search is the sphere one (closest triangle point to the centre,
     using the bounding radius); the gap along the wall normal n is sdf - h(n) and the
     contact is the support point -s(n).  Exact for a flat wall.
  grain-grain: two convex bodies overlap by
        delta = min over unit n of  h_i(n) + h_j(n) - n . (x_i - x_j)
     (the depth of x_i - x_j inside the Minkowski sum).  Minimised by projected gradient
     on the sphere from n0 = (x_i - x_j)/|.|; the step 1/|x_i - x_j| is exact for spheres,
     and for bean-like aspect ratios (< ~1.5) each step cuts the error ~3x -- ELL_ITERS
     steps.  Contact point: midway between the two support points.

The force law is SolverGranularDEM's (Hertz-Mindlin with history spring, rolling
resistance), with the lever arms of the actual contact points, and Hertz stiffness from
the volume-equivalent radius.  Bodies are integrated by granular_clumps.integrate_clumps
(world angular momentum + quaternion), each "clump" being one particle.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from granular_clumps import SolverGranularClumps, integrate_clumps
from granular_dem import (ShellCollider, WallGrid, WallTri, _avg_normal_from_list, _cell_of,
                          _entry, _history_slot, _rolling_torque, _rotational_damping,
                          _shell_sdf, _surface_velocity, _tangential_spring, _tri_closest)

# the adjoint of the wall kernel exceeds CUDA's 4 KB parameter limit (as for granular_dem)
wp.set_module_options({"enable_backward": False})

ELL_ITERS = wp.constant(6)


class EllipsoidShape:
    """Semi-axes a >= b >= c (metres) and density; quacks like a ClumpTemplate with one
    sphere so SolverGranularClumps.set_clumps can place the bodies."""

    def __init__(self, a, b, c, density):
        ax = np.sort(np.array([a, b, c], dtype=np.float64))[::-1]
        self.axes = ax
        self.volume = 4.0 / 3.0 * math.pi * ax.prod()
        self.mass = density * self.volume
        a, b, c = ax
        self.inertia = self.mass / 5.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
        self.n = 1
        self.offsets = np.zeros((1, 3))
        self.bound_radius = float(ax[0])
        self.radii = np.array([self.bound_radius])
        self.equiv_radius = float((ax.prod()) ** (1.0 / 3.0))
        self.equiv_diameter = 2.0 * self.equiv_radius

    def describe(self):
        return (f"ellipsoid {2*self.axes[0]*1e3:.2f} x {2*self.axes[1]*1e3:.2f} x "
                f"{2*self.axes[2]*1e3:.2f} mm, volume-equivalent dia "
                f"{self.equiv_diameter*1e3:.2f} mm, mass {self.mass*1e3:.4f} g")


@wp.func
def _ell_M(q: wp.quat, ax2: wp.vec3):
    R = wp.quat_to_matrix(q)
    return R * wp.diag(ax2) * wp.transpose(R)


@wp.func
def _h(M: wp.mat33, n: wp.vec3):
    return wp.sqrt(wp.max(wp.dot(n, M * n), 1.0e-30))


@wp.kernel
def eval_ellipsoid_pairs(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_inv_mass: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.quat),
    ax2: wp.vec3,
    r_eq: float,
    mu: float,
    mu_roll: float,
    rot_damp: float,
    max_radius: float,
    k_t: float,
    e_star: float,
    g_star: float,
    beta: float,
    dt: float,
    step_arr: wp.array(dtype=int),
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """Hertz-Mindlin between ellipsoids (ASSIGNS f and t, like eval_particle_contact_rot)."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        particle_f[i] = wp.vec3(0.0)
        particle_t[i] = wp.vec3(0.0)
        return
    step = step_arr[0]
    x = particle_x[i]
    v = particle_v[i]
    w = particle_w[i]
    ri = particle_radius[i]
    Mi = _ell_M(body_q[i], ax2)
    f = wp.vec3(0.0)
    t = wp.vec3(0.0)
    r_star = 0.5 * r_eq
    query = wp.hash_grid_query(grid, x, ri + max_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if (particle_flags[j] & newton.ParticleFlags.ACTIVE) == 0:
            continue
        u = x - particle_x[j]
        du = wp.length(u)
        if du < 1.0e-9 or du >= ri + particle_radius[j]:
            continue
        Mj = _ell_M(body_q[j], ax2)
        n = u / du
        for _k in range(ELL_ITERS):
            hi = _h(Mi, n)
            hj = _h(Mj, n)
            grad = Mi * n / hi + Mj * n / hj - u
            gt = grad - n * wp.dot(grad, n)
            n = wp.normalize(n - gt / du)
        hi = _h(Mi, n)
        hj = _h(Mj, n)
        overlap = hi + hj - wp.dot(n, u)
        if overlap <= 0.0:
            continue
        # surface points: i's in direction -n, j's in direction +n; contact midway
        pi = x - Mi * n / hi
        pj = particle_x[j] + Mj * n / hj
        cpt = 0.5 * (pi + pj)
        c_i = cpt - x
        c_j = cpt - particle_x[j]
        wj = particle_w[j]
        v_rel = (v + wp.cross(w, c_i)) - (particle_v[j] + wp.cross(wj, c_j))
        vn = wp.dot(v_rel, n)
        vt = v_rel - n * vn
        m_star = 1.0 / wp.max(particle_inv_mass[i] + particle_inv_mass[j], 1.0e-12)
        sq = wp.sqrt(r_star * overlap)
        s_n = 2.0 * e_star * sq
        s_t = 8.0 * g_star * sq
        kd_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_n * m_star)
        kf_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_t * m_star)
        fn = (4.0 / 3.0) * e_star * wp.sqrt(r_star) * overlap * wp.sqrt(overlap) - kd_eff * vn
        if fn <= 0.0:
            continue
        ft = wp.vec3(0.0)
        kt_eff = k_t * s_t
        if kt_eff > 0.0:
            slot = _history_slot(tang_partner, tang_stamp, i, j, step)
            if slot >= 0:
                ft = _tangential_spring(tang_xi, tang_partner, tang_stamp, i, slot, j, step,
                                        n, vt, dt, kt_eff, mu * fn)
            else:
                vl = wp.length(vt)
                if vl > 1.0e-8:
                    ft = -(vt / vl) * wp.min(kf_eff * vl, mu * fn)
        else:
            vl = wp.length(vt)
            if vl > 1.0e-8:
                ft = -(vt / vl) * wp.min(kf_eff * vl, mu * fn)
        fc = n * fn + ft
        f += fc
        t += wp.cross(c_i, fc)
        w_rel = w - wj
        if mu_roll > 0.0:
            t += _rolling_torque(mu_roll, fn, r_star, w_rel)
        if rot_damp > 0.0:
            t += _rotational_damping(rot_damp, kd_eff, r_star, w_rel)
    particle_f[i] = f
    particle_t[i] = t


@wp.func
def _wall_force_ell(
    collider: ShellCollider, m: int, n: wp.vec3, c: float, r_hz: float, c_arm: wp.vec3,
    v: wp.vec3, w: wp.vec3, inv_mass: float, mu_roll: float, rot_damp: float, k_t: float,
    e_star: float, g_star: float, beta: float, step: int, i: int, wid: int, cp: wp.vec3,
    tang_partner: wp.array2d(dtype=int), tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3), dt: float,
):
    """granular_dem._wall_force (Hertz path) with an arbitrary lever arm c_arm."""
    v_rel = v + wp.cross(w, c_arm)
    w_rel = w
    if collider.motion_type[m] != 0:
        v_surf, w_surf = _surface_velocity(collider, m, n, cp)
        v_rel = v_rel - v_surf
        w_rel = w - w_surf
    vn = wp.dot(v_rel, n)
    vt = v_rel - n * vn
    delta = -c
    m_star = 1.0 / wp.max(inv_mass, 1.0e-12)
    sq = wp.sqrt(r_hz * delta)
    s_n = 2.0 * e_star * sq
    s_t = 8.0 * g_star * sq
    kd_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_n * m_star)
    kf_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_t * m_star)
    fn = (4.0 / 3.0) * e_star * wp.sqrt(r_hz) * delta * wp.sqrt(delta) - vn * kd_eff
    fn = wp.max(fn, 0.0)
    ft = wp.vec3(0.0)
    kt_eff = k_t * s_t
    if kt_eff > 0.0:
        slot = _history_slot(tang_partner, tang_stamp, i, wid, step)
        if slot >= 0:
            ft = _tangential_spring(tang_xi, tang_partner, tang_stamp, i, slot, wid, step, n,
                                    vt, dt, kt_eff, collider.friction[m] * fn)
        else:
            vl = wp.length(vt)
            if vl > 1.0e-8:
                ft = -(vt / vl) * wp.min(kf_eff * vl, collider.friction[m] * fn)
    else:
        vl = wp.length(vt)
        if vl > 1.0e-8:
            ft = -(vt / vl) * wp.min(kf_eff * vl, collider.friction[m] * fn)
    f = n * fn + ft
    t = wp.cross(c_arm, f)
    if mu_roll > 0.0:
        t += _rolling_torque(mu_roll, fn, r_hz, w_rel)
    if rot_damp > 0.0:
        t += _rotational_damping(rot_damp, kd_eff, r_hz, w_rel)
    return f, t


@wp.kernel
def eval_ellipsoid_walls(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.quat),
    ax2: wp.vec3,
    r_eq: float,
    collider: ShellCollider,
    g: WallGrid,
    mu_roll: float,
    rot_damp: float,
    k_t: float,
    e_star: float,
    g_star: float,
    beta: float,
    step_arr: wp.array(dtype=int),
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    dt: float,
    grid: wp.uint64,
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """eval_wall_grid_rot's search (bounding radius), ellipsoid gap and lever arm."""
    i = wp.hash_grid_point_id(grid, wp.tid())
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        return
    step = step_arr[0]
    x = particle_q[i]
    cc = _cell_of(g, x)
    if cc[0] < 0 or cc[1] < 0 or cc[2] < 0 or cc[0] >= g.nx or cc[1] >= g.ny or cc[2] >= g.nz:
        return
    cell = (cc[2] * g.ny + cc[1]) * g.nx + cc[0]
    k0 = g.cell_start[cell]
    ncand = g.cell_start[cell + 1] - k0
    if ncand == 0:
        return
    radius = particle_radius[i]
    v = particle_qd[i]
    w = particle_w[i]
    M = _ell_M(body_q[i], ax2)
    f_total = wp.vec3(0.0)
    t_total = wp.vec3(0.0)
    hit = int(0)
    s_part = wp.vec4i(-1)
    s_tri = wp.vec4i(-1)
    s_seg0 = wp.vec4i(0)
    s_seg1 = wp.vec4i(0)
    s_sq = wp.vec4(0.0)
    ns = int(0)
    cur = int(-1)
    seg0 = int(0)
    best_sq = float(0.0)
    best_t = int(-1)
    for s in range(ncand + 1):
        t = int(-1)
        p = int(-1)
        r = WallTri()
        if s < ncand:
            t = k0 + s
            r = g.rec[t]
            p = r.part
        if p != cur:
            if best_t >= 0 and ns < 4:
                s_part[ns] = cur
                s_tri[ns] = best_t
                s_seg0[ns] = seg0
                s_seg1[ns] = s
                s_sq[ns] = best_sq
                ns += 1
            cur = p
            seg0 = s
            best_t = -1
            rr = radius + collider.thickness[wp.max(p, 0)]
            best_sq = rr * rr
            if p >= 0:
                if collider.active[p] == 0:
                    best_sq = 0.0
        if t >= 0:
            hpl = wp.dot(x - r.v0, r.n)
            if hpl * hpl < best_sq:
                sq, _cand = _tri_closest(r.v0, r.u, r.w, r.n, r.t2, x)
                if sq < best_sq:
                    best_sq = sq
                    best_t = t
    for jj in range(ns):
        m = s_part[jj]
        br = g.rec[s_tri[jj]]
        _sq, best_cp = _tri_closest(br.v0, br.u, br.w, br.n, br.t2, x)
        thick = collider.thickness[m]
        offset = x - best_cp
        d_unsigned = wp.sqrt(s_sq[jj])
        face_n = br.n
        sdf, n = _shell_sdf(collider.two_sided[m], thick, offset, d_unsigned, face_n)
        hn = _h(M, n)
        c = sdf - hn
        if c < 0.0 and c >= -hn:
            arm = -(M * n) / hn
            fw, tw = _wall_force_ell(collider, m, n, c, r_eq, arm, v, w, particle_inv_mass[i],
                                     mu_roll, rot_damp, k_t, e_star, g_star, beta, step, i,
                                     -(m + 2), best_cp, tang_partner, tang_stamp, tang_xi, dt)
            f_total += fw
            t_total += tw
            hit += 1
    if hit > 0:
        particle_f[i] = particle_f[i] + f_total
        particle_t[i] = particle_t[i] + t_total


class SolverGranularEllipsoids(SolverGranularClumps):
    """One rigid ellipsoid per particle (Hertz contact, baked wall grid, reference preset).
    Call set_ellipsoids() before stepping."""

    def __init__(self, model, collider, **kw):
        if not kw.get("hertz", False):
            raise NotImplementedError("ellipsoids: Hertz contact only")
        if kw.get("wall_grid") is None:
            raise NotImplementedError("ellipsoids: baked wall grid only")
        super().__init__(model, collider, **kw)

    def set_ellipsoids(self, shape: EllipsoidShape, com, quats, vel=None, kill_y=-1.0e9,
                       park=(0.0, 1.0e3, 0.0)):
        self.set_clumps(shape, com, quats, vel=vel, kill_y=kill_y, park=park)
        self.shape = shape
        self.ax2 = wp.vec3(*(shape.axes ** 2))
        self.r_eq = shape.equiv_radius

    def step(self, state_in, state_out, control, contacts, dt: float):
        if state_out is not state_in:
            raise ValueError("ellipsoids step in place (state_out must be state_in)")
        model = self.model
        self._step += 1
        model.particle_grid.build(state_in.particle_q, self.grid_cell)
        wp.launch(
            kernel=eval_ellipsoid_pairs, dim=model.particle_count,
            inputs=[model.particle_grid.id, state_in.particle_q, state_in.particle_qd,
                    self.particle_w, model.particle_radius, model.particle_flags,
                    model.particle_inv_mass, self.clump_q, self.ax2, self.r_eq,
                    model.particle_mu, self.mu_roll, self.rot_damp, model.particle_max_radius,
                    self.k_t, self.e_star_pp, self.g_star_pp, self.beta_pp, dt, self.step_arr,
                    self.tang_partner, self.tang_stamp, self.tang_xi],
            outputs=[state_in.particle_f, self.particle_t], device=model.device)
        wp.launch(
            kernel=eval_ellipsoid_walls, dim=model.particle_count,
            inputs=[state_in.particle_q, state_in.particle_qd, self.particle_w,
                    model.particle_radius, model.particle_inv_mass, model.particle_flags,
                    self.clump_q, self.ax2, self.r_eq, self.collider, self.wall_grid,
                    self.mu_roll_wall, self.rot_damp_wall, self.k_t, self.e_star_w,
                    self.g_star_w, self.beta_w, self.step_arr, self.tang_partner,
                    self.tang_stamp, self.tang_xi, dt, model.particle_grid.id],
            outputs=[state_in.particle_f, self.particle_t], device=model.device)
        wp.launch(
            kernel=integrate_clumps, dim=self.n_clumps,
            inputs=[state_in.particle_q, state_in.particle_qd, state_in.particle_f, self.particle_t,
                    self.particle_w, model.particle_flags, self.sub_offset, 1,
                    self.clump_x, self.clump_v, self.clump_q, self.clump_L,
                    self.inv_mass_c, self.inv_inertia_c, model.gravity, dt,
                    model.particle_max_velocity, self.max_spin, self.kill_y, self.park, self.step_arr],
            device=model.device)
