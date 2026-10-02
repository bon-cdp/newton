#!/usr/bin/env python3
"""
Multi-sphere clumps (rigid assemblies of overlapping spheres) on top of SolverGranularDEM
-- the first, opt-in step of issue #10.

Design (as in #10): the sub-spheres stay ordinary particles, so every contact kernel, the
hash grid, the baked wall grid and the contact history work unchanged.  Clumps are rigid
bodies on top:

  1. grain-grain contact as eval_particle_contact_rot, skipping pairs of the SAME clump;
     a clump's sub-spheres are contiguous (clump c owns particles [c*n, c*n + n)), so the
     clump of particle i is i // n and no extra array is read;
  2. grain-wall contact: the parent's kernels, unchanged;
  3. integrate_clumps, one thread per clump: F = sum f_s, T = sum (r_s x f_s + tau_s);
     translation as integrate_fused; rotation by the world-frame angular momentum
     (L += T dt, w = R I_body^-1 R^T L -- the gyroscopic term is implicit in carrying L),
     quaternion update q += dt/2 (w,0) q, renormalised; then the sub-spheres are written
     back: x_s = x + R o_s, v_s = v + w x (R o_s), w_s = w.

Every sub-sphere carries the CLUMP's inverse mass, so the Hertz damping and the wall
contact see the body's mass, as is usual for multi-sphere DEM.

Kept in its own module while the calibration sweeps run on granular_dem.py; spheres are
untouched.  Single-template only for now (one shape, any number of sub-spheres).
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from granular_dem import (SolverGranularDEM, _pair_contact, eval_shell_contact_forces_rot,
                          eval_wall_corners_rot, eval_wall_grid_rot)


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------
class ClumpTemplate:
    """Sub-sphere centres (body frame, metres) and radii.  Mass properties are integrated
    over the UNION of the spheres on a voxel grid (overlaps counted once); the template is
    then shifted to its centre of mass and rotated into its principal axes, so the body
    inertia is diagonal."""

    def __init__(self, offsets, radii, density: float, voxels: int = 96):
        o = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
        r = np.asarray(radii, dtype=np.float64).reshape(-1)
        if len(r) == 1 and len(o) > 1:
            r = np.full(len(o), r[0])
        lo = (o - r[:, None]).min(axis=0)
        hi = (o + r[:, None]).max(axis=0)
        h = float((hi - lo).max()) / voxels
        axes = [np.arange(lo[k] + 0.5 * h, hi[k], h) for k in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        inside = np.zeros(len(P), dtype=bool)
        for c, rr in zip(o, r):
            inside |= ((P - c) ** 2).sum(axis=1) <= rr * rr
        P = P[inside]
        dv = h ** 3
        self.volume = len(P) * dv
        self.mass = density * self.volume
        com = P.mean(axis=0)
        Q = P - com
        Iten = density * dv * (np.eye(3) * (Q ** 2).sum() - Q.T @ Q)
        evals, evecs = np.linalg.eigh(Iten)
        if np.linalg.det(evecs) < 0:
            evecs[:, 2] *= -1.0
        self.inertia = evals                        # principal moments (body frame)
        self.offsets = (o - com) @ evecs            # body frame = principal axes
        self.radii = r
        self.n = len(r)
        self.bound_radius = float((np.linalg.norm(self.offsets, axis=1) + r).max())
        self.equiv_diameter = (6.0 * self.volume / math.pi) ** (1.0 / 3.0)

    def describe(self):
        ext = [(self.offsets[:, k] + self.radii).max() - (self.offsets[:, k] - self.radii).min()
               for k in range(3)]
        return (f"{self.n} spheres, extents {', '.join(f'{e*1e3:.2f}' for e in ext)} mm, "
                f"volume-equivalent dia {self.equiv_diameter*1e3:.2f} mm, mass "
                f"{self.mass*1e3:.4f} g, I {', '.join(f'{v:.3g}' for v in self.inertia)} kg m2")


def random_quats(n, rng):
    """Uniform random unit quaternions (x, y, z, w) -- Shoemake."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    return np.stack([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2),
                     b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3)], 1)


def quat_to_R(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@wp.kernel
def eval_particle_contact_clump(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_inv_mass: wp.array(dtype=float),
    k_n: float,
    k_d: float,
    k_f: float,
    mu: float,
    mu_roll: float,
    rot_damp: float,
    max_radius: float,
    k_t: float,
    hertz: int,
    e_star: float,
    g_star: float,
    beta: float,
    dt: float,
    step_arr: wp.array(dtype=int),
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    clump_size: int,
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """eval_particle_contact_rot, skipping sub-spheres of the same clump."""
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
    ci = i // clump_size
    f = wp.vec3(0.0)
    t = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, x, ri + max_radius)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index // clump_size == ci:
            continue                       # itself, or a sub-sphere of the same body
        if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
            continue
        df, dtq = _pair_contact(i, index, x, v, w, ri, particle_x, particle_v, particle_w,
                                particle_radius, particle_inv_mass, k_n, k_d, k_f, mu,
                                mu_roll, rot_damp, k_t, hertz, e_star, g_star, beta, dt, step,
                                tang_partner, tang_stamp, tang_xi)
        f += df
        t += dtq
    particle_f[i] = f
    particle_t[i] = t


@wp.kernel
def integrate_clumps(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    sub_offset: wp.array(dtype=wp.vec3),
    clump_size: int,
    clump_x: wp.array(dtype=wp.vec3),
    clump_v: wp.array(dtype=wp.vec3),
    clump_q: wp.array(dtype=wp.quat),
    clump_L: wp.array(dtype=wp.vec3),
    inv_mass: float,
    inv_inertia: wp.vec3,
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    max_spin: float,
    kill_y: float,
    park: wp.vec3,
    step_arr: wp.array(dtype=int),
):
    c = wp.tid()
    if c == 0:
        step_arr[0] = step_arr[0] + 1      # every reader of this step has finished
    i0 = c * clump_size
    if (particle_flags[i0] & newton.ParticleFlags.ACTIVE) == 0:
        return
    xc = clump_x[c]
    F = wp.vec3(0.0)
    T = wp.vec3(0.0)
    for k in range(clump_size):
        i = i0 + k
        fi = particle_f[i]
        F += fi
        T += wp.cross(particle_x[i] - xc, fi) + particle_t[i]

    v1 = clump_v[c] + (F * inv_mass + gravity[0]) * dt
    sp = wp.length(v1)
    if sp > v_max:
        v1 *= v_max / sp
    x1 = xc + v1 * dt

    if x1[1] < kill_y:
        # fell off the belt: retire the whole body (it never comes back)
        for k in range(clump_size):
            i = i0 + k
            particle_flags[i] = wp.int32(0)
            particle_x[i] = park
            particle_v[i] = wp.vec3(0.0)
            particle_w[i] = wp.vec3(0.0)
        return

    q = clump_q[c]
    L = clump_L[c] + T * dt
    R = wp.quat_to_matrix(q)
    wb = wp.transpose(R) * L
    w = R * wp.cw_mul(inv_inertia, wb)
    ws = wp.length(w)
    if ws > max_spin:
        L = L * (max_spin / ws)
        w = w * (max_spin / ws)
    q1 = wp.normalize(q + wp.quat(w[0], w[1], w[2], 0.0) * q * (0.5 * dt))
    R1 = wp.quat_to_matrix(q1)
    w1 = R1 * wp.cw_mul(inv_inertia, wp.transpose(R1) * L)

    clump_x[c] = x1
    clump_v[c] = v1
    clump_q[c] = q1
    clump_L[c] = L
    for k in range(clump_size):
        i = i0 + k
        o = R1 * sub_offset[i]
        particle_x[i] = x1 + o
        particle_v[i] = v1 + wp.cross(w1, o)
        particle_w[i] = w1


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class SolverGranularClumps(SolverGranularDEM):
    """SolverGranularDEM whose particles are the sub-spheres of identical clumps.

    Supports the rotation path with the hash grid (no neighbour list) and either wall path
    -- the REFERENCE preset.  Call set_clumps() before stepping."""

    def __init__(self, model, collider, **kw):
        if kw.get("neighbor_every", 0):
            raise NotImplementedError("clumps: neighbour list not supported yet (reference preset only)")
        if not kw.get("rotation", True):
            raise NotImplementedError("clumps need rotation")
        super().__init__(model, collider, **kw)
        self.clump_size = 1

    def set_clumps(self, template: ClumpTemplate, com, quats, vel=None, kill_y=-1.0e9,
                   park=(0.0, 1.0e3, 0.0)):
        """Place n identical clumps (com: (n,3), quats: (n,4) x,y,z,w) on particles
        [0, n*template.n); writes positions, velocities, radii, inverse masses and ACTIVE
        flags of those particles.  Particles beyond them are left alone."""
        model, dev = self.model, self.model.device
        n, k = len(com), template.n
        if n * k > model.particle_count:
            raise ValueError(f"{n} clumps x {k} spheres > {model.particle_count} particles")
        com = np.asarray(com, dtype=np.float64)
        quats = np.asarray(quats, dtype=np.float64)
        vel = np.zeros((n, 3)) if vel is None else np.asarray(vel, dtype=np.float64)
        R = quat_to_R(quats)                                  # (n,3,3)
        off = np.einsum("nij,kj->nki", R, template.offsets)   # (n,k,3)
        x = (com[:, None, :] + off).reshape(-1, 3)
        npart = model.particle_count
        pq = self._state_q.numpy()
        pqd = self._state_qd.numpy()
        pq[: n * k] = x
        pqd[: n * k] = np.repeat(vel, k, axis=0)
        self._state_q.assign(pq.astype(np.float32))
        self._state_qd.assign(pqd.astype(np.float32))
        rad = model.particle_radius.numpy()
        rad[: n * k] = np.tile(template.radii, n)
        model.particle_radius.assign(rad.astype(np.float32))
        im = model.particle_inv_mass.numpy()
        im[: n * k] = 1.0 / template.mass
        model.particle_inv_mass.assign(im.astype(np.float32))
        fl = model.particle_flags.numpy()
        fl[: n * k] = int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(fl)
        model.particle_max_radius = float(max(model.particle_max_radius, template.radii.max()))
        offs = np.zeros((npart, 3), dtype=np.float32)
        offs[: n * k] = np.tile(template.offsets, (n, 1))
        self.sub_offset = wp.array(offs, dtype=wp.vec3, device=dev)
        self.clump_x = wp.array(com.astype(np.float32), dtype=wp.vec3, device=dev)
        self.clump_v = wp.array(vel.astype(np.float32), dtype=wp.vec3, device=dev)
        self.clump_q = wp.array(quats.astype(np.float32), dtype=wp.quat, device=dev)
        self.clump_L = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.n_clumps = n
        self.clump_size = k
        self.template = template
        self.inv_mass_c = 1.0 / template.mass
        self.inv_inertia_c = wp.vec3(*(1.0 / template.inertia))
        self.kill_y = float(kill_y)
        self.park = wp.vec3(*park)
        self.particle_w.zero_()
        self.invalidate_cache()

    def bind_state(self, state):
        """The state whose particle_q / particle_qd set_clumps writes (stepping is in place)."""
        self._state_q, self._state_qd = state.particle_q, state.particle_qd

    def step(self, state_in, state_out, control, contacts, dt: float):
        if state_out is not state_in:
            raise ValueError("clumps step in place (state_out must be state_in)")
        model = self.model
        self._step += 1
        model.particle_grid.build(state_in.particle_q, self.grid_cell)
        wp.launch(
            kernel=eval_particle_contact_clump,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id, state_in.particle_q, state_in.particle_qd,
                self.particle_w, model.particle_radius, model.particle_flags,
                model.particle_inv_mass,
                model.particle_ke, model.particle_kd, model.particle_kf,
                model.particle_mu, self.mu_roll, self.rot_damp, model.particle_max_radius,
                self.k_t, int(self.hertz), self.e_star_pp, self.g_star_pp, self.beta_pp,
                dt, self.step_arr,
                self.tang_partner, self.tang_stamp, self.tang_xi, self.clump_size,
            ],
            outputs=[state_in.particle_f, self.particle_t],
            device=model.device,
        )
        if self.wall_grid is not None:
            wp.launch(
                kernel=eval_wall_grid_rot,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q, state_in.particle_qd, self.particle_w,
                    model.particle_radius, model.particle_inv_mass, model.particle_flags,
                    self.collider, self.wall_grid,
                    self.mu_roll_wall, self.rot_damp_wall, self.k_t,
                    int(self.hertz), self.e_star_w, self.g_star_w, self.beta_w, self.step_arr,
                    self.tang_partner, self.tang_stamp, self.tang_xi,
                    dt, model.particle_grid.id,
                    0, self.wl, self.wl_count, self.skin_arr, self.x_build,
                ],
                outputs=[state_in.particle_f, self.particle_t],
                device=model.device,
            )
            if self.any_corners:
                wp.launch(
                    kernel=eval_wall_corners_rot,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q, state_in.particle_qd, self.particle_w,
                        model.particle_radius, model.particle_inv_mass, model.particle_flags,
                        self.collider, self.wall_grid,
                        self.mu_roll_wall, self.rot_damp_wall, self.k_t,
                        int(self.hertz), self.e_star_w, self.g_star_w, self.beta_w, self.step_arr,
                        self.tang_partner, self.tang_stamp, self.tang_xi,
                        dt, model.particle_grid.id,
                        0, self.wl, self.wl_count, self.wl_corner, self.skin_arr,
                        self.x_build,
                    ],
                    outputs=[state_in.particle_f, self.particle_t],
                    device=model.device,
                )
        else:
            self.contact_count.zero_()
            if not self.wall_cache:
                self.wall_slack.zero_()
            wp.launch(
                kernel=eval_shell_contact_forces_rot,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q, state_in.particle_qd, self.particle_w,
                    model.particle_radius, model.particle_inv_mass, model.particle_flags,
                    self.collider,
                    self.mu_roll_wall, self.rot_damp_wall, self.k_t,
                    int(self.hertz), self.e_star_w, self.g_star_w, self.beta_w, self.step_arr,
                    self.tang_partner, self.tang_stamp, self.tang_xi,
                    self.wall_slack, dt, model.particle_grid.id,
                ],
                outputs=[state_in.particle_f, self.particle_t, self.contact_count],
                device=model.device,
            )
        wp.launch(
            kernel=integrate_clumps,
            dim=self.n_clumps,
            inputs=[
                state_in.particle_q, state_in.particle_qd, state_in.particle_f, self.particle_t,
                self.particle_w, model.particle_flags, self.sub_offset, self.clump_size,
                self.clump_x, self.clump_v, self.clump_q, self.clump_L,
                self.inv_mass_c, self.inv_inertia_c, model.gravity, dt,
                model.particle_max_velocity, self.max_spin, self.kill_y, self.park, self.step_arr,
            ],
            device=model.device,
        )
