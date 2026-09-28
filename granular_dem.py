#!/usr/bin/env python3
"""
A soft-sphere DEM solver for Newton that works on open-shell STL geometry.

Newton already has the pieces for DEM; they just do not compose for chute geometry.
This assembles them and replaces the one part that does not work:

    particle-particle    newton.solvers.semi_implicit.kernels_contact.eval_particle_contact
                         (linear spring-dashpot + Coulomb, hash-grid neighbours) -- used as is
    particle-wall        REPLACED.  The stock `create_soft_contacts`
                         (newton/_src/geometry/kernels.py) takes the contact normal's sign from
                         `wp.mesh_query_point_sign_normal`, a winding-based inside/outside test
                         that assumes a closed, correctly wound mesh.  The BFA chute parts are
                         open shells, and measured against probe particles placed 3 mm off the
                         surface the sign comes out inverted on 15% of Def and 44% of Mid.  An
                         inverted normal makes `fn = n*c*ke` drive the particle *through* the
                         wall while adding energy.  This module uses unsigned distance plus an
                         average-face-normal sign, with a per-part two-sided option, which is
                         the approach already proven in this fork's MPM collider.
    integration          newton.solvers.SolverBase.integrate_particles (symplectic Euler)

Two traps this solver removes, both of which look exactly like "the particles behave like a
gas and leak out":

  * `model.particle_grid` is allocated by `finalize()` but nothing ever builds it, and
    `eval_particle_contact` silently returns when it is unbuilt -- so particle-particle
    forces vanish with no error.  `step()` rebuilds it every call.
  * DEM is unforgiving of initial overlap in a way MPM is not.  Two grains seeded 2 mm apart
    overlap by 10 mm; at k = 1000 N/m that is 10 N on a 0.9 g grain, or 11,000 m/s^2.  In a
    200-particle test, uniform-random seeding reached 137 m/s in 40 ms while lattice seeding
    stayed at 2.04 m/s.  Use `lattice_sites` (below) for injection, never uniform random.

Known limitation: Newton particles carry no angular state (`State` has particle_q, particle_qd,
particle_f and nothing else), so there is no rolling friction.  A DEM material calibrated with
sliding + rolling friction -- as BFA's corn is, at 0.09 and 0.30 for an angle of repose of
23.3 deg -- must fold both into an effective sliding coefficient.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.semi_implicit.kernels_contact import eval_particle_contact
from newton._src.solvers.solver import SolverBase

_EPS_NORMAL = wp.constant(1.0e-3)
_NO_CONTACT = wp.constant(1.0e9)


@wp.struct
class ShellCollider:
    """Open-shell collider set: per-part mesh, sidedness, thickness and material."""

    mesh: wp.array(dtype=wp.uint64)
    """Warp mesh id per part."""
    two_sided: wp.array(dtype=int)
    """1 = thin shell, repel from both faces (use when material flows over both sides).
    0 = single-sided, sign taken from the average face normal at the closest point."""
    thickness: wp.array(dtype=float)
    """Half-thickness of the shell (m); also seals seams narrower than 2*thickness."""
    friction: wp.array(dtype=float)
    """Coulomb coefficient per part."""
    ke: wp.array(dtype=float)
    """Normal contact stiffness per part (N/m)."""
    kd: wp.array(dtype=float)
    """Normal contact damping per part (N.s/m)."""
    kf: wp.array(dtype=float)
    """Tangential viscous coefficient per part (N.s/m), capped by Coulomb."""
    lower: wp.array(dtype=wp.vec3)
    """Per-part AABB lower corner, for broad-phase culling."""
    upper: wp.array(dtype=wp.vec3)
    """Per-part AABB upper corner."""
    max_dist: float
    """Query radius; must exceed particle radius + thickness."""


@wp.func
def _average_face_normal(mesh_id: wp.uint64, point: wp.vec3):
    """Average of the face normals meeting at ``point``.

    Taking a single face's normal makes the sign test flip erratically when the closest
    point lands on an edge shared by two steeply angled faces -- which is exactly where
    material was observed being pushed sideways along a seam instead of clear of it.
    """
    face_normal = wp.vec3(0.0)
    vidx = wp.mesh_get(mesh_id).indices
    points = wp.mesh_get(mesh_id).points
    eps_sq = _EPS_NORMAL * _EPS_NORMAL
    epsilon = wp.vec3(_EPS_NORMAL)
    q = wp.mesh_query_aabb(mesh_id, point - epsilon, point + epsilon)
    face_index = wp.int32(0)
    while wp.mesh_query_aabb_next(q, face_index):
        v0 = points[vidx[face_index * 3 + 0]]
        v1 = points[vidx[face_index * 3 + 1]]
        v2 = points[vidx[face_index * 3 + 2]]
        sq_dist, _c = fem.geometry.closest_point.project_on_tri_at_origin(point - v0, v1 - v0, v2 - v0)
        if sq_dist < eps_sq:
            face_normal += wp.mesh_eval_face_normal(mesh_id, face_index)
    return wp.normalize(face_normal)


@wp.kernel
def eval_shell_contact_forces(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    collider: ShellCollider,
    wall_slack: wp.array(dtype=float),
    dt: float,
    particle_f: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
):
    """Nearest-surface wall contact for each particle, with a correct outward normal.

    The particle *radius* is what makes this a DEM rather than a continuum: a gap
    narrower than a grain becomes impassable, which is the whole point.
    """
    i = wp.tid()
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        return

    x = particle_q[i]
    radius = particle_radius[i]
    v = particle_qd[i]

    # Temporal coherence.  The colliders are static, so a grain whose nearest surface
    # was D away cannot touch anything until it has travelled D - radius.  At dt = 24 us
    # a grain moves ~0.1 mm per step, so a 30 mm clearance buys ~200 query-free steps.
    #
    # The slack must only ever be credited from a distance actually MEASURED.  A BVH
    # query that misses proves the surface is farther than the search radius -- not that
    # it is infinitely far -- so the credit is capped at collider.max_dist below.  Getting
    # this wrong exempts every grain in open space from wall checks on its first step and
    # they fall straight through the geometry.
    slack = wall_slack[i] - wp.length(v) * dt
    if slack > 0.0:
        wall_slack[i] = slack
        return

    f_total = wp.vec3(0.0)
    nearest = float(_NO_CONTACT)
    hit = int(0)

    # Accumulate over EVERY part in contact rather than picking the minimum SDF.
    # Taking the min lets any one-sided part that believes the particle is behind it
    # win outright; in the cascade, small parts (SS, Def) sit inside the flow region,
    # so grains are routinely "behind" them.  With a force proportional to depth, one
    # such spurious 50 mm penetration is a 500 N kick on a 0.9 g grain.  A grain in a
    # corner genuinely touches two walls, so summing is also the physical answer.
    for m in range(collider.mesh.shape[0]):
        thick = collider.thickness[m]
        reach = radius + thick + collider.max_dist
        # Broad phase: the six cascade parts occupy a 0.34 x 0.47 x 0.20 m box at the
        # bottom, so almost every grain in the 6.8 m chute can skip them outright.
        # Without this every grain BVH-queries all seven meshes on every step.
        lo = collider.lower[m]
        hi = collider.upper[m]
        if (x[0] < lo[0] - reach or x[0] > hi[0] + reach or
            x[1] < lo[1] - reach or x[1] > hi[1] + reach or
            x[2] < lo[2] - reach or x[2] > hi[2] + reach):
            continue

        mesh = collider.mesh[m]
        query = wp.mesh_query_point_no_sign(mesh, x, reach)
        if not query.result:
            continue

        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        offset = x - cp
        d_unsigned = wp.length(offset)

        if collider.two_sided[m] == 1:
            sdf = d_unsigned - thick
            if d_unsigned < _EPS_NORMAL:
                n = wp.mesh_eval_face_normal(mesh, query.face)
            else:
                n = offset / d_unsigned
        else:
            face_n = _average_face_normal(mesh, cp)
            sign = wp.where(wp.dot(face_n, offset) > 0.0, 1.0, -1.0)
            sdf = d_unsigned * sign - thick
            if d_unsigned < _EPS_NORMAL:
                n = face_n
            else:
                n = (offset / d_unsigned) * sign

        nearest = wp.min(nearest, sdf)

        c = sdf - radius
        if c >= 0.0:
            continue

        # A physical overlap cannot exceed the grain radius.  Anything deeper is a
        # wrong-side artifact or a tunnelled grain, not a contact -- ignore it rather
        # than firing an unbounded spring.  Clamping alone is not enough: an unclamped
        # depth of 50 mm and a clamped one of 6 mm are both wrong, but only the second
        # is survivable.
        if c < -radius:
            continue

        vn = wp.dot(n, v)
        vt = v - n * vn
        fn = -c * collider.ke[m] - wp.min(vn, 0.0) * collider.kd[m]
        fn = wp.max(fn, 0.0)

        vt_len = wp.length(vt)
        if vt_len > 1.0e-8:
            f_total -= (vt / vt_len) * wp.min(collider.kf[m] * vt_len, collider.friction[m] * fn)
        f_total += n * fn
        hit += 1

    # Re-arm the cache.  Clamped at zero so a grain in contact re-queries every step.
    # Cap by max_dist: every part was searched to at least radius + thickness + max_dist,
    # and any part skipped by the AABB test is provably farther than that, so max_dist is
    # a sound lower bound on the true clearance regardless of what the queries returned.
    verified = wp.min(nearest, collider.max_dist + radius)
    wall_slack[i] = wp.max(verified - radius, 0.0)

    if hit > 0:
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(particle_f, i, f_total)


def lattice_sites(triangles: np.ndarray, spacing: float) -> np.ndarray:
    """Non-overlapping seed points covering a triangulated planar face.

    DEM explodes if particles are injected on top of each other, so injection sites are
    laid out on a lattice at ``spacing`` rather than sampled at random.  Returns the
    lattice points that fall inside the face.
    """
    pts = triangles.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    axis = int(np.argmin(hi - lo))  # the face is planar: its thin axis
    u, v = [k for k in range(3) if k != axis]

    gu = np.arange(lo[u] + spacing * 0.5, hi[u], spacing)
    gv = np.arange(lo[v] + spacing * 0.5, hi[v], spacing)
    U, V = np.meshgrid(gu, gv, indexing="ij")
    cand = np.zeros((U.size, 3))
    cand[:, u] = U.ravel()
    cand[:, v] = V.ravel()
    cand[:, axis] = 0.5 * (lo[axis] + hi[axis])

    # keep candidates inside any triangle (2D point-in-triangle on the face plane)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p = cand[:, [u, v]][:, None, :]
    a2, b2, c2 = a[:, [u, v]][None], b[:, [u, v]][None], c[:, [u, v]][None]

    def cross(o, q, r):
        return (q[..., 0] - o[..., 0]) * (r[..., 1] - o[..., 1]) - \
               (q[..., 1] - o[..., 1]) * (r[..., 0] - o[..., 0])

    d1, d2, d3 = cross(a2, b2, p), cross(b2, c2, p), cross(c2, a2, p)
    inside = (((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))).any(axis=1)
    return cand[inside]


def _simulate_restitution(v0, m_eff, dt, ke=None, hertz=None, kd=None, beta=None,
                          steps=20000):
    """Bounce one contact with the SOLVER'S OWN discrete update and return the delivered e.

    The analytic e = exp(-zeta*pi/sqrt(1-zeta^2)) does not hold here, because `fn` is
    clamped to >= 0 (no tensile force at separation).  That ends the contact early and
    truncates the rebound, so a linear spring asked for e = 0.20 actually delivers 0.33 --
    measured, and independent of dt (it converges to 0.326 as dt -> 0, so it is the clamp,
    not integration error).  Rather than trust a formula that does not apply, reproduce
    the discrete contact exactly and solve for the damping that lands on the target.

    Sign convention matches the kernels: vn is the SEPARATION rate, negative approaching.
    """
    d = 0.0                      # overlap
    vn = -abs(v0)
    for _ in range(steps):
        if hertz is None:
            fn = ke * d - kd * vn
        else:
            c, e_star, r_star = hertz
            s_n = 2.0 * e_star * math.sqrt(r_star * max(d, 0.0))
            gn = 2.0 * math.sqrt(5.0 / 6.0) * beta * math.sqrt(s_n * m_eff)
            fn = c * d * math.sqrt(max(d, 0.0)) - gn * vn
        fn = max(fn, 0.0)
        vn += (fn / m_eff) * dt
        d -= vn * dt
        if d <= 0.0:
            return max(vn, 0.0) / abs(v0)
    return float("nan")


def calibrate_damping_scale(target_e, v0, m_eff, dt, ke=None, hertz=None, kd=None,
                            beta=None, hi=6.0, n=60):
    """Find the damping multiplier that DELIVERS target_e, staying on the usable branch.

    e is NOT monotonic in damping.  Measured for the grain-wall Hertz contact at 2.4 m/s:
    scale 1.0 -> e 0.340, 2.0 -> 0.157, 4.0 -> 0.242, 8.0 -> 1.403.  Past the minimum the
    contact gets BOUNCIER, because a contact is detected at near-zero overlap where the
    spring term is ~0 and the dashpot term -kd*vn alone is a large impulse that ejects the
    grain faster than it arrived.  So there is a floor on achievable restitution, and a
    naive bisection either straddles the minimum or runs off into the unphysical branch.

    Scan upward from 1.0, stop at the minimum, then bisect only on the descending part.
    """
    def delivered(scale):
        e = _simulate_restitution(v0, m_eff, dt, ke=ke, hertz=hertz,
                                  kd=None if kd is None else kd * scale,
                                  beta=None if beta is None else beta * scale)
        return float("inf") if e != e else e

    grid = [1.0 + (hi - 1.0) * i / n for i in range(n + 1)]
    vals = [delivered(g) for g in grid]
    k_min = min(range(len(vals)), key=lambda i: vals[i])
    if vals[0] <= target_e:
        return 1.0                                  # already at or below target
    if vals[k_min] > target_e:                      # floor is above the target
        return grid[k_min]                          # give the most dissipative usable value
    lo, hi_b = 1.0, grid[k_min]
    for _ in range(60):
        mid = 0.5 * (lo + hi_b)
        if delivered(mid) > target_e:
            lo = mid
        else:
            hi_b = mid
    return 0.5 * (lo + hi_b)


class SolverGranularDEM(SolverBase):
    """Soft-sphere DEM over open-shell mesh colliders.

    Args:
        model: finalized Newton model carrying the particles.
        collider: a populated :class:`ShellCollider`.
        grid_cell: hash-grid cell size; ~2x the largest particle diameter works well.
    """

    def __init__(self, model: newton.Model, collider: ShellCollider,
                 grid_cell: float | None = None, keepalive=None, wall_cache: bool = True,
                 rotation: bool = True, mu_roll: float = 0.0, mu_roll_wall: float = 0.0,
                 rot_damp: float = 0.0, rot_damp_wall: float = 0.0,
                 tangential_ratio: float = 0.0,
                 max_spin: float = 1.0e4,
                 hertz: bool = False, youngs: float = 1.4220405e8,
                 poisson: float = 0.30, restitution: float = 0.20,
                 calibrate_restitution: bool = False, cal_dt: float = 2.4316429e-5,
                 cal_v0: float = 2.4):
        super().__init__(model=model)
        self.wall_cache = wall_cache
        self.rotation = rotation
        self.mu_roll = float(mu_roll)
        self.mu_roll_wall = float(mu_roll_wall)
        self.rot_damp = float(rot_damp)
        self.rot_damp_wall = float(rot_damp_wall)
        # k_t as a fraction of k_n.  2/7 is the Hertz-Mindlin equal-partition value and
        # the usual default in DEM codes; 0 disables the spring and restores the
        # viscous-Coulomb law, which is the A/B control.
        self.tangential_ratio = float(tangential_ratio)
        # In hertz mode the kernel multiplies this by the Mindlin St, which already carries
        # the stiffness, so the ratio must go through bare -- multiplying by ke as well made
        # the tangential spring ~2e4x too stiff.  In linear mode k_t IS a fraction of ke.
        self.k_t = self.tangential_ratio * (1.0 if hertz else float(model.particle_ke))

        # Hertz-Mindlin.  E* and G* are the reduced moduli: grain-grain has two compliant
        # bodies, grain-wall only one (steel is ~1000x stiffer than corn, so the plate's
        # compliance drops out), which makes the wall contact exactly twice as stiff.
        # beta comes from the same restitution the linear damping used, so the two modes
        # are calibrated to the same e and differ only in the SHAPE of the force law.
        self.hertz = bool(hertz)
        nu = float(poisson)
        ee = float(youngs)
        self.e_star_pp = ee / (2.0 * (1.0 - nu * nu))
        self.g_star_pp = ee / (4.0 * (1.0 + nu) * (2.0 - nu))
        self.e_star_w = ee / (1.0 - nu * nu)
        self.g_star_w = ee / (2.0 * (1.0 + nu) * (2.0 - nu))
        r = max(min(float(restitution), 0.999), 1.0e-4)
        self.beta = abs(math.log(r)) / math.sqrt(math.log(r) ** 2 + math.pi ** 2)
        self.beta_pp = self.beta
        self.beta_w = self.beta
        self.restitution = r
        self.calibrate_restitution = bool(calibrate_restitution)
        self.cal_dt = float(cal_dt)
        self.cal_v0 = float(cal_v0)
        self._calibrated = None
        self._step = 0
        self.max_spin = float(max_spin)
        self.collider = collider
        self._keepalive = keepalive
        # Cell size should match the neighbour query radius (2*r), not exceed it.  Cost
        # per query is (particles per cell) x (cells scanned) ~ rho*c^3*(2*ceil(rq/c)+1)^3,
        # which for c = 2*rq is 8x worse than for c = rq: the cell count stays at 27 while
        # each cell holds 8x more grains.
        self.grid_cell = grid_cell if grid_cell else 2.0 * float(model.particle_max_radius)
        if model.particle_grid is None:
            model.particle_grid = wp.HashGrid(128, 128, 128, device=model.device)
        self.contact_count = wp.zeros(1, dtype=int, device=model.device)
        self.wall_slack = wp.zeros(model.particle_count, dtype=float, device=model.device)
        ns = int(TANGENTIAL_SLOTS)
        self.tang_partner = wp.full((model.particle_count, ns), -1, dtype=int,
                                    device=model.device)
        self.tang_stamp = wp.full((model.particle_count, ns), -10, dtype=int,
                                  device=model.device)
        self.tang_xi = wp.zeros((model.particle_count, ns), dtype=wp.vec3,
                                device=model.device)

        n = model.particle_count
        self.particle_w = wp.zeros(n, dtype=wp.vec3, device=model.device)
        self.particle_t = wp.zeros(n, dtype=wp.vec3, device=model.device)
        # solid sphere: I = 0.4*m*r^2, so I^-1 = 2.5/(m*r^2).  Derived, not stored per se.
        mass = 1.0 / np.maximum(model.particle_inv_mass.numpy(), 1e-30)
        rad = np.maximum(model.particle_radius.numpy(), 1e-9)
        self.particle_inv_inertia = wp.array(
            (2.5 / (mass * rad * rad)).astype(np.float32), dtype=float, device=model.device)

        if self.calibrate_restitution:
            m_typ = float(np.median(mass))
            r_typ = float(np.median(rad))
            if self.hertz:
                c_pp = (4.0 / 3.0) * self.e_star_pp * math.sqrt(0.5 * r_typ)
                c_w = (4.0 / 3.0) * self.e_star_w * math.sqrt(r_typ)
                s_pp = calibrate_damping_scale(
                    self.restitution, self.cal_v0, 0.5 * m_typ, self.cal_dt,
                    hertz=(c_pp, self.e_star_pp, 0.5 * r_typ), beta=self.beta)
                s_w = calibrate_damping_scale(
                    self.restitution, self.cal_v0, m_typ, self.cal_dt,
                    hertz=(c_w, self.e_star_w, r_typ), beta=self.beta)
                self.beta_pp = self.beta * s_pp
                self.beta_w = self.beta * s_w
            else:
                ke = float(model.particle_ke)
                s_pp = calibrate_damping_scale(
                    self.restitution, self.cal_v0, 0.5 * m_typ, self.cal_dt,
                    ke=ke, kd=float(model.particle_kd))
                kdw = float(collider.kd.numpy().mean())
                s_w = calibrate_damping_scale(
                    self.restitution, self.cal_v0, m_typ, self.cal_dt, ke=ke, kd=kdw)
                model.particle_kd = float(model.particle_kd) * s_pp
                collider.kd = wp.array(collider.kd.numpy() * s_w, dtype=float,
                                       device=model.device)
                self._kd_keepalive = collider.kd
            self._calibrated = (s_pp, s_w)
            print(f"  restitution calibration  e {self.restitution:.3f} requested -> "
                  f"damping x {s_pp:.3f} grain-grain, x {s_w:.3f} grain-wall")

    def invalidate_cache(self):
        """Clear the wall-distance cache.  Call after teleporting particles (injection
        or recycling), otherwise a grain inherits the slack of whatever occupied its
        index before and can start inside a wall."""
        self.wall_slack.zero_()

    def step(self, state_in, state_out, control, contacts, dt: float):
        model = self.model
        self._step += 1
        state_in.clear_forces()

        # Rebuild every step: eval_particle_contact returns immediately on an unbuilt
        # grid, which silently removes all particle-particle forces.
        model.particle_grid.build(state_in.particle_q, self.grid_cell)

        if self.rotation:
            self.particle_t.zero_()
            wp.launch(
                kernel=eval_particle_contact_rot,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id, state_in.particle_q, state_in.particle_qd,
                    self.particle_w, model.particle_radius, model.particle_flags,
                    model.particle_inv_mass,
                    model.particle_ke, model.particle_kd, model.particle_kf,
                    model.particle_mu, self.mu_roll, self.rot_damp, model.particle_max_radius,
                    self.k_t, int(self.hertz), self.e_star_pp, self.g_star_pp, self.beta_pp,
                    dt, self._step,
                    self.tang_partner, self.tang_stamp, self.tang_xi,
                ],
                outputs=[state_in.particle_f, self.particle_t],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=eval_particle_contact,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_ke,
                    model.particle_kd,
                    model.particle_kf,
                    model.particle_mu,
                    model.particle_cohesion,
                    model.particle_max_radius,
                ],
                outputs=[state_in.particle_f],
                device=model.device,
            )

        self.contact_count.zero_()
        if not self.wall_cache:
            # Force a full query every step.  The cache is meant to be exact, so results
            # with it on and off must match -- that equality is the regression test.
            self.wall_slack.zero_()
        if self.rotation:
            wp.launch(
                kernel=eval_shell_contact_forces_rot,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q, state_in.particle_qd, self.particle_w,
                    model.particle_radius, model.particle_inv_mass, model.particle_flags,
                    self.collider,
                    self.mu_roll_wall, self.rot_damp_wall, self.k_t,
                    int(self.hertz), self.e_star_w, self.g_star_w, self.beta_w, self._step,
                    self.tang_partner, self.tang_stamp, self.tang_xi,
                    self.wall_slack, dt,
                ],
                outputs=[state_in.particle_f, self.particle_t, self.contact_count],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=eval_shell_contact_forces,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    self.collider,
                    self.wall_slack,
                    dt,
                ],
                outputs=[state_in.particle_f, self.contact_count],
                device=model.device,
            )

        self.integrate_particles(model, state_in, state_out, dt)

        if self.rotation:
            wp.launch(
                kernel=integrate_angular,
                dim=model.particle_count,
                inputs=[self.particle_w, self.particle_t, self.particle_inv_inertia,
                        model.particle_flags, dt, self.max_spin],
                device=model.device,
            )


def build_collider(parts, two_sided, friction, ke, kd, kf, thickness, max_dist, device):
    """Pack per-part meshes and materials into a :class:`ShellCollider`.

    ``parts`` is a list of (name, vertices, faces).  The other arguments are per-part
    sequences, in the same order.
    """
    meshes = []
    for _name, v, f in parts:
        pts = wp.array(np.asarray(v, dtype=np.float32), dtype=wp.vec3, device=device)
        idx = wp.array(np.asarray(f, dtype=np.int32).flatten(), dtype=int, device=device)
        meshes.append(wp.Mesh(pts, idx))

    c = ShellCollider()
    c.mesh = wp.array([m.id for m in meshes], dtype=wp.uint64, device=device)
    c.two_sided = wp.array([1 if t else 0 for t in two_sided], dtype=int, device=device)
    c.thickness = wp.array(thickness, dtype=float, device=device)
    c.friction = wp.array(friction, dtype=float, device=device)
    c.ke = wp.array(ke, dtype=float, device=device)
    c.kd = wp.array(kd, dtype=float, device=device)
    c.kf = wp.array(kf, dtype=float, device=device)
    lo = np.array([np.asarray(v).min(axis=0) for _n, v, _f in parts], dtype=np.float32)
    hi = np.array([np.asarray(v).max(axis=0) for _n, v, _f in parts], dtype=np.float32)
    c.lower = wp.array(lo, dtype=wp.vec3, device=device)
    c.upper = wp.array(hi, dtype=wp.vec3, device=device)
    c.max_dist = float(max_dist)
    # wp.Mesh objects must outlive the struct: it stores only their integer ids, so if
    # they are garbage collected the kernel reads freed BVHs.  Returned for the caller
    # to hold (SolverGranularDEM keeps them via its ``keepalive`` argument).
    return c, meshes


# ---------------------------------------------------------------------------
# Rotational DEM
#
# Newton's `State` carries only particle_q / particle_qd / particle_f -- no angular
# state -- so spheres can slide but never roll.  That is not a cosmetic omission for
# this material: BFA calibrates corn with sliding friction 0.09 AND rolling friction
# 0.30, and it is the pair that yields the 23.3 deg angle of repose.  It also has a
# mechanical consequence we measured: a sphere that cannot rotate can only slide past
# its neighbour, so at a constriction it arches.  At mu = 0.43 the chute/cascade
# junction plugged and passed 4.5 kg/s of an 8.61 kg/s feed; only by halving mu to
# 0.20 -- well below the material's real internal friction -- did it flow.
#
# Angular velocity lives on the solver, not on newton.State, so nothing in Newton's
# core needs to change.  Orientation is not tracked: for spheres with a viscous-Coulomb
# tangential law there is no need for a contact history spring.
#
# Inertia is derived rather than stored: a solid sphere has I = 0.4*m*r^2.
# ---------------------------------------------------------------------------


@wp.func
def _rolling_torque(mu_roll: float, fn: float, r_eff: float, w_rel: wp.vec3):
    """Constant-directional-torque rolling resistance (the LIGGGHTS/BFA model).

    Opposes relative rolling with a torque of fixed magnitude mu_r * |Fn| * r_eff,
    which is what BFA's "Rotating Friction" coefficient means.
    """
    w_len = wp.length(w_rel)
    if w_len < 1.0e-8:
        return wp.vec3(0.0)
    return -(w_rel / w_len) * (mu_roll * fn * r_eff)


TANGENTIAL_SLOTS = wp.constant(12)
"""Contact-history slots per grain.  Dense random sphere packings have a coordination
number of 6-8, so 12 covers grain-grain plus a couple of walls with margin."""


@wp.func
def _history_slot(
    partner: wp.array2d(dtype=int),
    stamp: wp.array2d(dtype=int),
    i: int,
    j: int,
    step: int,
):
    """Find grain i's history slot for partner j, or a reusable one.

    The tangential spring needs the accumulated slip of THIS contact, carried across
    steps -- but the hash grid renumbers neighbours every step, so the pairing has to be
    stored explicitly.  A slot is live if it was touched on the previous step; anything
    older is a contact that has since broken and may be recycled.  Walls are partners
    with negative ids, so the same table serves both.
    """
    free = int(-1)
    for k in range(TANGENTIAL_SLOTS):
        p = partner[i, k]
        if p == j and stamp[i, k] >= step - 1:
            return k
        if free < 0 and (p == -1 or stamp[i, k] < step - 1):
            free = k
    return free


@wp.func
def _tangential_spring(
    xi_arr: wp.array2d(dtype=wp.vec3),
    partner: wp.array2d(dtype=int),
    stamp: wp.array2d(dtype=int),
    i: int,
    slot: int,
    j: int,
    step: int,
    n: wp.vec3,
    vt: wp.vec3,
    dt: float,
    k_t: float,
    f_coulomb: float,
):
    """Mindlin-style tangential force: an elastic spring on accumulated slip, capped by
    Coulomb.

    This is what distinguishes sticking from creeping.  A viscous law (ft = kf*|vt|,
    capped) produces a force only while the contact is sliding, so a grain under load
    always creeps; a spring holds a stored displacement and pushes back even at zero
    slip rate, which is what lets force chains carry load in a dense slow layer.

    On yield the stored slip is rescaled back onto the friction cone rather than left to
    grow without bound, which is the standard Cundall-Strack treatment.
    """
    fresh = partner[i, slot] != j or stamp[i, slot] < step - 1
    xi = wp.vec3(0.0)
    if not fresh:
        xi = xi_arr[i, slot]
        # the contact normal rotates as grains move: keep the stored slip tangential
        xi = xi - n * wp.dot(xi, n)
    xi = xi + vt * dt

    ft = -xi * k_t
    mag = wp.length(ft)
    if mag > f_coulomb and mag > 1.0e-12:
        ft = ft * (f_coulomb / mag)
        if k_t > 0.0:
            xi = -ft / k_t

    partner[i, slot] = j
    stamp[i, slot] = step
    xi_arr[i, slot] = xi
    return ft


@wp.func
def _rotational_damping(eta: float, kd: float, r_eff: float, w_rel: wp.vec3):
    """Viscous torque opposing relative rotation -- the rotational analogue of the
    normal dashpot, which we already have and BFA also has (ContactR = 0.20).

    BFA additionally specifies RotatingR = 0.20 and TwistingR = 0.20 per contact pair,
    dissipation channels this solver had no equivalent for: rolling *friction* is a
    Coulomb-like torque of fixed magnitude, not something proportional to spin rate.  In
    a tumbling cascade the grains spin hard, so this is the sink that removes energy
    where the material should be slowing, without adding the tangential locking that
    makes higher sliding friction jam the constriction.

    Scaled as eta * kd * r_eff^2 so the units work out (N.s/m * m^2 = N.m.s) and the
    strength tracks the normal damping rather than being an unrelated free number.
    """
    return -w_rel * (eta * kd * r_eff * r_eff)


@wp.kernel
def eval_particle_contact_rot(
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
    step: int,
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """Grain-grain contact with rotation: relative velocity at the contact point,
    torque from the tangential force, and rolling resistance."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return  # grid not built -- see SolverGranularDEM.step
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    w = particle_w[i]
    ri = particle_radius[i]

    f = wp.vec3(0.0)
    t = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, ri + max_radius)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
            continue

        rj = particle_radius[index]
        d_vec = x - particle_x[index]
        d = wp.length(d_vec)
        if d < 1.0e-9 or d >= ri + rj:
            continue

        n = d_vec / d
        overlap = ri + rj - d

        # velocity at the contact point, not at the centres: this is what rotation adds
        c_i = -n * ri
        c_j = n * rj
        v_rel = (v + wp.cross(w, c_i)) - (particle_v[index] + wp.cross(particle_w[index], c_j))

        vn = wp.dot(v_rel, n)
        vt = v_rel - n * vn

        # Hertz: fn = (4/3) E* sqrt(R*) d^1.5, so the TANGENT stiffness rises as sqrt(d)
        # and the contact is soft under light load, stiff under impact.  A single linear
        # k_n cannot do both -- measured on this machine it spans 5.6x (scratchpad/
        # hertzmap.py), and matching it to the plug leaves the impact front ~9x too soft.
        kd_eff = k_d
        kt_eff = k_t
        kf_eff = k_f
        fn = float(0.0)
        if hertz == 1:
            r_star = ri * rj / (ri + rj)
            imi = particle_inv_mass[i]
            imj = particle_inv_mass[index]
            m_star = 1.0 / wp.max(imi + imj, 1.0e-12)
            sq = wp.sqrt(r_star * overlap)
            s_n = 2.0 * e_star * sq
            s_t = 8.0 * g_star * sq
            kd_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_n * m_star)
            kf_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_t * m_star)
            if k_t > 0.0:
                kt_eff = k_t * s_t          # ratio 1.0 is textbook Mindlin
            fn = (4.0 / 3.0) * e_star * wp.sqrt(r_star) * overlap * wp.sqrt(overlap) \
                - kd_eff * vn
        else:
            fn = k_n * overlap - k_d * vn
        if fn < 0.0:
            continue

        ft = wp.vec3(0.0)
        if kt_eff > 0.0:
            slot = _history_slot(tang_partner, tang_stamp, i, index, step)
            if slot >= 0:
                ft = _tangential_spring(tang_xi, tang_partner, tang_stamp, i, slot,
                                        index, step, n, vt, dt, kt_eff, mu * fn)
            else:
                # table full (rare): fall back to the viscous-Coulomb law
                vl = wp.length(vt)
                if vl > 1.0e-8:
                    ft = -(vt / vl) * wp.min(kf_eff * vl, mu * fn)
        else:
            vt_len = wp.length(vt)
            if vt_len > 1.0e-8:
                ft = -(vt / vt_len) * wp.min(kf_eff * vt_len, mu * fn)

        f += n * fn + ft
        t += wp.cross(c_i, ft)

        if mu_roll > 0.0 or rot_damp > 0.0:
            r_eff = ri * rj / (ri + rj)
            w_rel = w - particle_w[index]
            if mu_roll > 0.0:
                t += _rolling_torque(mu_roll, fn, r_eff, w_rel)
            if rot_damp > 0.0:
                t += _rotational_damping(rot_damp, kd_eff, r_eff, w_rel)

    wp.atomic_add(particle_f, i, f)
    wp.atomic_add(particle_t, i, t)


@wp.kernel
def eval_shell_contact_forces_rot(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    collider: ShellCollider,
    mu_roll: float,
    rot_damp: float,
    k_t: float,
    hertz: int,
    e_star: float,
    g_star: float,
    beta: float,
    step: int,
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    wall_slack: wp.array(dtype=float),
    dt: float,
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
):
    """Grain-wall contact with rotation.  Same geometry handling as the non-rotating
    version (unsigned distance, average-normal sign, per-part two-sided, AABB cull,
    wall-distance cache); the wall is static so it contributes no velocity."""
    i = wp.tid()
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        return

    x = particle_q[i]
    radius = particle_radius[i]
    v = particle_qd[i]
    w = particle_w[i]

    slack = wall_slack[i] - wp.length(v) * dt
    if slack > 0.0:
        wall_slack[i] = slack
        return

    f_total = wp.vec3(0.0)
    t_total = wp.vec3(0.0)
    nearest = float(_NO_CONTACT)
    hit = int(0)

    for m in range(collider.mesh.shape[0]):
        thick = collider.thickness[m]
        reach = radius + thick + collider.max_dist
        lo = collider.lower[m]
        hi = collider.upper[m]
        if (x[0] < lo[0] - reach or x[0] > hi[0] + reach or
            x[1] < lo[1] - reach or x[1] > hi[1] + reach or
            x[2] < lo[2] - reach or x[2] > hi[2] + reach):
            continue

        mesh = collider.mesh[m]
        query = wp.mesh_query_point_no_sign(mesh, x, reach)
        if not query.result:
            continue

        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        offset = x - cp
        d_unsigned = wp.length(offset)

        if collider.two_sided[m] == 1:
            sdf = d_unsigned - thick
            if d_unsigned < _EPS_NORMAL:
                n = wp.mesh_eval_face_normal(mesh, query.face)
            else:
                n = offset / d_unsigned
        else:
            face_n = _average_face_normal(mesh, cp)
            sign = wp.where(wp.dot(face_n, offset) > 0.0, 1.0, -1.0)
            sdf = d_unsigned * sign - thick
            if d_unsigned < _EPS_NORMAL:
                n = face_n
            else:
                n = (offset / d_unsigned) * sign

        nearest = wp.min(nearest, sdf)

        c = sdf - radius
        if c >= 0.0 or c < -radius:
            continue

        c_arm = -n * radius
        v_rel = v + wp.cross(w, c_arm)
        vn = wp.dot(v_rel, n)
        vt = v_rel - n * vn

        # grain against a rigid plate: R* = radius, m* = m (the wall never recoils)
        kd_eff = collider.kd[m]
        kt_eff = k_t
        kf_eff = collider.kf[m]
        fn = float(0.0)
        if hertz == 1:
            delta = -c
            m_star = 1.0 / wp.max(particle_inv_mass[i], 1.0e-12)
            sq = wp.sqrt(radius * delta)
            s_n = 2.0 * e_star * sq
            s_t = 8.0 * g_star * sq
            kd_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_n * m_star)
            kf_eff = 2.0 * wp.sqrt(5.0 / 6.0) * beta * wp.sqrt(s_t * m_star)
            if k_t > 0.0:
                kt_eff = k_t * s_t
            fn = (4.0 / 3.0) * e_star * wp.sqrt(radius) * delta * wp.sqrt(delta) \
                - vn * kd_eff
        else:
            fn = -c * collider.ke[m] - vn * collider.kd[m]
        fn = wp.max(fn, 0.0)

        ft = wp.vec3(0.0)
        if kt_eff > 0.0:
            # walls occupy the same history table, keyed by negative partner ids
            wid = -(m + 2)
            slot = _history_slot(tang_partner, tang_stamp, i, wid, step)
            if slot >= 0:
                ft = _tangential_spring(tang_xi, tang_partner, tang_stamp, i, slot,
                                        wid, step, n, vt, dt, kt_eff,
                                        collider.friction[m] * fn)
            else:
                vl = wp.length(vt)
                if vl > 1.0e-8:
                    ft = -(vt / vl) * wp.min(kf_eff * vl, collider.friction[m] * fn)
        else:
            vt_len = wp.length(vt)
            if vt_len > 1.0e-8:
                ft = -(vt / vt_len) * wp.min(kf_eff * vt_len, collider.friction[m] * fn)

        f_total += n * fn + ft
        t_total += wp.cross(c_arm, ft)
        if mu_roll > 0.0:
            t_total += _rolling_torque(mu_roll, fn, radius, w)
        if rot_damp > 0.0:
            # the wall does not rotate, so the relative spin is just the grain's
            t_total += _rotational_damping(rot_damp, kd_eff, radius, w)
        hit += 1

    verified = wp.min(nearest, collider.max_dist + radius)
    wall_slack[i] = wp.max(verified - radius, 0.0)

    if hit > 0:
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(particle_f, i, f_total)
        wp.atomic_add(particle_t, i, t_total)


@wp.kernel
def integrate_angular(
    particle_w: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
    particle_inv_inertia: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    max_spin: float,
):
    """w += I^-1 * torque * dt, for a solid sphere (I = 0.4*m*r^2)."""
    i = wp.tid()
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        particle_w[i] = wp.vec3(0.0)
        return
    w = particle_w[i] + particle_t[i] * particle_inv_inertia[i] * dt
    s = wp.length(w)
    if s > max_spin:
        w = w * (max_spin / s)
    particle_w[i] = w
