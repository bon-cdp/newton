#!/usr/bin/env python3
"""
A soft-sphere DEM solver for Newton that works on open-shell STL geometry, on the GPU.

Newton has DEM pieces (SolverSemiImplicit's spring-dashpot contact, a hash grid) but they do
not compose for chute geometry, and Newton particles carry no angular state.  This module
supplies what is missing; nothing in newton/_src needs to change.

    grain-grain      Hertz-Mindlin or linear spring-dashpot, Coulomb friction, Mindlin
                     tangential history spring, rolling friction (constant directional
                     torque) and rotational damping.  Angular velocity lives on the solver.
    grain-wall       Unsigned distance + average-face-normal sign with per-part two-sided
                     option.  The stock `create_soft_contacts` takes its sign from a winding
                     test that needs a closed mesh; on the BFA open shells it came out
                     inverted on 15% of Def and 44% of Mid, driving grains through walls.
    integration      symplectic Euler, linear and angular fused into one kernel.

Performance structure (measured on a Quadro P5000; see tools/dem_bench.py):

    WallGrid         walls are static, so the broad phase is baked once into a uniform grid
                     of candidate-triangle lists (CSR, triangle records inlined).  Per-step
                     BVH queries cost 1.2 ms of a 1.7 ms step; the grid ~0.15 ms.  Triangle
                     math runs in per-triangle orthonormal frames (_tri_closest): float32
                     closest-point on the Spout's 3 m sliver panels was off by up to 0.19 mm
                     and dropped 12% of Spout contacts in the old BVH path.
    neighbour list   Verlet list with a skin, rebuilt every N steps; grains that outrun it
                     fall back to an exact search.  Removes the per-step hash-grid rebuild.
    device state     step counter, injection schedule and free list all live on the device,
                     so blocks of steps capture into one CUDA graph and runs are bitwise
                     reproducible.

Two traps that look exactly like "the particles behave like a gas and leak out":

  * `model.particle_grid` is allocated by `finalize()` but nothing builds it, and
    `eval_particle_contact` silently returns when it is unbuilt -- particle-particle forces
    vanish with no error.  `step()` builds it.
  * DEM is unforgiving of initial overlap.  Two grains seeded 2 mm apart overlap by 10 mm; at
    k = 1000 N/m that is 10 N on a 0.9 g grain.  Uniform-random seeding reached 137 m/s in
    40 ms where lattice seeding stayed at 2 m/s.  Use `lattice_sites` for injection.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.semi_implicit.kernels_contact import eval_particle_contact
from newton._src.solvers.solver import SolverBase

# Forward kernels only.  Nothing here is differentiated, and the adjoint of the wall kernel
# needs a parameter block over CUDA's 4 KB limit (it failed to compile once the collider
# gained per-part active flags).  Also roughly halves compile time.
wp.set_module_options({"enable_backward": False})

_EPS_NORMAL = wp.constant(1.0e-3)
_NO_CONTACT = wp.constant(1.0e9)
# Two face contacts of one part closer than this in direction (cos 14 deg) are the same
# contact (a triangulated flat or gently curved surface); further apart, a concave corner.
_CORNER_COS = wp.constant(0.97)
# Corner search only for grains with at most this many wall candidates (see
# eval_wall_corners_rot).
_CORNER_MAX_LIST = wp.constant(16)


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
    active: wp.array(dtype=int)
    """1 = part takes part in contact, 0 = switched off (scenario time windows)."""
    motion_type: wp.array(dtype=int)
    """Surface motion on stationary geometry (issue #11): 0 static, 1 belt, 2 rotating."""
    motion_axis: wp.array(dtype=wp.vec3)
    """Belt: its width axis a (belt velocity = speed * normalize(a x n)).  Rotating: the
    unit rotation axis."""
    motion_point: wp.array(dtype=wp.vec3)
    """Rotating: a point on the axis."""
    motion_rate: wp.array(dtype=float)
    """Belt: speed (m/s).  Rotating: angular speed (rad/s)."""
    corners: wp.array(dtype=int)
    """1 = resolve concave creases within this part with a second contact (issue #13;
    eval_wall_corners_rot).  Off by default: it costs ~5-10% of wall-contact time."""
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


@wp.func
def _shell_sdf(two_sided: int, thick: float, offset: wp.vec3, d: float, face_n: wp.vec3):
    """Signed gap and outward contact normal from the closest point on a shell part.

    offset = x - closest point, d = |offset|, face_n = the face normal there (for one-sided
    parts the AVERAGE normal of the faces meeting at the closest point, which fixes the
    sign).  Two-sided parts repel from both faces; one-sided parts report a grain behind
    them as negative.  Shared by every wall kernel, so the BVH and grid paths differ
    only in how they find the closest point.
    """
    sdf = float(0.0)
    n = wp.vec3(0.0)
    if two_sided == 1:
        sdf = d - thick
        if d < _EPS_NORMAL:
            n = face_n
        else:
            n = offset / d
    else:
        sign = wp.where(wp.dot(face_n, offset) > 0.0, 1.0, -1.0)
        sdf = d * sign - thick
        if d < _EPS_NORMAL:
            n = face_n
        else:
            n = (offset / d) * sign
    return sdf, n


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
        if collider.active[m] == 0:
            continue
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

        face_n = wp.vec3(0.0)
        if collider.two_sided[m] == 1:
            face_n = wp.mesh_eval_face_normal(mesh, query.face)
        else:
            face_n = _average_face_normal(mesh, cp)
        sdf, n = _shell_sdf(collider.two_sided[m], thick, offset, d_unsigned, face_n)

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
                 cal_v0: float = 2.4, wall_grid: WallGrid | None = None,
                 hash_dims: tuple[int, int, int] | None = None,
                 neighbor_every: int = 0, skin: float = 0.003, skin_max: float = 0.006):
        super().__init__(model=model)
        # baked candidate-triangle grid; None falls back to per-step BVH queries
        self.wall_grid = wall_grid
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
        # The step counter lives on the device (the history table's staleness test reads
        # it) so that stepping never needs a host value -- a prerequisite for capturing
        # steps into a CUDA graph.  Holds the index of the step about to run.
        self.step_arr = wp.array([1], dtype=int, device=model.device)
        self.max_spin = float(max_spin)
        self.collider = collider
        self._keepalive = keepalive
        # Cell size should match the neighbour query radius (2*r), not exceed it.  Cost
        # per query is (particles per cell) x (cells scanned) ~ rho*c^3*(2*ceil(rq/c)+1)^3,
        # which for c = 2*rq is 8x worse than for c = rq: the cell count stays at 27 while
        # each cell holds 8x more grains.
        self.grid_cell = grid_cell if grid_cell else 2.0 * float(model.particle_max_radius)
        # The hash grid clears its whole cell table on every build: at Newton's default
        # 128^3 that is 2.1 M cells, 16 MB of memset per step for ~18k grains.  Cells hash
        # modulo these dims, so a smaller table only aliases cells a table-width apart,
        # and the distance test rejects those.
        if hash_dims is not None:
            model.particle_grid = wp.HashGrid(*hash_dims, device=model.device)
        elif model.particle_grid is None:
            model.particle_grid = wp.HashGrid(128, 128, 128, device=model.device)
        # grains touching a wall this step -- counted by the BVH wall kernels only
        self.contact_count = wp.zeros(1, dtype=int, device=model.device)

        # Verlet neighbour list (0 = off: rebuild the hash grid and scan it every step).
        # The grid is then built with cell 2*r_max + skin_max so a list query still spans
        # only 3x3x3 cells; the skin itself lives on the device so it can be retuned
        # between graph replays without re-capturing.
        self.neighbor_every = int(neighbor_every)
        self.skin_max = float(skin_max)
        self.skin_arr = wp.array([float(skin)], dtype=float, device=model.device)
        self.nbr_fallbacks = wp.zeros(1, dtype=int, device=model.device)
        self.nbr_overflow = wp.zeros(1, dtype=int, device=model.device)
        self._since_build = 0
        n_all = model.particle_count
        use_wl = self.neighbor_every > 0 and self.wall_grid is not None
        self.wl = wp.zeros((int(MAX_WALL_CANDIDATES), n_all) if use_wl else (1, 1), dtype=int,
                           device=model.device)
        self.wl_count = wp.zeros(n_all if use_wl else 1, dtype=int, device=model.device)
        self.wl_corner = wp.zeros(n_all if use_wl else 1, dtype=int, device=model.device)
        self.any_corners = bool(collider.corners.numpy().any())
        if self.neighbor_every > 0:
            self.nbr = wp.zeros((int(MAX_NEIGHBORS), n_all), dtype=int, device=model.device)
            self.nbr_count = wp.zeros(n_all, dtype=int, device=model.device)
            self.x_build = wp.zeros(n_all, dtype=wp.vec3, device=model.device)
        else:
            self.nbr = wp.zeros((1, 1), dtype=int, device=model.device)
            self.nbr_count = wp.zeros(1, dtype=int, device=model.device)
            self.x_build = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.wall_slack = wp.zeros(model.particle_count, dtype=float, device=model.device)
        # Laid out [slot, grain], not [grain, slot]: threads of a warp are consecutive
        # grains scanning the same slot k, so this makes each load one coalesced 128-byte
        # transaction instead of 32 strided ones.
        ns = int(TANGENTIAL_SLOTS)
        self.tang_partner = wp.full((ns, model.particle_count), -1, dtype=int,
                                    device=model.device)
        self.tang_stamp = wp.full((ns, model.particle_count), -10, dtype=int,
                                  device=model.device)
        self.tang_xi = wp.zeros((ns, model.particle_count), dtype=wp.vec3,
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

    @property
    def step_count(self) -> int:
        """Device step counter, read back (syncs -- not for the hot loop)."""
        return int(self.step_arr.numpy()[0])

    def set_part_active(self, flags):
        """Switch collider parts on/off (1/0 per part).  Takes effect on the next step;
        values live on the device, so captured graphs see them.  Turning a part ON clears
        the BVH path's distance cache -- slack banked while the part was absent would let
        grains coast through it."""
        new = np.asarray(flags, dtype=np.int32)
        old = self.collider.active.numpy()
        if not np.array_equal(new, old):
            self.collider.active.assign(new)
            if np.any(new > old):
                self.invalidate_cache()

    def request_rebuild(self):
        """Make the next step rebuild the neighbour list.  Call before eager steps that
        follow graph replays: replays do not advance the host's rebuild counter."""
        self._since_build = 0

    def set_step(self, n_done: int):
        """Restore after ``n_done`` completed steps (e.g. from a checkpoint)."""
        self.step_arr.assign(np.array([n_done + 1], dtype=np.int32))
        self._step = n_done

    def step(self, state_in, state_out, control, contacts, dt: float):
        """One DEM step.  state_out may be state_in: every kernel either finishes reading
        neighbour state before integration starts or touches only its own grain, so the
        update is safe in place -- and in place is what makes graph capture simple."""
        model = self.model
        self._step += 1                # host mirror; the kernels read step_arr

        use_list = self.neighbor_every > 0 and self.rotation
        if use_list:
            if self._since_build == 0:
                model.particle_grid.build(state_in.particle_q,
                                          2.0 * float(model.particle_max_radius) + self.skin_max)
                wp.launch(build_neighbor_list, dim=model.particle_count, inputs=[
                    model.particle_grid.id, state_in.particle_q, model.particle_radius,
                    model.particle_flags, model.particle_max_radius, self.skin_arr,
                    self.nbr, self.nbr_count, self.x_build, self.nbr_overflow],
                    device=model.device)
                if self.wall_grid is not None:
                    wp.launch(build_wall_list, dim=model.particle_count, inputs=[
                        state_in.particle_q, model.particle_flags, self.collider, self.wall_grid,
                        int(self.any_corners),
                        self.skin_arr, self.wl, self.wl_count, self.wl_corner],
                        device=model.device)
            self._since_build = (self._since_build + 1) % self.neighbor_every
        else:
            # Rebuild every step: eval_particle_contact returns immediately on an unbuilt
            # grid, which silently removes all particle-particle forces.
            model.particle_grid.build(state_in.particle_q, self.grid_cell)

        if not self.rotation:
            state_in.clear_forces()
            wp.launch(
                kernel=eval_particle_contact,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id, state_in.particle_q, state_in.particle_qd,
                    model.particle_radius, model.particle_flags, model.particle_ke,
                    model.particle_kd, model.particle_kf, model.particle_mu,
                    model.particle_cohesion, model.particle_max_radius,
                ],
                outputs=[state_in.particle_f],
                device=model.device,
            )
            self.contact_count.zero_()
            if not self.wall_cache:
                self.wall_slack.zero_()
            wp.launch(
                kernel=eval_shell_contact_forces,
                dim=model.particle_count,
                inputs=[state_in.particle_q, state_in.particle_qd, model.particle_radius,
                        model.particle_flags, self.collider, self.wall_slack, dt],
                outputs=[state_in.particle_f, self.contact_count],
                device=model.device,
            )
            self.integrate_particles(model, state_in, state_out, dt)
            wp.launch(_advance_step, dim=1, inputs=[self.step_arr], device=model.device)
            return

        # grain-grain: ASSIGNS force and torque for every grain (no clearing needed)
        if use_list:
            wp.launch(
                kernel=eval_particle_contact_list,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id, self.nbr, self.nbr_count,
                    state_in.particle_q, state_in.particle_qd,
                    self.particle_w, model.particle_radius, model.particle_flags,
                    model.particle_inv_mass,
                    model.particle_ke, model.particle_kd, model.particle_kf,
                    model.particle_mu, self.mu_roll, self.rot_damp,
                    self.k_t, int(self.hertz), self.e_star_pp, self.g_star_pp, self.beta_pp,
                    dt, self.step_arr,
                    self.tang_partner, self.tang_stamp, self.tang_xi,
                    model.particle_max_radius, self.skin_arr, self.x_build, self.nbr_fallbacks,
                ],
                outputs=[state_in.particle_f, self.particle_t],
                device=model.device,
            )
        else:
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
                    dt, self.step_arr,
                    self.tang_partner, self.tang_stamp, self.tang_xi,
                ],
                outputs=[state_in.particle_f, self.particle_t],
                device=model.device,
            )

        # grain-wall: ADDS to what the grain-grain kernel wrote
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
                    int(use_list), self.wl, self.wl_count, self.skin_arr, self.x_build,
                ],
                outputs=[state_in.particle_f, self.particle_t],
                device=model.device,
            )
            # second contacts in concave corners of one part -- see its docstring.  Only
            # launched when some part asks for it, so the default costs nothing.
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
                        int(use_list), self.wl, self.wl_count, self.wl_corner, self.skin_arr,
                        self.x_build,
                    ],
                    outputs=[state_in.particle_f, self.particle_t],
                    device=model.device,
                )
        else:
            self.contact_count.zero_()
            if not self.wall_cache:
                # Force a full query every step.  The cache is meant to be exact, so
                # results with it on and off must match -- that is the regression test.
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

        # linear + angular integration and the step counter, one launch
        wp.launch(
            kernel=integrate_fused,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q, state_in.particle_qd, state_in.particle_f,
                model.particle_inv_mass, self.particle_w, self.particle_t,
                self.particle_inv_inertia, model.particle_flags, model.gravity, dt,
                model.particle_max_velocity, self.max_spin, self.step_arr,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )


@wp.kernel
def _advance_step(step_arr: wp.array(dtype=int)):
    step_arr[0] = step_arr[0] + 1


def build_collider(parts, two_sided, friction, ke, kd, kf, thickness, max_dist, device,
                   corners=None, motions=None):
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
    c.active = wp.ones(len(parts), dtype=int, device=device)
    c.corners = wp.array([1 if x else 0 for x in (corners or [False] * len(parts))], dtype=int,
                         device=device)
    mt, ma, mp, mr = [], [], [], []
    for mo in (motions or [None] * len(parts)):
        mo = mo or (0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0)
        mt.append(int(mo[0]))
        ma.append(mo[1])
        mp.append(mo[2])
        mr.append(float(mo[3]))
    c.motion_type = wp.array(mt, dtype=int, device=device)
    c.motion_axis = wp.array(np.array(ma, dtype=np.float32), dtype=wp.vec3, device=device)
    c.motion_point = wp.array(np.array(mp, dtype=np.float32), dtype=wp.vec3, device=device)
    c.motion_rate = wp.array(mr, dtype=float, device=device)
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
        p = partner[k, i]
        if p == j and stamp[k, i] >= step - 1:
            return k
        if free < 0 and (p == -1 or stamp[k, i] < step - 1):
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
    fresh = partner[slot, i] != j or stamp[slot, i] < step - 1
    xi = wp.vec3(0.0)
    if not fresh:
        xi = xi_arr[slot, i]
        # the contact normal rotates as grains move: keep the stored slip tangential
        xi = xi - n * wp.dot(xi, n)
    xi = xi + vt * dt

    ft = -xi * k_t
    mag = wp.length(ft)
    if mag > f_coulomb and mag > 1.0e-12:
        ft = ft * (f_coulomb / mag)
        if k_t > 0.0:
            xi = -ft / k_t

    partner[slot, i] = j
    stamp[slot, i] = step
    xi_arr[slot, i] = xi
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


@wp.func
def _pair_contact(
    i: int,
    index: int,
    x: wp.vec3,
    v: wp.vec3,
    w: wp.vec3,
    ri: float,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    k_n: float,
    k_d: float,
    k_f: float,
    mu: float,
    mu_roll: float,
    rot_damp: float,
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
):
    """Force and torque on grain i from grain `index` (zero if not touching).  Shared by
    the hash-grid and neighbour-list kernels so they cannot drift apart physically."""
    rj = particle_radius[index]
    d_vec = x - particle_x[index]
    d = wp.length(d_vec)
    if d < 1.0e-9 or d >= ri + rj:
        return wp.vec3(0.0), wp.vec3(0.0)

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
    # k_n cannot do both -- measured on this machine it spans 5.6x (tools/
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
        return wp.vec3(0.0), wp.vec3(0.0)

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

    f = n * fn + ft
    t = wp.cross(c_i, ft)

    if mu_roll > 0.0 or rot_damp > 0.0:
        r_eff = ri * rj / (ri + rj)
        w_rel = w - particle_w[index]
        if mu_roll > 0.0:
            t += _rolling_torque(mu_roll, fn, r_eff, w_rel)
        if rot_damp > 0.0:
            t += _rotational_damping(rot_damp, kd_eff, r_eff, w_rel)
    return f, t


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
    step_arr: wp.array(dtype=int),
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
    # This kernel ASSIGNS rather than accumulates: every grain appears exactly once in
    # the grid's sorted order, so writing here replaces the two per-step memsets that
    # used to clear the force and torque arrays (and their atomics).
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        particle_f[i] = wp.vec3(0.0)
        particle_t[i] = wp.vec3(0.0)
        return
    step = step_arr[0]

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

        df, dtq = _pair_contact(i, index, x, v, w, ri, particle_x, particle_v, particle_w,
                                particle_radius, particle_inv_mass, k_n, k_d, k_f, mu,
                                mu_roll, rot_damp, k_t, hertz, e_star, g_star, beta, dt, step,
                                tang_partner, tang_stamp, tang_xi)
        f += df
        t += dtq

    particle_f[i] = f
    particle_t[i] = t


@wp.func
def _surface_velocity(collider: ShellCollider, m: int, n: wp.vec3, cp: wp.vec3):
    """(velocity, spin) of part m's SURFACE at contact point cp, normal n (pointing from
    the wall to the grain).  The geometry never moves; only its surface does (issue #11).

    Belt: speed * normalize(a x n), a = the belt's width axis.  The grain always sits
    outside the belt loop, so n points away from the loop and a x n is the running
    direction wherever it touches: +x on the carrying strand, full speed on troughed
    wings (the normalisation), downward round the head pulley, backward on the return
    strand -- the belt's path, with no geometry analysis.  Faces whose normal is nearly
    along a (belt edges) get no velocity.
    Rotating: omega * axis x (cp - point), spin omega * axis (pulleys, drums, rollers)."""
    mt = collider.motion_type[m]
    vs = wp.vec3(0.0)
    ws = wp.vec3(0.0)
    if mt == 1:
        t = wp.cross(collider.motion_axis[m], n)
        tl = wp.length(t)
        if tl > 0.2:
            vs = t * (collider.motion_rate[m] / tl)
    elif mt == 2:
        ws = collider.motion_axis[m] * collider.motion_rate[m]
        vs = wp.cross(ws, cp - collider.motion_point[m])
    return vs, ws


@wp.func
def _wall_force(
    collider: ShellCollider,
    m: int,
    n: wp.vec3,
    c: float,
    radius: float,
    v: wp.vec3,
    w: wp.vec3,
    inv_mass: float,
    mu_roll: float,
    rot_damp: float,
    k_t: float,
    hertz: int,
    e_star: float,
    g_star: float,
    beta: float,
    step: int,
    i: int,
    wid: int,
    cp: wp.vec3,
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    dt: float,
):
    """Force and torque on grain i from part m, given the outward normal n, the gap c
    (negative = overlap) and the wall's closest point cp.  Shared by every wall kernel
    so the paths cannot drift apart physically -- they differ only in how they FIND the
    wall.  Velocities are relative to the wall SURFACE (conveyors, pulleys: #11), so
    friction, the history spring and rolling resistance all act on the relative motion."""
    c_arm = -n * radius
    v_rel = v + wp.cross(w, c_arm)
    w_rel = w
    if collider.motion_type[m] != 0:
        # only moving surfaces touch this: keeping the static expression unchanged keeps
        # static walls bit-identical (even "- 0" lets the compiler fuse differently)
        v_surf, w_surf = _surface_velocity(collider, m, n, cp)
        v_rel = v_rel - v_surf
        w_rel = w - w_surf
    vn = wp.dot(v_rel, n)
    vt = v_rel - n * vn

    # grain against a rigid plate: R* = radius, m* = m (the wall never recoils)
    kd_eff = collider.kd[m]
    kt_eff = k_t
    kf_eff = collider.kf[m]
    fn = float(0.0)
    if hertz == 1:
        delta = -c
        m_star = 1.0 / wp.max(inv_mass, 1.0e-12)
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
        # walls occupy the same history table, keyed by negative partner ids (wid)
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

    f = n * fn + ft
    t = wp.cross(c_arm, ft)
    if mu_roll > 0.0:
        t += _rolling_torque(mu_roll, fn, radius, w_rel)
    if rot_damp > 0.0:
        # relative to the surface's own spin (zero unless it is a rotating part)
        t += _rotational_damping(rot_damp, kd_eff, radius, w_rel)
    return f, t


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
    step_arr: wp.array(dtype=int),
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    wall_slack: wp.array(dtype=float),
    dt: float,
    grid: wp.uint64,
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
):
    """Grain-wall contact with rotation.  Same geometry handling as the non-rotating
    version (unsigned distance, average-normal sign, per-part two-sided, AABB cull,
    wall-distance cache); the wall is static so it contributes no velocity."""
    # Visit grains in the hash grid's cell order, not pool order.  Pool order is
    # spatially random (indices are recycled), so a 32-thread warp would mix grains in
    # free fall with grains deep in the cascade and diverge through different BVHs.
    # Sorted, a warp's grains share cells: same parts, same BVH nodes, same branches.
    i = wp.hash_grid_point_id(grid, wp.tid())
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        return

    x = particle_q[i]
    radius = particle_radius[i]
    v = particle_qd[i]
    w = particle_w[i]

    step = step_arr[0]
    slack = wall_slack[i] - wp.length(v) * dt
    if slack > 0.0:
        wall_slack[i] = slack
        return

    f_total = wp.vec3(0.0)
    t_total = wp.vec3(0.0)
    nearest = float(_NO_CONTACT)
    hit = int(0)

    for m in range(collider.mesh.shape[0]):
        if collider.active[m] == 0:
            continue
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

        face_n = wp.vec3(0.0)
        if collider.two_sided[m] == 1:
            face_n = wp.mesh_eval_face_normal(mesh, query.face)
        else:
            face_n = _average_face_normal(mesh, cp)
        sdf, n = _shell_sdf(collider.two_sided[m], thick, offset, d_unsigned, face_n)

        nearest = wp.min(nearest, sdf)

        c = sdf - radius
        if c >= 0.0 or c < -radius:
            continue

        fw, tw = _wall_force(collider, m, n, c, radius, v, w, particle_inv_mass[i], mu_roll,
                             rot_damp, k_t, hertz, e_star, g_star, beta, step, i, -(m + 2), cp,
                             tang_partner, tang_stamp, tang_xi, dt)
        f_total += fw
        t_total += tw
        hit += 1

    verified = wp.min(nearest, collider.max_dist + radius)
    wall_slack[i] = wp.max(verified - radius, 0.0)

    if hit > 0:
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(particle_f, i, f_total)
        wp.atomic_add(particle_t, i, t_total)


# ---------------------------------------------------------------------------
# Baked wall grid
#
# The colliders never move, so the broad phase can be done ONCE instead of every step.
# A BVH point query is a pointer chase: each of ~13 levels is a dependent global-memory
# load, and a warp executes the UNION of its 32 threads' paths.  Measured on this machine:
# ~1,000 queries against the 6,372-triangle Top part take 300 us whatever the BVH builder,
# and the wall kernel cost 1.18 ms of a 1.73 ms step.
#
# Instead: a uniform grid over the collider bounding box where each cell lists every
# (part, triangle) that could lie within `reach` of ANY point in the cell (CSR layout:
# cell_start[c] .. cell_start[c+1] index into `entries`).  A grain computes its cell and
# walks a short contiguous list -- no tree, no stack.  A grain in open air reads an empty
# range and is done, which also makes the wall-distance cache unnecessary.
#
# Exactness: any triangle within `reach` of the grain is within reach + half_diag of its
# cell centre, so it is in the list.  The closest triangle, normal and force are therefore
# the same ones the BVH finds; only rounding in the closest-point arithmetic differs.
# ---------------------------------------------------------------------------


@wp.struct
class WallTri:
    """One list entry with its triangle inlined (64 bytes).  Storing the triangle IN the
    list rather than an index to it removes a dependent random gather per entry: a
    grain's walk becomes a contiguous read, two entries per 128-byte cache line."""
    v0: wp.vec3
    u: wp.vec3
    w: wp.vec3
    n: wp.vec3
    t2: wp.vec3
    part: int


@wp.struct
class WallGrid:
    origin: wp.vec3
    inv_h: float
    nx: int
    ny: int
    nz: int
    reach: float
    """Lists are complete for triangles within this distance of any point in a cell."""
    cell_start: wp.array(dtype=int)
    entries: wp.array(dtype=int)
    """Global triangle ids, grouped by part within each cell (ascending part order)."""
    tri_v0: wp.array(dtype=wp.vec3)
    tri_u: wp.array(dtype=wp.vec3)
    """Unit in-plane axis along edge v0->v1."""
    tri_w: wp.array(dtype=wp.vec3)
    """Unit in-plane axis n x u."""
    tri_n: wp.array(dtype=wp.vec3)
    tri_2d: wp.array(dtype=wp.vec3)
    """(|v1-v0|, c_u, c_w): the triangle in its own frame is (0,0), (b,0), (c_u,c_w)."""
    tri_part: wp.array(dtype=int)
    rec: wp.array(dtype=WallTri)
    """entries[k]'s triangle, inlined -- what the step kernel actually reads."""
    rec_sharp: wp.array(dtype=int)
    """1 if entries[k]'s triangle shares an edge with a neighbour > ~14 deg away (a crease).
    Kept out of WallTri so the main wall kernel's records stay 64 bytes."""


@wp.func
def _cell_of(g: WallGrid, x: wp.vec3):
    p = (x - g.origin) * g.inv_h
    return wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))


@wp.func
def _seg2(p: wp.vec2, a: wp.vec2, b: wp.vec2):
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-30), 0.0, 1.0)
    return a + ab * t


@wp.func
def _tri_closest(v0: wp.vec3, u: wp.vec3, w: wp.vec3, n: wp.vec3, t2: wp.vec3, x: wp.vec3):
    """Closest point on a triangle to x, computed in the triangle's own orthonormal frame.
    Returns (squared distance, closest point).

    Both Warp's native closest_point_to_triangle (Ericson's region test) and warp.fem's
    barycentric solve lose the answer in float32 on sliver triangles: the Spout is built
    from 3 m panels with aspect ratios up to 337, and their region tests are differences
    of products ~20 whose true value is tiny.  Measured against float64: 14 um error on a
    54 um Hertz overlap (a 30% force error), and on 12% of Spout contacts the sign test
    found no face at all and the contact was dropped.

    Here the out-of-plane distance is ONE dot product, and the in-plane problem is a 2D
    edge-sign test plus clamped segment projections -- all well conditioned whatever the
    triangle's shape.  Error is ~1 ulp of |x - v0| (~0.3 um), not of the products.
    """
    r = x - v0
    h = wp.dot(r, n)
    p = wp.vec2(wp.dot(r, u), wp.dot(r, w))
    a = wp.vec2(0.0, 0.0)
    b = wp.vec2(t2[0], 0.0)
    c = wp.vec2(t2[1], t2[2])
    # edge functions; the frame makes the triangle counter-clockwise, so inside = all >= 0
    e0 = b[0] * p[1]                                           # (b-a) x (p-a), a = origin
    e1 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
    e2 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
    q = p
    if e0 < 0.0 or e1 < 0.0 or e2 < 0.0:
        qa = _seg2(p, a, b)
        qb = _seg2(p, b, c)
        qc = _seg2(p, c, a)
        da = wp.length_sq(p - qa)
        db = wp.length_sq(p - qb)
        dc = wp.length_sq(p - qc)
        q = qa
        if db < da:
            q = qb
            da = db
        if dc < da:
            q = qc
    dq = p - q
    return h * h + wp.dot(dq, dq), v0 + u * q[0] + w * q[1]


@wp.func
def _tri_inside(v0: wp.vec3, u: wp.vec3, w: wp.vec3, t2: wp.vec3, x: wp.vec3):
    """1 if x projects inside the triangle (edge-function signs in its own frame).  Then
    the closest point is the projection: the contact distance is just the plane distance
    and the direction the face normal, with no edge/corner search needed."""
    r = x - v0
    p0 = wp.dot(r, u)
    p1 = wp.dot(r, w)
    b = t2[0]
    cu = t2[1]
    cv = t2[2]
    e0 = b * p1
    e1 = (cu - b) * p1 - cv * (p0 - b)
    e2 = -cu * (p1 - cv) + cv * (p0 - cu)
    if e0 >= 0.0 and e1 >= 0.0 and e2 >= 0.0:
        return 1
    return 0


@wp.kernel
def _bake_cells(
    meshes: wp.array(dtype=wp.uint64),
    part_lo: wp.array(dtype=wp.vec3),
    part_hi: wp.array(dtype=wp.vec3),
    tri_offset: wp.array(dtype=int),
    tv0: wp.array(dtype=wp.vec3),
    tu: wp.array(dtype=wp.vec3),
    tw: wp.array(dtype=wp.vec3),
    tn: wp.array(dtype=wp.vec3),
    t2: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    h: float,
    nx: int,
    ny: int,
    radius: float,
    reach: float,
    sub: int,
    write: int,
    cell_start: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    entries: wp.array(dtype=int),
):
    """Count (write=0) or fill (write=1) one cell's candidate list."""
    c = wp.tid()
    ix = c % nx
    iy = (c / nx) % ny
    iz = c / (nx * ny)
    corner = origin + wp.vec3(float(ix) * h, float(iy) * h, float(iz) * h)
    centre = corner + wp.vec3(0.5 * h)
    hs = h / float(sub)
    rs = reach + 0.5 * wp.sqrt(3.0) * hs
    ext = wp.vec3(radius)
    r2 = rs * rs
    k = int(0)
    base = int(0)
    if write == 1:
        base = cell_start[c]
    for m in range(meshes.shape[0]):
        lo = part_lo[m]
        hi = part_hi[m]
        if (centre[0] < lo[0] - radius or centre[0] > hi[0] + radius or
            centre[1] < lo[1] - radius or centre[1] > hi[1] + radius or
            centre[2] < lo[2] - radius or centre[2] > hi[2] + radius):
            continue
        mesh = meshes[m]
        q = wp.mesh_query_aabb(mesh, centre - ext, centre + ext)
        f = int(0)
        while wp.mesh_query_aabb_next(q, f):
            gt = tri_offset[m] + f
            # Test against a sub x sub x sub lattice of points inside the cell, each
            # covering a sphere of radius reach + h*sqrt(3)/(2*sub).  Any point of the
            # cell is within that of some lattice point, so the list stays complete, but
            # the slack over `reach` shrinks sub-fold -- and with it the longest lists,
            # which set the kernel's run time (a warp waits for its longest walk).
            hit = int(0)
            for sx in range(sub):
                for sy in range(sub):
                    for sz in range(sub):
                        if hit == 0:
                            pt = corner + wp.vec3((float(sx) + 0.5) * hs, (float(sy) + 0.5) * hs,
                                                  (float(sz) + 0.5) * hs)
                            sq, _cp = _tri_closest(tv0[gt], tu[gt], tw[gt], tn[gt], t2[gt], pt)
                            if sq <= r2:
                                hit = 1
            if hit == 1:
                if write == 1:
                    entries[base + k] = tri_offset[m] + f
                k += 1
    if write == 0:
        counts[c] = k


def build_wall_grid(parts, meshes, collider_lo, collider_hi, reach, cell, device, sub=4):
    """Bake the candidate-triangle grid for static colliders (see block comment above).

    ``reach`` must cover every distance at which a triangle can matter: contact
    (particle radius + shell thickness) plus the 1 mm neighbourhood used for the
    average-normal sign test.
    """
    allv = np.vstack([np.asarray(v, dtype=np.float64) for _n, v, _f in parts])
    lo = allv.min(axis=0) - reach - cell
    hi = allv.max(axis=0) + reach + cell
    dims = np.ceil((hi - lo) / cell).astype(int)
    ncell = int(np.prod(dims))
    half_diag = 0.5 * math.sqrt(3.0) * cell

    # global triangle table, parts concatenated in collider order
    # per-triangle orthonormal frames, computed in float64 (see _tri_closest)
    v0s, us, ws, ns, t2s, part_ids, offs = [], [], [], [], [], [], []
    off = 0
    for m, (_n, v, f) in enumerate(parts):
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        e1, e2 = b - a, c - a
        nrm = np.cross(e1, e2)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
        blen = np.linalg.norm(e1, axis=1)
        u = e1 / np.maximum(blen[:, None], 1e-30)
        w = np.cross(nrm, u)
        t2 = np.stack([blen, (e2 * u).sum(1), (e2 * w).sum(1)], axis=1)
        v0s.append(a); us.append(u); ws.append(w); ns.append(nrm); t2s.append(t2)
        part_ids.append(np.full(len(f), m))
        offs.append(off)
        off += len(f)
    tv0, tu, tw, tn, tt2 = (wp.array(np.concatenate(x).astype(np.float32), dtype=wp.vec3,
                                     device=device) for x in (v0s, us, ws, ns, t2s))

    def arr(x, dt=wp.vec3):
        return wp.array(np.concatenate(x).astype(np.float32 if dt is wp.vec3 else np.int32),
                        dtype=dt, device=device)

    mesh_ids = wp.array([mm.id for mm in meshes], dtype=wp.uint64, device=device)
    tri_offset = wp.array(np.array(offs, dtype=np.int32), dtype=int, device=device)
    counts = wp.zeros(ncell, dtype=int, device=device)
    cell_start = wp.zeros(ncell + 1, dtype=int, device=device)
    dummy = wp.zeros(1, dtype=int, device=device)
    origin = wp.vec3(*lo.astype(np.float32))
    common = [mesh_ids, collider_lo, collider_hi, tri_offset, tv0, tu, tw, tn, tt2, origin, float(cell),
              int(dims[0]), int(dims[1]), float(reach + half_diag), float(reach), int(sub)]
    wp.launch(_bake_cells, dim=ncell, inputs=common + [0, cell_start, counts, dummy], device=device)
    wp.utils.array_scan(counts, cell_start[1:], inclusive=True)
    total = int(cell_start[ncell:].numpy()[0])
    entries = wp.zeros(max(total, 1), dtype=int, device=device)
    wp.launch(_bake_cells, dim=ncell, inputs=common + [1, cell_start, counts, entries], device=device)

    g = WallGrid()
    g.origin = origin
    g.inv_h = float(1.0 / cell)
    g.nx, g.ny, g.nz = int(dims[0]), int(dims[1]), int(dims[2])
    g.reach = float(reach)
    g.cell_start = cell_start
    g.entries = entries
    g.tri_v0, g.tri_u, g.tri_w, g.tri_n, g.tri_2d = tv0, tu, tw, tn, tt2
    g.tri_part = arr(part_ids, dt=int)
    # crease flags: a triangle is "sharp" if it shares an edge with a neighbour whose
    # plane differs by more than ~14 deg (|cos| < _CORNER_COS).  Fillet facets, each a few
    # degrees apart, are not; trough and panel corners are.
    sharp = []
    for _n, v, f in parts:
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        v = np.asarray(v, dtype=np.float64)
        nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        fid = np.tile(np.arange(len(f)), 3)
        key = np.sort(e, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        key, fid = key[order], fid[order]
        same = np.all(key[1:] == key[:-1], axis=1)
        a, b = fid[:-1][same], fid[1:][same]
        crease = np.abs(np.einsum("ij,ij->i", nrm[a], nrm[b])) < float(_CORNER_COS)
        flag = np.zeros(len(f), dtype=np.int32)
        flag[a[crease]] = 1
        flag[b[crease]] = 1
        sharp.append(flag)
    sharp = np.concatenate(sharp)
    ent = entries.numpy()[:total] if total else np.zeros(0, dtype=np.int32)
    g.rec_sharp = wp.array(sharp[ent] if total else np.zeros(1, np.int32), dtype=int, device=device)
    g.rec = wp.empty(max(total, 1), dtype=WallTri, device=device)
    if total:
        wp.launch(_gather_records, dim=total, inputs=[entries, tv0, tu, tw, tn, tt2, g.tri_part,
                                                      g.rec], device=device)
    cnt = counts.numpy()
    occ = cnt[cnt > 0]
    info = dict(cells=ncell, dims=tuple(int(d) for d in dims), occupied=int(len(occ)),
                entries=total, mean_list=float(occ.mean()) if len(occ) else 0.0,
                max_list=int(occ.max()) if len(occ) else 0,
                mbytes=(4 * (ncell + 1 + total) + 64 * (off + total)) / 1e6)
    return g, info


@wp.kernel
def _gather_records(entries: wp.array(dtype=int), tv0: wp.array(dtype=wp.vec3),
                    tu: wp.array(dtype=wp.vec3), tw: wp.array(dtype=wp.vec3),
                    tn: wp.array(dtype=wp.vec3), t2: wp.array(dtype=wp.vec3),
                    tp: wp.array(dtype=int), rec: wp.array(dtype=WallTri)):
    k = wp.tid()
    t = entries[k]
    r = WallTri()
    r.v0 = tv0[t]
    r.u = tu[t]
    r.w = tw[t]
    r.n = tn[t]
    r.t2 = t2[t]
    r.part = tp[t]
    rec[k] = r


@wp.func
def _avg_normal_from_list(g: WallGrid, mode: int, wl: wp.array2d(dtype=int), i: int, k0: int,
                          s0: int, s1: int, cp: wp.vec3):
    """Average normal of the faces passing within 1 mm of cp, taken from the candidate
    list.  Same rule as _average_face_normal, without its second BVH query."""
    acc = wp.vec3(0.0)
    eps_sq = _EPS_NORMAL * _EPS_NORMAL
    for s in range(s0, s1):
        r = g.rec[_entry(mode, wl, i, k0, s)]
        sq, _q = _tri_closest(r.v0, r.u, r.w, r.n, r.t2, cp)
        if sq < eps_sq:
            acc += r.n
    return wp.normalize(acc)


MAX_WALL_CANDIDATES = wp.constant(96)


@wp.func
def _entry(mode: int, wl: wp.array2d(dtype=int), i: int, k0: int, s: int):
    """Record index of the s-th candidate: from the grain's own list (mode 1) or from the
    contiguous cell list starting at k0 (mode 0)."""
    if mode == 1:
        return wl[s, i]
    return k0 + s


@wp.kernel
def build_wall_list(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    collider: ShellCollider,
    g: WallGrid,
    any_corners: int,
    skin: wp.array(dtype=float),
    wl: wp.array2d(dtype=int),
    wl_count: wp.array(dtype=int),
    wl_corner: wp.array(dtype=int),
):
    """Per-grain wall candidates: the records of the grain's cell list that lie within
    reach + skin/2 of the GRAIN (the cell list covers the whole cell, so it is several
    times longer).  Walls are static, so only the grain's own motion can bring a new
    triangle into range.  A grain that moves more than skin/2 walks the full cell list.
    -1 = too many to store; the wall kernel then walks the full cell list."""
    i = wp.tid()
    wl_count[i] = 0
    wl_corner[i] = 0
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return
    x = particle_q[i]
    cc = _cell_of(g, x)
    if cc[0] < 0 or cc[1] < 0 or cc[2] < 0 or cc[0] >= g.nx or cc[1] >= g.ny or cc[2] >= g.nz:
        return
    cell = (cc[2] * g.ny + cc[1]) * g.nx + cc[0]
    rr = g.reach + 0.5 * skin[0]
    r2 = rr * rr
    c = int(0)
    # Corner flag: two FACE-reachable CREASE candidates (rec_sharp) of one part whose planes
    # differ by more than ~14 deg (|cos|, so the two faces of a thin plate do not count).
    # Smooth concave fillets are excluded: their closest point moves continuously, so one
    # contact does not chatter, and their long lists made the corner kernel cost 335 us.  Face-reachable
    # = the grain's projection is within skin/2 of the triangle in its plane, i.e. it may
    # project inside it before the next rebuild.  A concave corner needs two such faces; a
    # CONVEX edge has at most one, so grains on a panel near an edge or seam are not
    # flagged (flagging every grain with differing planes nearby ran the corner kernel
    # for most wall grains and cost 390 us a step).  Only flagged grains run
    # eval_wall_corners_rot.
    hs2 = 0.25 * skin[0] * skin[0]
    ref_part = int(-1)
    ref_n = wp.vec3(0.0)
    corner = int(0)
    for k in range(g.cell_start[cell], g.cell_start[cell + 1]):
        r = g.rec[k]
        h = wp.dot(x - r.v0, r.n)
        if h * h < r2:
            sq, _cp = _tri_closest(r.v0, r.u, r.w, r.n, r.t2, x)
            if sq < r2:
                if c < MAX_WALL_CANDIDATES:
                    wl[c, i] = k
                c += 1
                if any_corners == 1 and sq - h * h <= hs2 and g.rec_sharp[k] == 1 \
                        and collider.corners[r.part] == 1:
                    if r.part != ref_part:
                        ref_part = r.part
                        ref_n = r.n
                    elif wp.abs(wp.dot(r.n, ref_n)) < _CORNER_COS:
                        corner = 1
    if c > MAX_WALL_CANDIDATES:
        c = -1
    wl_count[i] = c
    # flag only grains the corner kernel will actually handle (see _CORNER_MAX_LIST)
    if c < 2 or c > _CORNER_MAX_LIST:
        corner = 0
    wl_corner[i] = corner


@wp.kernel
def eval_wall_grid_rot(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    collider: ShellCollider,
    g: WallGrid,
    mu_roll: float,
    rot_damp: float,
    k_t: float,
    hertz: int,
    e_star: float,
    g_star: float,
    beta: float,
    step_arr: wp.array(dtype=int),
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    dt: float,
    grid: wp.uint64,
    use_wl: int,
    wl: wp.array2d(dtype=int),
    wl_count: wp.array(dtype=int),
    skin: wp.array(dtype=float),
    x_build: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """Grain-wall contact from the baked grid.  Physics identical to
    eval_shell_contact_forces_rot; only the search differs."""
    i = wp.hash_grid_point_id(grid, wp.tid())
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        return
    step = step_arr[0]
    x = particle_q[i]
    mode = int(0)
    k0 = int(0)
    ncand = int(0)
    fast = int(0)
    if use_wl == 1:
        hs = 0.5 * skin[0]
        if wp.length_sq(x - x_build[i]) > hs * hs:
            fast = 1              # outran its filtered list: walk the full cell list
    if use_wl == 1 and wl_count[i] >= 0 and fast == 0:
        mode = 1                  # the grain's own filtered list
        ncand = wl_count[i]
    else:
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
    f_total = wp.vec3(0.0)
    t_total = wp.vec3(0.0)
    hit = int(0)

    # Pass 1 -- search only.  Entries are grouped by part; keep the closest triangle of
    # each part within contact range in up to MAX_WALL_CONTACTS register slots.  The
    # force law is deliberately NOT evaluated in here: lanes of a warp finish their part
    # segments at different iterations, so a heavy branch inside the loop would execute
    # serially once per distinct lane.  Measured, that divergence was most of the kernel.
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
            t = _entry(mode, wl, i, k0, s)
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
            best_sq = rr * rr        # only a triangle closer than contact range matters
            if p >= 0:
                if collider.active[p] == 0:
                    # switched-off part: a zero range rejects all its triangles in the
                    # test below, with no extra load or branch per triangle (a per-entry
                    # active check cost 14% of this kernel)
                    best_sq = 0.0
        if t >= 0:
            # cheap reject: the plane distance alone already exceeds the best so far
            h = wp.dot(x - r.v0, r.n)
            if h * h < best_sq:
                sq, _cand = _tri_closest(r.v0, r.u, r.w, r.n, r.t2, x)
                if sq < best_sq:
                    best_sq = sq
                    best_t = t

    # Pass 2 -- forces, one short uniform loop over the (at most 4) parts in contact.
    for j in range(ns):
        m = s_part[j]
        br = g.rec[s_tri[j]]            # s_tri holds the list position, not the tri id
        _sq, best_cp = _tri_closest(br.v0, br.u, br.w, br.n, br.t2, x)
        thick = collider.thickness[m]
        offset = x - best_cp
        d_unsigned = wp.sqrt(s_sq[j])
        face_n = br.n
        if collider.two_sided[m] == 0:
            face_n = _avg_normal_from_list(g, mode, wl, i, k0, s_seg0[j], s_seg1[j], best_cp)
        sdf, n = _shell_sdf(collider.two_sided[m], thick, offset, d_unsigned, face_n)
        c = sdf - radius
        if c < 0.0 and c >= -radius:
            fw, tw = _wall_force(collider, m, n, c, radius, v, w, particle_inv_mass[i],
                                 mu_roll, rot_damp, k_t, hertz, e_star, g_star, beta,
                                 step, i, -(m + 2), best_cp, tang_partner, tang_stamp,
                                 tang_xi, dt)
            f_total += fw
            t_total += tw
            hit += 1

    if hit > 0:
        # one thread owns grain i and the particle kernel has already finished, so a
        # plain read-modify-write is safe -- no atomic needed
        particle_f[i] = particle_f[i] + f_total
        particle_t[i] = particle_t[i] + t_total


# ---------------------------------------------------------------------------
# Verlet neighbour list
#
# Rebuilding the hash grid costs ~180 us a step (CUB radix sort, a 16 MB cell-table
# memset, two index kernels), and each grain then scans 27 cells -- ~30 candidates -- to
# find its ~6 contacts.  A neighbour list, standard in molecular dynamics, records every
# grain within contact distance PLUS A SKIN once every N steps; in between, each grain
# walks its ~6 listed neighbours.  The list is complete for every grain that has moved
# less than skin/2 since the build (two such grains cannot close by more than the skin).
# A grain that has moved further -- e.g. a rare one at the 30 m/s clamp -- falls back to an
# exact search that step, so the skin only needs to cover TYPICAL speeds; the fallback
# count is reported per run.  Measured: sizing it for 6 m/s at N = 4, dt = 97 us costs
# ~0.01% of grain-steps in fallbacks; for 4 m/s they explode into the millions.
# ---------------------------------------------------------------------------

MAX_NEIGHBORS = wp.constant(32)


@wp.kernel
def build_neighbor_list(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    max_radius: float,
    skin: wp.array(dtype=float),
    nbr: wp.array2d(dtype=int),
    nbr_count: wp.array(dtype=int),
    x_build: wp.array(dtype=wp.vec3),
    overflow: wp.array(dtype=int),
):
    """Neighbours of each active grain within (ri + rj + skin), stored slot-major
    ([slot, grain]) so consecutive threads reading slot k coalesce."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    x = particle_x[i]
    x_build[i] = x
    if (particle_flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        nbr_count[i] = 0
        return
    ri = particle_radius[i]
    sk = skin[0]
    c = int(0)
    query = wp.hash_grid_query(grid, x, ri + max_radius + sk)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if (particle_flags[j] & newton.ParticleFlags.ACTIVE) == 0:
            continue
        cut = ri + particle_radius[j] + sk
        if wp.length_sq(x - particle_x[j]) < cut * cut:
            if c < MAX_NEIGHBORS:
                nbr[c, i] = j
            c += 1
    if c > MAX_NEIGHBORS:
        wp.atomic_add(overflow, 0, 1)
        c = MAX_NEIGHBORS
    nbr_count[i] = c


@wp.kernel
def eval_particle_contact_list(
    grid: wp.uint64,
    nbr: wp.array2d(dtype=int),
    nbr_count: wp.array(dtype=int),
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
    max_radius: float,
    skin: wp.array(dtype=float),
    x_build: wp.array(dtype=wp.vec3),
    fallbacks: wp.array(dtype=int),
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """eval_particle_contact_rot driven by the neighbour list.  Visits grains in the
    order of the last grid build, which is still spatially coherent."""
    i = wp.hash_grid_point_id(grid, wp.tid())
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
    f = wp.vec3(0.0)
    t = wp.vec3(0.0)
    hs = 0.5 * skin[0]
    if wp.length_sq(x - x_build[i]) <= hs * hs:
        for k in range(nbr_count[i]):
            index = nbr[k, i]
            if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
                continue
            df, dtq = _pair_contact(i, index, x, v, w, ri, particle_x, particle_v, particle_w,
                                    particle_radius, particle_inv_mass, k_n, k_d, k_f, mu,
                                    mu_roll, rot_damp, k_t, hertz, e_star, g_star, beta, dt,
                                    step, tang_partner, tang_stamp, tang_xi)
            f += df
            t += dtq
    else:
        # This grain has outrun its list (e.g. a rare grain at the 30 m/s clamp, or one
        # spawned since the last build -- its x_build is stale).  Query the grid instead.
        # The grid holds BUILD-time positions; a partner that has moved <= skin/2 has its
        # build position within ri + rj + skin/2 of x, so radius ri + r_max + skin finds
        # it.  NOT exact in two cases, both left as known limits: a partner that has ALSO
        # outrun its list may be missed (needs two grains beyond skin/(N*dt), ~12 m/s in
        # the fast preset; measured flow tops out near 8), and slow partners, still on
        # their own lists, do not see this grain until the next rebuild (a one-sided
        # force for < N steps).
        wp.atomic_add(fallbacks, 0, 1)
        query = wp.hash_grid_query(grid, x, ri + max_radius + skin[0])
        index = int(0)
        while wp.hash_grid_query_next(query, index):
            if index == i:
                continue
            if (particle_flags[index] & newton.ParticleFlags.ACTIVE) == 0:
                continue
            df, dtq = _pair_contact(i, index, x, v, w, ri, particle_x, particle_v, particle_w,
                                    particle_radius, particle_inv_mass, k_n, k_d, k_f, mu,
                                    mu_roll, rot_damp, k_t, hertz, e_star, g_star, beta, dt,
                                    step, tang_partner, tang_stamp, tang_xi)
            f += df
            t += dtq
    particle_f[i] = f
    particle_t[i] = t


@wp.kernel
def eval_wall_corners_rot(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_w: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    collider: ShellCollider,
    g: WallGrid,
    mu_roll: float,
    rot_damp: float,
    k_t: float,
    hertz: int,
    e_star: float,
    g_star: float,
    beta: float,
    step_arr: wp.array(dtype=int),
    tang_partner: wp.array2d(dtype=int),
    tang_stamp: wp.array2d(dtype=int),
    tang_xi: wp.array2d(dtype=wp.vec3),
    dt: float,
    grid: wp.uint64,
    use_wl: int,
    wl: wp.array2d(dtype=int),
    wl_count: wp.array(dtype=int),
    wl_corner: wp.array(dtype=int),
    skin: wp.array(dtype=float),
    x_build: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
):
    """Second contacts in concave corners of one part (issue #13).

    eval_wall_grid_rot keeps one contact per part -- its closest triangle.  A grain in a
    concave corner of ONE part touches two faces, and with one contact the push flipped
    between them each step: settled corner grains moved 10-20x faster than with the faces
    as separate parts.  This adds, per part in contact, the closest FACE contact (grain
    projects inside the triangle) whose direction differs from the primary's by more than
    ~14 deg.  Edge/corner closest points never qualify, so a grain resting on a face near
    a CONVEX edge is not pushed twice.

    A separate kernel on purpose: folded into eval_wall_grid_rot, the extra state took it
    from 127 to 134-159 registers, over the 128 that fit two 256-thread blocks per SM on
    this GPU, and the halved occupancy cost 30-40% of that latency-bound kernel.  Here only
    grains flagged at wall-list build (two differing planes of one part nearby) do work;
    without wall lists, or for grains that outran theirs, every wall grain checks.

    The extra contact's friction history is keyed -(1000000 + 8 m + 1).  If the two faces
    swap which is closer, their histories swap too (a reset, not an error)."""
    # Launched over ALL grains in the hash grid's spatial order, not over a compact list
    # of flagged ones: packed densely, ~250 flagged grains fill only ~8 warps, and this
    # latency-bound work then has nothing to overlap with (137 us vs 76 us measured).
    i = wp.hash_grid_point_id(grid, wp.tid())
    if use_wl == 1:
        if wl_corner[i] == 0:
            return                    # the common case: one load and out
    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        return
    x = particle_q[i]
    mode = int(0)
    k0 = int(0)
    ncand = int(0)
    fast = int(0)
    if use_wl == 1:
        hs = 0.5 * skin[0]
        if wp.length_sq(x - x_build[i]) > hs * hs:
            fast = 1
    if use_wl == 1:
        # Only flagged grains with SHORT lists: real creases between flat panels (trough
        # bottom/wing, chute panel corners) have a handful of candidates.  Grains in dense
        # CAD detail (lists up to ~95) keep a single contact per part -- walking those
        # lists twice made this kernel's slowest warps cost ~290 us a step.  Grains that
        # outran their list skip it for those few steps.
        if fast == 1:
            return
        mode = 1
        ncand = wl_count[i]
    else:
        cc = _cell_of(g, x)
        if cc[0] < 0 or cc[1] < 0 or cc[2] < 0 or cc[0] >= g.nx or cc[1] >= g.ny or cc[2] >= g.nz:
            return
        cell = (cc[2] * g.ny + cc[1]) * g.nx + cc[0]
        k0 = g.cell_start[cell]
        ncand = g.cell_start[cell + 1] - k0
        if ncand > _CORNER_MAX_LIST:
            return
    if ncand < 2:
        return

    step = step_arr[0]
    radius = particle_radius[i]
    v = particle_qd[i]
    w = particle_w[i]
    f_total = wp.vec3(0.0)
    t_total = wp.vec3(0.0)
    hit = int(0)

    # walk the parts' segments: primary = closest triangle (as the main kernel found it),
    # then the closest face contact pointing >14 deg away from it
    cur = int(-1)
    seg0 = int(0)
    best_sq = float(0.0)
    best_t = int(-1)
    for s in range(ncand + 1):
        t = int(-1)
        p = int(-1)
        if s < ncand:
            t = _entry(mode, wl, i, k0, s)
            p = g.rec[t].part
        if p != cur:
            if best_t >= 0:
                m = cur
                pr = g.rec[best_t]
                _sq0, cp0 = _tri_closest(pr.v0, pr.u, pr.w, pr.n, pr.t2, x)
                d0 = pr.n
                if best_sq > _EPS_NORMAL * _EPS_NORMAL:
                    d0 = (x - cp0) / wp.sqrt(best_sq)
                rr = radius + collider.thickness[m]
                ex_sq = rr * rr
                ex_t = int(-1)
                for s2 in range(seg0, s):
                    t2 = _entry(mode, wl, i, k0, s2)
                    r2 = g.rec[t2]
                    h = wp.dot(x - r2.v0, r2.n)
                    if h * h < ex_sq:
                        d = r2.n
                        if h < 0.0:
                            d = -r2.n
                        if wp.dot(d, d0) < _CORNER_COS:
                            if _tri_inside(r2.v0, r2.u, r2.w, r2.t2, x) == 1:
                                ex_sq = h * h
                                ex_t = t2
                if ex_t >= 0:
                    er = g.rec[ex_t]
                    _sq1, cp1 = _tri_closest(er.v0, er.u, er.w, er.n, er.t2, x)
                    thick = collider.thickness[m]
                    offset = x - cp1
                    d_unsigned = wp.sqrt(ex_sq)
                    face_n = er.n
                    if collider.two_sided[m] == 0:
                        face_n = _avg_normal_from_list(g, mode, wl, i, k0, seg0, s, cp1)
                    sdf, n = _shell_sdf(collider.two_sided[m], thick, offset, d_unsigned, face_n)
                    c = sdf - radius
                    if c < 0.0 and c >= -radius:
                        fw, tw = _wall_force(collider, m, n, c, radius, v, w, particle_inv_mass[i],
                                             mu_roll, rot_damp, k_t, hertz, e_star, g_star, beta,
                                             step, i, -(1000000 + m * 8 + 1), cp1,
                                             tang_partner, tang_stamp, tang_xi, dt)
                        f_total += fw
                        t_total += tw
                        hit += 1
            cur = p
            seg0 = s
            best_t = -1
            rr0 = radius + collider.thickness[wp.max(p, 0)]
            best_sq = rr0 * rr0
            if p >= 0:
                if collider.active[p] == 0:
                    best_sq = 0.0
        if t >= 0:
            r = g.rec[t]
            h = wp.dot(x - r.v0, r.n)
            if h * h < best_sq:
                sq, _cand = _tri_closest(r.v0, r.u, r.w, r.n, r.t2, x)
                if sq < best_sq:
                    best_sq = sq
                    best_t = t

    if hit > 0:
        # one owner thread per grain, and the main wall kernel has finished
        particle_f[i] = particle_f[i] + f_total
        particle_t[i] = particle_t[i] + t_total


@wp.kernel
def integrate_fused(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    particle_w: wp.array(dtype=wp.vec3),
    particle_t: wp.array(dtype=wp.vec3),
    particle_inv_inertia: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    max_spin: float,
    step_arr: wp.array(dtype=int),
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    """Newton's integrate_particles and the angular update in one launch, with the same
    arithmetic in the same order as the separate kernels it replaced (bit-identical), plus
    the device-side step counter the contact history needs.  Keeping the counter on the
    device is what lets a whole block of steps be captured into one CUDA graph."""
    tid = wp.tid()
    if tid == 0:
        step_arr[0] = step_arr[0] + 1      # every reader of this step has finished
    x0 = x[tid]
    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        x_new[tid] = x0
        particle_w[tid] = wp.vec3(0.0)
        return
    v0 = v[tid]
    f0 = f[tid]
    im = inv_mass[tid]
    v1 = v0 + (f0 * im + gravity[0] * wp.step(-im)) * dt
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag
    x1 = x0 + v1 * dt
    x_new[tid] = x1
    v_new[tid] = v1

    wv = particle_w[tid] + particle_t[tid] * particle_inv_inertia[tid] * dt
    sp = wp.length(wv)
    if sp > max_spin:
        wv = wv * (max_spin / sp)
    particle_w[tid] = wv


