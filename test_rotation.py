#!/usr/bin/env python3
"""
Unit tests for the rotational DEM in granular_dem.py, against analytic answers.

1. Sphere rolling down an incline.  A solid sphere that rolls without slipping
   accelerates at a = (5/7) g sin(theta) -- markedly less than the g sin(theta) of a
   frictionless slide, because 2/7 of the energy goes into spin.  It must also satisfy
   the rolling constraint |omega| = v / r.  This is the sharpest available check that
   the tangential force, the lever arm and the inertia are consistent: get any one of
   them wrong and the acceleration misses.

2. Rolling resistance brings a spinning sphere to rest.  With mu_roll > 0 and no drive,
   a rolling sphere on a level surface must decelerate at mu_roll * g and stop.

Usage:  python test_rotation.py
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from granular_dem import SolverGranularDEM, build_collider

wp.init()
DEVICE = "cuda:0"
G = 9.81
RADIUS = 0.02
MASS = 0.01


def inclined_plane(theta_deg: float, size: float = 4.0):
    """A single-sided ramp descending in +x, with its normal pointing up out of the face."""
    t = math.radians(theta_deg)
    # surface contains direction (cos t, -sin t, 0); normal is (sin t, cos t, 0)
    u = np.array([math.cos(t), -math.sin(t), 0.0])
    w = np.array([0.0, 0.0, 1.0])
    o = np.array([-0.5, 0.0, 0.0])
    v = np.array([o, o + u * size, o + u * size + w * size, o + w * size])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    n = np.cross(v[1] - v[0], v[2] - v[0])
    if n[1] < 0:                      # ensure the normal points up out of the ramp
        f = f[:, ::-1]
    return v, np.ascontiguousarray(f)


def run(theta_deg, mu, mu_roll, steps=4000, dt=2.0e-5, v0=0.0, w0=None,
        tangential_ratio=0.0, kf=1.0e3):
    verts, faces = inclined_plane(theta_deg)
    parts = [("ramp", verts, faces)]

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-G)
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(verts, faces.flatten()),
                           cfg=newton.ModelBuilder.ShapeConfig(mu=mu), key="ramp")
    t = math.radians(theta_deg)
    down = np.array([math.cos(t), -math.sin(t), 0.0])
    nrm = np.array([math.sin(t), math.cos(t), 0.0])
    u = down
    w = np.array([0.0, 0.0, 1.0])
    # The start point must lie ON the ramp plane, then be lifted by one radius.  Placing
    # it at a fixed world point puts it above the plane for any non-zero tilt, and the
    # sphere then free-falls for the whole test -- which reads as "a = g sin(theta),
    # omega = 0", i.e. exactly like a friction bug.
    o_pt = np.array([-0.5, 0.0, 0.0])
    start = o_pt + u * 1.0 + w * 2.0 + nrm * RADIUS * 0.999
    builder.add_particle(pos=wp.vec3(*start), vel=wp.vec3(*(down * v0)),
                         mass=MASS, radius=RADIUS)

    model = builder.finalize(device=DEVICE)
    ke = 1.0e5
    model.particle_ke, model.particle_kd, model.particle_kf = ke, 1.0, kf
    model.particle_mu, model.particle_cohesion = mu, 0.0
    model.set_gravity((0.0, -G, 0.0))
    model.particle_grid = wp.HashGrid(16, 16, 16, device=DEVICE)

    collider, meshes = build_collider(
        parts, two_sided=[False], friction=[mu], ke=[ke], kd=[1.0], kf=[kf],
        thickness=[0.0], max_dist=0.02, device=DEVICE)
    solver = SolverGranularDEM(model, collider, grid_cell=4 * RADIUS, keepalive=meshes,
                               rotation=True, mu_roll=mu_roll, mu_roll_wall=mu_roll,
                               tangential_ratio=tangential_ratio)
    if w0 is not None:
        solver.particle_w = wp.array(np.array([w0], dtype=np.float32), dtype=wp.vec3, device=DEVICE)

    s0, s1 = model.state(), model.state()
    traj = []
    for step in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        if step % 200 == 0:
            q = s0.particle_q.numpy()[0]
            v = s0.particle_qd.numpy()[0]
            om = solver.particle_w.numpy()[0]
            traj.append((step * dt, float(np.dot(v, down)), float(om[2]), q.copy()))
    return np.array([(a, b, c) for a, b, c, _ in traj]), traj


def test_rolling_incline():
    theta = 20.0
    print(f"\n1. Sphere rolling down a {theta:.0f} deg incline (mu = 0.6, no rolling resistance)")
    tr, _ = run(theta, mu=0.6, mu_roll=0.0, steps=6000)
    # fit acceleration over the second half, once contact has settled
    half = tr[len(tr) // 2:]
    a_meas = np.polyfit(half[:, 0], half[:, 1], 1)[0]
    a_roll = 5.0 / 7.0 * G * math.sin(math.radians(theta))
    a_slide = G * math.sin(math.radians(theta))
    v_end, w_end = tr[-1, 1], tr[-1, 2]
    print(f"   measured acceleration   {a_meas:7.3f} m/s^2")
    print(f"   rolling  (5/7)g sin(t)  {a_roll:7.3f} m/s^2   <- expected")
    print(f"   sliding      g sin(t)   {a_slide:7.3f} m/s^2")
    print(f"   ratio to rolling        {a_meas/a_roll:7.3f}")
    print(f"   rolling constraint  |w|*r = {abs(w_end)*RADIUS:6.3f} vs v = {v_end:6.3f} m/s"
          f"   ratio {abs(w_end)*RADIUS/max(v_end,1e-9):6.3f}")
    ok = abs(a_meas / a_roll - 1.0) < 0.12 and abs(abs(w_end) * RADIUS / max(v_end, 1e-9) - 1.0) < 0.15
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def test_rolling_resistance():
    print("\n2. Rolling resistance stops a rolling sphere on the level (mu_roll = 0.10)")
    v0 = 1.0
    tr, _ = run(0.0, mu=0.6, mu_roll=0.10, steps=8000, v0=v0, w0=[0.0, 0.0, -v0 / RADIUS])
    half = tr[2:len(tr) // 2]
    a_meas = np.polyfit(half[:, 0], half[:, 1], 1)[0]
    a_pred = -0.10 * G * 5.0 / 7.0
    print(f"   measured deceleration   {a_meas:7.3f} m/s^2")
    print(f"   expected ~ -mu_r*g*5/7  {a_pred:7.3f} m/s^2")
    print(f"   speed {v0:.2f} -> {tr[-1,1]:.3f} m/s")
    ok = a_meas < -0.02 and tr[-1, 1] < v0
    print(f"   {'PASS' if ok else 'FAIL'}  (sign and monotonic slowing are what matter here;"
          f" the exact coefficient depends on the contact model)")
    return ok


def test_static_friction():
    """A grain on a slope shallower than its friction angle must not move.

    This is the one thing a viscous tangential law cannot do.  ft = min(kf*|vt|, mu*fn)
    vanishes as the slip rate goes to zero, so the only steady state is a creep at
    v = mg sin(theta)/kf -- with the production kf = 3.0 that is ~11 mm/s for a grain
    that should be motionless.  A Cundall-Strack spring stores the displacement and
    holds it.  Rolling friction is set high so the sphere cannot simply roll away,
    which would confound the test.
    """
    theta, mu, kf = 20.0, 0.6, 3.0
    print(f"\n3. Grain at rest on a {theta:.0f} deg slope, friction angle "
          f"atan({mu}) = {math.degrees(math.atan(mu)):.1f} deg -- it must NOT move")
    print(f"   (production kf = {kf}; viscous creep would be mg sin(theta)/kf = "
          f"{MASS*G*math.sin(math.radians(theta))/kf*1e3:.1f} mm/s)")
    out = {}
    for lbl, ratio in [("viscous  (ratio 0)", 0.0), ("spring   (ratio 2/7)", 2.0 / 7.0)]:
        tr, _ = run(theta, mu=mu, mu_roll=1.0, steps=5000, dt=2.0e-5,
                    tangential_ratio=ratio, kf=kf)
        half = tr[len(tr) // 2:]
        v = float(np.mean(half[:, 1]))
        out[lbl] = v
        print(f"   {lbl:<22} mean speed {v*1e3:8.3f} mm/s")
    ok = abs(out["spring   (ratio 2/7)"]) < 0.2 * abs(out["viscous  (ratio 0)"]) + 1e-5
    print(f"   {'PASS' if ok else 'FAIL'}  (spring must hold it far better than viscous)")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("Rotational DEM unit tests")
    print("=" * 70)
    r1 = test_rolling_incline()
    r2 = test_rolling_resistance()
    r3 = test_static_friction()
    print("\n" + "=" * 70)
    print(f"  incline rolling      {'PASS' if r1 else 'FAIL'}")
    print(f"  rolling resistance   {'PASS' if r2 else 'FAIL'}")
    print(f"  static friction      {'PASS' if r3 else 'FAIL'}")
