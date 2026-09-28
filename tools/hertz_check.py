"""Validate the Hertz branch before trusting a 10 s run.

Three things must hold, or the implementation is wrong regardless of how the flow looks:
  1. a grain dropped on a plate rebounds at the restitution it was given, in BOTH modes
  2. the peak overlap and contact duration match the closed-form Hertz values
  3. dt*sqrt(k_peak/m*) stays well under 2, or BFA's own timestep is unstable for us
"""
import math, sys
import numpy as np, warp as wp
sys.path.insert(0, '/home/s/Documents/newton-dem')
import newton
from granular_dem import SolverGranularDEM, build_collider

wp.init()
DEV = "cuda:0"
R, RHO = 0.006, 994.05
M = 4.0/3.0*math.pi*R**3*RHO
E, NU, EREST = 1.4220405e8, 0.30, 0.20
DT = 2.4316429e-5

def plate():
    h = 0.5
    v = np.array([[-h,0,-h],[h,0,-h],[h,0,h],[-h,0,h]], float)
    f = np.array([[0,1,2],[0,2,3]], np.int32)
    if np.cross(v[1]-v[0], v[2]-v[0])[1] < 0: f = f[:, ::-1]
    return v, np.ascontiguousarray(f)

def drop(hertz, v0=1.0, ke=2.0e4, dt=DT, kdmul=1.0, cal=False):
    verts, faces = plate()
    b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    b.add_shape_mesh(body=-1, mesh=newton.Mesh(verts, faces.flatten()),
                     cfg=newton.ModelBuilder.ShapeConfig(mu=0.0), key="p")
    b.add_particle(pos=wp.vec3(0.0, R*1.0001, 0.0), vel=wp.vec3(0.0,-v0,0.0),
                   mass=M, radius=R)
    m = b.finalize(device=DEV)
    zeta = -math.log(EREST)/math.sqrt(math.pi**2+math.log(EREST)**2)
    kd = kdmul*zeta*2.0*math.sqrt(M*ke)
    m.particle_ke, m.particle_kd, m.particle_kf = ke, kd, 0.0
    m.particle_mu, m.particle_cohesion = 0.0, 0.0
    m.set_gravity((0.0, 0.0, 0.0))            # isolate the contact from gravity
    m.particle_grid = wp.HashGrid(8,8,8, device=DEV)
    col, keep = build_collider([("p", verts, faces)], two_sided=[False], friction=[0.0],
                               ke=[ke], kd=[kd], kf=[0.0], thickness=[0.0],
                               max_dist=0.03, device=DEV)
    sol = SolverGranularDEM(m, col, grid_cell=2*R, keepalive=keep, rotation=True,
                            hertz=hertz, youngs=E, poisson=NU, restitution=EREST,
                            calibrate_restitution=cal, cal_dt=dt)
    s0, s1 = m.state(), m.state()
    ov, nc = 0.0, 0
    for _ in range(int(4000*DT/dt)):
        sol.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        y = float(s0.particle_q.numpy()[0][1])
        d = R - y
        if d > 0:
            ov = max(ov, d); nc += 1
        vy = float(s0.particle_qd.numpy()[0][1])
        if d <= 0 and vy > 0:
            return vy/v0, ov, nc*dt
    return float('nan'), ov, nc*dt

print(f"  grain r {R*1e3:.0f} mm  m {M*1e3:.4f} g   target restitution {EREST}")
print(f"\n  mode      v0      e_out   peak overlap   contact time   Hertz theory (d, tc)")
Estar_w = E/(1-NU**2); C = 4.0/3.0*Estar_w*math.sqrt(R)
for v0 in (0.5, 2.4, 6.3):
    for lbl, hz in (("linear", False), ("hertz ", True)):
        e_out, ov, tc = drop(hz, v0)
        dth = ((5.0*0.5*M*v0*v0)/(2.0*C))**0.4
        tcth = 2.87*(M**2/(C**2*v0))**0.2
        print(f"  {lbl}  {v0:5.2f}   {e_out:6.3f}   {ov*1e3:8.3f} mm   {tc*1e6:8.1f} us"
              f"     {dth*1e3:6.3f} mm  {tcth*1e6:7.1f} us")

k_peak = 1.5*C*math.sqrt(((5.0*0.5*M*6.3**2)/(2.0*C))**0.4)
print(f"\n  stability: dt*sqrt(k_peak/m) = {DT*math.sqrt(k_peak/M):.3f} at 6.3 m/s "
      f"(needs < 2; linear ke=2e4 gives {DT*math.sqrt(2.0e4/M):.3f})")


print("\n  is e=0.33 the force clamp or the timestep?  (linear, v0 = 2.4 m/s)")
for div in (1, 2, 4, 8):
    e_out, ov, tc = drop(False, 2.4, dt=DT/div)
    print(f"    dt/{div:<2d} = {DT/div*1e6:6.2f} us   e = {e_out:.3f}   ({tc/(DT/div):.0f} steps in contact)")

print("\n  kd multiplier needed to actually deliver e = 0.20 (v0 = 2.4 m/s)")
for kdm in (1.0, 1.3, 1.6, 2.0, 2.5):
    el, _, _ = drop(False, 2.4, kdmul=kdm)
    eh, _, _ = drop(True, 2.4, kdmul=kdm)
    print(f"    kd x {kdm:4.2f}   linear e {el:.3f}   hertz e {eh:.3f}")


print("\n  DOES CALIBRATION DELIVER THE REQUESTED e = 0.20?  (grain on a plate)")
print("    mode      v0     uncalibrated      calibrated")
for v0 in (0.5, 2.4, 6.3):
    for lbl, hz in (("linear", False), ("hertz ", True)):
        a, _, _ = drop(hz, v0, cal=False)
        b, _, _ = drop(hz, v0, cal=True)
        print(f"    {lbl}  {v0:5.2f}      {a:.3f}            {b:.3f}")
