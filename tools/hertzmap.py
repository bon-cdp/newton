"""How does BFA's Hertzian contact map onto our linear spring-dashpot, regime by regime?

Hertz:   Fn = (4/3) E* sqrt(R*) d^1.5          tangent stiffness  k = 1.5 (4/3) E* sqrt(R*) sqrt(d)
Linear:  Fn = ke d                             tangent stiffness  k = ke  (constant)

The point of the exercise: a Hertzian contact has NO single stiffness.  Its tangent
stiffness scales as sqrt(overlap), so it is soft under light load and stiff under heavy
load.  One linear ke has to pick a regime.  Which one did we pick, and how wrong is it
in the other?
"""
import numpy as np

E     = 1.4220405e8      # prj/lin, Pa
NU    = 0.30             # not in the prj; standard DEM value, flagged as an assumption
R     = 0.006
RHO   = 994.05
M     = 4.0/3.0*np.pi*R**3*RHO
G     = 9.81
KE_US = 2.0e4            # our linear normal stiffness

Estar = E/(2.0*(1.0-NU**2))          # two identical spheres
Gstar = E/(2.0*(2.0-NU)*(1.0+NU))    # Mindlin tangential
Rstar = R/2.0
Mstar = M/2.0
C     = 4.0/3.0*Estar*np.sqrt(Rstar)      # Fn = C d^1.5

def d_from_force(F):  return (F/C)**(2.0/3.0)
def d_from_impact(v): return ((5.0*0.5*Mstar*v*v)/(2.0*C))**(0.4)
def k_tan(d):         return 1.5*C*np.sqrt(d)
def kt_mindlin(d):    return 8.0*Gstar*np.sqrt(Rstar*d)

print(f"grain  r {R*1e3:.0f} mm   m {M*1e3:.4f} g      E {E:.3e} Pa   nu {NU} (ASSUMED)")
print(f"E* {Estar:.3e}   G* {Gstar:.3e}   R* {Rstar*1e3:.1f} mm   C = {C:.3e} N/m^1.5\n")

print("  regime                              overlap      Hertz k_n    vs our ke    Mindlin k_t  k_t/k_n")
print("  " + "-"*94)

rows = []
# quasi-static: force per contact ~ sigma * d^2, sigma = rho_bulk g h
for h in (0.05, 0.15, 0.30):
    sig = 710.1*G*h
    F   = sig*(2*R)**2
    rows.append((f"plug, {h*100:.0f} cm deep (sigma {sig:5.0f} Pa)", d_from_force(F)))
rows.append(("single grain weight (F = mg)", d_from_force(M*G)))
for v in (0.5, 2.4, 4.2, 6.3):
    rows.append((f"impact at {v:.1f} m/s", d_from_impact(v)))

for label, d in rows:
    kn = k_tan(d)
    kt = kt_mindlin(d)
    print(f"  {label:<34} {d*1e6:7.1f} um  {kn:10.3e}  {kn/KE_US:8.2f}x  {kt:10.3e}   {kt/kn:6.3f}")

print(f"\n  our linear model:                  {'--':>7}      {KE_US:10.3e}     1.00x  "
      f"{'0 (viscous)':>10}   {0.0:6.3f}")

lo = k_tan(d_from_force(710.1*G*0.30*(2*R)**2))
hi = k_tan(d_from_impact(6.3))
print(f"\n  Hertz spans {lo:.2e} -> {hi:.2e} N/m across this machine, a factor of {hi/lo:.1f}.")
print(f"  Our single ke = {KE_US:.1e} sits {KE_US/lo:.2f}x the plug value and {KE_US/hi:.3f}x the impact value.")

# contact time and timestep check
for label, d, v in (("plug (v ~ 0.1 m/s)", d_from_impact(0.1), 0.1),
                    ("chute impact 6.3 m/s", d_from_impact(6.3), 6.3)):
    tc_h = 2.87*(Mstar**2/(C**2*v))**0.2
    tc_l = np.pi*np.sqrt(Mstar/KE_US)
    print(f"  contact time {label:<22} Hertz {tc_h*1e6:7.2f} us   linear {tc_l*1e6:7.2f} us"
          f"   (dt = 24.32 us)")

print("\n  how deeply do grains interpenetrate at impact?  (grain diameter = 12.0 mm)")
print("  speed      Hertz d_max     linear d_max    ratio   linear as %% of diameter")
for v in (0.5, 2.4, 4.2, 6.3):
    dh = d_from_impact(v)
    dl = v*np.sqrt(Mstar/KE_US)
    print(f"  {v:4.1f} m/s   {dh*1e3:8.3f} mm     {dl*1e3:8.3f} mm    {dl/dh:5.2f}x   {dl/(2*R)*100:6.2f}%%")
