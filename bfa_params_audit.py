#!/usr/bin/env python3
"""
Decode the BulkFlowAnalyst project in 23087-25sim/ and cross-check every value
the Newton MPM replication depends on.

BFA writes reals as an integer mantissa with an exponent: "24316429E-012" means
24316429e-12 = 2.4316429e-5, NOT 2.4316429e-12.  Lengths are inches, densities
lb/ft3, velocities in/s, mass flow short tons/hour -- even with "Units: Metric"
set in the project header.

Run:  python bfa_params_audit.py
"""

from __future__ import annotations

import math
import os
import re

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BFA = os.path.join(SCRIPT_DIR, "23087-25sim")

IN = 0.0254            # m per inch
LBFT3 = 16.018463      # kg/m3 per lb/ft3
SHORT_TON = 907.18474  # kg per short ton
G = 9.81


def bfa_float(tok: str) -> float:
    """'24316429E-012' -> 2.4316429e-05  (integer mantissa, not a normalised one)."""
    mant, exp = tok.strip().split("E")
    return float(mant) * 10.0 ** int(exp)


def scan(path, keys, encoding="latin1"):
    """First occurrence of each 'Key:' in a BFA text file."""
    out = {}
    want = set(keys)
    with open(path, encoding=encoding, errors="replace") as fh:
        for line in fh:
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            if k in want and k not in out:
                out[k] = v.strip().rstrip(",")
    return out


def read_lin(path):
    """The .lin file is '# Label:' / value pairs."""
    lines = [ln.strip() for ln in open(path, encoding="latin1", errors="replace")]
    out = {}
    for i, ln in enumerate(lines):
        if ln.startswith("#") and i + 1 < len(lines):
            out.setdefault(ln.lstrip("# ").rstrip(":"), lines[i + 1])
    return out


def row(label, source, raw, decoded, used=None, ok=None):
    mark = "" if ok is None else ("  OK" if ok else "  <-- MISMATCH")
    u = "" if used is None else f"{used:>18}"
    print(f"  {label:<28} {source:<24} {raw:<18} {decoded:>18}{u}{mark}")


def close(a, b, rel=2e-3):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def main():
    prj = os.path.join(BFA, "New Simulation (1).prj")
    est = os.path.join(BFA, "New Simulation (1).est")
    lin = os.path.join(BFA, "NewSimulationX1X.lin")

    p = scan(prj, [
        "Material Intr Density", "Material Bulk Density", "Material Min_Rad", "Material Max_Rad",
        "Material Packing_Ratio", "Material Coefficient_Restitution", "Material Inter-particle Friction",
        "Material Rotating Friction", "Material Twisting Friction", "Material Cohesion_Coefficient",
        "Material Angle Of Repose (degrees)", "Material Max Velocity (in/s)",
        "Injection Box FlowRate", "Injection Box Injection Start", "Injection Extrusion Length",
        "Simulation Time", "Simulation FramesPerSecond", "Simulation TimeStepMultiplier",
        "Chute Friction", "Contact Mode", "Gravity Direction", "Units", "Simulation Contact Model",
        "Simulation Allows Rotating",
    ])
    e = dict(ln.split(":", 1) for ln in open(est) if ":" in ln)
    e = {k.strip(): v.strip() for k, v in e.items()}
    L = read_lin(lin)

    print("=" * 108)
    print("BulkFlowAnalyst project audit -- 23087 'Quarter Scale', New Simulation (1)")
    print("=" * 108)
    print(f"  {'quantity':<28} {'source':<24} {'raw':<18} {'decoded (SI)':>18}{'used by MPM':>18}")
    print("  " + "-" * 104)

    # --- material -----------------------------------------------------------
    rho_grain = bfa_float(p["Material Intr Density"]) * LBFT3
    rho_bulk = bfa_float(p["Material Bulk Density"]) * LBFT3
    packing = bfa_float(p["Material Packing_Ratio"])
    rad = bfa_float(p["Material Min_Rad"]) * IN
    rad_max = bfa_float(p["Material Max_Rad"]) * IN
    aor = bfa_float(p["Material Angle Of Repose (degrees)"])
    mu_pp = bfa_float(p["Material Inter-particle Friction"])
    mu_roll = bfa_float(p["Material Rotating Friction"])
    cor = bfa_float(p["Material Coefficient_Restitution"])
    coh = bfa_float(p["Material Cohesion_Coefficient"])
    young = bfa_float(L["Young's modulus (Pa)"])

    row("grain density", "prj Intr Density", p["Material Intr Density"], f"{rho_grain:.1f} kg/m3", "reporting only")
    row("bulk density", "prj Bulk Density", p["Material Bulk Density"], f"{rho_bulk:.1f} kg/m3", f"{710.1} kg/m3",
        close(rho_bulk, 710.1, 1e-3))
    row("packing ratio", "prj Packing_Ratio", p["Material Packing_Ratio"], f"{packing:.3f}",
        f"{rho_grain / rho_bulk:.3f} (chk)", close(packing, rho_grain / rho_bulk))
    row("grain radius", "prj Min_Rad = Max_Rad", p["Material Min_Rad"], f"{rad:.4f} m", "0.0060 m",
        close(rad, 0.006, 1e-3) and close(rad, rad_max))
    row("angle of repose", "prj Angle Of Repose", p["Material Angle Of Repose (degrees)"], f"{aor:.2f} deg",
        f"tan = {math.tan(math.radians(aor)):.4f}", close(math.tan(math.radians(aor)), 0.4307, 1e-3))
    row("inter-particle friction", "prj Inter-particle Fric", p["Material Inter-particle Friction"], f"{mu_pp:.3f}",
        "not used *")
    row("rolling friction", "prj Rotating Friction", p["Material Rotating Friction"], f"{mu_roll:.3f}", "not used *")
    row("restitution", "prj Coefficient_Restit", p["Material Coefficient_Restitution"], f"{cor:.3f}", "no analogue *")
    row("cohesion", "prj Cohesion_Coeff", p["Material Cohesion_Coefficient"], f"{coh:.3f}", "yield_stress 0",
        coh == 0.0)
    row("Young's modulus (DEM)", "lin Young's modulus", L["Young's modulus (Pa)"], f"{young:.4g} Pa", "not used *")

    # --- boundaries ---------------------------------------------------------
    wall = bfa_float(p["Chute Friction"])
    n_int = 0
    frics = []
    with open(prj, encoding="latin1", errors="replace") as fh:
        for line in fh:
            if line.startswith("Friction:"):
                frics.append(bfa_float(line.split(":", 1)[1]))
                n_int += 1
    row("wall friction", "prj Chute Friction", p["Chute Friction"], f"{wall:.3f}", f"{0.5:.3f}", close(wall, 0.5))
    if frics:
        pb = [f for f in frics if close(f, 0.5)]
        row("  ..material interactions", f"prj ({n_int} pairs)", "-",
            f"{len(pb)}/{n_int} at 0.500", f"pp = {frics[0]:.3f}")

    # --- injection ----------------------------------------------------------
    flow_stph = bfa_float(p["Injection Box FlowRate"])
    flow_kgs = flow_stph * SHORT_TON / 3600.0
    extrude = bfa_float(p["Injection Extrusion Length"]) * IN
    v_inj = math.sqrt(2.0 * G * extrude)
    row("mass flow", "prj Injection FlowRate", p["Injection Box FlowRate"],
        f"{flow_kgs:.4f} kg/s", "8.6111 kg/s", close(flow_kgs, 8.6111, 1e-4))
    row("  ..as short ton/h", "", "", f"{flow_stph:.4f} STPH", f"{flow_kgs * 3.6:.2f} t/h")
    row("injection extrusion", "prj Extrusion Length", p["Injection Extrusion Length"], f"{extrude:.4f} m",
        f"v = {v_inj:.4f} m/s", close(v_inj, 3.1321, 1e-3))

    # --- run control --------------------------------------------------------
    simtime = bfa_float(p["Simulation Time"])
    fps = float(p["Simulation FramesPerSecond"])
    dt_dem = bfa_float(L["Timestep"])
    nsteps = int(e["# of Timesteps"])
    row("simulated time", "prj Simulation Time", p["Simulation Time"], f"{simtime:.2f} s", "10.00 s",
        close(simtime, 10.0))
    row("output rate", "prj FramesPerSecond", p["Simulation FramesPerSecond"], f"{fps:.0f} fps", "15 fps", fps == 15)
    row("DEM timestep", "lin Timestep", L["Timestep"], f"{dt_dem:.4e} s", f"{nsteps:,} steps",
        close(dt_dem * nsteps, simtime, 1e-4))
    row("gravity", "prj Gravity Direction", p["Gravity Direction"], "-Y", "(0,-9.81,0)",
        p["Gravity Direction"] == "NegativeY")
    row("contact model", "prj Contact Mode", p["Contact Mode"], "Hertzian", "Drucker-Prager *")

    # --- .est cross-checks ---------------------------------------------------
    print("\n  cross-checks against 'New Simulation (1).est' (BFA's own estimates)")
    print("  " + "-" * 104)
    mass_grain = 4.0 / 3.0 * math.pi * rad**3 * rho_grain
    pps = flow_kgs / mass_grain
    row("min/max radius", "est Min/Max Radius", e["Min Radius"], f"{float(e['Min Radius']):.6f} m",
        f"{rad:.6f} m", close(float(e["Min Radius"]), rad, 1e-5))
    row("max velocity", "est MaxV", e["MaxV (m/s)"], f"{float(e['MaxV (m/s)']):.4f} m/s",
        f"{bfa_float(p['Material Max Velocity (in/s)']) * IN:.4f} m/s",
        close(float(e["MaxV (m/s)"]), bfa_float(p["Material Max Velocity (in/s)"]) * IN, 1e-6))
    row("est particle count", "est Est Prt Count", e["Est Prt Count"], f"{float(e['Est Prt Count']):.1f} /s",
        f"{pps:.1f} /s", close(float(e["Est Prt Count"]), pps, 5e-4))
    row("boundary elements", "est Element Count", e["Element Count"], f"{e['Element Count']} tris",
        "14461 loaded", int(e["Element Count"]) == 14461)

    # --- independent check against the binary particle output ----------------
    print("\n  independent verification from Local Output/*.por + *.his")
    print("  " + "-" * 104)
    try:
        from compare_bfa_mpm import read_bfa_frames, read_bfa_history
        frames = []
        for i, f in enumerate(read_bfa_frames()):
            frames.append(f)
            if i >= 2:
                break
        f0 = frames[0]
        fresh = f0["pos"][:, 1] > 3.97
        v_meas = f0["vel"][fresh][:, 1].mean()
        ke = 0.5 * (4 / 3 * np.pi * f0["rad"] ** 3 * rho_grain * np.sum(f0["vel"] ** 2, axis=1)).sum()
        his = read_bfa_history()
        row("injection velocity", "por frame 1, y>3.97", f"n={fresh.sum()}", f"{v_meas:.4f} m/s",
            f"{-v_inj:.4f} m/s", close(-v_meas, v_inj, 5e-3))
        row("frame-1 kinetic energy", "por, rho = grain", "-", f"{ke:.4f} J",
            f"his {his[1, 2]:.4f} J", close(ke, his[1, 2], 1e-3))
        row("grain mass", "derived", "-", f"{mass_grain * 1e3:.4f} g", f"{np.median(f0['rad']) * 1e3:.2f} mm r")
        steady = his[:, 0] >= 3.0
        row("steady holdup", "his (t>3 s)", "-", f"{his[steady, 1].mean():.2f} kg", "MPM target")
        row("steady kinetic energy", "his (t>3 s)", "-", f"{his[steady, 2].mean():.1f} J", "MPM target")
    except Exception as exc:  # pragma: no cover
        print(f"    (skipped: {exc})")

    print("""
  * not transferable to MPM:
      inter-particle friction 0.09 and rolling friction 0.30 are DEM contact-law
      numbers that only mean anything together; what they were calibrated to
      produce is the 23.3 deg angle of repose, and tan(23.3) = 0.4307 is what a
      Drucker-Prager continuum wants.  Restitution has no continuum analogue --
      implicit MPM is inelastic by construction.  The DEM Young's modulus
      1.42e8 Pa is a softened *contact* stiffness chosen to allow a 24.3 us
      timestep, not a bulk modulus, so it is not carried over either.
""")


if __name__ == "__main__":
    main()
