#!/usr/bin/env python3
"""
Analyze VTK particle output for stationary particle accumulation.
Usage: python analyze_vtk.py [vtk_dir]
       If no dir specified, uses most recent vtk_output_mpm_test_* directory.
"""

import os
import sys
import glob
import numpy as np

def parse_vtk_particles(filepath):
    """Parse VTK file and extract particle positions and velocities."""
    positions = []
    velocities = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    n_points = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('POINTS'):
            parts = line.split()
            n_points = int(parts[1])
            i += 1
            while len(positions) < n_points and i < len(lines):
                vals = lines[i].strip().split()
                for j in range(0, len(vals), 3):
                    if len(vals) >= j + 3:
                        positions.append([float(vals[j]), float(vals[j+1]), float(vals[j+2])])
                i += 1
        elif 'velocity' in line.lower() and 'VECTORS' in line:
            i += 1
            while len(velocities) < n_points and i < len(lines):
                vals = lines[i].strip().split()
                for j in range(0, len(vals), 3):
                    if len(vals) >= j + 3:
                        velocities.append([float(vals[j]), float(vals[j+1]), float(vals[j+2])])
                i += 1
        else:
            i += 1

    return np.array(positions), np.array(velocities)

def find_latest_vtk_dir():
    """Find the most recent vtk_output_mpm_test_* directory."""
    dirs = glob.glob("vtk_output_mpm_test_*")
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def analyze(vtk_dir):
    """Analyze stationary particles over time."""
    print(f"Analyzing: {vtk_dir}\n")
    print("Frame | Total | Still (<0.01 m/s) | Still % | Δ Still")
    print("-" * 55)

    prev_still = 0
    frames_to_check = [0, 5, 10, 20, 30, 40, 50]

    for frame in frames_to_check:
        filepath = os.path.join(vtk_dir, f"frame_{frame:04d}_particles.vtk")
        if not os.path.exists(filepath):
            continue

        pos, vel = parse_vtk_particles(filepath)

        if len(vel) == 0:
            print(f"{frame:5d} | {len(pos):5d} | No velocity data")
            continue

        speeds = np.linalg.norm(vel, axis=1)
        still_count = np.sum(speeds < 0.01)
        still_pct = 100.0 * still_count / len(pos) if len(pos) > 0 else 0
        delta = still_count - prev_still

        print(f"{frame:5d} | {len(pos):5d} | {still_count:17d} | {still_pct:6.1f}% | {delta:+6d}")
        prev_still = still_count

    # Summary at frame 50
    filepath = os.path.join(vtk_dir, "frame_0050_particles.vtk")
    if os.path.exists(filepath):
        pos, vel = parse_vtk_particles(filepath)
        if len(vel) > 0:
            speeds = np.linalg.norm(vel, axis=1)
            still_count = np.sum(speeds < 0.01)
            still_pct = 100.0 * still_count / len(pos)
            print(f"\n==> Frame 50 result: {still_pct:.1f}% stationary ({still_count}/{len(pos)})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vtk_dir = sys.argv[1]
    else:
        vtk_dir = find_latest_vtk_dir()

    if not vtk_dir or not os.path.isdir(vtk_dir):
        print("No VTK directory found. Specify one or run simulation first.")
        sys.exit(1)

    analyze(vtk_dir)
