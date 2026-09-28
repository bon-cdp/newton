#!/usr/bin/env python3
"""
Export the BulkFlowAnalyst DEM reference to VTK, in the same format as our runs.

Until now the only way to see BFA's result was its rendered .avi, which cannot be
compared like-for-like against a ParaView render of ours -- different camera, scale and
glyph size.  This writes the decoded .por frames as frame_XXXX_particles.vtk plus the
same geometry.vtk, so both can be loaded into one session and scrubbed together.

BFA writes 149 frames at 15 fps; ours are 150 at the same rate, so frame N lines up
with frame N+1 of ours (BFA has no t=0 frame).

Usage:  python export_bfa_vtk.py [outdir]
"""
from __future__ import annotations
import os, sys
import numpy as np
from compare_bfa_mpm import read_bfa_frames, DEM_MASS, DEM_RADIUS
from bfa_replication_mpm import COLLIDER_PARTS, load_part, write_geometry_vtk, write_particles_vtk

out = sys.argv[1] if len(sys.argv) > 1 else "bfa_reference_vtk"
os.makedirs(out, exist_ok=True)

parts = [(n, *load_part(n, fx, fl)) for n, ts, fx, fl in COLLIDER_PARTS]
write_geometry_vtk(os.path.join(out, "geometry.vtk"), parts)

n = 0
for i, f in enumerate(read_bfa_frames()):
    write_particles_vtk(os.path.join(out, f"frame_{i+1:04d}_particles.vtk"),
                        i + 1, f["pos"], f["vel"], DEM_RADIUS)
    n += 1
    if (i + 1) % 30 == 0:
        print(f"  {i+1} frames  ({f['n']} grains, {f['n']*DEM_MASS:.2f} kg)", flush=True)
print(f"wrote {n} frames + geometry.vtk to {out}/")
print(f"grain radius {DEM_RADIUS*1e3:.1f} mm -- set the ParaView glyph to this to match ours")
