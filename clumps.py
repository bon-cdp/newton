#!/usr/bin/env python3
"""Random multi-sphere clumps for the size distribution (issue #10).

A size class with "spheres": n > 1 is made of rigid clumps of n overlapping spheres
instead of single spheres.  Each clump takes one of a few random shapes (TEMPLATES per
class), every shape scaled so its volume -- the UNION of its spheres -- equals the class's
equivalent sphere, so the class keeps its grain mass and mass fractions stay exact.

A shape grows like a small aggregate: the first sphere at the origin, then each new one
touching a random earlier one at a centre distance of 0.8-1.3 radii (so they overlap and
the clump is connected) and no closer than 0.6 radii to any other.  Mass properties are
integrated over the union on a voxel grid; the shape is then shifted to its centre of mass
and rotated into its principal axes, so its inertia is diagonal (ClumpTemplate, as used for
the soybean calibration).
"""
from __future__ import annotations

import functools
import math

import numpy as np

TEMPLATES = 8          # random shapes per class


class ClumpTemplate:
    """Sphere centres (body frame, m) and radii; volume, mass and principal inertia of the
    union of the spheres (voxel grid, overlaps counted once).  Shifted to the centre of
    mass and rotated into the principal axes."""

    def __init__(self, offsets, radii, density: float, voxels: int = 64):
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

    def scaled(self, s: float, density_ratio: float = 1.0) -> "ClumpTemplate":
        """The same shape s times larger (volume s^3, inertia s^5)."""
        t = object.__new__(ClumpTemplate)
        t.volume = self.volume * s ** 3
        t.mass = self.mass * s ** 3 * density_ratio
        t.inertia = self.inertia * s ** 5 * density_ratio
        t.offsets = self.offsets * s
        t.radii = self.radii * s
        t.n = self.n
        t.bound_radius = self.bound_radius * s
        return t


def _grow(n, rng):
    """Centres of n unit spheres grown as a connected overlapping aggregate."""
    pts = [np.zeros(3)]
    while len(pts) < n:
        base = pts[rng.integers(len(pts))]
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        c = base + d * rng.uniform(0.8, 1.3)
        if min(np.linalg.norm(c - p) for p in pts) >= 0.6:
            pts.append(c)
    return np.array(pts)


@functools.lru_cache(maxsize=64)
def _unit_templates(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return tuple(ClumpTemplate(_grow(n, rng), [1.0], 1.0) for _ in range(TEMPLATES))


def templates(radius: float, n: int, density: float, seed: int = 0):
    """TEMPLATES random clumps of n spheres, each with the volume (and mass) of a sphere of
    `radius` at `density`."""
    v_target = 4.0 / 3.0 * math.pi * radius ** 3
    out = []
    for t in _unit_templates(int(n), int(seed)):
        s = (v_target / t.volume) ** (1.0 / 3.0)
        out.append(t.scaled(s, density))
    return out


def describe(tpls):
    r = np.concatenate([t.radii for t in tpls])
    b = [t.bound_radius for t in tpls]
    return (f"{len(tpls)} random shapes of {tpls[0].n} spheres, sphere radius "
            f"{r.min()*1e3:.2f}-{r.max()*1e3:.2f} mm, reach {min(b)*1e3:.2f}-{max(b)*1e3:.2f} mm")
