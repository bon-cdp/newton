#!/usr/bin/env python3
"""
Scenario files: everything that defines a DEM run, as data.

A scenario is a JSON file that `dem_run.py` executes.  It replaces the machine-specific
constants that used to live in bfa_dem.py / bfa_replication_mpm.py, so a new machine is a
new file, not new code.  `bfa_import.py` writes one from a BulkFlowAnalyst project.

    {
      "name": "...",
      "units": "m",                      # STL/coordinate unit; all other fields are SI
      "gravity": [0, -9.81, 0],
      "material": {...},                 # Material
      "parts": [{...}, ...],             # Part, one per STL
      "injectors": [{...}],              # Injector
      "domain": {"lo": [...], "hi": [...]},   # grains leaving it are recycled
      "regions": [{...}],                # Region: named boxes reported in history.csv
      "flow_planes": [{...}],            # FlowPlane: mass crossing a rectangle (portals)
      "solver": {...},                   # Solver
      "output": {...}                    # Output
    }

Relative paths (STL files) resolve against the scenario file's directory.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields

INF = 1.0e30
UNITS = {"m": 1.0, "mm": 1.0e-3, "cm": 1.0e-2, "in": 0.0254, "ft": 0.3048}


@dataclass
class Material:
    """One granular material (spheres; clumps and size distributions are issue #10)."""

    name: str = "material"
    radius: float = 0.006                 # m
    density: float = 994.05               # kg/m3, intrinsic (solid) density
    youngs: float = 1.4220405e8           # Pa, TRUE modulus; the solver may soften it
    poisson: float = 0.30
    restitution: float = 0.20
    friction: float = 0.09                # grain-grain sliding
    rolling_friction: float = 0.30        # grain-grain, constant directional torque
    wall_rolling_friction: float = 0.50
    rot_damp: float = 0.20
    rot_damp_wall: float = 0.20
    tangential_ratio: float = 0.0         # Mindlin k_t/k_n (0 = viscous-Coulomb)
    contact: str = "linear"               # "linear" or "hertz"
    ke: float = 2.0e4                     # N/m, linear contact stiffness
    kf: float = 3.0                       # N.s/m, tangential viscous coefficient
    kd: float | None = None               # override grain-grain damping (else from restitution)
    kd_wall: float | None = None          # override grain-wall damping
    calibrate_restitution: bool = False
    rotation: bool = True

    @property
    def mass(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3 * self.density


@dataclass
class Part:
    """One collider surface (an STL)."""

    name: str
    stl: str
    two_sided: bool = True                # thin shell repelling from both faces
    fix_normals: bool = False             # trimesh fix_normals on load (one-sided parts)
    flip: bool = False                    # reverse winding on load
    thickness: float = 0.0                # m, shell half-thickness (two-sided only)
    friction: float = 0.5                 # grain-wall Coulomb
    active: list[float] = field(default_factory=lambda: [0.0, INF])   # [on, off) seconds
    motion: dict | None = None            # surface motion (belt / rotating), see dem_run.surface_motion
    corners: bool = False                 # second contact in concave creases of this part (#13)


@dataclass
class Injector:
    """Grains spawned on a non-overlapping lattice: over a planar STL face, or filling a
    box placed relative to a belt or a horizontal plane (box, see dem_run.injector_box):

      {"reference": "belt", "part": "<belt part>", "position": m from its upstream end,
       "lateral": m from its centreline, "clearance": m above its surface,
       "length": m, "width": m, "height": m, "match_belt": true}
      {"reference": "plane", "plane_height": m, "center": [x, y, z], "direction": [..],
       "clearance": m, "length": m, "width": m, "height": m}
    """

    name: str
    face_stl: str = ""
    mass_rate: float = 10.0               # kg/s
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    start: float = 0.0
    stop: float = INF
    site_spacing: float = 2.2             # lattice pitch, in grain radii
    wall_clearance: float = 1.25          # drop sites closer than this many radii to a wall
    batch_fraction: float = 0.25          # at most this fraction of sites filled per event
    max_interval: float = 0.04            # s; smaller batches if events would be further apart
    set_coordinate: list[float] | None = None   # [axis, value]: place all sites on that plane
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # m, added to sites
    box: dict | None = None               # box injector instead of face_stl (see above)


@dataclass
class Region:
    """A named box whose grain mass / mean speed (and optionally spin) go to history.csv."""

    name: str
    lo: list[float] = field(default_factory=lambda: [-INF, -INF, -INF])
    hi: list[float] = field(default_factory=lambda: [INF, INF, INF])
    spin: bool = False


@dataclass
class FlowPlane:
    """Mass crossing a rectangle in an axis-aligned plane (BFA's "portals").

    axis: 0/1/2 = the plane's normal (x/y/z); value: its coordinate; lo/hi: the rectangle
    in the other two coordinates (the axis component is ignored).  Crossings in either
    direction are counted; history.csv gets the cumulative mass as flow_<name>_kg."""

    name: str
    axis: int
    value: float
    lo: list[float] = field(default_factory=lambda: [-INF, -INF, -INF])
    hi: list[float] = field(default_factory=lambda: [INF, INF, INF])


@dataclass
class Solver:
    """Defaults are the REFERENCE settings: the true Young's modulus and dt = 0.35 Rayleigh
    time (BFA's own timestep is 0.357 of it on the iron-ore project; the importer copies
    BFA's value exactly).  The FAST preset (validated on the corn and iron-ore projects,
    ~3x faster) sets youngs_divisor 10 and neighbor_every 4."""

    dt: float | str = "auto"              # s, or "auto" = rayleigh_fraction * Rayleigh time
    rayleigh_fraction: float = 0.35
    youngs_divisor: float = 1.0           # soften E for a larger stable dt (fast: 10)
    neighbor_every: int = 0               # Verlet list rebuild interval (0 = off; fast: 4)
    skin: float | None = None             # m; default 2 * skin_speed * N * dt
    skin_speed: float = 6.0               # m/s the default skin is sized for; faster grains
                                          # fall back to a direct search (12 m/s measured slower)
    wall_grid_cell: float | None = None   # m; None = the grain radius (>= 6 mm), enlarged
                                          # if needed to keep the grid within its memory budget
    bvh_walls: bool = False               # per-step BVH wall queries (reference path)
    wall_cache: bool = True               # BVH path only
    graph_steps: int = 256
    hash_dims: list[int] | None = None
    max_velocity: float = 30.0            # m/s safety clamp
    expected_holdup_kg: float | None = None  # sizes the particle pool; None = mass rate x
                                          # residence time (dem_run.holdup_estimate)
    pool_factor: float = 2.5
    min_pool: int = 8192
    simplify_mm: float = 0.0
    simplify_tol_mm: float | None = None
    device: str = "cuda:0"
    seed: int = 0


@dataclass
class Output:
    duration: float = 10.0                # s
    fps: float = 15.0
    vtk: bool = True
    checkpoint_at: float | None = None


@dataclass
class Scenario:
    name: str
    material: Material
    parts: list[Part]
    injectors: list[Injector]
    domain_lo: list[float]
    domain_hi: list[float]
    regions: list[Region] = field(default_factory=list)
    flow_planes: list[FlowPlane] = field(default_factory=list)
    solver: Solver = field(default_factory=Solver)
    output: Output = field(default_factory=Output)
    gravity: list[float] = field(default_factory=lambda: [0.0, -9.81, 0.0])
    units: str = "m"
    base_dir: str = "."                   # where relative paths resolve (not serialised)
    notes: dict = field(default_factory=dict)   # free-form provenance (importer remarks)

    # ------------------------------------------------------------------ paths
    def path(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base_dir, p))

    @property
    def unit_scale(self) -> float:
        return UNITS[self.units]

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("base_dir")
        d["domain"] = {"lo": d.pop("domain_lo"), "hi": d.pop("domain_hi")}
        return d

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: dict, base_dir: str = ".", validate: bool = True) -> "Scenario":
        d = dict(d)
        dom = d.pop("domain")
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        sc = cls(
            name=d["name"],
            material=_mk(Material, d.get("material", {})),
            parts=[_mk(Part, p) for p in d["parts"]],
            injectors=[_mk(Injector, i) for i in d.get("injectors", [])],
            domain_lo=list(dom["lo"]), domain_hi=list(dom["hi"]),
            regions=[_mk(Region, r) for r in d.get("regions", [])],
            flow_planes=[_mk(FlowPlane, f) for f in d.get("flow_planes", [])],
            solver=_mk(Solver, d.get("solver", {})),
            output=_mk(Output, d.get("output", {})),
            gravity=list(d.get("gravity", [0.0, -9.81, 0.0])),
            units=d.get("units", "m"),
            base_dir=base_dir,
            notes=d.get("notes", {}),
        )
        if validate:
            sc.validate()
        return sc

    @classmethod
    def load(cls, path: str) -> "Scenario":
        with open(path) as fh:
            return cls.from_dict(json.load(fh), base_dir=os.path.dirname(os.path.abspath(path)))

    # ------------------------------------------------------------------ checks
    def validate(self):
        errs = []
        if self.units not in UNITS:
            errs.append(f"units {self.units!r} not in {sorted(UNITS)}")
        m = self.material
        if m.contact not in ("linear", "hertz"):
            errs.append(f"material.contact {m.contact!r} must be 'linear' or 'hertz'")
        for k in ("radius", "density", "youngs"):
            if getattr(m, k) <= 0:
                errs.append(f"material.{k} must be > 0")
        if not self.parts:
            errs.append("no parts")
        if not self.injectors:
            errs.append("no injector")
        names = [p.name for p in self.parts]
        if len(set(names)) != len(names):
            errs.append("part names must be unique")
        for p in self.parts:
            if not os.path.exists(self.path(p.stl)):
                errs.append(f"part {p.name!r}: STL not found: {self.path(p.stl)}")
            if len(p.active) != 2 or p.active[0] >= p.active[1]:
                errs.append(f"part {p.name!r}: active must be [on, off) with on < off")
        part_files = {os.path.normpath(self.path(p.stl)) for p in self.parts}
        for inj in self.injectors:
            if inj.box:
                bx = inj.box
                for k in ("length", "width", "height"):
                    if float(bx.get(k, 0)) <= 0:
                        errs.append(f"injector {inj.name!r}: box {k} must be > 0")
                if bx.get("reference") == "belt":
                    p = next((p for p in self.parts if p.name == bx.get("part")), None)
                    if p is None or not p.motion or p.motion.get("type") != "belt":
                        errs.append(f"injector {inj.name!r}: box reference {bx.get('part')!r} is "
                                    f"not a conveyor belt part")
                continue
            if not inj.face_stl:
                errs.append(f"injector {inj.name!r}: needs a face STL or a box")
                continue
            if os.path.normpath(self.path(inj.face_stl)) in part_files:
                errs.append(f"injector {inj.name!r}: its face {inj.face_stl} is also a collider part "
                            f"(a wall exactly where grains spawn) -- remove that part")
            if not os.path.exists(self.path(inj.face_stl)):
                errs.append(f"injector {inj.name!r}: face STL not found: {self.path(inj.face_stl)}")
            if inj.mass_rate <= 0:
                errs.append(f"injector {inj.name!r}: mass_rate must be > 0")
        if any(a >= b for a, b in zip(self.domain_lo, self.domain_hi)):
            errs.append("domain lo must be below hi on every axis")
        s = self.solver
        if not (s.dt == "auto" or (isinstance(s.dt, (int, float)) and s.dt > 0)):
            errs.append("solver.dt must be 'auto' or a positive number")
        if errs:
            raise ValueError("invalid scenario:\n  " + "\n  ".join(errs))


def rayleigh_time(radius: float, density: float, youngs: float, poisson: float) -> float:
    """Rayleigh wave time across a grain -- the usual DEM timestep scale."""
    g = youngs / (2.0 * (1.0 + poisson))
    return math.pi * radius * math.sqrt(density / g) / (0.1631 * poisson + 0.8766)


def _mk(cls, d):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def _json_default(o):
    if isinstance(o, float) and math.isinf(o):
        return INF
    raise TypeError(f"not JSON serialisable: {type(o)}")
