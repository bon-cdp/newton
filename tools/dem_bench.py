#!/usr/bin/env python3
"""
Benchmark SolverGranularDEM from a frozen steady-state checkpoint.

    python bfa_dem.py <config> --duration 3 --checkpoint-at 3 --no-vtk --out runs/dem/ckpt
    python tools/dem_bench.py --ckpt runs/dem/ckpt/checkpoint.npz <same config> [--steps 2000]

Every variant starts from the same grains in the same contacts, so wall time per step is
comparable, and the state after --steps can be diffed against a reference (--save-ref /
--ref) to prove an optimisation did not change the physics.

Reports:
  * wall ms/step with host launch overhead included (what a run actually pays)
  * a per-kernel device-time breakdown from Warp's CUDA activity timer
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bfa_dem  # noqa: E402


def load_checkpoint(S, path):
    ck = np.load(path)
    q = ck["q"].copy()
    idle = (ck["flags"] & 1) == 0
    q[idle] = np.array(S.park_lo)          # idle grains carry no physics; use this build's park
    S.s0.particle_q.assign(q)
    S.s0.particle_qd.assign(ck["qd"])
    S.model.particle_flags.assign(ck["flags"])
    sv = S.solver
    sv.particle_w.assign(ck["w"])
    for name in ("tang_partner", "tang_stamp", "tang_xi"):
        a = ck[name]
        dst = getattr(sv, name)
        if a.shape[:2] != tuple(dst.shape):     # checkpoints from the old [grain, slot] layout
            a = np.ascontiguousarray(np.swapaxes(a, 0, 1))
        dst.assign(a)
    sv.wall_slack.assign(ck["wall_slack"])
    sv._step = int(ck["solver_step"])
    if hasattr(sv, "set_step"):
        sv.set_step(int(ck["solver_step"]))
    if hasattr(sv, "request_rebuild"):
        sv.request_rebuild()
    return ck


def main():
    argv = sys.argv[1:]
    own = {}
    for flag, conv, default in (("--ckpt", str, None), ("--steps", int, 2000),
                                ("--save-ref", str, None), ("--ref", str, None),
                                ("--breakdown", int, 1), ("--pingpong", int, 0),
                                ("--graph", int, 0)):
        own[flag] = default
        if flag in argv:
            k = argv.index(flag)
            own[flag] = conv(argv[k + 1])
            del argv[k:k + 2]
    scratch = os.environ.get("BENCH_OUT", "/tmp/dem_bench_out")
    args = bfa_dem.parse_args(argv + ["--no-vtk", "--out", scratch])
    wp.init()
    S = bfa_dem.build(args, quiet=True)
    load_checkpoint(S, own["--ckpt"])
    act = int((S.model.particle_flags.numpy() & 1).sum())
    solver, s0, s1, dt = S.solver, S.s0, S.s1, S.dt
    n = own["--steps"]
    if not own["--pingpong"]:
        s1 = s0                 # in place: the solver's step() supports state_out is state_in

    # warm-up: JIT, module load, allocator
    for _ in range(3):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    load_checkpoint(S, own["--ckpt"])
    s0, s1 = S.s0, (S.s1 if own["--pingpong"] else S.s0)
    wp.synchronize()

    K = own["--graph"]
    if K:
        S.model.particle_grid.reserve(S.n_pool)
        solver.request_rebuild()
        with wp.ScopedCapture(force_module_load=False) as cap:
            for _ in range(K):
                solver.step(s0, s0, None, None, dt)
        load_checkpoint(S, own["--ckpt"])          # capture recorded, but ran nothing
        wp.synchronize()
        t = time.perf_counter()
        for _ in range(n // K):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        wall = (time.perf_counter() - t) / (n // K * K)
    else:
        t = time.perf_counter()
        for _ in range(n):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        wp.synchronize()
        wall = (time.perf_counter() - t) / n

    q = s0.particle_q.numpy()
    flags = S.model.particle_flags.numpy() & 1
    if own["--save-ref"]:
        np.savez(own["--save-ref"], q=q, qd=s0.particle_qd.numpy(), w=solver.particle_w.numpy(),
                 flags=flags)
    drift = ""
    if own["--ref"]:
        r = np.load(own["--ref"])
        m = (flags == 1) & (r["flags"] == 1)
        dq = np.linalg.norm(q[m] - r["q"][m], axis=1)
        drift = (f"   vs ref: |dq| median {np.median(dq)*1e3:.4f} mm  p99 {np.percentile(dq, 99)*1e3:.3f}"
                 f" mm  max {dq.max()*1e3:.2f} mm  (grain d = 12 mm)")

    if getattr(solver, "neighbor_every", 0):
        drift += (f"   [nbr fallbacks {int(solver.nbr_fallbacks.numpy()[0])}, "
                  f"overflow {int(solver.nbr_overflow.numpy()[0])}]")
    print(f"active {act:,} / pool {S.n_pool:,}   {n} steps   wall {wall*1e3:.3f} ms/step   "
          f"-> {wall / dt:.1f} s per simulated second{drift}")

    if own["--breakdown"]:
        m = 200
        wp.timing_begin(cuda_filter=wp.TIMING_ALL)
        for _ in range(m):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        wp.synchronize()
        res = wp.timing_end()
        agg = {}
        for r in res:
            key = r.name.split("(")[0][:60]
            a = agg.setdefault(key, [0.0, 0])
            a[0] += r.elapsed
            a[1] += 1
        tot = sum(v[0] for v in agg.values())
        print(f"  device time {tot / m * 1e3:.1f} us/step  (the rest of wall time is host launch overhead)")
        for k, (el, c) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"    {el / m * 1e3:8.1f} us  {c / m:5.1f}x  {k}")


if __name__ == "__main__":
    main()
