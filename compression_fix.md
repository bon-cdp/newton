 Plan: Fix MPM High-Speed Granular Flow - Deep Technical Analysis

 Executive Summary

 After deep exploration of Newton's MPM source code, we've identified the real root cause: Newton's implicit MPM solver was not designed for high-speed confined flows (20+ m/s through spouts). The issue is a combination of:

 1. Unconstrained P2G velocity transfer - Grid velocities are weighted averages of particle velocities with NO clamping
 2. APIC velocity gradient amplification - The affine term can overshoot with large C matrices
 3. Over-conservative collision projection - 65mm "stay-away" zone creates artificial dams
 4. critical_fraction with inverted mesh - May be miscounting interior volume

 ---
 Test Results History

 | Test                     | Parameters             | Frame 50 Still % | Result             |
 |--------------------------|------------------------|------------------|--------------------|
 | Baseline                 | cf=0.0                 | 37.6%            | -                  |
 | cf=0.55                  | critical_fraction only | 43.6%            | WORSE              |
 | cf=0.35, 100mm           | larger particles       | 60.1%            | WORSE              |
 | cf=0.05, 50mm            | stricter               | 33.3%            | Slight improvement |
 | cf=0.01, h=50            | extreme                | UNSTABLE         | Crashed            |
 | cf=0.35, h=5, voxel=0.05 | "correct" ratio        | 35.8% at F40     | Still accumulating |

 ---
 ROOT CAUSE ANALYSIS

 1. The P2G Transfer Problem (PRIMARY)

 Location: solver_implicit_mpm.py lines 2059-2115

 # Step 1: Accumulate weighted particle velocities
 velocity_int = fem.integrate(integrate_velocity, ...)  # SUM(phi * rho * v_p)

 # Step 2: Add APIC correction (ADDITIVE!)
 velocity_int += fem.integrate(integrate_velocity_apic, ...)  # SUM(phi * C * offset)

 # Step 3: Divide by mass → UNCONSTRAINED VELOCITY
 velocity_avg[i] = velocity_int[i] / (pmass + drag)  # NO CLAMPING!

 The Problem: Particles at 20 m/s create grid velocities of 20+ m/s. With voxel_size=0.05m and dt=0.0167s:
 - Distance per step: 20 * 0.0167 = 0.33m = 6.6 voxels!
 - CFL condition requires: velocity * dt < voxel_size → max velocity = 3 m/s

 The implicit solver should handle this, but the initial unconstrained velocity seeds the solve with extreme values that the iterative solver struggles to correct.

 2. The APIC Amplification Problem

 Location: solver_implicit_mpm.py lines 142-159

 vel_apic = velocity_gradients[particle] * node_offset  # C * (x_node - x_particle)

 The velocity gradient C is extracted from the previous step's grid solve (line 321):
 p_vel_grad = fem.grad(grid_vel, s)  # NOT BOUNDED

 For high-speed impacts:
 - C can have large eigenvalues
 - node_offset is up to voxel_size/2 = 0.025m
 - vel_apic can exceed the actual particle velocity

 3. The Collision Projection Problem

 Location: rasterized_collisions.py lines 254-264

 sdf_end = sdf - dot(sdf_vel, gradient) * dt + projection_threshold

 Your setup:
 - projection_threshold default = 0.5 * voxel_size = 0.025m
 - collider_thicknesses=[0.04] = additional 40mm
 - Total exclusion zone: 65mm

 At shelf corners, this creates a 65mm "no-go zone" where particles cannot settle, creating artificial dams.

 4. The critical_fraction with Inverted Mesh Problem

 Location: solver_implicit_mpm.py lines 435-452

 spherical_part = max_fraction * (node_volume - collider_volume) - particle_volume

 With inverted mesh:
 - Interior of spout may be counted as "collider_volume"
 - Available space = node_volume - collider_volume ≈ 0
 - ANY particle triggers strain offset → particles get pushed out
 - Creates "bouncing" at shelf entrance

 ---
 NUMERICAL ANALYSIS

 CFL Condition for Your Scenario

 Spout drop height: ~20m
 Terminal velocity: sqrt(2 * 10 * 20) ≈ 20 m/s
 Current voxel_size: 0.05m
 Current dt: 1/60 = 0.0167s

 Particles move: 20 * 0.0167 = 0.33m per step
 Voxels crossed: 0.33 / 0.05 = 6.6 voxels per step

 For CFL ≤ 1: dt_max = voxel_size / velocity = 0.05 / 20 = 0.0025s
 Required SUBSTEPS: 0.0167 / 0.0025 = 6.7 → 8 substeps minimum

 Why Implicit Solver Doesn't Save Us

 The implicit solver ensures stability (won't explode) but not accuracy:
 - Initial grid velocity is unconstrained (seeded with 20+ m/s)
 - Iterative solve tries to project onto yield surface
 - With only 500 iterations and tolerance 1e-7, may not fully converge
 - Particles advect with partially-corrected velocities
 - Compression still occurs because constraints aren't fully satisfied

 ---
 PROPOSED FIXES (Ordered by Impact)

 FIX 1: Add Grid Velocity Clamping After P2G (HIGHEST IMPACT)

 File: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
 Location: After line 2115 (after free_velocity kernel)

 @wp.kernel
 def clamp_grid_velocities(
     velocity: wp.array(dtype=wp.vec3),
     max_velocity: float,
 ):
     i = wp.tid()
     v_norm_sq = wp.length_sq(velocity[i])
     if v_norm_sq > max_velocity * max_velocity:
         velocity[i] = velocity[i] * max_velocity / wp.sqrt(v_norm_sq)

 # Add after free_velocity launch:
 wp.launch(
     clamp_grid_velocities,
     dim=vel_node_count,
     inputs=[state_out.velocity_field.dof_values, max_grid_velocity],
 )

 Set: max_grid_velocity = 2.0 * voxel_size / dt (CFL = 2)

 FIX 2 (SKIPPED): Increase SUBSTEPS

 User has already tried this - does not want to pursue

 FIX 3: Remove Over-Conservative Collision Threshold

 File: shiploader_mpm_test.py

 mpm_model.setup_collider(
     body_mass=wp.zeros_like(model.body_mass),
     collider_thicknesses=[0.0],  # REMOVE the 40mm margin
     ground_height=-1000.0
 )

 FIX 4: Disable critical_fraction (For Now)

 CRITICAL_FRACTION = 0.0  # Disable until mesh volume issue is resolved

 The inverted mesh may be causing incorrect volume calculations that trigger artificial repulsion.

 FIX 5: Regularize APIC Velocity Gradient

 File: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
 Location: Before APIC transfer (line 2074)

 @wp.kernel
 def clamp_velocity_gradients(
     vel_grad: wp.array(dtype=wp.mat33),
     max_grad: float,  # e.g., 100 s^-1
 ):
     i = wp.tid()
     # Clamp each component
     for row in range(3):
         for col in range(3):
             vel_grad[i][row, col] = wp.clamp(vel_grad[i][row, col], -max_grad, max_grad)

 FIX 6: Increase Air Drag for High-Speed Damping

 mpm_options.air_drag = 10.0  # Instead of 1.0

 This adds damping: inv_mass = 1 / (mass + drag * dt)

 ---
 IMPLEMENTATION PLAN

 Phase 1: Newton Code Modifications (PRIMARY)

 User wants to modify Newton directly rather than use substeps.

 1. Add grid velocity clamping kernel after P2G transfer
   - Location: solver_implicit_mpm.py after line 2115
   - Clamp to CFL-safe velocity: max_vel = 2 * voxel_size / dt
 2. Add velocity gradient clamping kernel before APIC
   - Location: solver_implicit_mpm.py before line 2074
   - Clamp C matrix eigenvalues to prevent amplification
 3. Add CFL monitoring (warn if violated)
   - Print max velocity each frame for debugging

 Phase 2: Parameter Adjustments

 1. collider_thicknesses = [0.0] - Remove artificial exclusion zone
 2. CRITICAL_FRACTION = 0.0 - Disable until mesh volume issue resolved
 3. air_drag = 5.0 - Add some damping
 4. PARTICLE_RADIUS = 0.030 - 60mm particles (faster testing)

 Phase 3: Advanced (If Needed)

 1. Velocity-dependent yield surface (reduce yield_pressure for fast particles)
 2. Impact detection and special handling for high-speed collisions

 ---
 FILES TO MODIFY

 Immediate (shiploader_mpm_test.py)

 - Line 29: SUBSTEPS = 8
 - Line 24: PARTICLE_RADIUS = 0.030
 - Line 36: CRITICAL_FRACTION = 0.0
 - Line 332: collider_thicknesses=[0.0]
 - mpm_options: air_drag = 5.0

 Newton Code (if Phase 2)

 - newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
   - Add clamp_grid_velocities kernel (~line 180)
   - Add clamp_velocity_gradients kernel (~line 180)
   - Launch after P2G transfer (~line 2115)
   - Launch before APIC transfer (~line 2074)

 ---
 VERIFICATION

 After Phase 1 changes, run and check:
 python analyze_vtk.py

 Target metrics:
 - Frame 50 Still % < 15%
 - Stable particle count (no crashes)
 - Smooth flow through shelf region

 ---
 KEY INSIGHT

 Newton's examples use small drops (1-4m) with velocities under 10 m/s.

 Your 20m spout with 20+ m/s velocities is outside Newton's validated regime.

 The implicit solver provides stability (no explosion) but not accuracy at these speeds. The fix is either:
 1. Use many substeps to satisfy CFL (expensive but easy)
 2. Add explicit velocity clamping to prevent extreme grid states (requires code mod)
 3. Both (recommended)

 ---
 SUMMARY OF RECOMMENDED CHANGES

 Newton Code Changes (solver_implicit_mpm.py)

 # Add after line ~180 (with other kernels):

 @wp.kernel
 def clamp_grid_velocities(
     velocity: wp.array(dtype=wp.vec3),
     max_velocity: float,
 ):
     """Clamp grid velocities to CFL-safe values after P2G transfer."""
     i = wp.tid()
     v_norm_sq = wp.length_sq(velocity[i])
     if v_norm_sq > max_velocity * max_velocity:
         velocity[i] = velocity[i] * max_velocity / wp.sqrt(v_norm_sq)

 @wp.kernel
 def clamp_velocity_gradients(
     vel_grad: wp.array(dtype=wp.mat33),
     max_grad: float,
 ):
     """Clamp velocity gradient (C matrix) to prevent APIC amplification."""
     i = wp.tid()
     for row in range(3):
         for col in range(3):
             vel_grad[i][row, col] = wp.clamp(vel_grad[i][row, col], -max_grad, max_grad)

 # Add after line 2115 (after free_velocity launch):

 # CFL-based velocity clamping
 max_grid_velocity = 2.0 * self.mpm_model.options.voxel_size / dt
 wp.launch(
     clamp_grid_velocities,
     dim=vel_node_count,
     inputs=[state_out.velocity_field.dof_values, max_grid_velocity],
 )

 # Add before line 2074 (before APIC transfer):

 # Clamp velocity gradients to prevent APIC amplification
 max_vel_grad = 100.0  # 100 s^-1 max gradient
 wp.launch(
     clamp_velocity_gradients,
     dim=model.particle_count,
     inputs=[state_in.particle_qd_grad, max_vel_grad],
 )

 Parameter Changes (shiploader_mpm_test.py)

 # Particle sizing - faster testing
 PARTICLE_RADIUS = 0.030  # 60mm (was 25mm)
 VOXEL_SIZE = 0.06  # 60mm for ratio=2

 # Plasticity - disable problematic features
 CRITICAL_FRACTION = 0.0  # Disable (was 0.35) - mesh volume issues

 # Damping - add stability
 mpm_options.air_drag = 5.0  # Was 1.0

 # Collision - remove artificial margins
 mpm_model.setup_collider(
     body_mass=wp.zeros_like(model.body_mass),
     collider_thicknesses=[0.0],  # Was 0.04
     ground_height=-1000.0
 )

 This combination clamps high-speed grid velocities at the source (P2G transfer), prevents APIC amplification, and removes artificial dams.
