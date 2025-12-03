#!/usr/bin/env python3
"""
Shiploader MPM Test - Minimal particle injection to test STL interaction
Based on newton/examples/mpm/example_mpm_granular.py
Tests if MPM solver can handle particle-mesh interaction with Assemr2.STL spout
"""

import numpy as np
import warp as wp
import trimesh
import newton
from newton.solvers import SolverImplicitMPM
import os
import datetime

# MPM Parameters (from working granular example)
DENSITY = 800.0  # kg/m³ (wheat bulk density)
YOUNG_MODULUS = 1.0e15  # Pa (MPM default for granular)
POISSON_RATIO = 0.3
FRICTION_COEFF = 0.68  # From MPM example
DAMPING = 0.0
VOXEL_SIZE = 0.06  # 60mm grid cells (controls particle size in MPM)
PARTICLE_RADIUS = 0.020  # Target: 40mm diameter particles

# Simulation parameters
DURATION = 10.0  # seconds - short test
FPS = 60.0
SUBSTEPS = 1

# Particle streaming parameters
FEED_RATE_MTPH = 600.0  # Metric tons per hour
INJECTION_CENTER = wp.vec3(18.75, 25.8, 5.05)  # Above spout entrance
INJECTION_RADIUS = 0.7  # 0.7m radius circular injection
PARTICLE_POOL_MULTIPLIER = 3.0  # Pre-allocate 3x particles for streaming

# VTK Output
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VTK_OUTPUT_DIR = f"/home/shakil/newton/vtk_output_mpm_test_{timestamp}"

# Calculate derived parameters
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SUBSTEPS

def load_shiploader_mesh():
    """Load and process the shiploader STL file."""
    mesh = trimesh.load('/home/shakil/newton/Assemr2.STL', force='mesh')

    # CRITICAL FIX: Invert normals so interior channel is treated as exterior
    # This prevents particles from disappearing when entering the spout interior
    print(f"Mesh watertight before inversion: {mesh.is_watertight}")
    mesh.invert()
    print(f"Mesh inverted - interior surfaces now treated as collision exteriors")

    vertices = mesh.vertices
    indices = mesh.faces.flatten()
    newton_mesh = newton.Mesh(vertices, indices)
    return newton_mesh, mesh.bounds, mesh

@wp.kernel
def activate_particle_stream_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    spawn_center: wp.vec3,
    spawn_radius: float,
    spawn_velocity: wp.vec3,
    rand_floats: wp.array(dtype=float)
):
    """Activate particles in circular streaming pattern."""
    tid = wp.tid()
    idx = indices[tid]

    # Circular spawn pattern (polar coordinates)
    r = wp.sqrt(rand_floats[tid * 3 + 0]) * spawn_radius
    theta = rand_floats[tid * 3 + 1] * 2.0 * wp.pi

    x_offset = r * wp.cos(theta)
    z_offset = r * wp.sin(theta)
    y_jitter = (rand_floats[tid * 3 + 2] - 0.5) * 0.04  # ±2cm vertical jitter

    spawn_pos = spawn_center + wp.vec3(x_offset, y_jitter, z_offset)
    particle_q[idx] = spawn_pos
    particle_qd[idx] = spawn_velocity

def export_frame_vtk(frame, positions, radii, mesh_vertices, mesh_faces, filename):
    """Export particles AND mesh geometry to VTK format (ParaView compatible)."""
    num_particles = len(positions)
    num_mesh_verts = len(mesh_vertices)
    num_mesh_faces = len(mesh_faces)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"MPM Test frame {frame}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
        f.write(f"POINTS {num_particles + num_mesh_verts} float\n")
        for pos in positions:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")
        for v in mesh_vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        total_cells = num_particles + num_mesh_faces
        cell_size = num_particles * 2 + num_mesh_faces * 4
        f.write(f"\nCELLS {total_cells} {cell_size}\n")
        for i in range(num_particles):
            f.write(f"1 {i}\n")
        for face in mesh_faces:
            v0, v1, v2 = face
            f.write(f"3 {v0 + num_particles} {v1 + num_particles} {v2 + num_particles}\n")

        f.write(f"\nCELL_TYPES {total_cells}\n")
        for i in range(num_particles):
            f.write("1\n")
        for i in range(num_mesh_faces):
            f.write("5\n")

        f.write(f"\nPOINT_DATA {num_particles + num_mesh_verts}\n")
        f.write("SCALARS radius float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for r in radii:
            f.write(f"{r}\n")
        for i in range(num_mesh_verts):
            f.write("0.0\n")

def calculate_streaming_params():
    """Calculate particle streaming parameters from feed rate."""
    feed_rate_kgs = (FEED_RATE_MTPH * 1000.0) / 3600.0  # kg/s
    total_mass_kg = feed_rate_kgs * DURATION

    # Particle properties
    volume_per_particle = (4.0/3.0) * np.pi * (PARTICLE_RADIUS ** 3)
    mass_per_particle = volume_per_particle * DENSITY

    # Total particles needed
    target_particles = int(total_mass_kg / mass_per_particle)
    max_particles = int(target_particles * PARTICLE_POOL_MULTIPLIER)

    particles_per_second = feed_rate_kgs / mass_per_particle
    particles_per_step = particles_per_second * SIM_DT

    print(f"Streaming parameters:")
    print(f"  Feed rate: {FEED_RATE_MTPH} MTPH ({feed_rate_kgs:.1f} kg/s)")
    print(f"  Particle mass: {mass_per_particle*1000:.2f} g")
    print(f"  Target particles: {target_particles:,}")
    print(f"  Pool size (pre-allocated): {max_particles:,}")
    print(f"  Injection rate: {particles_per_second:.1f} particles/s")
    print(f"  Particles per step: {particles_per_step:.2f}")

    return max_particles, mass_per_particle, particles_per_step

def emit_particle_pool(builder: newton.ModelBuilder, max_particles: int, mass_per_particle: float):
    """
    Pre-allocate particle pool for streaming.
    Particles start far away (Y=1000) and are moved to injection point when activated.
    """
    # Calculate grid dimensions for particle pool
    dim = int(np.ceil(max_particles ** (1.0/3.0)))
    spacing = VOXEL_SIZE

    # Place inactive particles far away
    inactive_pos = wp.vec3(0.0, 1000.0, 0.0)

    print(f"Particle pool:")
    print(f"  Grid: {dim} x {dim} x {dim}")
    print(f"  Spacing: {spacing*1000:.0f} mm")
    print(f"  Initial position: Y=1000m (inactive)")

    builder.add_particle_grid(
        pos=inactive_pos,
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass_per_particle,
        jitter=spacing * 0.1,
        radius_mean=PARTICLE_RADIUS,
    )

def main():
    print("=" * 60)
    print("SHIPLOADER MPM TEST - PARTICLE STREAMING")
    print("=" * 60)
    print(f"Testing MPM solver with Assemr2.STL spout geometry")
    print(f"Duration: {DURATION}s at {FPS} FPS")
    print(f"Voxel size: {VOXEL_SIZE*1000:.0f} mm")
    print()

    # Calculate streaming parameters
    max_particles, mass_per_particle, particles_per_step = calculate_streaming_params()
    print()

    # Initialize Warp
    wp.init()
    device = "cuda:0"

    # Build model
    builder = newton.ModelBuilder()

    # Load shiploader STL as collision mesh
    print("Loading Assemr2.STL...")
    shiploader_mesh, bounds, trimesh_obj = load_shiploader_mesh()
    print(f"Mesh bounds:")
    print(f"  X: {bounds[0][0]:.2f} to {bounds[1][0]:.2f} m")
    print(f"  Y: {bounds[0][1]:.2f} to {bounds[1][1]:.2f} m (vertical)")
    print(f"  Z: {bounds[0][2]:.2f} to {bounds[1][2]:.2f} m")
    print()

    builder.add_shape_mesh(
        body=-1,
        mesh=shiploader_mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            mu=FRICTION_COEFF,
            ke=YOUNG_MODULUS
        )
    )

    # Create particle pool for streaming
    emit_particle_pool(builder, max_particles, mass_per_particle)

    # Finalize model
    model = builder.finalize(device=device)
    model.particle_mu = FRICTION_COEFF
    model.set_gravity((0.0, -9.81, 0.0))  # Y-up gravity

    # Disable model's particle material parameters for MPM
    model.particle_ke = None
    model.particle_kd = None
    model.particle_cohesion = None
    model.particle_adhesion = None

    # Create MPM options
    mpm_options = SolverImplicitMPM.Options()
    mpm_options.density = DENSITY
    mpm_options.young_modulus = YOUNG_MODULUS
    mpm_options.poisson_ratio = POISSON_RATIO
    mpm_options.friction_coeff = FRICTION_COEFF
    mpm_options.damping = DAMPING
    mpm_options.voxel_size = VOXEL_SIZE
    mpm_options.grid_type = "sparse"
    mpm_options.max_iterations = 250
    mpm_options.tolerance = 1.0e-6

    # Create MPM model and solver
    mpm_model = SolverImplicitMPM.Model(model, mpm_options)

    # CRITICAL FIX: Setup collider for mesh collision detection
    print("Setting up collider for mesh collision...")
    # Enable collision on both inside and outside of mesh surfaces
    mpm_model.setup_collider(
        body_mass=wp.zeros_like(model.body_mass),  # Static/kinematic meshes
        collider_thicknesses=[0.15],  # 15cm collision margin for bidirectional contact
        ground_height=-1000.0  # Disable ground plane (using mesh instead)
    )

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverImplicitMPM(mpm_model, mpm_options)
    solver.enrich_state(state_0)
    solver.enrich_state(state_1)

    print(f"\nModel initialized:")
    print(f"  Particles: {model.particle_count:,}")
    print(f"  Shapes: {model.shape_count}")
    print()

    # Create VTK output directory
    os.makedirs(VTK_OUTPUT_DIR, exist_ok=True)
    print(f"VTK output: {VTK_OUTPUT_DIR}\n")

    # Get mesh data for VTK export
    mesh_vertices = trimesh_obj.vertices
    mesh_faces = trimesh_obj.faces

    # Simulation loop with streaming
    print("Running simulation with particle streaming...")
    print(f"  Press Ctrl+C to stop early")
    print()

    num_steps = int(DURATION / SIM_DT)
    sim_time = 0.0
    frame_count = 0
    VTK_EXPORT_INTERVAL = max(1, num_steps // 50)  # ~50 frames

    # Streaming state
    last_particle_idx = 0
    particle_accumulator = 0.0
    spawn_velocity = wp.vec3(0.0, -0.5, 0.0)  # Small downward velocity

    import time
    start_time = time.time()

    try:
        for step in range(num_steps):
            # Activate new particles for streaming
            particle_accumulator += particles_per_step
            num_to_spawn = int(particle_accumulator)

            if num_to_spawn > 0 and last_particle_idx + num_to_spawn < model.particle_count:
                particle_accumulator -= num_to_spawn

                # Generate indices for new particles
                indices = np.arange(last_particle_idx, last_particle_idx + num_to_spawn, dtype=np.int32)
                indices_wp = wp.array(indices, dtype=int, device=device)

                # Generate random values for circular spawn pattern
                rand_floats = np.random.rand(num_to_spawn * 3).astype(np.float32)
                rand_wp = wp.array(rand_floats, dtype=float, device=device)

                # Activate particles in circular pattern
                wp.launch(
                    kernel=activate_particle_stream_kernel,
                    dim=num_to_spawn,
                    inputs=[
                        state_0.particle_q,
                        state_0.particle_qd,
                        indices_wp,
                        INJECTION_CENTER,
                        INJECTION_RADIUS,
                        spawn_velocity,
                        rand_wp
                    ],
                    device=device
                )

                last_particle_idx += num_to_spawn

            # MPM simulation step with substeps
            for _ in range(SUBSTEPS):
                solver.step(state_0, state_1, None, None, SIM_DT)
                solver.project_outside(state_1, state_1, SIM_DT)
                state_0, state_1 = state_1, state_0

            sim_time += FRAME_DT

            # Export VTK frame
            if step % VTK_EXPORT_INTERVAL == 0 or step == num_steps - 1:
                # Get active particle positions (exclude those above Y=90)
                positions = state_0.particle_q.numpy()
                particle_y = positions[:, 1]
                active_indices = np.where((particle_y < 90.0) & (particle_y > -20.0))[0]
                active_positions = positions[active_indices]
                radii = np.full(len(active_positions), PARTICLE_RADIUS)

                vtk_filename = os.path.join(VTK_OUTPUT_DIR, f"frame_{frame_count:04d}.vtk")
                if len(active_positions) > 0:
                    export_frame_vtk(frame_count, active_positions, radii, mesh_vertices, mesh_faces, vtk_filename)

                frame_count += 1

            # Progress update
            if step % (FPS * 1) == 0 or step == 0:  # Every 1 second
                elapsed = time.time() - start_time
                progress = (step + 1) / num_steps * 100
                print(f"  Step {step+1:5d}/{num_steps} ({progress:5.1f}%) | "
                      f"Sim time: {sim_time:5.2f}s | "
                      f"Real time: {elapsed:5.1f}s | "
                      f"Frames: {frame_count}")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

    # Final analysis
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")

    total_time = time.time() - start_time
    print(f"Real time: {total_time:.1f}s")
    print(f"Simulated time: {sim_time:.2f}s")
    print()

    # Analyze final particle positions
    final_positions = state_0.particle_q.numpy()
    print("Final particle analysis:")
    print(f"  Min Y: {np.min(final_positions[:, 1]):.2f} m")
    print(f"  Max Y: {np.max(final_positions[:, 1]):.2f} m")
    print(f"  Y range: {np.max(final_positions[:, 1]) - np.min(final_positions[:, 1]):.2f} m")

    # Check if particles fell (Y decreased significantly)
    initial_y_avg = (EMIT_LO[1] + EMIT_HI[1]) / 2
    final_y_avg = np.mean(final_positions[:, 1])
    y_drop = initial_y_avg - final_y_avg

    print(f"\nParticle movement:")
    print(f"  Initial Y (avg): {initial_y_avg:.2f} m")
    print(f"  Final Y (avg): {final_y_avg:.2f} m")
    print(f"  Y drop: {y_drop:.2f} m")

    if y_drop > 1.0:
        print("\n✓ SUCCESS: Particles fell significantly!")
        print("  MPM solver is working with STL mesh")
    elif y_drop > 0.1:
        print("\n⚠ PARTIAL: Particles moved but not much")
        print("  May need longer simulation or check collision")
    else:
        print("\n✗ ISSUE: Particles didn't fall")
        print("  Check gravity, collision, or initial conditions")

    print()

if __name__ == "__main__":
    main()
