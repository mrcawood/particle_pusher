"""
This file contains CUDA-accelerated functions for particle initialization.
"""
import cupy as cp
from numba import cuda
import math

@cuda.jit
def _plummer_kernel(positions, velocities, masses, scale_radius, G, rotation_factor):
    """
    CUDA kernel to generate Plummer model particle properties in parallel.
    This version is instrumented to debug a Numba typing error.
    """
    i = cuda.grid(1)
    if i >= masses.shape[0]:
        return

    # --- Isolate the failing operation ---
    # Step 1: Power operation
    mass_val = masses[i]
    power_val = mass_val**(-2.0/3.0)
    
    # Step 2: Subtraction
    sub_val = power_val - 1.0
    
    # Step 3: Square root
    sqrt_val = math.sqrt(sub_val)

    # Step 4: Division
    radius = scale_radius / sqrt_val
    # --- End isolation block ---

    # Spherical coordinates from more random numbers
    theta = positions[i, 0] # Re-using array for random numbers
    phi = positions[i, 1]
    
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    positions[i, 0] = x
    positions[i, 1] = y
    positions[i, 2] = z

    # Velocities
    X = velocities[i, 0]
    q = math.sqrt(x*x + y*y + z*z)
    if q == 0:
        esc_vel_sq = 0.0
    else:
        esc_vel_sq = 2.0 * G * masses.shape[0] / q
    
    v_mag = X * math.sqrt(esc_vel_sq)
    
    v_theta = velocities[i, 2] * 2.0 * math.pi
    v_phi = math.acos(2.0 * X - 1.0)
    
    vx = v_mag * math.sin(v_theta) * math.cos(v_phi)
    vy = v_mag * math.sin(v_theta) * math.sin(v_phi)
    vz = v_mag * math.cos(v_theta)

    # Add a rigid body rotation
    if rotation_factor > 0:
        velocities[i, 0] = vx - rotation_factor * y
        velocities[i, 1] = vy + rotation_factor * x
        velocities[i, 2] = vz
    else:
        velocities[i, 0] = vx
        velocities[i, 1] = vy
        velocities[i, 2] = vz


def initialize_plummer_gpu(num_particles, scale_radius, G, rotation_factor):
    """
    Generates particles on the GPU according to a Plummer model.
    """
    d_masses_rand = cp.random.rand(num_particles, dtype=cp.float64)
    d_pos_rand = cp.random.rand(num_particles, 2, dtype=cp.float64) * cp.array([2.0 * cp.pi, cp.pi], dtype=cp.float64)
    d_vel_rand = cp.random.rand(num_particles, 3, dtype=cp.float64)

    d_positions = cp.empty((num_particles, 3), dtype=cp.float64)
    d_velocities = cp.empty((num_particles, 3), dtype=cp.float64)
    d_masses = cp.full(num_particles, 1.0 / num_particles, dtype=cp.float64)
    
    d_positions[:, 0] = d_pos_rand[:, 0]
    d_positions[:, 1] = d_pos_rand[:, 1]
    d_velocities[:, :] = d_vel_rand[:, :]
    d_masses[:] = d_masses_rand[:]

    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block
    
    _plummer_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d_masses, scale_radius, G, rotation_factor
    )
    
    d_masses.fill(1.0 / num_particles)

    return d_positions, d_velocities, d_masses
