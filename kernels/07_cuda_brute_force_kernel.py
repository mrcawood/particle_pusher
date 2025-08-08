"""
This file contains a brute-force O(N^2) implementation for the N-body problem.
It is primarily used for validation and benchmarking against more complex algorithms.
"""
import numpy as np
import cupy as cp
from numba import cuda

@cuda.jit
def _brute_force_kernel(positions, masses, forces, G, epsilon, num_particles, 
                      debug_particle_idx, debug_out):
    """
    Calculates the gravitational force on each particle by iterating through all other particles.
    """
    i = cuda.grid(1)
    if i >= num_particles:
        return

    force_x, force_y, force_z = 0.0, 0.0, 0.0
    px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
    mass_i = masses[i]

    for j in range(num_particles):
        if i == j:
            continue

        if i == debug_particle_idx:
            debug_out[j] = 1
        
        # Vector from particle i to particle j
        rij_x = positions[j, 0] - px
        rij_y = positions[j, 1] - py
        rij_z = positions[j, 2] - pz

        # Squared distance with softening factor
        d_sq = rij_x*rij_x + rij_y*rij_y + rij_z*rij_z + epsilon*epsilon

        # Inverse distance cubed
        inv_d = d_sq**(-0.5) 
        inv_dist_cubed = inv_d * inv_d * inv_d

        # Force magnitude
        force_mag = G * mass_i * masses[j] * inv_dist_cubed

        # Accumulate force components
        force_x += force_mag * rij_x
        force_y += force_mag * rij_y
        force_z += force_mag * rij_z

    forces[i, 0] = force_x
    forces[i, 1] = force_y
    forces[i, 2] = force_z

def get_gflops(num_particles, time_ms):
    """
    Estimates the GFLOPS for the brute-force N-body calculation.
    """
    if time_ms == 0:
        return 0.0
    
    # Estimate of floating point operations per interaction
    flops_per_interaction = 20 
    
    # Total interactions in a brute-force approach
    total_interactions = num_particles * (num_particles - 1)
    
    # Total FLOPs
    total_flops = total_interactions * flops_per_interaction
    
    # GFLOPS
    gflops = total_flops / (time_ms * 1e-3) / 1e9
    return gflops

def calculate_forces(positions, masses, G, epsilon, debug_particle_idx=-1, **kwargs):
    """
    The main interface for the brute-force N-body simulation on the GPU.
    """
    num_particles = positions.shape[0]
    if num_particles <= 1:
        return {"forces": cp.zeros_like(positions), "timings": {}, "metrics": {"gflops": 0}}

    # Ensure data is on the GPU
    d_positions = cp.asarray(positions)
    d_masses = cp.asarray(masses)
    d_forces = cp.zeros_like(d_positions)
    
    # Always pass a device array to the kernel to satisfy the compiler.
    # Use a 1-length dummy buffer when debugging is disabled.
    if debug_particle_idx != -1:
        debug_out = cp.zeros(num_particles, dtype=cp.int32)
    else:
        debug_out = cp.zeros(1, dtype=cp.int32)

    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    _brute_force_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_masses, d_forces, G, epsilon, num_particles,
        debug_particle_idx, debug_out
    )
    end_event.record()
    end_event.synchronize()
    force_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)

    gflops = get_gflops(num_particles, force_time_ms)
    
    timings = {"force_calculation": force_time_ms}
    result = {"forces": d_forces, "timings": timings, "metrics": {"gflops": gflops}}
    
    if debug_particle_idx != -1:
        result["debug_out"] = debug_out

    return result
