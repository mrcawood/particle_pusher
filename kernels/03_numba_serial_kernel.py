import numpy as np
from numba import njit
import time

def get_gflops(num_particles, time_ms):
    """
    Estimates the GFLOPS for the brute-force N-body calculation.
    """
    if time_ms == 0:
        return 0.0
    
    # Estimate of floating point operations per interaction
    flops_per_interaction = 20 
    
    # Total interactions in a brute-force approach with symmetric calculations
    total_interactions = num_particles * (num_particles - 1) / 2
    
    # Total FLOPs
    total_flops = total_interactions * flops_per_interaction
    
    # GFLOPS
    gflops = total_flops / (time_ms * 1e-3) / 1e9
    return gflops

@njit(fastmath=True)
def _calculate_forces_numba(positions, masses, G, epsilon, forces):
    """
    Numba-jitted function for serial force calculation.
    """
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz + epsilon**2
            
            inv_dist = 1.0 / np.sqrt(dist_sq)
            inv_dist_cubed = inv_dist**3
            
            force_magnitude = G * masses[i] * masses[j] * inv_dist_cubed
            
            fx = force_magnitude * dx
            fy = force_magnitude * dy
            fz = force_magnitude * dz

            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz

            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz

def calculate_forces(positions, masses, G, epsilon, **kwargs):
    """
    Wrapper for the Numba serial kernel to include timing and metrics.
    """
    num_particles = positions.shape[0]
    forces = np.zeros_like(positions)

    start_time = time.time()
    _calculate_forces_numba(positions, masses, G, epsilon, forces)
    force_calc_time = (time.time() - start_time) * 1000  # in ms

    gflops = get_gflops(num_particles, force_calc_time)

    return {
        "forces": forces,
        "timings": {"force_calculation": force_calc_time},
        "metrics": {"gflops": gflops}
    }
