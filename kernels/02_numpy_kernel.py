import numpy as np
import time

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

def calculate_forces(positions, masses, G, epsilon, **kwargs):
    """
    Calculates the gravitational forces between all particles using NumPy.

    Args:
        positions (np.array): (N, 3) array of particle positions.
        masses (np.array): (N,) array of particle masses.
        G (float): Gravitational constant.
        epsilon (float): Softening length to avoid singularities.

    Returns:
        dict: A dictionary containing the forces array, timings, and GFLOPS.
    """
    start_time = time.time()
    
    num_particles = positions.shape[0]

    rij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    
    dist_sq = np.sum(rij**2, axis=2)
    dist_sq += epsilon**2
    
    inv_dist = 1.0 / np.sqrt(dist_sq)
    inv_dist_cubed = inv_dist**3
    
    np.fill_diagonal(inv_dist_cubed, 0.)
    
    mass_product = masses[:, np.newaxis] * masses[np.newaxis, :]
    
    force_magnitudes = G * mass_product * inv_dist_cubed
    
    forces = np.sum(force_magnitudes[:, :, np.newaxis] * rij, axis=1)
    
    force_calc_time = (time.time() - start_time) * 1000  # in ms

    gflops = get_gflops(num_particles, force_calc_time)

    return {
        "forces": forces,
        "timings": {"force_calculation": force_calc_time},
        "metrics": {"gflops": gflops}
    }
