import numpy as np
from numba import njit, prange

@njit(parallel=True)
def calculate_forces(positions, masses, G, epsilon):
    """
    Calculates the gravitational forces between all particles.
    Uses Numba's @njit(parallel=True) decorator to compile and parallelize the function.

    Args:
        positions (np.array): (N, 3) array of particle positions.
        masses (np.array): (N,) array of particle masses.
        G (float): Gravitational constant.
        epsilon (float): Softening length to avoid singularities.

    Returns:
        np.array: (N, 3) array of forces on each particle.
    """
    num_particles = positions.shape[0]
    forces = np.zeros_like(positions)

    # Use prange to parallelize the outer loop
    for i in prange(num_particles):
        for j in range(num_particles):
            if i == j:
                continue

            # Calculate distance vector components
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            # Squared distance with softening
            dist_sq = dx*dx + dy*dy + dz*dz + epsilon**2
            
            # Inverse cube of distance
            inv_dist_cubed = dist_sq**(-1.5)
            
            # Force magnitude
            force_magnitude = G * masses[i] * masses[j] * inv_dist_cubed
            
            # Update forces component-wise
            forces[i, 0] += force_magnitude * dx
            forces[i, 1] += force_magnitude * dy
            forces[i, 2] += force_magnitude * dz

    return forces
