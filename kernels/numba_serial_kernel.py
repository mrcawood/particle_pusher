import numpy as np
from numba import njit

@njit
def calculate_forces(positions, masses, G, epsilon):
    """
    Calculates the gravitational forces between all particles.
    Uses Numba's @njit decorator to compile the function for serial execution.

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

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
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
            
            # Update forces component-wise to avoid temporary arrays
            fx = force_magnitude * dx
            fy = force_magnitude * dy
            fz = force_magnitude * dz

            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz

            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz

    return forces
