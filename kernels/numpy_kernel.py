import numpy as np

def calculate_forces(positions, masses, G, epsilon):
    """
    Calculates the gravitational forces between all particles.
    Uses a vectorized approach for performance.

    Args:
        positions (np.array): (N, 3) array of particle positions.
        masses (np.array): (N,) array of particle masses.
        G (float): Gravitational constant.
        epsilon (float): Softening length to avoid singularities.

    Returns:
        np.array: (N, 3) array of forces on each particle.
    """
    num_particles = positions.shape[0]

    # Calculate all pairwise position differences.
    rij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # Calculate squared distances.
    dist_sq = np.sum(rij**2, axis=2)

    # Add softening factor to avoid division by zero.
    dist_sq += epsilon**2

    # Calculate 1 / r^3 for the force calculation.
    inv_dist_cubed = dist_sq**(-1.5)
    
    # Set diagonal to zero to avoid self-interaction.
    np.fill_diagonal(inv_dist_cubed, 0.)

    # Calculate the product of masses for each pair using broadcasting.
    mass_product = masses[:, np.newaxis] * masses[np.newaxis, :]

    # Calculate the magnitude of the force for each pair.
    force_magnitudes = G * mass_product * inv_dist_cubed

    # Calculate the force vectors.
    forces = np.sum(force_magnitudes[:, :, np.newaxis] * rij, axis=1)
    
    return forces
