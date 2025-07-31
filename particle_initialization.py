"""
Provides functions for generating initial conditions for N-body simulations.
"""
import numpy as np

def initialize_plummer(num_particles, scale_radius=1.0, total_mass=1.0, G=1.0, random_seed=None):
    """
    Generates particle positions and velocities for a Plummer model.

    The Plummer model is a spherical density profile that is often used to
    represent star clusters.

    Args:
        num_particles (int): The number of particles to generate.
        scale_radius (float): The scale radius 'a' of the Plummer sphere. This
                              parameter defines the characteristic size of the
                              cluster core.
        total_mass (float): The total mass of the particle system.
        random_seed (int, optional): Seed for the random number generator for
                                     reproducibility.

    Returns:
        tuple: A tuple containing:
            - positions (np.ndarray): (N, 3) array of particle positions.
            - velocities (np.ndarray): (N, 3) array of particle velocities.
            - masses (np.ndarray): (N,) array of particle masses.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. Assign equal mass to all particles
    masses = np.full(num_particles, total_mass / num_particles)

    # 2. Generate positions based on the Plummer density profile
    # We use inverse transform sampling from the mass profile M(r).
    # The formula to generate a radius r is: r = a / sqrt(X^(-2/3) - 1)
    # where X is a random number uniformly distributed in [0, 1].
    x_rand = np.random.rand(num_particles)
    radii = scale_radius / np.sqrt(x_rand**(-2/3) - 1)

    # Generate isotropic random directions for the positions
    theta = np.arccos(2 * np.random.rand(num_particles) - 1)
    phi = 2 * np.pi * np.random.rand(num_particles)
    
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    
    positions = np.vstack([x, y, z]).T

    # 3. Generate velocities to be in approximate dynamical equilibrium.
    # We sample from a distribution that produces a stable system. The velocity
    # of a particle is drawn from a probability distribution f(v) proportional
    # to (v_e^2 - v^2)^(7/2), where v_e is the escape velocity.
    # This is done via rejection sampling, as described by Aarseth, Hénon, & Wielen (1974).
    
    # Calculate escape velocity squared at each particle's radius
    # v_e^2 = 2 * G * M / sqrt(r^2 + a^2)
    # For simplicity in the model, we can set G=1 if we scale units appropriately.
    # Here, we'll use a G that makes sense for the scale. Let M_total = 1, a = 1.
    # We'll set G such that a typical velocity is ~1.
    escape_vel_sq = (2 * G * total_mass) / np.sqrt(radii**2 + scale_radius**2)

    # Rejection sampling for velocity magnitudes
    vel_mag_sq = np.zeros(num_particles)
    for i in range(num_particles):
        while True:
            # Draw two random numbers g1, g2
            g1 = np.random.rand()
            g2 = np.random.rand()
            # If g2 < g1^2 * (1 - g1^2)^(7/2), accept v^2 = g1 * v_e^2
            # We use a simplified condition that is faster to compute:
            # if 0.1 * g2 < g1**2 * (1 - g1**2)**3.5, we accept.
            # The 0.1 factor is because the peak of x^2*(1-x^2)^3.5 is ~0.09
            if 0.1 * g2 < g1**2 * (1 - g1**2)**3.5:
                vel_mag_sq[i] = g1 * escape_vel_sq[i]
                break
    
    vel_magnitudes = np.sqrt(vel_mag_sq)

    # Generate isotropic random directions for the velocities
    theta_v = np.arccos(2 * np.random.rand(num_particles) - 1)
    phi_v = 2 * np.pi * np.random.rand(num_particles)

    vx = vel_magnitudes * np.sin(theta_v) * np.cos(phi_v)
    vy = vel_magnitudes * np.sin(theta_v) * np.sin(phi_v)
    vz = vel_magnitudes * np.cos(theta_v)

    velocities = np.vstack([vx, vy, vz]).T

    # Center the system
    positions -= np.mean(positions, axis=0)
    velocities -= np.mean(velocities, axis=0)

    return positions, velocities, masses
