"""
Provides functions for generating initial conditions for N-body simulations.
"""
import numpy as np

def initialize_plummer(num_particles, scale_radius=1.0, total_mass=1.0, G=1.0, rotation_factor=0.0, seed=None, dtype=np.float64):
    """
    Generates particle positions and velocities for a Plummer model.

    Args:
        num_particles (int): The number of particles to generate.
        scale_radius (float): The scale radius 'a' of the Plummer sphere.
        total_mass (float): The total mass of the particle system.
        G (float): The gravitational constant.
        rotation_factor (float): Factor for solid-body rotation.
        random_seed (int, optional): Seed for the random number generator.
        dtype (np.dtype): The data type (e.g., np.float32) for the arrays.

    Returns:
        tuple: A tuple containing positions, velocities, and masses arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    masses = np.full(num_particles, total_mass / num_particles, dtype=dtype)
    
    x_rand = np.random.rand(num_particles).astype(dtype)
    radii = scale_radius / np.sqrt(x_rand**(-2/3) - 1)
    
    theta = np.arccos(2 * np.random.rand(num_particles).astype(dtype) - 1)
    phi = 2 * np.pi * np.random.rand(num_particles).astype(dtype)
    
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    
    positions = np.vstack([x, y, z]).T.astype(dtype)

    escape_vel_sq = (2 * G * total_mass) / np.sqrt(radii**2 + scale_radius**2)

    vel_mag_sq = np.zeros(num_particles, dtype=dtype)
    for i in range(num_particles):
        while True:
            g1 = np.random.rand()
            g2 = np.random.rand()
            if 0.1 * g2 < g1**2 * (1 - g1**2)**3.5:
                vel_mag_sq[i] = g1 * escape_vel_sq[i]
                break
    
    vel_magnitudes = np.sqrt(vel_mag_sq)

    theta_v = np.arccos(2 * np.random.rand(num_particles).astype(dtype) - 1)
    phi_v = 2 * np.pi * np.random.rand(num_particles).astype(dtype)

    vx = vel_magnitudes * np.sin(theta_v) * np.cos(phi_v)
    vy = vel_magnitudes * np.sin(theta_v) * np.sin(phi_v)
    vz = vel_magnitudes * np.cos(theta_v)

    velocities = np.vstack([vx, vy, vz]).T.astype(dtype)

    if rotation_factor > 0:
        omega_vector = np.array([0, 0, rotation_factor], dtype=dtype)
        rotational_velocities = np.cross(omega_vector, positions)
        
        t_rand = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        t_rot = 0.5 * np.sum(masses[:, np.newaxis] * rotational_velocities**2)

        if t_rot >= t_rand:
            print("Warning: Rotation is too high for a stable system. Clamping rotation.")
            scale_rot = np.sqrt((t_rand * 0.5) / t_rot)
            rotational_velocities *= scale_rot
            t_rot = 0.5 * t_rand
        
        alpha = np.sqrt(1.0 - t_rot / t_rand)
        velocities *= alpha
        velocities += rotational_velocities

    positions -= np.mean(positions, axis=0)
    velocities -= np.mean(velocities, axis=0)

    return positions, velocities, masses

def initialize_grid(num_particles, spacing=1.0, total_mass=1.0, dtype=np.float64):
    """
    Generates particles arranged in a 3D grid.

    Args:
        num_particles (int): The number of particles to generate.
        spacing (float): The distance between adjacent particles in the grid.
        total_mass (float): The total mass of the particle system.
        dtype (np.dtype): The data type for the arrays.

    Returns:
        tuple: A tuple containing positions, velocities, and masses arrays.
    """
    # Find the smallest cube that can contain num_particles
    side_len = int(np.ceil(num_particles**(1/3.0)))
    
    # Create grid coordinates
    x_coords = np.linspace(-side_len / 2 + 0.5, side_len / 2 - 0.5, side_len) * spacing
    y_coords = np.linspace(-side_len / 2 + 0.5, side_len / 2 - 0.5, side_len) * spacing
    z_coords = np.linspace(-side_len / 2 + 0.5, side_len / 2 - 0.5, side_len) * spacing
    
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords)
    
    positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    
    # Trim excess particles
    positions = positions[:num_particles].astype(dtype)
    
    velocities = np.zeros_like(positions, dtype=dtype)
    masses = np.full(num_particles, total_mass / num_particles, dtype=dtype)
    
    # Center the system
    positions -= np.mean(positions, axis=0)
    
    return positions, velocities, masses
