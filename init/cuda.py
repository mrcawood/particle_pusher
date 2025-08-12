import cupy as cp
from numba import cuda
import math

@cuda.jit
def _plummer_kernel(positions, velocities, masses, scale_radius, G, rotation_factor):
    """
    CUDA kernel to generate Plummer model particle properties in parallel.
    """
    i = cuda.grid(1)
    if i >= masses.shape[0]:
        return

    mass_val = masses[i]
    power_val = mass_val**(-2.0/3.0)
    sub_val = power_val - 1.0
    sqrt_val = math.sqrt(sub_val)
    radius = scale_radius / sqrt_val

    theta = positions[i, 0] 
    phi = positions[i, 1]
    
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    positions[i, 0] = x
    positions[i, 1] = y
    positions[i, 2] = z

    X = velocities[i, 0]
    q = math.sqrt(x*x + y*y + z*z)
    esc_vel_sq = 2.0 * G * masses.shape[0] / q if q > 0 else 0.0
    
    v_mag = X * math.sqrt(esc_vel_sq)
    
    v_theta = velocities[i, 2] * 2.0 * math.pi
    v_phi = math.acos(2.0 * X - 1.0)
    
    vx = v_mag * math.sin(v_theta) * math.cos(v_phi)
    vy = v_mag * math.sin(v_theta) * math.sin(v_phi)
    vz = v_mag * math.cos(v_theta)

    if rotation_factor > 0:
        velocities[i, 0] = vx - rotation_factor * y
        velocities[i, 1] = vy + rotation_factor * x
        velocities[i, 2] = vz
    else:
        velocities[i, 0] = vx
        velocities[i, 1] = vy
        velocities[i, 2] = vz

def initialize_plummer_gpu(num_particles, scale_radius, G, rotation_factor, dtype=cp.float64, seed=None):
    """
    Generates particles on the GPU according to a Plummer model with specified precision.
    """
    if seed is not None:
        cp.random.seed(seed)
        
    d_masses_rand = cp.random.rand(num_particles, dtype=dtype)
    d_pos_rand = cp.random.rand(num_particles, 2, dtype=dtype) * cp.array([2.0 * cp.pi, cp.pi], dtype=dtype)
    d_vel_rand = cp.random.rand(num_particles, 3, dtype=dtype)
    
    d_positions = cp.empty((num_particles, 3), dtype=dtype)
    d_velocities = cp.empty((num_particles, 3), dtype=dtype)
    
    d_positions[:, 0] = d_pos_rand[:, 0]
    d_positions[:, 1] = d_pos_rand[:, 1]
    d_velocities[:, :] = d_vel_rand[:, :]
    d_masses = d_masses_rand

    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block
    
    _plummer_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d_masses, scale_radius, G, rotation_factor
    )
    
    d_masses.fill(1.0 / num_particles)

    return d_positions, d_velocities, d_masses

@cuda.jit
def _grid_kernel(positions, spacing, side_len):
    """
    CUDA kernel to generate grid positions in parallel.
    """
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return

    # De-linearize the 1D index to 3D grid coordinates
    z = i // (side_len * side_len)
    xy_rem = i % (side_len * side_len)
    y = xy_rem // side_len
    x = xy_rem % side_len

    positions[i, 0] = (x - side_len / 2 + 0.5) * spacing
    positions[i, 1] = (y - side_len / 2 + 0.5) * spacing
    positions[i, 2] = (z - side_len / 2 + 0.5) * spacing

def initialize_grid_gpu(num_particles, spacing=1.0, total_mass=1.0, dtype=cp.float64):
    """
    Generates particles on the GPU arranged in a 3D grid.
    """
    side_len = int(cp.ceil(num_particles**(1/3.0)))

    d_positions = cp.empty((num_particles, 3), dtype=dtype)
    d_velocities = cp.zeros((num_particles, 3), dtype=dtype)
    d_masses = cp.full(num_particles, total_mass / num_particles, dtype=dtype)
    
    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block

    _grid_kernel[blocks_per_grid, threads_per_block](d_positions, spacing, side_len)
    
    # Center the system on the GPU
    mean_pos = cp.mean(d_positions, axis=0)
    d_positions -= mean_pos

    return d_positions, d_velocities, d_masses
