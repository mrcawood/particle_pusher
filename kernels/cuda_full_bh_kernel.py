"""
This file will contain the fully GPU-resident Barnes-Hut implementation.
"""
import numpy as np
import cupy as cp
from numba import cuda

# A constant to indicate an empty node or child
_EMPTY = -1

@cuda.jit(device=True)
def _expand_bits(v):
    """
    Spreads the bits of a 32-bit integer so that there are two empty bits
    between each original bit. This is a key part of Morton code calculation.
    Example: ...000111 -> ...001001001
    """
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

@cuda.jit
def _morton_code_kernel(
    positions, morton_codes, particle_indices,
    min_coord, inv_box_size
):
    """
    Calculates the 3D Morton code for each particle.
    - Normalizes particle positions to a 32-bit integer grid.
    - Spreads the bits of each coordinate.
    - Interleaves the bits to form the Morton code.
    """
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return

    # Normalize position to a 32-bit integer grid within the bounding box
    # We use a small epsilon to handle particles exactly on the max boundary
    norm_x = int(((positions[i, 0] - min_coord[0]) * inv_box_size) * 2147483647.0)
    norm_y = int(((positions[i, 1] - min_coord[1]) * inv_box_size) * 2147483647.0)
    norm_z = int(((positions[i, 2] - min_coord[2]) * inv_box_size) * 2147483647.0)

    # Spread the bits and interleave them to create the Morton code
    morton_codes[i] = (_expand_bits(norm_x) | 
                       _expand_bits(norm_y) << 1 | 
                       _expand_bits(norm_z) << 2)
    
    particle_indices[i] = i


def calculate_forces(positions, masses, G, epsilon, theta=0.5):
    """
    The main interface for the fully GPU-resident Barnes-Hut simulation.
    """
    num_particles = positions.shape[0]
    if num_particles == 0:
        return np.zeros_like(positions)

    # --- 0. Transfer Host Data to Device ---
    d_positions = cp.asarray(positions)
    d_masses = cp.asarray(masses)

    # --- 1. Morton Coding ---
    # Calculate bounding box for Morton coding
    min_coord = d_positions.min(axis=0)
    max_coord = d_positions.max(axis=0)
    box_size = (max_coord - min_coord).max()
    inv_box_size = 1.0 / box_size

    # Allocate memory on the GPU for the results
    d_morton_codes = cp.empty(num_particles, dtype=cp.uint64)
    d_particle_indices = cp.empty(num_particles, dtype=cp.int32)

    # Configure and launch the kernel
    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block
    
    _morton_code_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_morton_codes, d_particle_indices,
        min_coord, inv_box_size
    )
    
    # --- Verification Step ---
    # Copy results back to host to check them.
    # We will remove this in the final implementation.
    print("--- Phase 1: Morton Coding Verification ---")
    h_morton_codes = d_morton_codes.get()
    h_particle_indices = d_particle_indices.get()
    print(f"Calculated {len(h_morton_codes)} Morton codes.")
    print(f"First 5 codes: {h_morton_codes[:5]}")
    print(f"Particle indices: {h_particle_indices[:5]}... (should be [0 1 2 3 4])")
    print("-----------------------------------------")
    
    # Return zero forces for now so the simulation can run without crashing.
    return np.zeros_like(positions)
