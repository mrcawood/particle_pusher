"""
This file will contain the fully GPU-resident Barnes-Hut implementation.
"""
import numpy as np
import cupy as cp
from numba import cuda, uint64
import math

# A constant to indicate an empty node or child
_EMPTY = -1

@cuda.jit(device=True)
def _expand_bits(v):
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
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return

    norm_x = int(((positions[i, 0] - min_coord[0]) * inv_box_size) * 2147483647.0)
    norm_y = int(((positions[i, 1] - min_coord[1]) * inv_box_size) * 2147483647.0)
    norm_z = int(((positions[i, 2] - min_coord[2]) * inv_box_size) * 2147483647.0)

    morton_codes[i] = (_expand_bits(norm_x) | 
                       _expand_bits(norm_y) << 1 | 
                       _expand_bits(norm_z) << 2)
    particle_indices[i] = i


def calculate_forces(positions, masses, G, epsilon, theta=0.5):
    """
    The main interface for the fully GPU-resident Barnes-Hut simulation.
    """
    num_particles = positions.shape[0]
    if num_particles <= 1:
        return np.zeros_like(positions)
    
    threads_per_block = 256
    
    # --- GPU Timers ---
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    # --- 0. Transfer Host Data to Device ---
    d_positions = cp.asarray(positions)
    d_masses = cp.asarray(masses)

    # --- 1. Morton Coding and Sorting ---
    start_event.record()
    blocks_per_grid_particles = (num_particles + (threads_per_block - 1)) // threads_per_block
    min_coord = d_positions.min(axis=0)
    max_coord = d_positions.max(axis=0)
    box_size = (max_coord - min_coord).max().item()
    if box_size == 0: box_size = 1.0
    inv_box_size = 1.0 / box_size
    d_morton_codes = cp.empty(num_particles, dtype=cp.uint64)
    d_particle_indices = cp.empty(num_particles, dtype=cp.int32)
    _morton_code_kernel[blocks_per_grid_particles, threads_per_block](
        d_positions, d_morton_codes, d_particle_indices, min_coord, inv_box_size
    )
    sort_order = cp.argsort(d_morton_codes)
    d_sorted_morton_codes = d_morton_codes[sort_order]
    d_sorted_particle_indices = d_particle_indices[sort_order]
    end_event.record()
    end_event.synchronize()
    t_sort = cp.cuda.get_elapsed_time(start_event, end_event)

    # --- 2. Parallel Tree Build ---
    start_event.record()
    num_internal_nodes = num_particles - 1
    d_node_parent = cp.full(num_internal_nodes, _EMPTY, dtype=cp.int32)
    d_active_nodes = cp.arange(num_internal_nodes, dtype=cp.int32)
    
    num_active = len(d_active_nodes)
    while num_active > 1:
        num_pairs = (num_active + 1) // 2
        blocks_per_grid_pairs = (num_pairs + threads_per_block - 1) // threads_per_block
        d_next_active_nodes = cp.full(num_pairs, _EMPTY, dtype=cp.int32)
        _tree_build_reduction_kernel[blocks_per_grid_pairs, threads_per_block](
            d_node_parent, d_active_nodes, d_next_active_nodes, num_active
        )
        d_active_nodes = d_next_active_nodes
        num_active = len(d_active_nodes)
    
    max_nodes = num_particles + num_internal_nodes
    d_node_children = cp.full((max_nodes, 8), _EMPTY, dtype=cp.int32)
    child_set_blocks = (num_internal_nodes + threads_per_block - 1) // threads_per_block
    _set_children_kernel[child_set_blocks, threads_per_block](
        d_node_parent, d_node_children, d_sorted_morton_codes, num_particles
    )
    end_event.record()
    end_event.synchronize()
    t_tree = cp.cuda.get_elapsed_time(start_event, end_event)

    # --- 3. Center of Mass Calculation ---
    start_event.record()
    d_node_com = cp.zeros((max_nodes, 3), dtype=cp.float64)
    d_node_mass = cp.zeros(max_nodes, dtype=cp.float64)
    
    for _ in range(int(math.log2(num_particles)) + 2):
        _com_sweep_kernel[blocks_per_grid_particles, threads_per_block](
            d_node_parent, d_node_children, d_node_com, d_node_mass,
            d_sorted_particle_indices, d_positions, d_masses, num_particles
        )
    end_event.record()
    end_event.synchronize()
    t_com = cp.cuda.get_elapsed_time(start_event, end_event)

    # --- 4. Force Calculation ---
    start_event.record()
    d_forces = cp.zeros_like(d_positions)
    root_nodes = cp.where(d_node_parent == _EMPTY)[0]
    if len(root_nodes) == 0: # Guard against no root node being found
        root_node_idx = 0
    else:
        root_node_idx = root_nodes[0].item() # Use .item() to get a Python scalar

    d_node_bounds = cp.zeros((max_nodes, 6), dtype=cp.float64) 

    _force_calculation_kernel[blocks_per_grid_particles, threads_per_block](
        d_positions, d_masses, d_forces, G, epsilon, theta,
        root_node_idx, d_node_children, d_node_com, d_node_mass, d_node_bounds, num_particles, d_sorted_particle_indices
    )
    end_event.record()
    end_event.synchronize()
    t_force = cp.cuda.get_elapsed_time(start_event, end_event)
    
    # --- FLOPs Calculation ---
    flops_force = (27 * 25 * math.log2(num_particles) * num_particles) if num_particles > 1 else 0
    gflops = (flops_force / (t_force / 1000)) / 1e9 if t_force > 0 else 0


    print(f"    CUDA Full BH Timings: "
          f"Sort: {t_sort:.2f}ms, "
          f"Tree Build: {t_tree:.2f}ms, "
          f"CoM Calc: {t_com:.2f}ms, "
          f"Force Calc: {t_force:.2f}ms | "
          f"GFLOPS: {gflops:.2f}")

    return d_forces.get()

@cuda.jit(device=True)
def _common_prefix_length(a, b):
    if a == b: return 64
    xor_val = a ^ b
    if xor_val == 0: return 64
    if xor_val >> 32: return cuda.clz(uint64(xor_val >> 32))
    else: return cuda.clz(uint64(xor_val & 0xFFFFFFFF)) + 32

@cuda.jit
def _tree_build_reduction_kernel(
    node_parent, active_nodes, next_active_nodes, num_active
):
    pair_idx = cuda.grid(1)
    if pair_idx >= (num_active + 1) // 2: return

    node1_idx_in_active = pair_idx * 2
    node1 = active_nodes[node1_idx_in_active]

    if node1_idx_in_active + 1 < num_active:
        node2 = active_nodes[node1_idx_in_active + 1]
        parent_idx = min(node1, node2)
        child_idx = max(node1, node2)
        node_parent[child_idx] = parent_idx
        next_active_nodes[pair_idx] = parent_idx
    else:
        next_active_nodes[pair_idx] = node1

@cuda.jit(device=True)
def _get_octant_gpu(code1, code2):
    prefix_len = _common_prefix_length(code1, code2)
    if prefix_len >= 61: return 0
    shift = 60 - prefix_len
    return (code2 >> int(shift)) & 7

@cuda.jit
def _set_children_kernel(node_parent, node_children, sorted_morton_codes, num_particles):
    i = cuda.grid(1)
    if i >= num_particles - 1: return
    
    parent = node_parent[i]
    if parent != _EMPTY:
        octant = _get_octant_gpu(sorted_morton_codes[parent], sorted_morton_codes[i])
        cuda.atomic.cas(node_children, (parent, octant), _EMPTY, i)

    leaf_base = num_particles - 1
    octant1 = _get_octant_gpu(sorted_morton_codes[i], sorted_morton_codes[i])
    cuda.atomic.cas(node_children, (i, octant1), _EMPTY, leaf_base + i)
    octant2 = _get_octant_gpu(sorted_morton_codes[i], sorted_morton_codes[i + 1])
    cuda.atomic.cas(node_children, (i, octant2), _EMPTY, leaf_base + i + 1)


@cuda.jit
def _com_sweep_kernel(
    node_parent, node_children, node_com, node_mass,
    sorted_particle_indices, positions, masses, num_particles
):
    i = cuda.grid(1)
    if i >= num_particles - 1: return

    # Simplified bottom-up sweep
    # Level 0: Leaves
    leaf_idx = (num_particles - 1) + i
    particle_idx = sorted_particle_indices[i]
    node_com[leaf_idx, 0] = positions[particle_idx, 0]
    node_com[leaf_idx, 1] = positions[particle_idx, 1]
    node_com[leaf_idx, 2] = positions[particle_idx, 2]
    node_mass[leaf_idx] = masses[particle_idx]

    # Internal nodes
    total_mass = 0.0
    wx, wy, wz = 0.0, 0.0, 0.0
    
    for child_idx in range(8):
        child = node_children[i, child_idx]
        if child != _EMPTY:
            mass = node_mass[child]
            if mass > 0:
                total_mass += mass
                wx += node_com[child, 0] * mass
                wy += node_com[child, 1] * mass
                wz += node_com[child, 2] * mass
    
    if total_mass > 0:
        inv_mass = 1.0 / total_mass
        node_com[i, 0] = wx * inv_mass
        node_com[i, 1] = wy * inv_mass
        node_com[i, 2] = wz * inv_mass
        node_mass[i] = total_mass


@cuda.jit(device=True)
def _calculate_force_on_particle_gpu(
    particle_idx, root_node_idx, positions, masses, G, epsilon, theta,
    node_children, node_com, node_mass, node_bounds, sorted_particle_indices
):
    force_x, force_y, force_z = 0.0, 0.0, 0.0
    
    # The particle index we are calculating for is from the original array
    orig_particle_idx = sorted_particle_indices[particle_idx]
    px, py, pz = positions[orig_particle_idx, 0], positions[orig_particle_idx, 1], positions[orig_particle_idx, 2]
    
    stack = cuda.local.array(shape=64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = root_node_idx
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]
        
        is_leaf = True
        num_children = 0
        for i in range(8):
            if node_children[node_idx, i] != _EMPTY:
                is_leaf = False
                num_children += 1

        if is_leaf:
             continue

        s = 1.0 
        rij_x = node_com[node_idx, 0] - px
        rij_y = node_com[node_idx, 1] - py
        rij_z = node_com[node_idx, 2] - pz
        d_sq = rij_x*rij_x + rij_y*rij_y + rij_z*rij_z + epsilon*epsilon

        if (s*s) < (d_sq * theta * theta):
            if d_sq > 0:
                inv_dist_cubed = d_sq**(-1.5)
                force_mag = G * masses[orig_particle_idx] * node_mass[node_idx] * inv_dist_cubed
                force_x += force_mag * rij_x
                force_y += force_mag * rij_y
                force_z += force_mag * rij_z
        else:
            for i in range(8):
                child_idx = node_children[node_idx, i]
                if child_idx != _EMPTY:
                    if stack_ptr < 64:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1
    
    return force_x, force_y, force_z

@cuda.jit
def _force_calculation_kernel(
    positions, masses, forces, G, epsilon, theta,
    root_node_idx, node_children, node_com, node_mass, node_bounds, num_particles,
    sorted_particle_indices
):
    i = cuda.grid(1)
    if i >= num_particles: return

    fx, fy, fz = _calculate_force_on_particle_gpu(
        i, root_node_idx, positions, masses, G, epsilon, theta,
        node_children, node_com, node_mass, node_bounds, sorted_particle_indices
    )
    
    orig_particle_idx = sorted_particle_indices[i]
    forces[orig_particle_idx, 0] = fx
    forces[orig_particle_idx, 1] = fy
    forces[orig_particle_idx, 2] = fz
