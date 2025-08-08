import numpy as np
import cupy as cp
from numba import cuda, uint64
import math

_EMPTY = -1

@cuda.jit(device=True)
def _expand_bits(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

@cuda.jit
def _morton_code_kernel(positions, morton_codes, particle_indices, min_coord, inv_box_size):
    i = cuda.grid(1)
    if i >= positions.shape[0]: return
    norm_x = int(((positions[i, 0] - min_coord[0]) * inv_box_size) * 2147483647.0)
    norm_y = int(((positions[i, 1] - min_coord[1]) * inv_box_size) * 2147483647.0)
    norm_z = int(((positions[i, 2] - min_coord[2]) * inv_box_size) * 2147483647.0)
    morton_codes[i] = (_expand_bits(norm_x) | _expand_bits(norm_y) << 1 | _expand_bits(norm_z) << 2)
    particle_indices[i] = i

def calculate_forces(positions, masses, G, epsilon, theta=0.5, dtype=np.float64, **kwargs):
    num_particles = positions.shape[0]
    if num_particles <= 1:
        return {"forces": cp.zeros_like(positions), "timings": {}, "metrics": {"interaction_count": 0}}

    threads_per_block = 256
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    
    d_positions, d_masses = positions, masses

    start_event.record()
    blocks_per_grid_particles = (num_particles + (threads_per_block - 1)) // threads_per_block
    min_coord, max_coord = d_positions.min(axis=0), d_positions.max(axis=0)
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
        d_active_nodes, num_active = d_next_active_nodes, len(d_next_active_nodes)
    
    max_nodes = num_particles + num_internal_nodes
    d_node_children = cp.full((max_nodes, 8), _EMPTY, dtype=cp.int32)
    
    num_pairs = num_internal_nodes * 3
    d_parent_child_pairs = cp.full((num_pairs, 2), _EMPTY, dtype=cp.int32)
    pair_creation_blocks = (num_internal_nodes + threads_per_block - 1) // threads_per_block
    _create_parent_child_pairs_kernel[pair_creation_blocks, threads_per_block](
        d_node_parent, num_particles, d_parent_child_pairs
    )
    sort_keys, sort_values = d_parent_child_pairs[:, 0], d_parent_child_pairs[:, 1]
    valid_mask = sort_keys != _EMPTY
    valid_keys, valid_values = sort_keys[valid_mask], sort_values[valid_mask]
    sort_indices = cp.argsort(valid_keys)
    d_sorted_keys, d_sorted_values = valid_keys[sort_indices], valid_values[sort_indices]
    set_children_blocks = (len(d_sorted_keys) + threads_per_block - 1) // threads_per_block
    _set_children_from_sorted_pairs_kernel[set_children_blocks, threads_per_block](
        d_sorted_keys, d_sorted_values, d_sorted_morton_codes, d_node_children, num_particles
    )
    end_event.record()
    end_event.synchronize()
    t_tree = cp.cuda.get_elapsed_time(start_event, end_event)

    start_event.record()
    d_node_com = cp.zeros((max_nodes, 3), dtype=dtype)
    d_node_mass = cp.zeros(max_nodes, dtype=dtype)
    d_node_bounds = cp.zeros((max_nodes, 6), dtype=dtype)

    _initialize_leaves_kernel[blocks_per_grid_particles, threads_per_block](
        d_node_com, d_node_mass, d_node_bounds, d_sorted_particle_indices,
        d_positions, d_masses, num_particles
    )

    blocks_per_grid_internal = (num_internal_nodes + threads_per_block - 1) // threads_per_block
    for _ in range(int(math.log2(num_particles)) + 2):
        _calculate_internal_com_kernel[blocks_per_grid_internal, threads_per_block](
            d_node_children, d_node_com, d_node_mass, num_particles
        )
        _calculate_internal_bounds_kernel[blocks_per_grid_internal, threads_per_block](
            d_node_children, d_node_bounds, num_particles
        )
    
    end_event.record()
    end_event.synchronize()
    t_com = cp.cuda.get_elapsed_time(start_event, end_event)

    start_event.record()
    d_forces = cp.zeros_like(d_positions)
    d_interaction_counts = cp.zeros(num_particles, dtype=cp.int64)
    root_nodes = cp.where(d_node_parent == _EMPTY)[0]
    root_node_idx = root_nodes[0].item() if len(root_nodes) > 0 else 0

    _force_calculation_kernel[blocks_per_grid_particles, threads_per_block](
        d_positions, d_masses, d_forces, G, epsilon, theta,
        root_node_idx, d_node_children, d_node_com, d_node_mass, d_node_bounds, 
        num_particles, d_sorted_particle_indices, d_interaction_counts
    )
    end_event.record()
    end_event.synchronize()
    t_force = cp.cuda.get_elapsed_time(start_event, end_event)
    
    interaction_count = int(cp.sum(d_interaction_counts))

    timings = {"sort": t_sort, "tree_build": t_tree, "com_calc": t_com, "force_calculation": t_force}
    
    return {"forces": d_forces, "timings": timings, "metrics": {"interaction_count": interaction_count}}

@cuda.jit(device=True)
def _common_prefix_length(a, b):
    if a == b: return 64
    xor_val = a ^ b
    if xor_val == 0: return 64
    if xor_val >> 32: return cuda.clz(uint64(xor_val >> 32))
    else: return cuda.clz(uint64(xor_val & 0xFFFFFFFF)) + 32

@cuda.jit
def _tree_build_reduction_kernel(node_parent, active_nodes, next_active_nodes, num_active):
    pair_idx = cuda.grid(1)
    if pair_idx >= (num_active + 1) // 2: return
    node1_idx_in_active = pair_idx * 2
    node1 = active_nodes[node1_idx_in_active]
    if node1_idx_in_active + 1 < num_active:
        node2 = active_nodes[node1_idx_in_active + 1]
        parent_idx, child_idx = min(node1, node2), max(node1, node2)
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
def _create_parent_child_pairs_kernel(node_parent, num_particles, parent_child_pairs):
    i = cuda.grid(1)
    num_internal_nodes = num_particles - 1
    if i >= num_internal_nodes: return
    parent = node_parent[i]
    if parent != _EMPTY:
        parent_child_pairs[i, 0], parent_child_pairs[i, 1] = parent, i
    leaf_base = num_internal_nodes
    pair_idx_leaf1 = num_internal_nodes + i
    parent_child_pairs[pair_idx_leaf1, 0], parent_child_pairs[pair_idx_leaf1, 1] = i, leaf_base + i
    if i + 1 < num_particles:
       pair_idx_leaf2 = (2 * num_internal_nodes) + i
       parent_child_pairs[pair_idx_leaf2, 0], parent_child_pairs[pair_idx_leaf2, 1] = i, leaf_base + i + 1

@cuda.jit(device=True)
def _get_morton_code_for_node(node_idx, sorted_morton_codes, num_particles):
    return sorted_morton_codes[node_idx] if node_idx < num_particles - 1 else sorted_morton_codes[node_idx - (num_particles - 1)]

@cuda.jit
def _set_children_from_sorted_pairs_kernel(sorted_keys, sorted_values, sorted_morton_codes, node_children, num_particles):
    i = cuda.grid(1)
    if i >= sorted_keys.shape[0]: return
    parent, child = sorted_keys[i], sorted_values[i]
    parent_morton = _get_morton_code_for_node(parent, sorted_morton_codes, num_particles)
    child_morton = _get_morton_code_for_node(child, sorted_morton_codes, num_particles)
    octant = _get_octant_gpu(parent_morton, child_morton)
    node_children[parent, octant] = child

@cuda.jit
def _initialize_leaves_kernel(node_com, node_mass, node_bounds, sorted_particle_indices, positions, masses, num_particles):
    i = cuda.grid(1)
    if i >= num_particles: return
    leaf_idx = num_particles - 1 + i
    particle_idx = sorted_particle_indices[i]
    
    px, py, pz = positions[particle_idx, 0], positions[particle_idx, 1], positions[particle_idx, 2]
    
    node_com[leaf_idx, 0], node_com[leaf_idx, 1], node_com[leaf_idx, 2] = px, py, pz
    node_mass[leaf_idx] = masses[particle_idx]
    
    node_bounds[leaf_idx, 0], node_bounds[leaf_idx, 1] = px, px
    node_bounds[leaf_idx, 2], node_bounds[leaf_idx, 3] = py, py
    node_bounds[leaf_idx, 4], node_bounds[leaf_idx, 5] = pz, pz

@cuda.jit
def _calculate_internal_com_kernel(node_children, node_com, node_mass, num_particles):
    i = cuda.grid(1)
    if i >= num_particles - 1: return
    total_mass, wx, wy, wz = 0.0, 0.0, 0.0, 0.0
    for child_idx_in_octant in range(8):
        child = node_children[i, child_idx_in_octant]
        if child != _EMPTY:
            mass = node_mass[child]
            if mass > 0:
                total_mass += mass
                wx += node_com[child, 0] * mass
                wy += node_com[child, 1] * mass
                wz += node_com[child, 2] * mass
    if total_mass > 0:
        inv_mass = 1.0 / total_mass
        node_com[i, 0], node_com[i, 1], node_com[i, 2] = wx * inv_mass, wy * inv_mass, wz * inv_mass
        node_mass[i] = total_mass

@cuda.jit
def _calculate_internal_bounds_kernel(node_children, node_bounds, num_particles):
    i = cuda.grid(1)
    if i >= num_particles - 1: return

    min_x, max_x = 1e10, -1e10
    min_y, max_y = 1e10, -1e10
    min_z, max_z = 1e10, -1e10

    for child_idx_in_octant in range(8):
        child = node_children[i, child_idx_in_octant]
        if child != _EMPTY:
            c_min_x, c_max_x = node_bounds[child, 0], node_bounds[child, 1]
            if c_min_x < c_max_x: # Check if child bound is valid
                min_x = min(min_x, c_min_x)
                max_x = max(max_x, c_max_x)

            c_min_y, c_max_y = node_bounds[child, 2], node_bounds[child, 3]
            if c_min_y < c_max_y:
                min_y = min(min_y, c_min_y)
                max_y = max(max_y, c_max_y)

            c_min_z, c_max_z = node_bounds[child, 4], node_bounds[child, 5]
            if c_min_z < c_max_z:
                min_z = min(min_z, c_min_z)
                max_z = max(max_z, c_max_z)
    
    if min_x < max_x:
        node_bounds[i, 0], node_bounds[i, 1] = min_x, max_x
        node_bounds[i, 2], node_bounds[i, 3] = min_y, max_y
        node_bounds[i, 4], node_bounds[i, 5] = min_z, max_z

@cuda.jit(device=True)
def _calculate_force_on_particle_gpu(particle_idx, root_node_idx, positions, masses, G, epsilon, theta,
                                   node_children, node_com, node_mass, node_bounds, sorted_particle_indices, num_particles):
    force_x, force_y, force_z, interactions = 0.0, 0.0, 0.0, 0
    orig_particle_idx = sorted_particle_indices[particle_idx]
    px, py, pz = positions[orig_particle_idx, 0], positions[orig_particle_idx, 1], positions[orig_particle_idx, 2]
    
    stack = cuda.local.array(shape=64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = root_node_idx
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]
        
        if node_mass[node_idx] == 0:
            continue

        is_leaf = True
        for i in range(8):
            if node_children[node_idx, i] != _EMPTY:
                is_leaf = False
                break
        
        if is_leaf:
            if node_mass[node_idx] > 0 and node_idx != (num_particles - 1 + particle_idx):
                rij_x = node_com[node_idx, 0] - px
                rij_y = node_com[node_idx, 1] - py
                rij_z = node_com[node_idx, 2] - pz
                d_sq = rij_x*rij_x + rij_y*rij_y + rij_z*rij_z + epsilon*epsilon
                inv_dist_cubed = d_sq**(-1.5)
                force_mag = G * masses[orig_particle_idx] * node_mass[node_idx] * inv_dist_cubed
                force_x += force_mag * rij_x
                force_y += force_mag * rij_y
                force_z += force_mag * rij_z
                interactions += 1
            continue

        s = node_bounds[node_idx, 1] - node_bounds[node_idx, 0]
        rij_x, rij_y, rij_z = node_com[node_idx, 0] - px, node_com[node_idx, 1] - py, node_com[node_idx, 2] - pz
        d_sq = rij_x*rij_x + rij_y*rij_y + rij_z*rij_z + epsilon*epsilon

        if s > 0 and (s*s / d_sq) < (theta * theta):
            inv_dist_cubed = d_sq**(-1.5)
            force_mag = G * masses[orig_particle_idx] * node_mass[node_idx] * inv_dist_cubed
            force_x += force_mag * rij_x
            force_y += force_mag * rij_y
            force_z += force_mag * rij_z
            interactions += 1
        else:
            for i in range(8):
                child_idx = node_children[node_idx, i]
                if child_idx != _EMPTY and stack_ptr < 64:
                    stack[stack_ptr] = child_idx
                    stack_ptr += 1
    
    return force_x, force_y, force_z, interactions

@cuda.jit
def _force_calculation_kernel(positions, masses, forces, G, epsilon, theta,
                            root_node_idx, node_children, node_com, node_mass, node_bounds, num_particles,
                            sorted_particle_indices, interaction_counts):
    i = cuda.grid(1)
    if i >= num_particles: return
    fx, fy, fz, interactions = _calculate_force_on_particle_gpu(
        i, root_node_idx, positions, masses, G, epsilon, theta,
        node_children, node_com, node_mass, node_bounds, sorted_particle_indices, num_particles
    )
    orig_particle_idx = sorted_particle_indices[i]
    forces[orig_particle_idx, 0] = fx
    forces[orig_particle_idx, 1] = fy
    forces[orig_particle_idx, 2] = fz
    interaction_counts[i] = interactions
