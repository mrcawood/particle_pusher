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
    norm_x = int(((positions[i, 0] - min_coord[0]) * inv_box_size) * 1023.0)
    norm_y = int(((positions[i, 1] - min_coord[1]) * inv_box_size) * 1023.0)
    norm_z = int(((positions[i, 2] - min_coord[2]) * inv_box_size) * 1023.0)
    morton_codes[i] = (_expand_bits(norm_x) | _expand_bits(norm_y) << 1 | _expand_bits(norm_z) << 2)
    particle_indices[i] = i

@cuda.jit(device=True)
def _common_prefix_length(a, b):
    if a == b: return 64
    xor_val = a ^ b
    if xor_val == 0: return 64
    return cuda.clz(uint64(xor_val))

@cuda.jit(device=True)
def _delta(codes, n, i, j):
    if j < 0 or j >= n:
        return -1
    return _common_prefix_length(codes[i], codes[j])

@cuda.jit
def _build_tree_topology_kernel(codes, parents, n, internal_left, internal_right):
    i = cuda.grid(1)
    if i >= n - 1:
        return
    lcp_next = _delta(codes, n, i, i + 1)
    lcp_prev = _delta(codes, n, i, i - 1)
    d = 1 if lcp_next > lcp_prev else -1
    delta_min = _delta(codes, n, i, i - d)
    l_max = 2
    while _delta(codes, n, i, i + l_max * d) > delta_min:
        l_max *= 2
    l = 0
    t = l_max // 2
    while t > 0:
        if i + (l + t) * d >= 0 and i + (l + t) * d < n:
            if _delta(codes, n, i, i + (l + t) * d) > delta_min:
                l += t
        t //= 2
    j = i + l * d
    left = min(i, j)
    right = max(i, j)
    common_prefix = _common_prefix_length(codes[left], codes[right])
    s = left
    step = right - left
    while step > 1:
        step = (step + 1) >> 1
        new_s = s + step
        if new_s < right:
            split_prefix = _common_prefix_length(codes[left], codes[new_s])
            if split_prefix > common_prefix:
                s = new_s
    leaf_base = n - 1
    left_child_idx = (leaf_base + s) if (s == left) else s
    right_child_idx = (leaf_base + s + 1) if ((s + 1) == right) else (s + 1)
    parents[left_child_idx] = i
    parents[right_child_idx] = i
    internal_left[i] = left
    internal_right[i] = right

@cuda.jit
def _connect_children_kernel(codes, parents, internal_left, internal_right, children, n):
    i = cuda.grid(1)
    if i >= n * 2 - 1: return
    parent = parents[i]
    if parent != _EMPTY:
        Lp = internal_left[parent]
        Rp = internal_right[parent]
        parent_prefix = _common_prefix_length(codes[Lp], codes[Rp])
        child_code = codes[internal_left[i]] if i < (n - 1) else codes[i - (n - 1)]
        level = parent_prefix // 3
        if level > 9: level = 9
        shift = 3 * (9 - level)
        octant = (child_code >> int(shift)) & 7
        children[parent, octant] = i

@cuda.jit
def _initialize_leaves_kernel(node_com, node_mass, node_bounds, sorted_particle_indices, positions, masses, num_particles):
    i = cuda.grid(1)
    if i >= num_particles: return
    leaf_idx = num_particles - 1 + i
    particle_idx = sorted_particle_indices[i]
    px, py, pz = positions[particle_idx, 0], positions[particle_idx, 1], positions[particle_idx, 2]
    node_com[leaf_idx, 0], node_com[leaf_idx, 1], node_com[leaf_idx, 2] = px, py, pz
    node_mass[leaf_idx] = masses[particle_idx]
    
    # For leaves, center is the particle's position, size is 0
    node_bounds[leaf_idx, 0] = px
    node_bounds[leaf_idx, 1] = py
    node_bounds[leaf_idx, 2] = pz
    node_bounds[leaf_idx, 3] = 0.0

@cuda.jit
def _com_sweep_kernel(node_children, node_com, node_mass, num_particles):
    i = cuda.grid(1)
    if i >= num_particles - 1: return
    
    total_mass, wx, wy, wz = 0.0, 0.0, 0.0, 0.0
    
    for child_idx_in_octant in range(8):
        child = node_children[i, child_idx_in_octant]
        if child != _EMPTY:
            mass = node_mass[child]
            
            # This is the fix: only consider children who have valid mass.
            # A non-leaf child's mass will be > 0 only after it has been processed.
            is_leaf_child = child >= (num_particles - 1)
            if not is_leaf_child and mass == 0.0:
                continue

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
def _calculate_internal_bounds_kernel(node_children, node_bounds, node_mass, num_particles):
    i = cuda.grid(1)
    if i >= num_particles - 1: return

    min_x, max_x = 1e10, -1e10
    min_y, max_y = 1e10, -1e10
    min_z, max_z = 1e10, -1e10

    found_valid_child = False
    for child_idx_in_octant in range(8):
        child = node_children[i, child_idx_in_octant]
        if child != _EMPTY:
            child_mass = node_mass[child]
            
            is_leaf_child = child >= (num_particles - 1)
            if not is_leaf_child and child_mass == 0.0:
                continue

            found_valid_child = True
            child_cx, child_cy, child_cz = node_bounds[child, 0], node_bounds[child, 1], node_bounds[child, 2]
            child_s = node_bounds[child, 3]
            
            min_x = min(min_x, child_cx - child_s)
            max_x = max(max_x, child_cx + child_s)
            min_y = min(min_y, child_cy - child_s)
            max_y = max(max_y, child_cy + child_s)
            min_z = min(min_z, child_cz - child_s)
            max_z = max(max_z, child_cz + child_s)

    if found_valid_child:
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0
        
        size = 0.0
        size = max(size, max_x - center_x)
        size = max(size, center_x - min_x)
        size = max(size, max_y - center_y)
        size = max(size, center_y - min_y)
        size = max(size, max_z - center_z)
        size = max(size, center_z - min_z)
        
        node_bounds[i, 0] = center_x
        node_bounds[i, 1] = center_y
        node_bounds[i, 2] = center_z
        node_bounds[i, 3] = size

@cuda.jit(device=True)
def _is_leaf(node_idx, node_children, num_particles):
    if node_idx >= num_particles - 1:
        return True
    
    for i in range(8):
        if node_children[node_idx, i] != _EMPTY:
            return False
    return True

@cuda.jit(device=True)
def _calculate_force_on_particle_gpu(particle_idx, root_node_idx, positions, masses, G, epsilon, theta,
                                  node_children, node_com, node_mass, node_bounds, sorted_particle_indices, num_particles):
    force_x, force_y, force_z, interactions = 0.0, 0.0, 0.0, 0
    approx_count = 0
    max_stack_depth = 0
    
    orig_particle_idx = sorted_particle_indices[particle_idx]
    px, py, pz = positions[orig_particle_idx, 0], positions[orig_particle_idx, 1], positions[orig_particle_idx, 2]
    
    stack = cuda.local.array(shape=64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = root_node_idx
    stack_ptr += 1

    while stack_ptr > 0:
        max_stack_depth = max(max_stack_depth, stack_ptr)
        stack_ptr -= 1
        node_idx = stack[stack_ptr]
        
        if node_mass[node_idx] == 0:
            continue

        is_leaf_node = _is_leaf(node_idx, node_children, num_particles)
        
        if is_leaf_node:
            if node_idx != (num_particles - 1 + particle_idx):
                rij_x = node_com[node_idx, 0] - px
                rij_y = node_com[node_idx, 1] - py
                rij_z = node_com[node_idx, 2] - pz
                d_sq = rij_x*rij_x + rij_y*rij_y + rij_z*rij_z + epsilon*epsilon
                inv_d = d_sq**(-0.5)
                inv_dist_cubed = inv_d * inv_d * inv_d
                force_mag = G * masses[orig_particle_idx] * node_mass[node_idx] * inv_dist_cubed
                force_x += force_mag * rij_x
                force_y += force_mag * rij_y
                force_z += force_mag * rij_z
                interactions += 1
            continue

        s = node_bounds[node_idx, 3]
        
        rij_x, rij_y, rij_z = node_com[node_idx, 0] - px, node_com[node_idx, 1] - py, node_com[node_idx, 2] - pz
        d_sq = rij_x*rij_x + rij_y*rij_y + rij_z*rij_z

        if s > 0 and (s*s / d_sq) < (theta * theta):
            approx_count += 1
            d_sq += epsilon*epsilon
            inv_d = d_sq**(-0.5)
            inv_dist_cubed = inv_d * inv_d * inv_d
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
    
    return force_x, force_y, force_z, interactions, max_stack_depth, approx_count

@cuda.jit
def _force_calculation_kernel(positions, masses, forces, G, epsilon, theta,
                            root_node_idx, node_children, node_com, node_mass, node_bounds, num_particles,
                            sorted_particle_indices, interaction_counts, debug_output):
    i = cuda.grid(1)
    if i >= num_particles: return
    
    fx, fy, fz, interactions, max_stack_depth, approx_count = _calculate_force_on_particle_gpu(
        i, root_node_idx, positions, masses, G, epsilon, theta,
        node_children, node_com, node_mass, node_bounds, sorted_particle_indices, num_particles
    )

    orig_particle_idx = sorted_particle_indices[i]
    forces[orig_particle_idx, 0] = fx
    forces[orig_particle_idx, 1] = fy
    forces[orig_particle_idx, 2] = fz
    interaction_counts[i] = interactions
    debug_output[i, 0] = max_stack_depth
    debug_output[i, 1] = approx_count


def get_gflops(total_interactions, time_ms):
    if time_ms == 0:
        return 0.0
    
    flops_per_interaction = 20
    total_flops = total_interactions * flops_per_interaction
    gflops = total_flops / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0
    return gflops

def calculate_forces(positions, masses, G, epsilon, theta=0.5, dtype=np.float64, **kwargs):
    num_particles = positions.shape[0]
    if num_particles <= 1:
        return {"forces": cp.zeros_like(positions), "timings": {}, "metrics": {"gflops": 0}}

    threads_per_block = 256
    blocks_per_grid_particles = (num_particles + (threads_per_block - 1)) // threads_per_block
    timings = {}

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    min_coord, max_coord = positions.min(axis=0), positions.max(axis=0)
    box_size = (max_coord - min_coord).max().item()
    if box_size == 0: box_size = 1.0
    inv_box_size = 1.0 / box_size
    d_morton_codes = cp.empty(num_particles, dtype=cp.uint64)
    d_particle_indices = cp.empty(num_particles, dtype=cp.int32)
    _morton_code_kernel[blocks_per_grid_particles, threads_per_block](
        positions, d_morton_codes, d_particle_indices, min_coord, inv_box_size
    )
    sort_order = cp.argsort(d_morton_codes)
    d_sorted_morton_codes = d_morton_codes[sort_order]
    d_sorted_particle_indices = d_particle_indices[sort_order]
    end_event.record()
    end_event.synchronize()
    timings["morton_sort"] = cp.cuda.get_elapsed_time(start_event, end_event)

    start_event.record()
    num_internal_nodes = num_particles - 1
    max_nodes = num_particles + num_internal_nodes
    d_node_parent = cp.full(max_nodes, _EMPTY, dtype=cp.int32)
    d_internal_left = cp.full(num_internal_nodes, -1, dtype=cp.int32)
    d_internal_right = cp.full(num_internal_nodes, -1, dtype=cp.int32)
    d_node_children = cp.full((max_nodes, 8), _EMPTY, dtype=cp.int32)

    if num_internal_nodes > 0:
        tree_build_blocks = (num_internal_nodes + threads_per_block - 1) // threads_per_block
        _build_tree_topology_kernel[tree_build_blocks, threads_per_block](
            d_sorted_morton_codes, d_node_parent, num_particles, d_internal_left, d_internal_right
        )
        connect_children_blocks = (max_nodes + threads_per_block - 1) // threads_per_block
        _connect_children_kernel[connect_children_blocks, threads_per_block](
            d_sorted_morton_codes, d_node_parent, d_internal_left, d_internal_right, d_node_children, num_particles
        )
    end_event.record()
    end_event.synchronize()
    timings["tree_build"] = cp.cuda.get_elapsed_time(start_event, end_event)
    
    start_event.record()
    d_node_com = cp.zeros((max_nodes, 3), dtype=dtype)
    d_node_mass = cp.zeros(max_nodes, dtype=dtype)
    d_node_bounds = cp.zeros((max_nodes, 4), dtype=dtype)
    _initialize_leaves_kernel[blocks_per_grid_particles, threads_per_block](
        d_node_com, d_node_mass, d_node_bounds, d_sorted_particle_indices,
        positions, masses, num_particles
    )
    if num_internal_nodes > 0:
        blocks_per_grid_internal = (num_internal_nodes + threads_per_block - 1) // threads_per_block
        for _ in range(int(math.log2(num_particles)) + 2):
            _com_sweep_kernel[blocks_per_grid_internal, threads_per_block](
                d_node_children, d_node_com, d_node_mass, num_particles
            )
            _calculate_internal_bounds_kernel[blocks_per_grid_internal, threads_per_block](
                d_node_children, d_node_bounds, d_node_mass, num_particles
            )
    end_event.record()
    end_event.synchronize()
    timings["com_calc"] = cp.cuda.get_elapsed_time(start_event, end_event)

    start_event.record()
    d_forces = cp.zeros_like(positions)
    d_interaction_counts = cp.zeros(num_particles, dtype=cp.int64)
    d_debug_output = cp.zeros((num_particles, 2), dtype=cp.int32)
    root_nodes = cp.where(d_node_parent == _EMPTY)[0]
    root_node_idx = root_nodes[0].item() if len(root_nodes) > 0 else 0
    
    _force_calculation_kernel[blocks_per_grid_particles, threads_per_block](
        positions, masses, d_forces, G, epsilon, theta,
        root_node_idx, d_node_children, d_node_com, d_node_mass, d_node_bounds, 
        num_particles, d_sorted_particle_indices, d_interaction_counts, d_debug_output
    )
    end_event.record()
    end_event.synchronize()
    force_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    timings["force_calculation"] = force_time_ms

    interaction_count = int(cp.sum(d_interaction_counts))
    gflops = get_gflops(interaction_count, force_time_ms)
    
    return {
        "forces": d_forces, 
        "timings": timings, 
        "metrics": {"gflops": gflops},
        "debug_out": d_debug_output,
        "tree_data": {
            "node_mass": d_node_mass,
            "node_bounds": d_node_bounds,
            "node_children": d_node_children
        }
    }
