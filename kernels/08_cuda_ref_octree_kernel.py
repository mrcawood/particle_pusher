import numpy as np
import cupy as cp
from numba import cuda

_EMPTY = -1


@cuda.jit
def _reset_tree_arrays_kernel(node_parent, node_children_flat, node_center, node_half, next_internal,
                              root_cx, root_cy, root_cz, root_half, max_nodes):
    i = cuda.grid(1)
    if i < max_nodes:
        node_parent[i] = _EMPTY
        # zero children for node i
        base = i * 8
        for k in range(8):
            node_children_flat[base + k] = _EMPTY
        if i < node_center.shape[0]:
            node_center[i, 0] = 0.0
            node_center[i, 1] = 0.0
            node_center[i, 2] = 0.0
            node_half[i] = 0.0
    if i == 0:
        node_center[0, 0] = root_cx
        node_center[0, 1] = root_cy
        node_center[0, 2] = root_cz
        node_half[0] = root_half
        next_internal[0] = 1  # root occupies internal index 0


@cuda.jit(device=True)
def _compute_child_octant(px, py, pz, cx, cy, cz):
    octant = 0
    if px < cx:
        octant |= 1
    else:
        pass
    if py < cy:
        octant |= 2
    else:
        pass
    if pz < cz:
        octant |= 4
    else:
        pass
    return octant


@cuda.jit
def _build_octree_insertion_kernel(positions, masses, node_parent, node_children_flat, node_center, node_half,
                                   node_mass, node_com_sum, next_internal, num_particles, max_internal):
    pid = cuda.grid(1)
    if pid >= num_particles:
        return

    # Leaf index mapping: N-1 + pid
    leaf_idx = (num_particles - 1) + pid
    # Start at root
    current = 0
    cx, cy, cz = node_center[current, 0], node_center[current, 1], node_center[current, 2]
    half = node_half[current]

    px = positions[pid, 0]
    py = positions[pid, 1]
    pz = positions[pid, 2]

    while True:
        # Accumulate mass and weighted position into current internal node
        cuda.atomic.add(node_mass, current, masses[pid])
        cuda.atomic.add(node_com_sum, (current, 0), masses[pid] * px)
        cuda.atomic.add(node_com_sum, (current, 1), masses[pid] * py)
        cuda.atomic.add(node_com_sum, (current, 2), masses[pid] * pz)
        # Determine octant relative to current center
        octant = _compute_child_octant(px, py, pz, cx, cy, cz)
        idx = current * 8 + octant
        child_val = node_children_flat[idx]

        if child_val == _EMPTY:
            # Try to place leaf directly
            old = cuda.atomic.cas(node_children_flat, idx, _EMPTY, leaf_idx)
            if old == _EMPTY:
                node_parent[leaf_idx] = current
                # Set leaf mass and COM sums so leaf contributes in traversal
                cuda.atomic.add(node_mass, leaf_idx, masses[pid])
                cuda.atomic.add(node_com_sum, (leaf_idx, 0), masses[pid] * px)
                cuda.atomic.add(node_com_sum, (leaf_idx, 1), masses[pid] * py)
                cuda.atomic.add(node_com_sum, (leaf_idx, 2), masses[pid] * pz)
                return
            else:
                child_val = old  # fall through to handle

        if child_val < num_particles - 1 and child_val >= 0:
            # Descend into existing internal node
            current = child_val
            cx, cy, cz = node_center[current, 0], node_center[current, 1], node_center[current, 2]
            half = node_half[current]
            continue

        # Conflict with an existing leaf - expand this slot into a new internal node
        old_leaf = child_val
        # Attempt to lock the slot with sentinel -2
        locked = cuda.atomic.cas(node_children_flat, idx, old_leaf, -2)
        if locked != old_leaf:
            # Lost the race; retry
            continue

        # Allocate a new internal node index
        new_internal = cuda.atomic.add(next_internal, 0, 1)
        if new_internal >= max_internal:
            # Out of space; restore slot to previous state and abort
            node_children_flat[idx] = old_leaf
            return

        # Set new internal node region (child cube of current)
        child_half = half * 0.5
        child_cx = cx + (child_half if (octant & 1) == 0 else -child_half)
        child_cy = cy + (child_half if (octant & 2) == 0 else -child_half)
        child_cz = cz + (child_half if (octant & 4) == 0 else -child_half)
        node_center[new_internal, 0] = child_cx
        node_center[new_internal, 1] = child_cy
        node_center[new_internal, 2] = child_cz
        node_half[new_internal] = child_half

        # Initialize new internal node's children to EMPTY
        base_new = new_internal * 8
        for k in range(8):
            node_children_flat[base_new + k] = _EMPTY

        # Place the old leaf under the new internal
        if old_leaf >= num_particles - 1:
            old_pid = old_leaf - (num_particles - 1)
            opx = positions[old_pid, 0]
            opy = positions[old_pid, 1]
            opz = positions[old_pid, 2]
            old_oct = _compute_child_octant(opx, opy, opz, child_cx, child_cy, child_cz)
            node_children_flat[base_new + old_oct] = old_leaf
            node_parent[old_leaf] = new_internal
            # Accumulate the old leaf's mass/com into the new internal as well
            cuda.atomic.add(node_mass, new_internal, masses[old_pid])
            cuda.atomic.add(node_com_sum, (new_internal, 0), masses[old_pid] * opx)
            cuda.atomic.add(node_com_sum, (new_internal, 1), masses[old_pid] * opy)
            cuda.atomic.add(node_com_sum, (new_internal, 2), masses[old_pid] * opz)
        else:
            # Unexpected: should not encounter an internal here
            node_children_flat[base_new + 0] = old_leaf
            node_parent[old_leaf] = new_internal
        # Link the new internal into the parent slot
        node_children_flat[idx] = new_internal
        node_parent[new_internal] = current

        # Continue insertion for our leaf under the new internal
        current = new_internal
        cx, cy, cz = child_cx, child_cy, child_cz
        half = child_half
@cuda.jit
def _finalize_com_kernel(node_mass, node_com_sum, node_com):
    i = cuda.grid(1)
    if i >= node_mass.shape[0]:
        return
    m = node_mass[i]
    if m > 0.0:
        inv = 1.0 / m
        node_com[i, 0] = node_com_sum[i, 0] * inv
        node_com[i, 1] = node_com_sum[i, 1] * inv
        node_com[i, 2] = node_com_sum[i, 2] * inv


@cuda.jit(device=True)
def _push(stack, sp, val):
    stack[sp[0]] = val
    sp[0] += 1


@cuda.jit(device=True)
def _pop(stack, sp):
    sp[0] -= 1
    return stack[sp[0]]


@cuda.jit
def _force_traversal_kernel(positions, masses, forces, node_children_flat, node_com, node_half,
                            node_mass, num_particles, theta, epsilon, interaction_counts):
    i = cuda.grid(1)
    if i >= num_particles:
        return
    # Root is node 0
    px = positions[i, 0]
    py = positions[i, 1]
    pz = positions[i, 2]
    mi = masses[i]
    fx = 0.0
    fy = 0.0
    fz = 0.0
    interactions = 0
    # Simple stack
    stack = cuda.local.array(64, dtype=np.int32)
    sp = cuda.local.array(1, dtype=np.int32)
    sp[0] = 0
    _push(stack, sp, 0)
    while sp[0] > 0:
        node = _pop(stack, sp)
        # Leaf node
        if node >= (num_particles - 1):
            j = node - (num_particles - 1)
            if j != i:
                dx = positions[j, 0] - px
                dy = positions[j, 1] - py
                dz = positions[j, 2] - pz
                d2 = dx*dx + dy*dy + dz*dz + epsilon*epsilon
                inv_r3 = d2 ** (-1.5)
                f = mi * masses[j] * inv_r3
                fx += f * dx
                fy += f * dy
                fz += f * dz
                interactions += 1
            continue
        # Internal node
        cx = node_com[node, 0]
        cy = node_com[node, 1]
        cz = node_com[node, 2]
        dx = cx - px
        dy = cy - py
        dz = cz - pz
        d2 = dx*dx + dy*dy + dz*dz + epsilon*epsilon
        size = node_half[node] * 2.0
        if (size * size) / d2 < (theta * theta):
            m = node_mass[node]
            if m > 0.0:
                inv_r3 = d2 ** (-1.5)
                f = mi * m * inv_r3
                fx += f * dx
                fy += f * dy
                fz += f * dz
                interactions += 1
        else:
            base = node * 8
            for k in range(8):
                child = node_children_flat[base + k]
                if child != _EMPTY and sp[0] < 64:
                    _push(stack, sp, child)
    forces[i, 0] = fx
    forces[i, 1] = fy
    forces[i, 2] = fz
    interaction_counts[i] = interactions


def get_gflops(total_interactions, time_ms):
    """
    Estimates the GFLOPS for the Barnes-Hut calculation based on the actual number of interactions.
    """
    if time_ms == 0:
        return 0.0
    
    flops_per_interaction = 20
    total_flops = total_interactions * flops_per_interaction
    gflops = total_flops / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0
    return gflops

def calculate_forces(positions, masses, G, epsilon, theta=0.5, dtype=np.float64,
                     debug_tree_structure=False, **kwargs):
    num_particles = positions.shape[0]
    if num_particles <= 1:
        return {"forces": cp.zeros_like(positions), "timings": {}, "metrics": {"gflops": 0}}

    threads_per_block = 256
    timings = {}

    # Assume positions, masses are already on device (cp.ndarray)
    d_positions = positions

    # --- Timing for Bounding Box Calculation ---
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    
    min_coord = d_positions.min(axis=0)
    max_coord = d_positions.max(axis=0)
    box_center = (min_coord + max_coord) / 2.0
    box_size = (max_coord - min_coord).max().item()
    if box_size == 0:
        box_size = 1.0
    root_half = (box_size * 1.05) / 2.0

    end_event.record()
    end_event.synchronize()
    timings["bounding_box"] = cp.cuda.get_elapsed_time(start_event, end_event)

    num_internal = num_particles - 1
    max_nodes = num_particles + num_internal

    d_node_parent = cp.full(max_nodes, _EMPTY, dtype=cp.int32)
    d_node_children_flat = cp.full(max_nodes * 8, _EMPTY, dtype=cp.int32)
    d_node_children_view = d_node_children_flat.reshape(max_nodes, 8)
    d_node_center = cp.zeros((max_nodes, 3), dtype=cp.float64)
    d_node_half = cp.zeros(max_nodes, dtype=cp.float64)
    d_node_mass = cp.zeros(max_nodes, dtype=cp.float64)
    d_node_com_sum = cp.zeros((max_nodes, 3), dtype=cp.float64)
    d_node_com = cp.zeros((max_nodes, 3), dtype=cp.float64)
    d_next_internal = cp.zeros(1, dtype=cp.int32)

    # --- Timing for Tree Initialization ---
    start_event.record()
    blocks = (max_nodes + threads_per_block - 1) // threads_per_block
    _reset_tree_arrays_kernel[blocks, threads_per_block](
        d_node_parent, d_node_children_flat, d_node_center, d_node_half, d_next_internal,
        float(box_center[0].item()), float(box_center[1].item()), float(box_center[2].item()),
        float(root_half), max_nodes
    )
    end_event.record()
    end_event.synchronize()
    timings["tree_reset"] = cp.cuda.get_elapsed_time(start_event, end_event)

    # --- Timing for Octree Construction ---
    start_event.record()
    blocks_particles = (num_particles + threads_per_block - 1) // threads_per_block
    _build_octree_insertion_kernel[blocks_particles, threads_per_block](
        d_positions, masses, d_node_parent, d_node_children_flat, d_node_center, d_node_half,
        d_node_mass, d_node_com_sum, d_next_internal, num_particles, num_internal
    )
    end_event.record()
    end_event.synchronize()
    timings["tree_build"] = cp.cuda.get_elapsed_time(start_event, end_event)

    # --- Timing for Center of Mass Calculation ---
    start_event.record()
    blocks_nodes = (max_nodes + threads_per_block - 1) // threads_per_block
    _finalize_com_kernel[blocks_nodes, threads_per_block](d_node_mass, d_node_com_sum, d_node_com)
    end_event.record()
    end_event.synchronize()
    timings["com_finalize"] = cp.cuda.get_elapsed_time(start_event, end_event)

    # --- Timing for Force Calculation ---
    start_event.record()
    forces = cp.zeros_like(d_positions)
    interaction_counts = cp.zeros(num_particles, dtype=cp.int64)
    _force_traversal_kernel[blocks_particles, threads_per_block](
        d_positions, masses, forces, d_node_children_flat, d_node_com, d_node_half,
        d_node_mass, num_particles, theta, epsilon, interaction_counts
    )
    if G != 1.0:
        forces *= G
    end_event.record()
    end_event.synchronize()
    force_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    timings["force_calculation"] = force_time_ms
    
    total_interactions = int(cp.sum(interaction_counts))
    gflops = get_gflops(total_interactions, force_time_ms)
    
    result = {"forces": forces, "timings": timings, "metrics": {"gflops": gflops}}
    if debug_tree_structure:
        result["node_parent"] = d_node_parent
        result["node_children"] = d_node_children_view
        result["node_mass"] = d_node_mass
        result["node_center"] = d_node_center
        result["node_half"] = d_node_half
    return result


