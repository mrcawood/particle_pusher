import numpy as np
from numba import njit, prange, cuda
import time

# A constant to indicate an empty node or child
_EMPTY = -1

@njit
def _get_octant(particle_pos, node_center):
    """
    Determine which of the 8 octants a particle belongs to relative to a node's center.
    """
    octant = 0
    if particle_pos[0] >= node_center[0]: octant |= 1
    if particle_pos[1] >= node_center[1]: octant |= 2
    if particle_pos[2] >= node_center[2]: octant |= 4
    return octant

@njit
def _create_child_node(parent_node_idx, octant, next_node_idx, node_bounds):
    """
    Creates a new child node, calculates its bounding box, and returns its index.
    """
    child_node_idx = next_node_idx
    px_min, px_max, py_min, py_max, pz_min, pz_max = node_bounds[parent_node_idx]
    cx, cy, cz = (px_min + px_max) / 2, (py_min + py_max) / 2, (pz_min + pz_max) / 2
    
    if octant & 1: cx_min, cx_max = cx, px_max
    else:          cx_min, cx_max = px_min, cx
    if octant & 2: cy_min, cy_max = cy, py_max
    else:          cy_min, cy_max = py_min, cy
    if octant & 4: cz_min, cz_max = cz, pz_max
    else:          cz_min, cz_max = pz_min, cz
    
    node_bounds[child_node_idx] = np.array([cx_min, cx_max, cy_min, cy_max, cz_min, cz_max])
    return child_node_idx

@njit
def _insert_particle(
    current_node_idx, particle_idx, next_node_idx, positions,
    node_bounds, node_children, node_is_leaf, leaf_particle_index
):
    """
    Recursively inserts a particle into the octree. Returns the next available node index.
    """
    if not node_is_leaf[current_node_idx]:
        # It's an internal node, find the correct child and recurse or create a new leaf.
        node_center = (node_bounds[current_node_idx, 0:2].sum() / 2,
                       node_bounds[current_node_idx, 2:4].sum() / 2,
                       node_bounds[current_node_idx, 4:6].sum() / 2)
        
        octant = _get_octant(positions[particle_idx], node_center)
        child_idx = node_children[current_node_idx, octant]

        if child_idx == _EMPTY:
            new_child_idx = _create_child_node(current_node_idx, octant, next_node_idx, node_bounds)
            next_node_idx += 1
            node_children[current_node_idx, octant] = new_child_idx
            node_is_leaf[new_child_idx] = True
            leaf_particle_index[new_child_idx] = particle_idx
        else:
            next_node_idx = _insert_particle(child_idx, particle_idx, next_node_idx, positions, node_bounds, node_children, node_is_leaf, leaf_particle_index)
        
        return next_node_idx

    else:
        # It is a leaf, so it must be split into an internal node.
        node_is_leaf[current_node_idx] = False
        existing_particle_idx = leaf_particle_index[current_node_idx]
        leaf_particle_index[current_node_idx] = _EMPTY

        node_center = (node_bounds[current_node_idx, 0:2].sum() / 2,
                       node_bounds[current_node_idx, 2:4].sum() / 2,
                       node_bounds[current_node_idx, 4:6].sum() / 2)
        
        # Re-insert the existing particle.
        octant_old = _get_octant(positions[existing_particle_idx], node_center)
        child_idx_old = _create_child_node(current_node_idx, octant_old, next_node_idx, node_bounds)
        next_node_idx += 1
        node_children[current_node_idx, octant_old] = child_idx_old
        node_is_leaf[child_idx_old] = True
        leaf_particle_index[child_idx_old] = existing_particle_idx
        
        # Insert the new particle.
        next_node_idx = _insert_particle(current_node_idx, particle_idx, next_node_idx, positions, node_bounds, node_children, node_is_leaf, leaf_particle_index)
        return next_node_idx

@njit
def _compute_centers_of_mass_pass(node_idx, node_children, node_total_mass, node_center_of_mass, node_is_leaf, leaf_particle_index, positions, masses):
    """
    Recursively computes the center of mass for each node in a post-order traversal.
    """
    if node_is_leaf[node_idx]:
        particle_idx = leaf_particle_index[node_idx]
        if particle_idx != _EMPTY:
            node_total_mass[node_idx] = masses[particle_idx]
            node_center_of_mass[node_idx] = positions[particle_idx]
        return

    total_mass = 0.0
    weighted_pos = np.zeros(3, dtype=np.float64)
    for i in range(8):
        child_idx = node_children[node_idx, i]
        if child_idx != _EMPTY:
            _compute_centers_of_mass_pass(child_idx, node_children, node_total_mass, node_center_of_mass, node_is_leaf, leaf_particle_index, positions, masses)
            total_mass += node_total_mass[child_idx]
            weighted_pos += node_center_of_mass[child_idx] * node_total_mass[child_idx]

    if total_mass > 0:
        node_total_mass[node_idx] = total_mass
        node_center_of_mass[node_idx] = weighted_pos / total_mass

@cuda.jit(device=True)
def _calculate_force_on_particle_cuda(particle_idx, root_node_idx, positions, masses, G, epsilon, theta,
                                     node_bounds, node_center_of_mass, node_total_mass,
                                     node_children, node_is_leaf, leaf_particle_index):
    """
    This is a CUDA DEVICE function. It iteratively traverses the tree to calculate force
    using a manually-managed stack to avoid recursion.
    """
    force_x = 0.0
    force_y = 0.0
    force_z = 0.0

    # Manually managed stack for tree traversal. Max depth of 64 is sufficient for most cases.
    stack = cuda.local.array(shape=64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = root_node_idx
    stack_ptr += 1

    px = positions[particle_idx, 0]
    py = positions[particle_idx, 1]
    pz = positions[particle_idx, 2]

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        # If the node is a leaf, calculate direct force.
        is_leaf = node_is_leaf[node_idx]
        if is_leaf:
            other_particle_idx = leaf_particle_index[node_idx]
            if other_particle_idx != particle_idx and other_particle_idx != _EMPTY:
                rij_x = node_center_of_mass[node_idx, 0] - px
                rij_y = node_center_of_mass[node_idx, 1] - py
                rij_z = node_center_of_mass[node_idx, 2] - pz
                dist_sq = rij_x**2 + rij_y**2 + rij_z**2 + epsilon**2
                inv_dist_cubed = dist_sq**(-1.5)
                force_magnitude = G * masses[particle_idx] * node_total_mass[node_idx] * inv_dist_cubed
                force_x += force_magnitude * rij_x
                force_y += force_magnitude * rij_y
                force_z += force_magnitude * rij_z
            continue 

        # It's an internal node. Check the Barnes-Hut approximation condition.
        s = node_bounds[node_idx, 1] - node_bounds[node_idx, 0]
        rij_x = node_center_of_mass[node_idx, 0] - px
        rij_y = node_center_of_mass[node_idx, 1] - py
        rij_z = node_center_of_mass[node_idx, 2] - pz
        d_sq = rij_x**2 + rij_y**2 + rij_z**2

        if (s*s) < (d_sq * theta * theta):
            if d_sq > 0:
                inv_dist_cubed = d_sq**(-1.5)
                force_magnitude = G * masses[particle_idx] * node_total_mass[node_idx] * inv_dist_cubed
                force_x += force_magnitude * rij_x
                force_y += force_magnitude * rij_y
                force_z += force_magnitude * rij_z
        else:
            # Node is too close, push its children onto the stack.
            for i in range(8):
                child_idx = node_children[node_idx, i]
                if child_idx != _EMPTY:
                    if stack_ptr < 64:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1
    
    return force_x, force_y, force_z

@cuda.jit
def _calculate_forces_bh_cuda(positions, masses, G, epsilon, theta, forces, 
                             node_bounds, node_center_of_mass, node_total_mass, 
                             node_children, node_is_leaf, leaf_particle_index):
    """
    This is the main CUDA KERNEL. It is launched from the CPU.
    Each thread calculates the force for one particle.
    """
    i = cuda.grid(1)
    num_particles = positions.shape[0]
    
    if i < num_particles:
        root_node_idx = 0
        force_x, force_y, force_z = _calculate_force_on_particle_cuda(
            i, root_node_idx, positions, masses, G, epsilon, theta,
            node_bounds, node_center_of_mass, node_total_mass, 
            node_children, node_is_leaf, leaf_particle_index
        )
        forces[i, 0] = force_x
        forces[i, 1] = force_y
        forces[i, 2] = force_z

def calculate_forces(positions, masses, G, epsilon, theta=0.5):
    """
    The main interface function for the hybrid CUDA Barnes-Hut kernel.
    It builds the tree on the CPU and calculates forces on the GPU.
    """
    num_particles = positions.shape[0]
    if num_particles == 0:
        return np.zeros_like(positions)

    # --- 1. Initialize Tree Data Structures (on CPU) ---
    max_nodes = 2 * num_particles
    node_bounds = np.zeros((max_nodes, 6), dtype=np.float64)
    node_center_of_mass = np.zeros((max_nodes, 3), dtype=np.float64)
    node_total_mass = np.zeros(max_nodes, dtype=np.float64)
    node_children = np.full((max_nodes, 8), _EMPTY, dtype=np.int32)
    node_is_leaf = np.zeros(max_nodes, dtype=np.bool_)
    leaf_particle_index = np.full(max_nodes, _EMPTY, dtype=np.int32)
    
    # --- 2. Build Tree (on CPU) ---
    t0 = time.time()
    min_pos = np.min(positions, axis=0); max_pos = np.max(positions, axis=0)
    box_center = (min_pos + max_pos) / 2.0
    box_size = np.max(max_pos - min_pos) * 1.05
    half_size = box_size / 2.0
    node_bounds[0] = [box_center[0] - half_size, box_center[0] + half_size,
                      box_center[1] - half_size, box_center[1] + half_size,
                      box_center[2] - half_size, box_center[2] + half_size]
    
    root_node_idx = 0
    node_is_leaf[root_node_idx] = True
    # The first particle is inserted manually to initialize the root.
    if num_particles > 0:
        leaf_particle_index[root_node_idx] = 0
    next_node_idx = 1
    
    for i in range(1, num_particles):
        next_node_idx = _insert_particle(
            root_node_idx, i, next_node_idx, positions, node_bounds,
            node_children, node_is_leaf, leaf_particle_index
        )
    t1 = time.time()

    # --- 3. Compute Centers of Mass (on CPU) ---
    _compute_centers_of_mass_pass(
        root_node_idx, node_children, node_total_mass, node_center_of_mass,
        node_is_leaf, leaf_particle_index, positions, masses
    )
    t2 = time.time()

    # --- 4. Calculate Forces (on GPU) ---
    # Transfer data from host (CPU) to device (GPU)
    d_positions = cuda.to_device(positions)
    d_masses = cuda.to_device(masses)
    d_node_bounds = cuda.to_device(node_bounds)
    d_node_center_of_mass = cuda.to_device(node_center_of_mass)
    d_node_total_mass = cuda.to_device(node_total_mass)
    d_node_children = cuda.to_device(node_children)
    d_node_is_leaf = cuda.to_device(node_is_leaf)
    d_leaf_particle_index = cuda.to_device(leaf_particle_index)
    
    # Create an empty array on the device to store the results
    d_forces = cuda.device_array_like(positions)

    # Configure the kernel launch grid
    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block

    # Launch the kernel
    _calculate_forces_bh_cuda[blocks_per_grid, threads_per_block](
        d_positions, d_masses, G, epsilon, theta, d_forces,
        d_node_bounds, d_node_center_of_mass, d_node_total_mass,
        d_node_children, d_node_is_leaf, d_leaf_particle_index
    )
    
    # Copy the results back from device to host
    forces = d_forces.copy_to_host()
    t3 = time.time()
    
    print(f"    CUDA BH Timings: "
          f"Tree Build: {(t1 - t0)*1000:.2f}ms, "
          f"CoM Calc: {(t2 - t1)*1000:.2f}ms, "
          f"Force Calc (GPU): {(t3 - t2)*1000:.2f}ms")

    return forces
