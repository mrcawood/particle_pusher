import numpy as np
from numba import njit, prange
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

@njit
def _calculate_force_on_particle(particle_idx, node_idx, positions, masses, G, epsilon, theta, tree_data):
    """
    Recursively traverses the tree to calculate the force on a single particle.
    """
    (node_bounds, node_center_of_mass, node_total_mass, node_children, node_is_leaf, leaf_particle_index) = tree_data
    
    force = np.zeros(3, dtype=np.float64)

    # If the node is a leaf and not the particle itself, calculate direct force.
    if node_is_leaf[node_idx]:
        other_particle_idx = leaf_particle_index[node_idx]
        if other_particle_idx != particle_idx and other_particle_idx != _EMPTY:
            rij = node_center_of_mass[node_idx] - positions[particle_idx]
            dist_sq = np.sum(rij**2) + epsilon**2
            inv_dist_cubed = dist_sq**(-1.5)
            force_magnitude = G * masses[particle_idx] * node_total_mass[node_idx] * inv_dist_cubed
            return force_magnitude * rij
        return force

    # It's an internal node. Check the Barnes-Hut approximation condition.
    s = node_bounds[node_idx, 1] - node_bounds[node_idx, 0] # width of the node
    rij = node_center_of_mass[node_idx] - positions[particle_idx]
    d_sq = np.sum(rij**2)
    
    if (s*s / d_sq) < (theta * theta):
        # Node is far enough away, approximate it as a single macro-particle.
        inv_dist_cubed = d_sq**(-1.5)
        force_magnitude = G * masses[particle_idx] * node_total_mass[node_idx] * inv_dist_cubed
        return force_magnitude * rij
    else:
        # Node is too close, must recurse into its children.
        for i in range(8):
            child_idx = node_children[node_idx, i]
            if child_idx != _EMPTY:
                force += _calculate_force_on_particle(particle_idx, child_idx, positions, masses, G, epsilon, theta, tree_data)
        return force

@njit(parallel=True)
def _calculate_forces_bh(positions, masses, G, epsilon, theta, tree_data):
    """
    Calculates forces on all particles by traversing the Barnes-Hut tree in parallel.
    """
    num_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    root_node_idx = 0

    for i in prange(num_particles):
        forces[i] = _calculate_force_on_particle(i, root_node_idx, positions, masses, G, epsilon, theta, tree_data)
    
    return forces

def calculate_forces(positions, masses, G, epsilon, theta=0.5):
    """
    The main interface function for the Barnes-Hut kernel.
    """
    num_particles = positions.shape[0]
    if num_particles == 0:
        return np.zeros_like(positions)

    # --- 1. Initialize Tree Data Structures ---
    max_nodes = 2 * num_particles
    node_bounds = np.zeros((max_nodes, 6), dtype=np.float64)
    node_center_of_mass = np.zeros((max_nodes, 3), dtype=np.float64)
    node_total_mass = np.zeros(max_nodes, dtype=np.float64)
    node_children = np.full((max_nodes, 8), _EMPTY, dtype=np.int32)
    node_is_leaf = np.zeros(max_nodes, dtype=np.bool_)
    leaf_particle_index = np.full(max_nodes, _EMPTY, dtype=np.int32)
    
    # --- 2. Build Tree ---
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
    leaf_particle_index[root_node_idx] = 0
    next_node_idx = 1
    
    for i in range(1, num_particles):
        next_node_idx = _insert_particle(
            root_node_idx, i, next_node_idx, positions, node_bounds,
            node_children, node_is_leaf, leaf_particle_index
        )
    t1 = time.time()

    # --- 3. Compute Centers of Mass ---
    _compute_centers_of_mass_pass(
        root_node_idx, node_children, node_total_mass, node_center_of_mass,
        node_is_leaf, leaf_particle_index, positions, masses
    )
    t2 = time.time()

    # --- 4. Calculate Forces ---
    tree_data = (node_bounds, node_center_of_mass, node_total_mass, node_children, node_is_leaf, leaf_particle_index)
    forces = _calculate_forces_bh(positions, masses, G, epsilon, theta, tree_data)
    t3 = time.time()
    
    print(f"    Barnes-Hut Timings: "
          f"Tree Build: {(t1 - t0)*1000:.2f}ms, "
          f"CoM Calc: {(t2 - t1)*1000:.2f}ms, "
          f"Force Calc: {(t3 - t2)*1000:.2f}ms")

    return forces
