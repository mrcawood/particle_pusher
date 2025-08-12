import numpy as np
from numba import njit, prange
import time

_EMPTY = -1

@njit
def _get_octant(particle_pos, node_center):
    octant = 0
    if particle_pos[0] >= node_center[0]: octant |= 1
    if particle_pos[1] >= node_center[1]: octant |= 2
    if particle_pos[2] >= node_center[2]: octant |= 4
    return octant

@njit
def _create_child_node(parent_node_idx, octant, next_node_idx, node_bounds, node_parent):
    child_node_idx = next_node_idx
    node_parent[child_node_idx] = parent_node_idx
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
def _insert_particle(current_node_idx, particle_idx, next_node_idx, positions,
                   node_bounds, node_children, node_is_leaf, leaf_particle_index, node_parent):
    if not node_is_leaf[current_node_idx]:
        node_center = (node_bounds[current_node_idx, 0:2].sum() / 2,
                       node_bounds[current_node_idx, 2:4].sum() / 2,
                       node_bounds[current_node_idx, 4:6].sum() / 2)
        
        octant = _get_octant(positions[particle_idx], node_center)
        child_idx = node_children[current_node_idx, octant]

        if child_idx == _EMPTY:
            new_child_idx = _create_child_node(current_node_idx, octant, next_node_idx, node_bounds, node_parent)
            next_node_idx += 1
            node_children[current_node_idx, octant] = new_child_idx
            node_is_leaf[new_child_idx] = True
            leaf_particle_index[new_child_idx] = particle_idx
        else:
            next_node_idx = _insert_particle(child_idx, particle_idx, next_node_idx, positions, node_bounds, node_children, node_is_leaf, leaf_particle_index, node_parent)
        
        return next_node_idx

    else:
        node_is_leaf[current_node_idx] = False
        existing_particle_idx = leaf_particle_index[current_node_idx]
        leaf_particle_index[current_node_idx] = _EMPTY

        node_center = (node_bounds[current_node_idx, 0:2].sum() / 2,
                       node_bounds[current_node_idx, 2:4].sum() / 2,
                       node_bounds[current_node_idx, 4:6].sum() / 2)
        
        # Place existing particle in a new child node
        octant_old = _get_octant(positions[existing_particle_idx], node_center)
        child_idx_old = _create_child_node(current_node_idx, octant_old, next_node_idx, node_bounds, node_parent)
        next_node_idx += 1
        node_children[current_node_idx, octant_old] = child_idx_old
        node_is_leaf[child_idx_old] = True
        leaf_particle_index[child_idx_old] = existing_particle_idx
        
        # Re-insert the new particle into the current node (which is now an internal node)
        next_node_idx = _insert_particle(current_node_idx, particle_idx, next_node_idx, positions, node_bounds, node_children, node_is_leaf, leaf_particle_index, node_parent)
        return next_node_idx

@njit
def _compute_centers_of_mass_pass(node_idx, node_children, node_total_mass, node_center_of_mass, node_is_leaf, leaf_particle_index, positions, masses):
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
    (node_bounds, node_center_of_mass, node_total_mass, node_children, node_is_leaf, leaf_particle_index) = tree_data
    
    force = np.zeros(3, dtype=np.float64)
    interactions = 0

    if node_is_leaf[node_idx]:
        other_particle_idx = leaf_particle_index[node_idx]
        if other_particle_idx != particle_idx and other_particle_idx != _EMPTY:
            rij = node_center_of_mass[node_idx] - positions[particle_idx]
            dist_sq = np.sum(rij**2) + epsilon**2
            inv_d = dist_sq**(-0.5)  # Equivalent to 1.0 / sqrt(d_sq)
            inv_dist_cubed = inv_d * inv_d * inv_d
            force_magnitude = G * masses[particle_idx] * node_total_mass[node_idx] * inv_dist_cubed
            interactions += 1
            return force_magnitude * rij, interactions
        return force, interactions

    s = node_bounds[node_idx, 1] - node_bounds[node_idx, 0]
    rij = node_center_of_mass[node_idx] - positions[particle_idx]
    d_sq = np.sum(rij**2)
    
    if (s*s / d_sq) < (theta * theta):
        inv_d = d_sq**(-0.5)  # Equivalent to 1.0 / sqrt(d_sq)
        inv_dist_cubed = inv_d * inv_d * inv_d
        force_magnitude = G * masses[particle_idx] * node_total_mass[node_idx] * inv_dist_cubed
        interactions += 1
        return force_magnitude * rij, interactions
    else:
        for i in range(8):
            child_idx = node_children[node_idx, i]
            if child_idx != _EMPTY:
                child_force, child_interactions = _calculate_force_on_particle(particle_idx, child_idx, positions, masses, G, epsilon, theta, tree_data)
                force += child_force
                interactions += child_interactions
        return force, interactions

@njit(parallel=True)
def _calculate_forces_bh(positions, masses, G, epsilon, theta, tree_data):
    num_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    interaction_counts = np.zeros(num_particles, dtype=np.int64)
    root_node_idx = 0

    for i in prange(num_particles):
        forces[i], interaction_counts[i] = _calculate_force_on_particle(i, root_node_idx, positions, masses, G, epsilon, theta, tree_data)
    
    return forces, np.sum(interaction_counts)

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

def calculate_forces(positions, masses, G, epsilon, theta=0.5, debug_tree_structure=False):
    num_particles = positions.shape[0]
    if num_particles == 0:
        return {
            "forces": np.zeros_like(positions),
            "timings": {},
            "metrics": {"gflops": 0}
        }

    t0 = time.time()
    max_nodes = 8 * num_particles 
    node_bounds = np.zeros((max_nodes, 6), dtype=np.float64)
    node_center_of_mass = np.zeros((max_nodes, 3), dtype=np.float64)
    node_total_mass = np.zeros(max_nodes, dtype=np.float64)
    node_children = np.full((max_nodes, 8), _EMPTY, dtype=np.int32)
    node_is_leaf = np.zeros(max_nodes, dtype=np.bool_)
    leaf_particle_index = np.full(max_nodes, _EMPTY, dtype=np.int32)
    node_parent = np.full(max_nodes, _EMPTY, dtype=np.int32)
    
    min_pos = np.min(positions, axis=0); max_pos = np.max(positions, axis=0)
    box_center = (min_pos + max_pos) / 2.0
    box_size = np.max(max_pos - min_pos) * 1.05
    half_size = box_size / 2.0
    node_bounds[0] = [box_center[0] - half_size, box_center[0] + half_size,
                      box_center[1] - half_size, box_center[1] + half_size,
                      box_center[2] - half_size, box_center[2] + half_size]
    
    root_node_idx = 0
    node_parent[root_node_idx] = _EMPTY
    node_is_leaf[root_node_idx] = True
    if num_particles > 0:
        leaf_particle_index[root_node_idx] = 0
    next_node_idx = 1
    
    for i in range(1, num_particles):
        next_node_idx = _insert_particle(
            root_node_idx, i, next_node_idx, positions, node_bounds,
            node_children, node_is_leaf, leaf_particle_index, node_parent
        )
    t1 = time.time()

    _compute_centers_of_mass_pass(
        root_node_idx, node_children, node_total_mass, node_center_of_mass,
        node_is_leaf, leaf_particle_index, positions, masses
    )
    t2 = time.time()

    tree_data = (node_bounds, node_center_of_mass, node_total_mass, node_children, node_is_leaf, leaf_particle_index)
    forces, interaction_count = _calculate_forces_bh(positions, masses, G, epsilon, theta, tree_data)
    t3 = time.time()
    
    force_time_ms = (t3 - t2) * 1000
    gflops = get_gflops(interaction_count, force_time_ms)

    timings = {
        "tree_build": (t1 - t0) * 1000,
        "com_calc": (t2 - t1) * 1000,
        "force_calculation": force_time_ms
    }

    result = {
        "forces": forces,
        "timings": timings,
        "metrics": {"gflops": gflops}
    }
    
    if debug_tree_structure:
        result["tree_data"] = {
            "node_parent": node_parent,
            "node_children": node_children,
            "next_node_idx": next_node_idx
        }
    
    return result
