# Plan for a Fully GPU-Resident Barnes-Hut Implementation

## Objective

Move the entire N-body calculation pipeline—including octree construction, center-of-mass calculation, and force calculation—to the GPU. This will eliminate CPU bottlenecks and costly CPU-GPU data transfers that currently dominate the simulation time, enabling significant performance scaling with a large number of particles.

## Key Challenges & Mitigation Strategy

1.  **Challenge: Algorithm Complexity.** Parallel algorithms for sorting and tree construction are non-trivial and fundamentally different from their serial counterparts.
    *   **Mitigation:** We will implement well-known, peer-reviewed parallel algorithms rather than inventing new ones. Each stage of the plan will focus on a standard parallel primitive, which we can verify independently.

2.  **Challenge: Debugging on the GPU.** Verifying the integrity of complex, pointer-based data structures like an octree entirely on the GPU is difficult.
    *   **Mitigation:** Our phased implementation is the core of our debugging strategy. After developing each GPU kernel, we will write a corresponding test. This involves copying the data from that stage back to the CPU and validating its correctness before proceeding to the next stage. This ensures a correct foundation for subsequent steps.

3.  **Challenge: Ecosystem Limitations.** Numba, while powerful, does not have a native, high-performance parallel sorting algorithm, which is critical for our plan.
    *   **Mitigation:** We will integrate the `CuPy` library into our project. CuPy provides a highly optimized `cupy.sort()` function that operates on GPU arrays and interoperates seamlessly with Numba kernels.

## The GPU Pipeline: A Phased Implementation

Our implementation will be a pipeline of GPU kernels where the output of one kernel serves as the input for the next, keeping all data resident on the GPU.

### Input Data
- Particle Positions (as a `cupy.ndarray`)
- Particle Masses (as a `cupy.ndarray`)

---

### 1. Morton Coding Kernel
-   **Input:** Particle positions.
-   **Action:** For each particle, a GPU kernel will calculate a 3D Morton code. This code is a single integer that linearizes the particle's 3D position along a Z-order space-filling curve, placing particles that are close in 3D space near each other in the 1D sorted list.
-   **Output:**
    -   A `cupy.ndarray` of Morton codes for all particles.
    -   A `cupy.ndarray` of original particle indices (`[0, 1, 2, ...]`).

---

### 2. Parallel Sort (via CuPy)
-   **Input:** The Morton codes and the array of particle indices.
-   **Action:** Use `cupy.sort()` to perform a key-value sort. The Morton codes are the keys, and the particle indices are the values. This reorders the particles based on their spatial location.
-   **Output:** A `cupy.ndarray` containing the particle indices, now sorted according to their Morton codes.

---

### 3. Parallel Tree Build Kernel
-   **Input:** The sorted list of particle indices.
-   **Action:** This kernel is the most complex. It will construct the octree structure in parallel from the sorted list. The core idea is that each thread can determine the parent-child relationships for a given node by examining the Morton codes of its neighbors in the sorted list.
-   **Output:** A complete octree data structure residing on the GPU. This will likely be represented by several arrays (e.g., `node_parent`, `node_children`, `node_is_leaf`).

---

### 4. Center of Mass (CoM) Calculation Kernel
-   **Input:** The GPU-resident octree and the original particle data (positions and masses).
-   **Action:** Perform a parallel "bottom-up" sweep of the tree. The CoM for each leaf node is the particle it contains. Then, in parallel passes, the CoM for each internal node is calculated from the CoM of its children.
-   **Output:** The octree data structure, now augmented with `node_center_of_mass` and `node_total_mass` arrays.

---

### 5. Force Calculation Kernel
-   **Input:** The complete, CoM-annotated octree and particle data.
-   **Action:** This kernel will be a slightly modified version of our existing `_calculate_force_on_particle_cuda` device function. Each GPU thread will traverse the GPU-resident tree to calculate the force on one particle.
-   **Output:** A `cupy.ndarray` containing the final calculated forces for each particle.
