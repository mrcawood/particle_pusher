### MPI Implementation Plan for N-Body Simulation

#### 1. Objective

To extend the existing single-GPU Barnes-Hut N-body simulation to run efficiently on multiple GPUs, potentially across multiple nodes, using the Message Passing Interface (MPI). The primary goal is to scale the simulation to handle a larger number of particles than can be managed by a single GPU.

#### 2. Core Strategy: Spatial Domain Decomposition

The parallelization will be based on a **spatial domain decomposition** strategy. This approach is well-suited to the existing Barnes-Hut algorithm and its use of Morton codes.

*   **Partitioning:** Particles will be distributed among MPI ranks according to their Morton codes. This ensures that each rank is responsible for a spatially contiguous region of the simulation domain, which is key to minimizing communication.
*   **Force Calculation:** A **"Locally Essential Tree" (LET)** approach will be implemented. Each rank will build its own local octree. To calculate forces, ranks will exchange a summary of their local trees (i.e., the top-level nodes). For interactions near the boundaries of other domains, more detailed node information will be requested and exchanged as needed.

#### 3. Phased Implementation and Validation

The project will be broken down into distinct phases. Each phase has specific development goals and validation points to ensure the implementation is correct before we proceed to the next, more complex stage.

---

### **Phase 1: MPI Scaffolding and Basic Data Distribution**

**Goal:** Integrate MPI into the application and distribute the initial particle data from a root rank to all other ranks.

*   **Development Steps:**
    1.  Modify `pusher.py` to initialize and finalize the MPI environment.
    2.  Assign a unique GPU device to each MPI rank to ensure ranks do not compete for the same hardware.
    3.  Refactor the code to ensure all console output and file I/O operations are performed only by rank 0.
    4.  Implement a basic data distribution mechanism:
        *   Rank 0 will create or load the full particle dataset.
        *   Rank 0 will use `comm.scatter` to send a unique, contiguous chunk of the particle arrays to every other rank.

*   **Validation Points:**
    1.  **Single-Rank Sanity Check:** Run the code with `mpirun -n 1 ...`. The simulation should run correctly and produce the same results as the original, non-MPI version.
    2.  **Multi-Rank Distribution Check:** Run with multiple ranks (`-n 4`, for instance). Add temporary print statements to verify that each rank receives the correct number of particles and is running on its assigned GPU.

---

### **Phase 2: Global Morton Sort and Particle Redistribution**

**Goal:** Sort all particles across all ranks based on their Morton codes to achieve spatial locality.

*   **Development Steps:**
    1.  **Global Bounding Box:** To ensure Morton codes are consistent across all ranks, a global bounding box must be computed. This will involve each rank finding its local min/max coordinates, followed by an `MPI_Allreduce` operation to establish the global min and max coordinates for all particles.
    2.  **Local Morton Code Calculation:** Each rank will use the existing CUDA kernel to calculate Morton codes for its local particles, using the global bounding box.
    3.  **Global Sort (Initial Version):** We will start with a simple, centralized parallel sort:
        *   All ranks will `gather` their local particles and Morton codes onto rank 0.
        *   Rank 0 will concatenate the data, perform a global sort based on the Morton codes, and then `scatter` the newly sorted particles back to all ranks. Each rank will now hold a globally sorted, contiguous block of particles.

*   **Validation Points:**
    1.  **Code Integrity:** Add a `--save-morton-codes` debug flag that saves the Morton codes from each rank to a separate file (`codes_rank_0.npy`, `codes_rank_1.npy`, etc.).
    2.  **Verification Script:** Create a simple, separate Python script to load these files. The script will concatenate the codes and assert that the resulting global array is monotonically increasing. This confirms the global sort is working correctly.

---

### **Phase 3: Local Tree Construction**

**Goal:** Ensure each rank can independently build a valid octree from its local, sorted particle data.

*   **Development Steps:**
    1.  The existing GPU-based tree-building kernels (`_build_tree_topology_kernel`, `_com_sweep_kernel`, etc.) will be used on each rank. No major modifications to these kernels should be necessary, as they will simply operate on the local data.

*   **Validation Points:**
    1.  At this stage, force calculations will be *incorrect* because they don't account for remote particles. The primary validation is to confirm that tree construction completes without errors on all ranks.
    2.  We can add logging to check that the size and depth of the tree on each rank are reasonable given the number of particles it owns.

---

### **Phase 4: Distributed Force Calculation (Locally Essential Tree)**

**Goal:** Correctly calculate forces by accounting for interactions between particles on different ranks. This is the most critical and complex phase.

*   **Development Steps:**
    1.  **Tree Data Exchange:** Implement the communication for ranks to exchange the high-level nodes of their local trees. This data (e.g., center of mass, total mass, bounding box of key nodes) will be broadcast to form the "Locally Essential Tree" for each rank.
    2.  **Modify Force Kernel:** The main force calculation kernel (`_calculate_force_on_particle_gpu`) must be extended. Its traversal logic will now have to iterate through both its own local tree and the collected summary nodes from remote ranks.
    3.  **(Advanced) Direct Interaction for Boundary Conditions:** For cases where a particle is too close to a remote domain for the tree approximation to be valid, a more direct interaction is needed. This will likely require a second, more targeted communication step where ranks can request specific, fine-grained node or particle data from other ranks. This can be deferred until after the main LET logic is working.

*   **Validation Points:**
    1.  **Conservation Laws:** The most important validation. Run a long simulation and plot the total energy and momentum of the system. These values should be conserved to a high degree of precision. Any significant drift points to a bug in the force calculation.
    2.  **Comparative Analysis:** Run a small-N simulation and compare the final force arrays computed by the MPI version against the single-GPU version. The results should be identical within floating-point error tolerances.

---

### **Phase 5: Performance Optimization**

**Goal:** Profile and optimize the MPI implementation to ensure it scales effectively.

*   **Development Steps:**
    1.  **Asynchronous Communication:** Replace blocking MPI calls (`Send`, `Recv`) with their non-blocking counterparts (`Isend`, `Irecv`) to overlap communication with computation wherever possible.
    2.  **Improved Parallel Sort:** If the centralized sort from Phase 2 proves to be a bottleneck, replace it with a more scalable, fully decentralized parallel sorting algorithm (e.g., a sample sort or radix sort).
    3.  **Profiling:** Use `nsys`, `ncu`, and MPI-specific profiling tools to identify and address bottlenecks.

*   **Validation Points:**
    1.  **Scalability Benchmarking:** Measure the wall-clock time of the simulation with a fixed number of particles per rank, while increasing the number of ranks. Plot the results to analyze the weak scaling of the implementation.
    2.  **Result Verification:** Ensure that all optimizations do not introduce errors. The results must continue to match the unoptimized version.
