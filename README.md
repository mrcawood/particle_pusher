# Particle Pusher: A Python HPC Tutorial

This project is a simple N-body particle simulation designed to be a practical, hands-on example for a tutorial on High-Performance Computing (HPC) with Python. It demonstrates a clear progression of performance optimization techniques, from basic scientific computing libraries to advanced parallelization and algorithmic improvements.

The primary driver is `benchmark.py`, which allows for interactively running the simulation with different computational "kernels."

---

## How to Run the Simulation

All simulations are run from the command line using `benchmark.py`. The key options are:

*   `--kernel`: (Required) Specifies which computational kernel to use.
*   `--particles`: The number of particles to simulate (e.g., `1000`).
*   `--steps`: The number of time steps to run the simulation for (e.g., `20`).
*   `--animate`: (Optional) If included, displays a real-time 2D animation of the simulation (requires `matplotlib`).

### Example Commands

```bash
# Run with the basic NumPy kernel for 1000 particles
python benchmark.py --kernel numpy --particles 1000 --steps 20

# Run with the Numba serial-optimized kernel
python benchmark.py --kernel numba_serial --particles 1000 --steps 20

# Run with the Numba parallel-optimized kernel
python benchmark.py --kernel numba_parallel --particles 1000 --steps 20

# Run with the advanced Barnes-Hut algorithm for a large number of particles
python benchmark.py --kernel barnes_hut --particles 20000 --steps 20

# Run with the hybrid CPU/GPU Barnes-Hut kernel
python benchmark.py --kernel cuda_bh --particles 50000 --steps 20

# Run with animation (best with fewer particles to see movement)
python benchmark.py --kernel barnes_hut --particles 500 --steps 100 --animate
```

---

## Tutorial Narrative & Kernel Breakdown

This section provides a narrative that can be used for a tutorial presentation, explaining the purpose and performance characteristics of each kernel.

### Part 1: The Baseline - `numpy`

*   **Kernel:** `kernels/numpy_kernel.py`
*   **Concept:** This is our starting point. It uses the `NumPy` library, the foundation of scientific computing in Python. The force calculation is "vectorized," meaning we perform operations on entire arrays at once, avoiding slow, explicit `for` loops in Python.
*   **Demonstration:** Run with a moderate number of particles (`--particles 1000`). It's reasonably fast, but as you increase the particle count, its O(n²) complexity becomes apparent. The runtime will increase quadratically with the number of particles.

### Part 2: Easy Speed - `numba_serial`

*   **Kernel:** `kernels/numba_serial_kernel.py`
*   **Concept:** What if we need more performance but want to write simple, clear `for` loops? This is where `Numba` comes in. This kernel uses explicit `for` loops, which are typically slow in Python. However, by adding the `@njit` (nopython JIT) decorator, Numba compiles this Python code into highly optimized, C-like machine code just before it runs.
*   **Demonstration:** Compare `numba_serial` to `numpy` with the same particle count. You'll often find that Numba's JIT-compiled loops can be even faster than NumPy's generalized vectorized operations for this specific problem.

### Part 3: Unlocking Your CPU - `numba_parallel`

*   **Kernel:** `kernels/numba_parallel_kernel.py`
*   **Concept:** Modern CPUs have multiple cores, but standard Python code only uses one. Numba makes it trivial to unlock this power. By simply adding `parallel=True` to the decorator (`@njit(parallel=True)`) and changing `range` to `prange`, Numba automatically parallelizes the outer loop across all available CPU cores.
*   **Demonstration:** Run `numba_parallel` and compare it to `numba_serial` with a high particle count (e.g., `--particles 5000`). The speedup should be significant, often scaling with the number of physical cores in the machine. This demonstrates the power of data parallelism.

### Part 4: A Smarter Algorithm - `barnes_hut`

*   **Kernel:** `kernels/barnes_hut_kernel.py`
*   **Concept:** So far, we've only optimized *how* we do the calculation. Now, we change *what* we calculate. The previous kernels were all O(n²), calculating every single particle-particle interaction. The Barnes-Hut algorithm is a more intelligent, O(n log n) approximation. It groups distant particles into clusters and treats them as a single "macro-particle," saving enormous amounts of computation.
*   **Demonstration:** This is the key takeaway.
    1.  Run `numba_parallel` and `barnes_hut` with a small number of particles (`--particles 1000`). Observe that **Barnes-Hut is slower**. The overhead of building its octree data structure isn't worth it for small `n`.
    2.  Now, run both kernels with a large number of particles (`--particles 20000`). Observe that **Barnes-Hut is now significantly faster**. This perfectly illustrates the concept of algorithmic complexity and crossover points in performance. It's the most powerful optimization we've made.

### Part 5: Hybrid Computing - `cuda_bh`

*   **Kernel:** `kernels/cuda_bh_kernel.py`
*   **Concept:** This kernel introduces a powerful, real-world HPC concept: **hybrid computing**. Not all parts of a problem are suited for the same processor. Here, we use a hybrid strategy that leverages the strengths of both the CPU and the GPU.
    *   The complex, sequential task of building the Barnes-Hut octree remains on the **CPU**, where its branching logic is handled efficiently by Numba's JIT compiler.
    *   The massively parallel task of calculating the forces on thousands of particles is offloaded to the **GPU**. We use Numba's CUDA backend to write a GPU kernel that calculates the force for every particle simultaneously.
*   **Demonstration:** Run `numba_parallel` and `cuda_bh` with a very large number of particles (`--particles 50000`). The CPU is still doing the smart algorithmic work of building the tree, but the GPU is crushing the number-crunching part of the force calculation. This demonstrates how to use the right tool for the right job to achieve performance that neither the CPU nor the GPU could manage alone.

