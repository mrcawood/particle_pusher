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

# Run with the Numba parallel-optimized kernel
python benchmark.py --kernel numba_parallel --particles 10000 --steps 20

# Run with the advanced Barnes-Hut algorithm
python benchmark.py --kernel barnes_hut --particles 50000 --steps 20

# Run with the hybrid CPU/GPU Barnes-Hut kernel
python benchmark.py --kernel cuda_bh --particles 100000 --steps 20

# Run with the fully GPU-resident Barnes-Hut kernel for a massive simulation
python benchmark.py --kernel cuda_full_bh --particles 1000000 --steps 20
```

---

## Performance Results

The following table shows the average time per step for a simulation with **1 million particles**, demonstrating the effectiveness of each optimization stage.

| Kernel | Description | Avg. Time per Step | Speedup (vs. CPU BH) |
| :--- | :--- | :---: | :---: |
| `barnes_hut` | Parallel CPU (Numba) | ~12.06 s | 1x |
| `cuda_bh` | Hybrid (CPU Tree + GPU Force) | ~2.79 s | **4.3x** |
| `cuda_full_bh` | Full GPU (GPU Tree + GPU Force) | **~0.03 s** | **~400x** |

---

## Tutorial Narrative & Kernel Breakdown

This section provides a narrative that can be used for a tutorial presentation, explaining the purpose and performance characteristics of each kernel.

### Part 1: The Baseline - `numpy`

*   **Kernel:** `kernels/numpy_kernel.py`
*   **Concept:** This is our starting point. It uses `NumPy` for "vectorized" calculations, avoiding slow Python loops.
*   **Demonstration:** Fast for a small number of particles, but its O(n²) complexity makes it slow for large simulations.

### Part 2: Easy Speed - `numba_serial`

*   **Kernel:** `kernels/numba_serial_kernel.py`
*   **Concept:** Uses Numba's `@njit` decorator to compile simple Python `for` loops into highly optimized machine code.
*   **Demonstration:** Often faster than NumPy for this specific problem, showing the power of Just-in-Time (JIT) compilation.

### Part 3: Unlocking Your CPU - `numba_parallel`

*   **Kernel:** `kernels/numba_parallel_kernel.py`
*   **Concept:** Trivial to implement with Numba (`parallel=True` and `prange`), this kernel utilizes all available CPU cores.
*   **Demonstration:** Shows a significant speedup over the serial version, demonstrating the power of data parallelism on a multi-core CPU.

### Part 4: A Smarter Algorithm - `barnes_hut`

*   **Kernel:** `kernels/barnes_hut_kernel.py`
*   **Concept:** Changes *what* we calculate. The Barnes-Hut algorithm is an O(n log n) approximation that groups distant particles, saving enormous amounts of computation compared to the O(n²) direct-force method.
*   **Demonstration:** Illustrates the power of algorithmic complexity. While slower for small `n` due to tree-building overhead, it is dramatically faster for large `n`.

### Part 5: Hybrid Computing - `cuda_bh`

*   **Kernel:** `kernels/cuda_bh_kernel.py`
*   **Concept:** Introduces a hybrid CPU/GPU strategy. The complex, serial-logic task of building the octree remains on the CPU, while the massively parallel task of force calculation is offloaded to the GPU.
*   **Demonstration:** Shows a significant performance leap by using the right tool for the right job, but reveals that the CPU tree-build is now the primary bottleneck.

### Part 6: The Summit - `cuda_full_bh`

*   **Kernel:** `kernels/cuda_full_bh_kernel.py`
*   **Concept:** The culmination of our optimization efforts. The entire simulation pipeline—Morton coding, sorting, parallel tree construction, and force calculation—is performed entirely on the GPU. This eliminates the CPU bottleneck and minimizes data transfer between the CPU and GPU.
*   **Demonstration:** The performance results speak for themselves. By moving the entire workload to the GPU, we achieve a **~93x speedup** over the hybrid kernel and a **~400x speedup** over the highly optimized parallel CPU version for a 1-million-particle simulation, effectively utilizing the massive parallelism of the hardware.
