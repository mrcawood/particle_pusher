# Particle Pusher: A Python HPC Tutorial

This project is a simple N-body particle simulation designed to be a practical, hands-on example for a tutorial on High-Performance Computing (HPC) with Python. It demonstrates a clear progression of performance optimization techniques, from basic NumPy to advanced GPU parallelism and algorithmic improvements.

The primary driver is `pusher.py`, which allows for interactively running the simulation with different computational "kernels."

---

## How to Run the Simulation

All simulations are run from the command line using `pusher.py`. The key options are:

*   `--kernel`: (Required) Specifies which computational kernel to use.
*   `--particles`: The number of particles to simulate (e.g., `10000`).
*   `--steps`: The number of time steps to run the simulation for (e.g., `50`).
*   `--animate`: (Optional) Displays a real-time 2D animation (requires `matplotlib`).

### Example Commands

```bash
# Run with the basic NumPy kernel for 10,000 particles
python pusher.py --kernel numpy --particles 10000 --steps 20

# Run with the Numba parallel-optimized kernel
python pusher.py --kernel numba_parallel --particles 50000 --steps 20

# Run with the advanced Barnes-Hut algorithm on the CPU
python pusher.py --kernel barnes_hut --particles 100000 --steps 20

# Run with the fully GPU-resident, optimized Barnes-Hut kernel
python pusher.py --kernel cuda_optimized --particles 500000 --steps 20
```

---

## Performance Metrics

The benchmark script reports several key metrics:

*   **Average Step Time**: The mean time taken for a single simulation step, in milliseconds.
*   **Particle Updates/sec**: The number of particles simulated per second of wall-clock time (`num_particles / avg_step_time`). This is a direct measure of the simulation's algorithmic throughput.
*   **Estimated GFLOPS**: An estimate of the number of Giga-Floating-Point-Operations per Second. This provides a standardized way to compare the computational work done by each kernel, independent of the algorithm's complexity.

The output will also include a breakdown of the average time spent in different sub-tasks for the more complex kernels (e.g., Tree Build, Force Calculation).

---

## Tutorial Narrative & Kernel Breakdown

This section provides a narrative that can be used for a tutorial presentation, explaining the purpose and performance characteristics of each kernel.

### Part 1: The Obvious (and Slow) Approach - `loop`

*   **Kernel:** `kernels/01_loop_kernel.py`
*   **Concept:** This is the most straightforward implementation, using nested Python `for` loops to calculate the force between every pair of particles. It represents the "naive" or "obvious" solution that a beginner might write.
*   **Demonstration:** This kernel serves as a critical baseline. Its performance is extremely poor because interpreted Python loops carry significant overhead. It teaches the most fundamental lesson in scientific Python: for performance-critical code, avoid explicit Python loops over large arrays whenever possible. This motivates the need for optimized solutions like NumPy.

### Part 2: The Baseline - `numpy`

*   **Kernel:** `kernels/02_numpy_kernel.py`
*   **Concept:** This is our starting point. It uses `NumPy` for "vectorized" calculations, a common practice in scientific Python. It's clean, readable, and avoids explicit `for` loops in Python.
*   **Demonstration:** Its O(N²) complexity means the workload grows quadratically with the number of particles. It's efficient for small N, but quickly becomes a bottleneck, teaching the limits of simple vectorization.

### Part 3: Easy Speed - `numba_serial`

*   **Kernel:** `kernels/03_numba_serial_kernel.py`
*   **Concept:** Uses Numba's `@njit` decorator to compile Python `for` loops into highly optimized machine code. This is often the first and easiest step in accelerating numerical Python code.
*   **Demonstration:** This kernel significantly outperforms the `numpy` version, showcasing the power of Numba for loop-heavy numerical tasks. The performance gain comes from several factors:
    *   **Just-in-Time (JIT) Compilation:** The `@njit` decorator compiles the explicit Python `for` loops into highly optimized, low-level machine code, completely removing the overhead of the Python interpreter from the calculation.
    *   **Memory & Algorithmic Efficiency:** Unlike the `numpy` kernel, which creates massive intermediate arrays that can strain memory bandwidth, this version uses a C-style loop (`for j in range(i + 1, ...)`). This approach is far more memory-efficient and allows for an important algorithmic optimization: by using Newton's third law (`F_ij = -F_ji`), we only compute the force for each unique pair of particles *once*, effectively halving the number of calculations. This also leads to much better CPU cache utilization.
    *   This teaches a key lesson: for algorithms that are difficult or inefficient to express in a purely vectorized form, JIT compilation with Numba offers a simple and powerful path to high performance.

### Part 4: Unlocking Your CPU - `numba_parallel`

*   **Kernel:** `kernels/04_numba_parallel_kernel.py`
*   **Concept:** With a simple `parallel=True` argument and `prange`, Numba can automatically parallelize the loops across all available CPU cores.
*   **Demonstration:** This step shows a significant speedup over the serial version, highlighting the importance of data parallelism and leveraging the multi-core architecture of modern CPUs.

*   **Scaling Analysis:** This kernel also provides a perfect opportunity to explore the difference between **algorithmic throughput** and **hardware throughput**. Because it is an `O(N²)` algorithm, its performance characteristics change dramatically with the problem size.

    Consider these two runs:

    ```bash
    # 10,000 particles
    python pusher.py --kernel numba_parallel --particles 10000
    --- NUMBA_PARALLEL BENCHMARK RESULTS (FP64) ---
    Particles: 10000, Steps: 5
    Average Step Time: 9.91ms
    Particle Updates/sec: 1.01e+06
    Estimated GFLOPS: 102.30
    --------------------------------------

    # 100,000 particles
    python pusher.py --kernel numba_parallel --particles 100000
    --- NUMBA_PARALLEL BENCHMARK RESULTS (FP64) ---
    Particles: 100000, Steps: 5
    Average Step Time: 876.49ms
    Particle Updates/sec: 1.14e+05
    Estimated GFLOPS: 114.22
    --------------------------------------
    ```

    **Observations:**
    1.  **`Particle Updates/sec` Decreased:** When the particle count increased 10x, the workload increased by roughly 100x (since it's N²). The simulation throughput dropped dramatically, which correctly tells us that this brute-force algorithm does not scale well.
    2.  **`Estimated GFLOPS` Increased:** The GFLOPS *increased* slightly. This metric shows how effectively the hardware is being used. The larger problem size was better at keeping all the CPU cores busy, overcoming the fixed overhead of parallel task management.

    This teaches a critical lesson: `Particle Updates/sec` tells us about our *algorithmic efficiency*, while `GFLOPS` tells us about our *hardware efficiency*. A high GFLOPS value doesn't always mean a better simulation rate if the underlying algorithm is inefficient.

### Part 5: The Limits of Brute Force - `cuda_brute_force`

*   **Kernel:** `kernels/07_cuda_brute_force_kernel.py`
*   **Concept:** This kernel takes the same O(N²) brute-force algorithm from the `numba_parallel` version and ports it to CUDA to run on the GPU. The goal is to see what happens when we apply massive hardware parallelism to a naive algorithm.
*   **Demonstration:** This kernel highlights a critical distinction between raw computational throughput (GFLOPS) and true scientific progress (Particle Updates/sec). While the GPU can execute the O(N²) calculations much faster than the CPU, the underlying inefficiency of the algorithm itself imposes a severe limit on scalability.

    Let's compare the `numba_parallel` (CPU) and `cuda_brute_force` (GPU) kernels at 200,000 particles:

    ```bash
    # Numba Parallel on CPU
    --- NUMBA_PARALLEL BENCHMARK RESULTS (FP64) ---
    Particles: 200,000
    Average Step Time: 3498.30ms
    Particle Updates/sec: 5.72e+04
    Estimated GFLOPS: 114.40
    --------------------------------------

    # CUDA Brute-Force on GPU
    --- CUDA_BRUTE_FORCE BENCHMARK RESULTS (FP64) ---
    Particles: 200,000
    Average Step Time: 411.77ms
    Particle Updates/sec: 4.86e+05
    Estimated GFLOPS: 1944.41
    --------------------------------------
    ```

    **Observations:**
    1.  **Hardware Throughput (`Estimated GFLOPS`):** The GPU achieves over 1900 GFLOPS, an order of magnitude higher than the multi-core CPU. This demonstrates the immense raw power of the GPU for simple, parallelizable math.
    2.  **Algorithmic Throughput (`Particle Updates/sec`):** While the GPU is much faster, the `Particle Updates/sec` metric tells a more nuanced story. The simulation is faster, but the benefit is not as large as the GFLOPS number would suggest.

    Now, let's see what happens when we increase the problem size to 1,000,000 particles for the `cuda_brute_force` kernel:

    ```bash
    --- CUDA_BRUTE_FORCE BENCHMARK RESULTS (FP64) ---
    Particles: 1,000,000
    Average Step Time: 9423.78ms
    Particle Updates/sec: 1.06e+05
    Estimated GFLOPS: 2122.37
    --------------------------------------
    ```
    **Scaling Analysis:**
    *   The particle count increased 5x (from 200k to 1M), but the workload increased by 25x (due to N² complexity).
    *   The `Particle Updates/sec` has plummeted from `4.86e+05` down to `1.06e+05`. The simulation has become much less efficient.
    *   The `Estimated GFLOPS` remained high, because the GPU is still busy doing work, but it's doing an enormous amount of redundant work.

    This teaches a crucial lesson: **throwing more hardware at an inefficient algorithm yields diminishing returns.** The GPU is doing a phenomenal job at executing the instructions it's given, but the O(N²) algorithm is simply not a scalable solution. This provides the motivation to move beyond brute-force and explore smarter, algorithmically-efficient methods on the GPU.

### Part 6: A Smarter Algorithm - `barnes_hut`

*   **Kernel:** `kernels/05_barnes_hut_kernel.py`
*   **Concept:** This marks a crucial shift from computational optimization to **algorithmic optimization**. The Barnes-Hut algorithm is an O(N log N) approximation that groups distant particles into single "macro-particles," dramatically reducing the number of required force calculations.
*   **Demonstration:** This kernel teaches the most critical lesson in performance optimization: a better algorithm will almost always outperform a brute-force approach, no matter how well the latter is optimized. For small N, the overhead of building the octree makes `barnes_hut` slower than the direct-summation `numba_parallel` kernel. However, as N grows, its superior `O(N log N)` complexity takes over, and it becomes vastly more efficient.

*   **Performance Deep Dive: Algorithmic vs. Hardware Efficiency**

    Let's compare the `numba_parallel` and `barnes_hut` kernels at 200,000 particles to see this effect clearly.

    ```bash
    # Brute-Force N² Kernel
    --- NUMBA_PARALLEL BENCHMARK RESULTS (FP64) ---
    Particles: 200000, Steps: 5
    Average Step Time: 3498.30ms
    Particle Updates/sec: 5.72e+04
    Estimated GFLOPS: 114.40
    --------------------------------------

    # Barnes-Hut N log N Kernel
    --- BARNES_HUT BENCHMARK RESULTS (FP64) ---
    Particles: 200000, Steps: 5
    Average Step Time: 1837.39ms
    Particle Updates/sec: 1.09e+05
    Estimated GFLOPS: 6.14
    --------------------------------------
    ```

    **Observations:**

    1.  **Algorithmic Throughput (`Particle Updates/sec`):** The `barnes_hut` kernel is nearly twice as fast (`1.09e+05` vs. `5.72e+04`). It gets more work done in the same amount of time. When we increased the particle count from 100k to 200k, the runtime for `numba_parallel` increased by `4x` (as expected from O(N²)), while `barnes_hut` only increased by `2.3x`, demonstrating its superior scalability.

    2.  **Hardware Throughput (`Estimated GFLOPS`):** This is the most fascinating insight. The `numba_parallel` kernel reports extremely high GFLOPS (114.40), while `barnes_hut` reports a tiny fraction of that (6.14). How can the "slower" kernel (in terms of GFLOPS) produce the faster simulation?
        *   The `numba_parallel` kernel achieves high GFLOPS because its brute-force algorithm is computationally dense. It keeps the CPU cores 100% busy doing floating-point math, but much of that work is unnecessary.
        *   The `barnes_hut` kernel's primary goal is to *avoid* computation. By grouping distant particles, it intelligently prunes the majority of calculations. It does less work, so the GFLOPS are naturally lower, but the work it does is *smarter*.

    This teaches the ultimate lesson: **`Particle Updates/sec` tells us how fast our simulation is, while `GFLOPS` tells us how busy our hardware is.** The goal of HPC is not just to keep the hardware busy, but to solve the problem efficiently. A superior algorithm that does less work is better than a brute-force algorithm that does the wrong work faster.

### Part 7: Hybrid Computing - `cuda_bh`

*   **Kernel:** `kernels/06_cuda_bh_kernel.py`
*   **Concept:** Introduces a hybrid CPU/GPU strategy. The complex, serial-style logic of building the octree remains on the CPU, while the massively parallel task of force calculation is offloaded to the GPU using CUDA.
*   **Demonstration:** This kernel provides a classic lesson in performance optimization. An analysis of the pure CPU `barnes_hut` kernel reveals that the **Force Calculation** step consumes the vast majority of the runtime (e.g., ~1,400ms out of ~1,850ms, or over 75% of the total time). This makes it the obvious and ideal target for GPU acceleration.

    The `cuda_bh` kernel does exactly this. Let's compare the detailed timings at 200,000 particles to see the impact:

    | Task                | `barnes_hut` (CPU) | `cuda_bh` (Hybrid) | Analysis                                                                        |
    | ------------------- | ------------------ | ------------------ | ------------------------------------------------------------------------------- |
    | **Tree Build**      | `367.36ms`         | `367.23ms`         | **Unchanged.** This task still runs on the CPU.                                 |
    | **COM Calc**        | `90.17ms`          | `98.02ms`          | **Unchanged.** This also remains on the CPU.                                    |
    | **Force Calculation** | `1398.71ms`        | `21.17ms`          | **~66x Faster!** The GPU's massive parallelism crushes this part of the problem. |
    | **Total Step Time** | `~1858ms`          | `~555ms`           | **~3.3x Faster Overall.**                                                       |

    This result is a textbook example of **Amdahl's Law**. We achieved a massive speedup on the portion of our code that we parallelized, but the overall application speedup is now limited by the remaining serial components. The `Tree Build` and `COM Calc`, which were previously responsible for only a fraction of the runtime, have now become the primary performance bottleneck. This perfectly motivates the next step in the tutorial: moving the entire simulation pipeline to the GPU to eliminate these new bottlenecks.

### Part 8: A Direct Approach to a GPU-Resident Tree - `cuda_ref_octree`

*   **Kernel:** `kernels/08_cuda_ref_octree_kernel.py`
*   **Concept:** Following Amdahl's Law, the next logical step is to move the entire simulation to the GPU, eliminating the CPU bottlenecks and data transfer overhead. This kernel represents a direct, "lift-and-shift" approach to that problem. The tree-building logic is parallelized by launching one thread per particle, where each thread traverses the global tree structure from the root to insert itself.
*   **Demonstration:** This implementation is a significant step forward, but it reveals a new, more subtle performance challenge: **contention**. When many particles are in the same region of space, their corresponding threads will try to modify the same nodes in the tree simultaneously. This requires expensive atomic operations and causes threads that lose the "race" to retry their insertion, leading to inefficient execution and high variance in step times. This kernel teaches a valuable lesson in parallel algorithm design: a straightforward translation of a serial algorithm to a parallel architecture can expose new bottlenecks that weren't apparent before.

### Part 9: Algorithmic Refinement for the GPU - `cuda_morton_bh`

*   **Kernel:** `kernels/09_cuda_morton_bh_kernel.py`
*   **Concept:** This kernel is a much more sophisticated, GPU-native implementation of the Barnes-Hut algorithm. Instead of a direct, contentious insertion, it uses a multi-stage, contention-free process:
    1.  **Morton Codes:** It first converts the 3D particle positions into 1D Morton codes, a type of space-filling curve that preserves data locality.
    2.  **Sort:** These codes are then sorted, which efficiently groups nearby particles together in memory.
    3.  **Tree Build:** The tree's parent-child relationships are then built in parallel by inspecting the sorted codes.
*   **Demonstration:** This approach is dramatically faster and more stable than the `cuda_ref_octree` kernel. The performance gain comes from two key areas:
    *   **Contention-Free Tree Construction:** The sort-based method completely avoids the atomic collisions inherent in the direct-insertion approach.
    *   **Data Locality:** By processing particles in sorted order, GPU threads that execute together work on particles that are physically close. This results in extremely high cache efficiency during the force calculation, which is the most expensive part of the simulation. This kernel shows that the best performance comes from adapting the algorithm to the specific strengths of the hardware architecture.

