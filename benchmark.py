import time
import argparse
import numpy as np
# Conditional import for animation
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def initialize_particles(num_particles):
    """
    Initializes particles with random positions, velocities, and masses.
    """
    positions = np.random.rand(num_particles, 3)
    velocities = (np.random.rand(num_particles, 3) - 0.5) * 0.1
    masses = np.random.rand(num_particles) + 0.1
    return positions, velocities, masses

def run_simulation(positions, velocities, masses, calculate_forces, num_steps, dt, kernel_name, num_particles, animate=False):
    """
    Run the simulation with optional animation.
    """
    if animate:
        # Set up the plot for animation
        fig, ax = plt.subplots()
        scatter = ax.scatter(positions[:, 0], positions[:, 1], s=0.2, c='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('N-Body Simulation')
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    start_time = time.time()
    step_times = []

    for step in range(num_steps):
        step_start = time.time()
        
        # Leapfrog Integrator: Drift (update position)
        positions += velocities * dt
        
        # Leapfrog Integrator: Kick (update velocity)
        forces = calculate_forces(positions, masses, G=1e-4, epsilon=1e-3)
        acceleration = forces / masses[:, np.newaxis]
        velocities += acceleration * dt
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Print timing for each step
        print(f"Step {step + 1}: {step_time:.4f}s")
        
        # Update animation if enabled
        if animate:
            scatter.set_offsets(positions[:, :2])
            plt.draw()
            plt.pause(0.001)  # Minimal pause just for GUI events

    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    force_calculations_per_second = (num_particles * num_particles * num_steps) / total_time if total_time > 0 else 0

    print(f"\n--- {kernel_name.upper()} Benchmark Results ---")
    print(f"Particles: {num_particles}, Steps: {num_steps}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per step: {avg_time_per_step:.6f}s")
    print(f"Min step time: {min_step_time:.6f}s")
    print(f"Max step time: {max_step_time:.6f}s")
    print(f"Force calculations per second: {force_calculations_per_second:.2e}")
    print("--------------------------------------\n")

    if animate:
        plt.close()  # Close the matplotlib window

    print("Simulation finished.")

def main(kernel_name, num_particles, num_steps, animate=False):
    """
    Main driver for the N-body simulation benchmark.
    """
    # Dynamically import the selected kernel
    if kernel_name == 'numpy':
        from kernels.numpy_kernel import calculate_forces
        print("Using NumPy kernel.")
    elif kernel_name == 'numba_serial':
        from kernels.numba_serial_kernel import calculate_forces
        print("Using Numba (serial) kernel.")
    elif kernel_name == 'numba_parallel':
        from kernels.numba_parallel_kernel import calculate_forces
        print("Using Numba (parallel) kernel.")
    elif kernel_name == 'barnes_hut':
        from kernels.barnes_hut_kernel import calculate_forces
        print("Using Barnes-Hut kernel.")
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    dt = 0.01  # Time step

    print(f"Initializing {num_particles} particles...")
    positions, velocities, masses = initialize_particles(num_particles)

    # For JIT kernels, perform a warm-up run
    if 'numba' in kernel_name:
        print("Performing initial run for Numba JIT compilation...")
        calculate_forces(positions, masses, G=1e-4, epsilon=1e-3)
        print("Compilation complete.")

    # Leapfrog Integrator: Initial half-step kick for velocity
    initial_forces = calculate_forces(positions, masses, G=1e-4, epsilon=1e-3)
    initial_acceleration = initial_forces / masses[:, np.newaxis]
    velocities += initial_acceleration * (dt / 2.0)

    if animate and not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for animation. Install with: pip install matplotlib")
        return
    
    print(f"Running simulation for {num_steps} steps...")
    run_simulation(positions, velocities, masses, calculate_forces, num_steps, dt, kernel_name, num_particles, animate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-body simulation benchmark driver.')
    parser.add_argument('--kernel', type=str, required=True, 
                        choices=['numpy', 'numba_serial', 'numba_parallel', 'barnes_hut'],
                        help='The computational kernel to use.')
    parser.add_argument('--particles', type=int, default=1000, 
                        help='Number of particles to simulate.')
    parser.add_argument('--steps', type=int, default=20, 
                        help='Number of simulation steps to run.')
    parser.add_argument('--animate', action='store_true',
                        help='Enable real-time animation of the simulation.')
    
    args = parser.parse_args()
    main(args.kernel, args.particles, args.steps, args.animate)
