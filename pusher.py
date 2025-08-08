import time
import argparse
import numpy as np
import cupy as cp
import importlib
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def print_benchmark_results(kernel_name, num_particles, num_steps, total_time, all_results, precision):
    """Prints a formatted summary of the benchmark results."""
    avg_step_time = total_time / num_steps if num_steps > 0 else 0
    
    # Calculate Particle Updates per second
    particle_updates_per_sec = num_particles / avg_step_time if avg_step_time > 0 else 0

    # Sum GFLOPS from all steps
    total_gflops = sum(r['metrics'].get('gflops', 0) for r in all_results)
    avg_gflops = total_gflops / num_steps if num_steps > 0 else 0

    print(f"\n--- {kernel_name.upper()} BENCHMARK RESULTS ({precision}) ---")
    print(f"Particles: {num_particles}, Steps: {num_steps}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Step Time: {avg_step_time * 1000:.2f}ms")
    print(f"Particle Updates/sec: {particle_updates_per_sec:.2e}")
    print(f"Estimated GFLOPS: {avg_gflops:.2f}")

    if all_results and 'timings' in all_results[0]:
        avg_timings = defaultdict(float)
        for result in all_results:
            for key, value in result['timings'].items():
                avg_timings[key] += value
        
        print("\nAverage timings per step:")
        for key, value in avg_timings.items():
            print(f"  - {key.replace('_', ' ').title()}: {value / num_steps:.2f}ms")
    
    print("--------------------------------------\n")

def run_simulation(positions, velocities, masses, calculate_forces, num_steps, dt, G, 
                   kernel_name, is_gpu_kernel, dtype, animate=False, particle_size=0.5, 
                   particle_alpha=0.7, theta=0.5, debug_particle_idx=-1):
    """Main simulation loop for both CPU and GPU."""
    if animate and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots()
        pos_host = positions.get() if is_gpu_kernel else positions
        scatter = ax.scatter(pos_host[:, 0], pos_host[:, 1], s=particle_size, c='k', alpha=particle_alpha)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title(f'N-Body Simulation: {kernel_name}')
        
        x_min, x_max = np.min(pos_host[:, 0]), np.max(pos_host[:, 0])
        y_min, y_max = np.min(pos_host[:, 1]), np.max(pos_host[:, 1])
        padding = (max(x_max - x_min, y_max - y_min)) * 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        plt.ion()
        plt.show(block=False)

    xp = cp if is_gpu_kernel else np
    all_results = []
    
    result = calculate_forces(positions, masses, G=G, epsilon=1e-3, theta=theta, dtype=dtype, debug_particle_idx=debug_particle_idx)
    forces = result['forces']
    acceleration = forces / masses[:, xp.newaxis]
    velocities += acceleration * (dt / 2.0)

    start_time = time.time()

    for step in range(num_steps):
        step_start_time = time.time()
        positions += velocities * dt
        
        result = calculate_forces(positions, masses, G=G, epsilon=1e-3, theta=theta, dtype=dtype, debug_particle_idx=debug_particle_idx)
        forces = result['forces']
        all_results.append(result)

        acceleration = forces / masses[:, xp.newaxis]
        velocities += acceleration * dt
        
        if is_gpu_kernel:
            cp.cuda.Stream.null.synchronize()
            
        step_time = time.time() - step_start_time
        print(f"Step {step + 1}/{num_steps}: {step_time * 1000:.2f}ms")

        if animate and MATPLOTLIB_AVAILABLE:
            pos_host = positions.get() if is_gpu_kernel else positions
            scatter.set_offsets(pos_host[:, :2])
            plt.draw()
            plt.pause(0.001)
            
    total_time = time.time() - start_time
    
    if animate and MATPLOTLIB_AVAILABLE:
        plt.close()
        
    return total_time, all_results

def main(args):
    """Main driver for the N-body simulation benchmark."""
    
    dtype = np.float32 if args.precision == '32' else np.float64
    gpu_kernels = ['cuda_bh', 'cuda_brute_force', 'cuda_ref_octree', 'cuda_morton_bh']
    is_gpu_kernel = args.kernel in gpu_kernels


    kernel_map = {
        'loop': 'kernels.01_loop_kernel',
        'numpy': 'kernels.02_numpy_kernel',
        'numba_serial': 'kernels.03_numba_serial_kernel',
        'numba_parallel': 'kernels.04_numba_parallel_kernel',
        'barnes_hut': 'kernels.05_barnes_hut_kernel',
        'cuda_bh': 'kernels.06_cuda_bh_kernel',
        'cuda_brute_force': 'kernels.07_cuda_brute_force_kernel',
        'cuda_ref_octree': 'kernels.08_cuda_ref_octree_kernel',
        'cuda_morton_bh': 'kernels.09_cuda_morton_bh_kernel',
    }

    if args.kernel not in kernel_map:
        raise ValueError(f"Unknown kernel: {args.kernel}")

    try:
        kernel_module = importlib.import_module(kernel_map[args.kernel])
        calculate_forces = kernel_module.calculate_forces
    except ImportError:
        raise ImportError(f"Could not import kernel: {args.kernel}")

    print(f"Using {args.kernel} kernel.")

    if args.load_initial_conditions:
        print(f"Loading initial conditions from {args.load_initial_conditions}...")
        try:
            positions = np.load(f"{args.load_initial_conditions}_pos.npy")
            velocities = np.load(f"{args.load_initial_conditions}_vel.npy")
            masses = np.load(f"{args.load_initial_conditions}_mass.npy")
            
            # Ensure the loaded particle count matches the argument, if it was also provided
            if args.particles != len(positions):
                print(f"Warning: --particles argument ({args.particles}) "
                      f"is overridden by loaded file ({len(positions)} particles).")
                args.particles = len(positions)

        except FileNotFoundError:
            print(f"Error: Could not find initial condition files with prefix '{args.load_initial_conditions}'.")
            return
    elif is_gpu_kernel and args.init_method == 'plummer':
        print("Initializing particles directly on the GPU...")
        from init.cuda import initialize_plummer_gpu
        positions, velocities, masses = initialize_plummer_gpu(
            args.particles, scale_radius=args.scale_radius, G=args.G, 
            rotation_factor=args.rotation, dtype=dtype, seed=args.seed
        )
    else:
        print(f"Initializing {args.particles} particles using '{args.init_method}' method on CPU...")
        if args.init_method == 'random':
            if args.seed is not None:
                np.random.seed(args.seed)
            positions = np.random.rand(args.particles, 3).astype(dtype)
            velocities = (np.random.rand(args.particles, 3) - 0.5).astype(dtype) * 0.1
            masses = (np.random.rand(args.particles) + 0.1).astype(dtype)
        elif args.init_method == 'plummer':
            from init.cpu import initialize_plummer
            positions, velocities, masses = initialize_plummer(
                args.particles, scale_radius=args.scale_radius, G=args.G, 
                rotation_factor=args.rotation, dtype=dtype, seed=args.seed
            )
        else:
            raise ValueError(f"Unknown initialization method: {args.init_method}")

    if args.save_initial_conditions:
        print(f"Saving initial conditions to {args.save_initial_conditions}...")
        
        # If data is on GPU, bring it to CPU before saving
        pos_to_save = positions.get() if isinstance(positions, cp.ndarray) else positions
        vel_to_save = velocities.get() if isinstance(velocities, cp.ndarray) else velocities
        mass_to_save = masses.get() if isinstance(masses, cp.ndarray) else masses

        np.save(f"{args.save_initial_conditions}_pos.npy", pos_to_save)
        np.save(f"{args.save_initial_conditions}_vel.npy", vel_to_save)
        np.save(f"{args.save_initial_conditions}_mass.npy", mass_to_save)

    if 'numba' in args.kernel:
        print("Performing initial run for Numba JIT compilation...")
        calculate_forces(positions, masses, G=args.G, epsilon=1e-3, theta=0.5, dtype=dtype)
        print("Compilation complete.")

    if is_gpu_kernel and isinstance(positions, np.ndarray):
        print("Transferring initial data to GPU...")
        positions = cp.asarray(positions)
        velocities = cp.asarray(velocities)
        masses = cp.asarray(masses)

    print(f"Running simulation for {args.steps} steps...")
    total_time, all_results = run_simulation(
        positions, velocities, masses, calculate_forces, args.steps, args.dt, args.G,
        args.kernel, is_gpu_kernel, dtype, args.animate, args.particle_size, args.particle_alpha,
        theta=args.theta, debug_particle_idx=args.debug_particle_idx
    )
    
    print_benchmark_results(args.kernel, args.particles, args.steps, total_time, all_results, f"FP{args.precision}")
    
    if args.save_forces:
        # Save the forces from the last step
        final_forces = all_results[-1]['forces']
        forces_to_save = final_forces.get() if isinstance(final_forces, cp.ndarray) else final_forces
        print(f"Saving final forces to {args.save_forces}...")
        np.save(args.save_forces, forces_to_save)

        # Also save the debug output if it exists
        if 'debug_out' in all_results[-1] and all_results[-1]['debug_out'] is not None:
            debug_out = all_results[-1]['debug_out']
            debug_to_save = debug_out.get() if isinstance(debug_out, cp.ndarray) else debug_out
            debug_filename = args.save_forces.replace('.npy', '.debug.npy')
            print(f"Saving debug output to {debug_filename}...")
            np.save(debug_filename, debug_to_save)


    print("Simulation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-body simulation benchmark driver.')
    parser.add_argument('--kernel', type=str, required=True, 
                        choices=['loop', 'numpy', 'numba_serial', 'numba_parallel', 'barnes_hut', 'cuda_bh', 'cuda_full_bh', 'cuda_optimized', 'cuda_brute_force', 'cuda_ref_octree', 'cuda_morton_bh'],
                        help='The computational kernel to use.')
    parser.add_argument('--particles', type=int, default=5000, help='Number of particles.')
    parser.add_argument('--steps', type=int, default=5, help='Number of simulation steps.')
    parser.add_argument('--init-method', type=str, default='plummer', choices=['random', 'plummer'], help='Particle initialization method.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for particle initialization.')
    parser.add_argument('--animate', action='store_true', help='Enable real-time animation.')
    parser.add_argument('--precision', type=str, default='64', choices=['32', '64'], help='Floating point precision (32 or 64 bit).')
    
    parser.add_argument('--G', type=float, default=1e-4, help='Gravitational constant.')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step for the simulation.')
    parser.add_argument('--rotation', type=float, default=0.0, help='Rotation factor for Plummer model.')
    parser.add_argument('--scale-radius', type=float, default=1.0, help='Scale radius for Plummer model.')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta value for Barnes-Hut.')
    parser.add_argument('--debug-particle-idx', type=int, default=-1, help='Index of the particle to debug.')

    parser.add_argument('--particle-size', type=float, default=0.5, help='Particle size for animation.')
    parser.add_argument('--particle-alpha', type=float, default=0.7, help='Particle opacity for animation.')

    parser.add_argument('--save-initial-conditions', type=str, default=None, help='Save the initial particle conditions to the specified file path prefix (e.g., "initial_conditions").')
    parser.add_argument('--load-initial-conditions', type=str, default=None, help='Load the initial particle conditions from the specified file path prefix (e.g., "initial_conditions").')
    parser.add_argument('--save-forces', type=str, default=None, help='Save the final force array to the specified .npy file.')

    args = parser.parse_args()
    main(args)
