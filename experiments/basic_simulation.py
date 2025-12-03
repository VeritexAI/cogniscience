"""
Basic simulation experiment for the Cognitive Engine.

Runs a simulation with N=1000 nodes and d=10 dimensions, demonstrating:
- Energy minimization and convergence
- Graph self-organization and clustering
- Hebbian learning and consolidation
- Stochastic exploration (dreaming)

This reproduces the experimental setup from Section 7 of the paper,
scaled up to 1000 nodes × 10 dimensions.
"""

import numpy as np
import time
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cognigraph import CognitiveEngine
from cognigraph.generators import create_default_generator


def run_basic_simulation(N: int = 1000, d: int = 10, 
                        num_steps: int = 2000,
                        dream_steps: int = 100,
                        generator_mode: str = 'temporal',
                        eta1: float = 0.02,
                        eta2: float = 0.01,
                        eta3: float = 0.01,
                        lam: float = 0.005,
                        random_seed: int = 42,
                        verbose: bool = True):
    """
    Run basic simulation experiment.
    
    Args:
        N: Number of nodes/concepts
        d: Embedding dimensionality  
        num_steps: Number of learning steps
        dream_steps: Number of dreaming steps at end
        generator_mode: Cache generator mode
        eta1: Cache consolidation rate
        eta2: Vector update rate
        eta3: Graph update rate (Hebbian)
        lam: Decay/pruning coefficient
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        dict: Simulation results including engine state and metrics
    """
    
    if verbose:
        print("="*70)
        print("COGNITIVE ENGINE - Basic Simulation")
        print("="*70)
        print(f"Configuration:")
        print(f"  Nodes (N): {N}")
        print(f"  Dimensions (d): {d}")
        print(f"  Learning steps: {num_steps}")
        print(f"  Dreaming steps: {dream_steps}")
        print(f"  Generator mode: {generator_mode}")
        print(f"  Learning rates: η₁={eta1}, η₂={eta2}, η₃={eta3}")
        print(f"  Decay coefficient: λ={lam}")
        print(f"  Random seed: {random_seed}")
        print("="*70)
    
    # Initialize engine
    if verbose:
        print("\n[1/4] Initializing Cognitive Engine...")
    
    start_time = time.time()
    
    engine = CognitiveEngine(
        N=N,
        d=d,
        eta1=eta1,
        eta2=eta2,
        eta3=eta3,
        lam=lam,
        random_seed=random_seed
    )
    
    init_time = time.time() - start_time
    if verbose:
        print(f"  ✓ Initialized in {init_time:.2f}s")
        print(f"  Initial state: {engine}")
    
    # Create cache generator
    if verbose:
        print(f"\n[2/4] Creating {generator_mode} cache generator...")
    
    cache_generator = create_default_generator(
        N=N,
        d=d,
        mode=generator_mode,
        random_seed=random_seed
    )
    
    if verbose:
        print(f"  ✓ Generator created")
    
    # Run learning simulation
    if verbose:
        print(f"\n[3/4] Running learning simulation ({num_steps} steps)...")
        print("  Progress:")
    
    learning_start = time.time()
    
    results = engine.run_simulation(
        num_steps=num_steps,
        cache_generator=cache_generator,
        verbose=verbose,
        log_interval=200
    )
    
    learning_time = time.time() - learning_start
    
    if verbose:
        print(f"\n  ✓ Learning complete in {learning_time:.2f}s")
        print(f"  Final energy: {results[-1]['energy']['E_total']:.4f}")
        print(f"  Final gradient norm: {results[-1]['gradient_norm']:.6f}")
        print(f"  Final graph density: {results[-1]['metrics']['density']:.3f}")
    
    # Run dreaming phase
    if verbose:
        print(f"\n[4/4] Running dreaming phase ({dream_steps} steps)...")
    
    dream_start = time.time()
    
    dream_results, bridges = engine.dream(
        num_steps=dream_steps,
        sigma_n=0.002,
        verbose=verbose
    )
    
    dream_time = time.time() - dream_start
    
    if verbose:
        print(f"  ✓ Dreaming complete in {dream_time:.2f}s")
    
    # Compile results
    total_time = time.time() - start_time
    
    final_state = engine.get_state()
    
    simulation_results = {
        'config': {
            'N': N,
            'd': d,
            'num_steps': num_steps,
            'dream_steps': dream_steps,
            'generator_mode': generator_mode,
            'random_seed': random_seed
        },
        'timing': {
            'total': total_time,
            'initialization': init_time,
            'learning': learning_time,
            'dreaming': dream_time
        },
        'learning_results': results,
        'dream_results': dream_results,
        'creative_bridges': bridges,
        'final_state': final_state
    }
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Total time: {total_time:.2f}s")
        print(f"Steps per second: {(num_steps + dream_steps) / total_time:.1f}")
        print("\nFinal Metrics:")
        print(f"  Energy: {final_state['energy_history'][-1]:.4f}")
        print(f"  Graph density: {final_state['metrics_history'][-1]['density']:.3f}")
        print(f"  Mean degree: {final_state['metrics_history'][-1]['mean_degree']:.2f}")
        print(f"  Creative bridges formed: {len(bridges)}")
        
        # Energy decrease
        initial_energy = final_state['energy_history'][0]
        final_energy = final_state['energy_history'][-1]
        energy_decrease = initial_energy - final_energy
        percent_decrease = (energy_decrease / abs(initial_energy)) * 100
        
        print(f"\nEnergy Evolution:")
        print(f"  Initial: {initial_energy:.4f}")
        print(f"  Final: {final_energy:.4f}")
        print(f"  Decrease: {energy_decrease:.4f} ({percent_decrease:.1f}%)")
        
        print("\n" + "="*70)
    
    return simulation_results


def quick_test(verbose: bool = True):
    """
    Quick test with smaller parameters for rapid validation.
    
    Args:
        verbose: Whether to print output
        
    Returns:
        dict: Results
    """
    if verbose:
        print("\n🔬 Running quick test (N=100, d=5, 500 steps)...\n")
    
    return run_basic_simulation(
        N=100,
        d=5,
        num_steps=500,
        dream_steps=50,
        generator_mode='temporal',
        random_seed=42,
        verbose=verbose
    )


def main():
    """Main entry point for basic simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Cognitive Engine basic simulation'
    )
    parser.add_argument('--nodes', '-N', type=int, default=1000,
                      help='Number of nodes (default: 1000)')
    parser.add_argument('--dims', '-d', type=int, default=10,
                      help='Embedding dimensionality (default: 10)')
    parser.add_argument('--steps', '-s', type=int, default=2000,
                      help='Number of learning steps (default: 2000)')
    parser.add_argument('--dream-steps', type=int, default=100,
                      help='Number of dreaming steps (default: 100)')
    parser.add_argument('--generator', '-g', type=str, default='temporal',
                      choices=['temporal', 'random_walk', 'mixed', 'adaptive'],
                      help='Cache generator mode (default: temporal)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    # Learning parameters
    parser.add_argument('--eta1', type=float, default=0.02,
                      help='Cache consolidation rate (default: 0.02)')
    parser.add_argument('--eta2', type=float, default=0.01,
                      help='Vector update rate (default: 0.01)')
    parser.add_argument('--eta3', type=float, default=0.01,
                      help='Graph update rate (default: 0.01)')
    parser.add_argument('--lam', type=float, default=0.005,
                      help='Decay/pruning coefficient (default: 0.005)')
    
    parser.add_argument('--quick-test', action='store_true',
                      help='Run quick test with small parameters')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Suppress output')
    
    args = parser.parse_args()
    
    if args.quick_test:
        results = quick_test(verbose=not args.quiet)
    else:
        results = run_basic_simulation(
            N=args.nodes,
            d=args.dims,
            num_steps=args.steps,
            dream_steps=args.dream_steps,
            generator_mode=args.generator,
            eta1=args.eta1,
            eta2=args.eta2,
            eta3=args.eta3,
            lam=args.lam,
            random_seed=args.seed,
            verbose=not args.quiet
        )
    
    # Optionally save results
    # import pickle
    # with open('simulation_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    return results


if __name__ == '__main__':
    main()
