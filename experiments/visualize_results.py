"""
Visualize results from a Cognitive Engine simulation.

This script creates comprehensive visualizations of simulation results,
including energy evolution, graph structure, and vector space organization.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognigraph import CognitiveEngine
from cognigraph.generators import create_default_generator
from visualization.energy_plots import (
    plot_energy_evolution,
    plot_energy_components,
    plot_metrics_dashboard
)
from visualization.graph_viz import (
    plot_graph_structure,
    plot_edge_distribution,
    plot_degree_distribution
)
from visualization.vector_space import (
    plot_vector_projection_2d,
    plot_similarity_matrix
)


def visualize_from_engine(engine, results, output_dir='./results'):
    """
    Create all visualizations from an engine and its results.
    
    Args:
        engine: CognitiveEngine instance
        results: List of step results from simulation
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📊 Generating visualizations...")
    
    # 1. Energy evolution
    print("  [1/7] Energy evolution...")
    fig = plot_energy_evolution(
        engine.energy_history,
        save_path=f'{output_dir}/energy_evolution.png'
    )
    plt.close(fig)
    
    # 2. Energy components
    print("  [2/7] Energy components...")
    fig = plot_energy_components(
        results,
        save_path=f'{output_dir}/energy_components.png'
    )
    plt.close(fig)
    
    # 3. Metrics dashboard
    print("  [3/7] Metrics dashboard...")
    fig = plot_metrics_dashboard(
        results,
        save_path=f'{output_dir}/metrics_dashboard.png'
    )
    plt.close(fig)
    
    # 4. Graph structure (for smaller graphs)
    adjacency = engine.graph_memory.get_adjacency()
    N = len(adjacency)
    
    if N <= 200:  # Only plot full graph for smaller systems
        print("  [4/7] Graph structure...")
        fig = plot_graph_structure(
            adjacency,
            threshold=0.1,
            save_path=f'{output_dir}/graph_structure.png'
        )
        plt.close(fig)
    else:
        print("  [4/7] Graph structure (skipped - too large, use edge distribution instead)...")
    
    # 5. Edge distribution
    print("  [5/7] Edge weight distribution...")
    fig = plot_edge_distribution(
        adjacency,
        save_path=f'{output_dir}/edge_distribution.png'
    )
    plt.close(fig)
    
    # 6. Degree distribution
    print("  [6/7] Degree distribution...")
    fig = plot_degree_distribution(
        adjacency,
        threshold=0.01,
        save_path=f'{output_dir}/degree_distribution.png'
    )
    plt.close(fig)
    
    # 7. Vector space projection
    print("  [7/7] Vector space projection...")
    vectors = engine.vector_memory.get_vectors()
    fig, coords = plot_vector_projection_2d(
        vectors,
        method='pca',
        save_path=f'{output_dir}/vector_space_2d.png'
    )
    plt.close(fig)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - energy_evolution.png")
    print("  - energy_components.png")
    print("  - metrics_dashboard.png")
    if N <= 200:
        print("  - graph_structure.png")
    print("  - edge_distribution.png")
    print("  - degree_distribution.png")
    print("  - vector_space_2d.png")


def quick_visualization():
    """
    Run a quick simulation and visualize results.
    """
    print("🔬 Running quick simulation for visualization demo...")
    print("   (N=100, d=5, 500 steps)\n")
    
    # Initialize and run
    engine = CognitiveEngine(N=100, d=5, random_seed=42)
    generator = create_default_generator(N=100, d=5, mode='temporal', random_seed=42)
    
    results = engine.run_simulation(
        num_steps=500,
        cache_generator=generator,
        verbose=False
    )
    
    # Visualize
    visualize_from_engine(engine, results, output_dir='./results')
    
    print("\n🖼️  Open the PNG files in the 'results/' folder to view!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize Cognitive Engine simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo (runs simulation and visualizes)
  python visualize_results.py --quick
  
  # Run custom simulation and visualize
  python visualize_results.py --nodes 500 --dims 8 --steps 1000
  
  # Adjust learning parameters (stronger pruning)
  python visualize_results.py --nodes 500 --dims 10 --steps 2000 --lam 0.05
  
  # Full parameter control
  python visualize_results.py -N 1000 -d 20 -s 3000 --eta1 0.01 --eta2 0.005 --eta3 0.01 --lam 0.02
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo simulation (N=100, 500 steps)')
    parser.add_argument('--nodes', '-N', type=int, default=500,
                       help='Number of nodes (default: 500)')
    parser.add_argument('--dims', '-d', type=int, default=8,
                       help='Embedding dimensionality (default: 8)')
    parser.add_argument('--steps', '-s', type=int, default=1000,
                       help='Number of steps (default: 1000)')
    parser.add_argument('--generator', '-g', type=str, default='temporal',
                       choices=['temporal', 'random_walk', 'mixed', 'adaptive'],
                       help='Cache generator mode (default: temporal)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', '-o', type=str, default='./results',
                       help='Output directory for plots (default: ./results)')
    
    # Learning parameters
    parser.add_argument('--eta1', type=float, default=0.02,
                       help='Cache consolidation rate (default: 0.02)')
    parser.add_argument('--eta2', type=float, default=0.01,
                       help='Vector update rate (default: 0.01)')
    parser.add_argument('--eta3', type=float, default=0.01,
                       help='Graph update rate (default: 0.01)')
    parser.add_argument('--lam', type=float, default=0.005,
                       help='Decay/pruning coefficient (default: 0.005)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_visualization()
    else:
        print(f"🔬 Running simulation...")
        print(f"   N={args.nodes}, d={args.dims}, steps={args.steps}")
        print(f"   η₁={args.eta1}, η₂={args.eta2}, η₃={args.eta3}, λ={args.lam}\n")
        
        engine = CognitiveEngine(
            N=args.nodes,
            d=args.dims,
            eta1=args.eta1,
            eta2=args.eta2,
            eta3=args.eta3,
            lam=args.lam,
            random_seed=args.seed
        )
        
        generator = create_default_generator(
            N=args.nodes,
            d=args.dims,
            mode=args.generator,
            random_seed=args.seed
        )
        
        results = engine.run_simulation(
            num_steps=args.steps,
            cache_generator=generator,
            verbose=True,
            log_interval=max(args.steps // 10, 100)
        )
        
        visualize_from_engine(engine, results, output_dir=args.output)
        print(f"\n🖼️  Open the PNG files in '{args.output}/' to view!")


if __name__ == '__main__':
    main()
