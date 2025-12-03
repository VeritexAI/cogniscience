# Cognitive Engine Matrix

An implementation of the **Cognitive Engine Matrix: An Energy-Based Framework for Hybrid Memory Systems** by Russ Tolsma (November 2025).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Cognitive Engine is a **living knowledge system** that continuously learns, consolidates, and reorganizes information through energy minimization. Unlike traditional static memory systems, it models knowledge as a self-organizing energy field with three interacting memory strata:

- **Cache (C_t)**: Short-term working memory (volatile, high-energy)
- **Vector (V)**: Long-term semantic embeddings (stable, normalized)
- **Graph (G)**: Relational structure (dynamic, symmetric)

The system evolves by minimizing total energy **E_t = E_C + E_V + E_G** through gradient descent, producing emergent properties like:

- ✨ **Hebbian Learning**: Co-activated concepts strengthen their connections
- 🧹 **Synaptic Pruning**: Unused associations naturally decay
- 💭 **Consolidation**: Short-term experiences integrate into long-term memory
- 🌙 **Creative Dreaming**: Stochastic exploration discovers novel patterns

## Mathematical Foundation

### Energy Function

The total energy combines three components:

**1. Cache-Vector Alignment (E_C)**
```
E_C = (1/2) Σ ||c_i - v̂_i||²
```
Measures misalignment between working memory and long-term embeddings.

**2. Vector-Field Coherence (E_V)**
```
E_V = -(1/2) Σ_{i,j} σ(v_i, v_j)
```
Measures semantic coherence via cosine similarity. Negative sign means high similarity lowers energy.

**3. Graph-Structural Energy (E_G)**
```
E_G = -(1/2) Σ_{i,j} G_ij σ(v_i, v_j)
```
Measures graph-vector alignment. Strong edges between similar nodes lower energy.

### Update Dynamics

The system evolves through gradient descent:

```
Δv_i = η₁(c_i - v_i) + η₂ Σ_j (1 + G_ij) ∂σ/∂v_i
ΔG_ij = η₃ σ(v_i, v_j) - λG_ij  (Hebbian + decay)
```

With normalization constraints:
- Vectors: `v_i ← v_i/||v_i||` (unit sphere)
- Graph: `G_ij ← tanh(G_ij)` (bounded weights)

### Theoretical Guarantees

- **Lyapunov Stability**: Energy decreases monotonically (dE_t/dt ≤ 0)
- **Convergence**: System reaches stable equilibria under bounded norms
- **Hebbian-Gradient Equivalence**: Local plasticity = global energy descent

See the [paper](Cognitive_Paper.tex) for complete mathematical derivations and proofs.

## Installation

### From Source

```bash
git clone https://github.com/yourusername/cognigraph.git
cd cognigraph

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Requirements

- Python 3.9+
- NumPy >= 1.24
- matplotlib >= 3.7
- networkx >= 3.0
- plotly >= 5.14 (optional, for interactive viz)
- scikit-learn >= 1.3

## Quick Start

### Basic Simulation

```python
from cognigraph import CognitiveEngine
from cognigraph.generators import create_default_generator

# Initialize engine (1000 nodes, 10 dimensions)
engine = CognitiveEngine(
    N=1000,
    d=10,
    eta1=0.02,  # Cache consolidation rate
    eta2=0.01,  # Vector update rate
    eta3=0.01,  # Graph update rate
    lam=0.005,  # Decay coefficient
    random_seed=42
)

# Create cache input generator (temporal clustering)
generator = create_default_generator(N=1000, d=10, mode='temporal')

# Run simulation
results = engine.run_simulation(
    num_steps=2000,
    cache_generator=generator,
    verbose=True,
    log_interval=200
)

# Access final state
state = engine.get_state()
print(f"Final energy: {state['energy_history'][-1]:.4f}")
print(f"Graph density: {state['metrics_history'][-1]['density']:.3f}")
```

### Dreaming Phase

```python
# Run stochastic exploration
dream_results, bridges = engine.dream(
    num_steps=100,
    sigma_n=0.002,  # Noise amplitude
    verbose=True
)

print(f"Creative bridges formed: {len(bridges)}")
```

### Command-Line Interface

```bash
# Quick test (N=100, 500 steps)
python experiments/basic_simulation.py --quick-test

# Full simulation (N=1000, 2000 steps)
python experiments/basic_simulation.py --nodes 1000 --dims 10 --steps 2000

# Custom configuration
python experiments/basic_simulation.py \
    --nodes 500 \
    --dims 8 \
    --steps 1000 \
    --generator mixed \
    --seed 123
```

## Visualization

### Energy Dynamics

```python
from visualization.energy_plots import (
    plot_energy_evolution,
    plot_energy_components,
    plot_metrics_dashboard
)

# Plot total energy
fig = plot_energy_evolution(
    engine.energy_history,
    save_path='energy.png'
)

# Plot energy components
fig = plot_energy_components(
    results,
    save_path='components.png'
)

# Comprehensive dashboard
fig = plot_metrics_dashboard(
    results,
    save_path='dashboard.png'
)
```

### Graph Structure

```python
from visualization.graph_viz import (
    plot_graph_structure,
    plot_edge_distribution,
    plot_degree_distribution
)

adjacency = engine.graph_memory.get_adjacency()

# Visualize graph
fig = plot_graph_structure(
    adjacency,
    threshold=0.1,
    save_path='graph.png'
)

# Edge weight distribution
fig = plot_edge_distribution(
    adjacency,
    save_path='edges.png'
)
```

### Vector Space

```python
from visualization.vector_space import (
    plot_vector_projection_2d,
    plot_similarity_matrix
)

vectors = engine.vector_memory.get_vectors()

# 2D projection (PCA)
fig, coords = plot_vector_projection_2d(
    vectors,
    method='pca',
    save_path='vectors_2d.png'
)

# Similarity heatmap
fig = plot_similarity_matrix(
    vectors,
    save_path='similarity.png'
)
```

### Interactive Visualizations

```python
from visualization.energy_plots import plot_energy_interactive
from visualization.graph_viz import plot_graph_interactive
from visualization.vector_space import plot_vector_interactive_3d

# Interactive energy plot
fig = plot_energy_interactive(engine.energy_history)
fig.show()

# Interactive graph
fig = plot_graph_interactive(adjacency, vectors, threshold=0.1)
fig.show()

# Interactive 3D vector space
fig = plot_vector_interactive_3d(vectors, method='pca')
fig.show()
```

## Architecture

```
cognigraph/
├── src/cognigraph/
│   ├── memory/
│   │   ├── cache.py          # CacheMemory (working context)
│   │   ├── vector.py         # VectorMemory (semantic embeddings)
│   │   └── graph.py          # GraphMemory (relational structure)
│   ├── energy/
│   │   ├── similarity.py     # Cosine similarity metrics
│   │   ├── functions.py      # E_C, E_V, E_G computations
│   │   └── gradients.py      # Gradient calculations
│   ├── dynamics/
│   │   ├── updates.py        # Update rules & normalization
│   │   └── dreaming.py       # Stochastic exploration
│   ├── engine.py             # Main CognitiveEngine class
│   ├── generators.py         # Cache input generators
│   └── utils.py              # Metrics & analysis tools
├── visualization/
│   ├── energy_plots.py       # Energy visualizations
│   ├── graph_viz.py          # Graph visualizations
│   └── vector_space.py       # Vector space visualizations
├── experiments/
│   └── basic_simulation.py   # Main experiment script
└── tests/
    ├── test_energy.py        # Energy function tests
    ├── test_gradients.py     # Gradient tests
    └── test_memory.py        # Memory class tests
```

## Cache Generators

The system supports multiple cache input patterns:

### Temporal Clustering
```python
generator = create_default_generator(N, d, mode='temporal')
```
Focuses on one semantic cluster at a time for sustained periods.

### Random Walk
```python
generator = create_default_generator(N, d, mode='random_walk')
```
Transitions between clusters with probability p.

### Mixed Exploration
```python
generator = create_default_generator(N, d, mode='mixed')
```
Balances focused learning with exploratory sampling.

### Adaptive (High-Energy)
```python
generator = create_default_generator(N, d, mode='adaptive')
```
Samples from regions with highest gradient norms (fastest change).

## Testing

The implementation includes comprehensive unit tests (57 tests):

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_energy.py -v
pytest tests/test_gradients.py -v
pytest tests/test_memory.py -v

# With coverage
pytest tests/ --cov=cognigraph --cov-report=html
```

**Test Coverage**:
- ✅ Energy functions (E_C, E_V, E_G formulas)
- ✅ Gradient computations (finite difference validation)
- ✅ Lyapunov property (energy monotonicity)
- ✅ Memory constraints (normalization, symmetry, bounds)
- ✅ Hebbian-gradient equivalence

## Performance

Benchmarks on MacBook Pro (M1):

| Configuration | Steps/Second | Memory Usage |
|--------------|--------------|--------------|
| N=100, d=5   | ~1,200      | <100 MB      |
| N=1000, d=10 | ~100        | ~500 MB      |
| N=5000, d=20 | ~10         | ~5 GB        |

## Examples & Use Cases

### Knowledge Graph Learning
Model evolving knowledge bases where concepts strengthen through use.

### Semantic Memory Systems
Build memory systems that consolidate and organize information automatically.

### Creative AI
Explore novel connections through dreaming phases.

### Cognitive Modeling
Simulate aspects of biological memory and learning.

## Theoretical Background

This implementation faithfully reproduces the mathematical framework described in the paper:

- **Section 4**: Energy function formulation
- **Section 5**: Hebbian learning and gradient descent
- **Section 6**: Stability and convergence analysis
- **Section 7**: Experimental validation
- **Appendix A**: Complete energy derivations
- **Appendix B**: Hebbian-gradient equivalence proofs
- **Appendix C**: Stability proofs

All formulas, update rules, and constraints match the paper specifications exactly.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tolsma2025cognitive,
  title={Cognitive Engine Matrix: An Energy-Based Framework for Hybrid Memory Systems},
  author={Tolsma, Russ},
  journal={Unpublished manuscript},
  year={2025},
  month={November}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Mathematical framework inspired by energy-based models (Hopfield, Boltzmann)
- Biological principles from neuroscience (LTP, synaptic pruning, consolidation)
- Graph theory and semantic embedding techniques

## Contact

Russ Tolsma - Independent Researcher, Cognitive Systems Architect

For questions or collaboration: [contact info]

---

**Built with ❤️ for advancing cognitive AI systems**
