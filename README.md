# Cogniscience

> Energy-based cognitive architecture with hierarchical memory consolidation

An implementation of hybrid vector-graph dynamics inspired by cognitive neuroscience research, featuring gradient descent-based memory consolidation, hierarchical concept extraction, and recency-enhanced dreaming.

[![Tests](https://img.shields.io/badge/tests-74%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview

Cogniscience implements an energy-based cognitive engine that models memory formation and consolidation through:

- **Hybrid Memory System**: Combines vector embeddings (semantic content) with dynamic graph structure (relational knowledge)
- **Energy Minimization**: Uses gradient descent to consolidate memories by minimizing total system energy
- **Hierarchical Concepts**: Automatically extracts abstract concepts from memory clusters via HDBSCAN
- **Recency-Enhanced Dreaming**: Prioritizes recent experiences during consolidation with temporal proximity gradients
- **Real-World Streaming**: Processes live data from OpenAI embeddings with PostgreSQL/pgvector persistence

## Key Features

### Core Cognitive Engine
- **Energy Functions**: E_C (cache-vector), E_V (vector coherence), E_G (graph structure)
- **Gradient Descent**: Adaptive consolidation with Hebbian updates
- **Graph Dynamics**: Symmetric adjacency with decay and normalization
- **Dreaming Phase**: Stochastic exploration with controlled noise

### Memory Management
- **Temporal Activation**: Exponential decay with boost on access
- **Similarity Linking**: Automatic detection of related memories
- **Dual Storage**: In-memory cognitive engine + persistent Postgres
- **Activation Tracking**: `last_accessed`, `access_history`, decay rates

### Hierarchical Concepts
- **HDBSCAN Clustering**: Extracts abstract concepts from memory embeddings
- **Concept Graph**: Separate layer above memory-level with weighted edges
- **Coherence Scoring**: Measures cluster tightness (0-1)
- **Keyword Extraction**: TF-based concept naming
- **Activation Propagation**: Concepts inherit member memory activations

### Recency-Enhanced Dreaming
- **Pre-Dream Rumination**: Boosts top-k activated memories before sleep
- **Weighted Noise**: Activation-biased exploration focuses on recent experiences
- **Temporal Gradients**: Gaussian decay encourages connections between co-accessed memories
- **Combined Dynamics**: Blends semantic + temporal gradients (α=0.3)
- **Creative Bridges**: Detects novel connections formed during dreaming

## Installation

```bash
# Clone the repository
git clone https://github.com/VeritexAI/cogniscience.git
cd cogniscience

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Postgres (for persistence)
docker-compose up -d postgres

# Configure environment
cp .env.example .env
# Add your OpenAI API key to .env
```

## Quick Start

### Basic Cognitive Engine

```python
from cognigraph import CognitiveEngine

# Initialize engine
engine = CognitiveEngine(
    N=100,          # Max nodes
    d=64,           # Embedding dimension
    eta1=0.01,      # Cache learning rate
    eta2=0.005,     # Vector learning rate
    eta3=0.01,      # Graph learning rate
    lam=0.01        # Decay coefficient
)

# Set cache inputs (working memory)
cache_inputs = np.random.randn(5, 64)
cache_indices = [0, 1, 2, 3, 4]
engine.set_cache(cache_inputs, cache_indices)

# Run consolidation
for _ in range(100):
    result = engine.step()
    print(f"Energy: {result['energy']['E_total']:.4f}")

# Run dreaming phase
results, bridges = engine.dream(num_steps=50)
print(f"Creative bridges: {len(bridges)}")
```

### Streaming Pipeline with Concepts

```python
import asyncio
from cognigraph.ingestion.embeddings import OpenAIEmbeddingGenerator
from cognigraph.storage.postgres_store import PostgresVectorStore, PostgresGraphStore
from cognigraph.memory.semantic_memory import SemanticMemoryManager
from cognigraph.knowledge.concepts import ConceptExtractor

# Initialize components
embedder = OpenAIEmbeddingGenerator(model="text-embedding-3-small")
vector_store = PostgresVectorStore(database_url)
graph_store = PostgresGraphStore(database_url)

engine = CognitiveEngine(N=200, d=1536)

memory_manager = SemanticMemoryManager(
    vector_memory=engine.vector_memory,
    graph_memory=engine.graph_memory,
    vector_store=vector_store,
    graph_store=graph_store
)

concept_extractor = ConceptExtractor(min_cluster_size=3)

# Process streaming text
async def process_text(text):
    # Generate embedding
    embedding = await embedder.embed_single(text)
    
    # Add to memory with activation tracking
    node_idx, is_new, similar = memory_manager.add_or_link_memory(text, embedding)
    memory_manager.access_memory(node_idx, boost_type='cache')
    
    # Run consolidation
    for _ in range(10):
        engine.step()
    
    return node_idx

# Extract concepts
embeddings = [engine.vector_memory.vectors[i] for i in range(n_memories)]
texts = [memory_manager.nodes[i].text for i in range(n_memories)]
concepts = concept_extractor.extract_concepts(embeddings, texts, list(range(n_memories)))

print(f"Extracted {len(concepts)} concepts")
```

### Recency-Enhanced Dreaming

```python
from cognigraph.dynamics.dreaming import (
    pre_dream_rumination,
    controlled_dreaming_step_recency,
    compute_combined_gradient
)

# Get current activation state
activations = memory_manager.get_activation_weights()
last_accessed = np.array([node.last_accessed for node in memory_manager.nodes.values()])

# Pre-dream rumination
activations = pre_dream_rumination(activations, top_k=10, rumination_boost=0.2)

# Dream with recency bias
for step in range(100):
    vectors = engine.vector_memory.get_vectors()
    adjacency = engine.graph_memory.get_adjacency()
    
    # Compute gradients
    grads = compute_gradients(vectors, adjacency, cache, cache_indices)
    
    # Combine semantic + temporal
    grad_G_combined = compute_combined_gradient(
        grads['grad_G'], last_accessed, adjacency, temporal_weight=0.3
    )
    
    # Dream step with weighted noise
    new_V, new_G = controlled_dreaming_step_recency(
        vectors, adjacency, grads['grad_V'], grad_G_combined, activations,
        eta=0.005, sigma_n=0.002, lam=0.01, recency_bias=0.4
    )
    
    engine.vector_memory.set_vectors(new_V)
    engine.graph_memory.set_adjacency(new_G)
```

## Visualization

```python
from visualization.concept_viz import (
    plot_concept_graph,
    plot_activation_heatmap,
    plot_dual_level_graph
)

# Concept graph with activation coloring
plot_concept_graph(concept_extractor, save_path="concepts.png")

# Memory activation heatmap
activations = memory_manager.get_activation_weights()
plot_activation_heatmap(activations, save_path="activation.png")

# Dual-level (memory + concept) view
adjacency = engine.graph_memory.get_adjacency()
plot_dual_level_graph(adjacency, concept_extractor, activations, save_path="dual.png")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_recency_features.py -v

# Run with coverage
pytest tests/ --cov=src/cognigraph --cov-report=html
```

**Test Coverage**: 74 tests covering:
- Energy functions and gradients
- Vector/graph memory operations
- Temporal activation tracking
- Concept extraction and clustering
- Recency-enhanced dreaming
- Temporal proximity gradients

## 🏗️ Architecture

```
cogniscience/
├── src/cognigraph/
│   ├── engine.py              # Core cognitive engine
│   ├── energy/                # Energy functions (E_C, E_V, E_G)
│   ├── dynamics/              # Update rules & dreaming
│   ├── memory/                # Vector, graph, cache, semantic
│   ├── knowledge/             # Concept extraction (NEW)
│   ├── storage/               # Postgres persistence
│   └── ingestion/             # OpenAI embeddings, text processing
├── visualization/
│   ├── concept_viz.py         # Concept graph visualization (NEW)
│   ├── graph_viz.py           # Memory graph visualization
│   ├── energy_plots.py        # Energy evolution plots
│   └── vector_space.py        # PCA/t-SNE projections
├── experiments/
│   └── streaming_demo_enhanced.py  # Full pipeline demo (NEW)
└── tests/
    └── test_recency_features.py    # New feature tests (NEW)
```

## 🔬 Research Background

This implementation is based on energy-based models for hybrid memory systems:

**Energy Function**:
```
E_total = E_C + E_V + E_G
E_C = Σ ||c_i - v_i||²          (cache alignment)
E_V = Σ ||v_i||² - N            (vector normalization)
E_G = Σ G_ij² - Σ σ(v_i,v_j)·G_ij  (graph structure)
```

**Gradient Descent**:
```
v_i ← v_i - η₁∇_{v_i}E_C - η₂∇_{v_i}E_V - η₂∇_{v_i}E_G
G_ij ← G_ij - η₃∇_{G_ij}E_G + η₃σ(v_i,v_j) - λG_ij
```

**Recency-Enhanced Dreaming** (NEW):
```
grad_combined = (1-α)·grad_semantic + α·grad_temporal
temporal_proximity(i,j) = exp(-Δt²/(2σ²))
```

## Contributing

Contributions welcome! Areas of interest:
- Additional concept extraction algorithms (Louvain, spectral clustering)
- Enhanced temporal dynamics (circadian rhythms, sleep stages)
- Multi-modal embeddings (vision, audio)
- Attention mechanisms for cache selection
- Distributed/federated learning

## 🔗 Links

- **Repository**: https://github.com/VeritexAI/cogniscience
- **Documentation**: [Coming soon]
- **Issues**: https://github.com/VeritexAI/cogniscience/issues

## 📧 Contact

Questions? Open an issue or reach out to the VeritexAI team.

---

Built by [VeritexAI](https://github.com/VeritexAI)
