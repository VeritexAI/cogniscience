"""
Cognitive Engine Matrix: An Energy-Based Framework for Hybrid Memory Systems

This package implements the mathematical framework described in:
'Cognitive Engine Matrix: An Energy-Based Framework for Hybrid Memory Systems'
by Russ Tolsma (November 2025)

The system models knowledge as a self-organizing energy field composed of three
interacting memory strata:
- Cache (C_t): Short-term working memory
- Vector (V): Long-term semantic embeddings
- Graph (G): Relational structure

The system evolves by minimizing total energy E_t = E_C + E_V + E_G through
gradient descent, producing emergent properties like Hebbian learning,
synaptic pruning, consolidation, and creative exploration.
"""

__version__ = "0.1.0"
__author__ = "Russ Tolsma"

from cognigraph.engine import CognitiveEngine

__all__ = ["CognitiveEngine"]
