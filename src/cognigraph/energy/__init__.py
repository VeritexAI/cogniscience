"""Energy functions and gradients for the Cognitive Engine."""

from cognigraph.energy.functions import compute_total_energy, energy_cache, energy_graph, energy_vector
from cognigraph.energy.gradients import compute_gradients
from cognigraph.energy.similarity import cosine_similarity_matrix

__all__ = [
    "energy_cache",
    "energy_vector", 
    "energy_graph",
    "compute_total_energy",
    "compute_gradients",
    "cosine_similarity_matrix",
]
