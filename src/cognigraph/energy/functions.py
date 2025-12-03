"""
Energy functions for the Cognitive Engine.

Implements the three components of total energy E_t = E_C + E_V + E_G:

1. E_C: Cache-Vector alignment energy
   E_C = (1/2) Σ ||c_i - v̂_i||²
   
2. E_V: Vector-field coherence energy
   E_V = -(1/2) Σ_{i,j} σ(v_i, v_j)
   
3. E_G: Graph-structural energy
   E_G = -(1/2) Σ_{i,j} G_ij σ(v_i, v_j)

All formulas from Section 4 and Appendix A of the paper.
"""

import numpy as np
from cognigraph.energy.similarity import cosine_similarity_matrix, cosine_similarity_with_cache


def energy_cache(vectors: np.ndarray, cache: np.ndarray, 
                 cache_indices: np.ndarray) -> float:
    """
    Compute cache-vector misalignment energy E_C.
    
    E_C = (1/2) Σ ||c_i - v̂_i||²
    
    where v̂_i is the vector most similar to cache entry c_i (or the
    vector at the specified index if cache_indices are provided).
    
    Args:
        vectors: Shape (N, d) - normalized vector memory
        cache: Shape (k, d) - cache entries
        cache_indices: Shape (k,) - indices of associated vectors (-1 for unassociated)
        
    Returns:
        float: Cache-vector alignment energy
    """
    if len(cache) == 0:
        return 0.0
    
    energy = 0.0
    
    for i, c_i in enumerate(cache):
        # Get associated vector
        idx = cache_indices[i]
        
        if idx >= 0 and idx < len(vectors):
            # Use specified vector
            v_hat = vectors[idx]
        else:
            # Find nearest vector by similarity
            similarities = vectors @ c_i  # (N,) dot products
            idx = np.argmax(similarities)
            v_hat = vectors[idx]
        
        # ||c_i - v̂_i||²
        diff = c_i - v_hat
        energy += 0.5 * np.dot(diff, diff)
    
    return energy


def energy_vector(vectors: np.ndarray) -> float:
    """
    Compute vector-field coherence energy E_V.
    
    E_V = -(1/2) Σ_{i,j} σ(v_i, v_j)
    
    The negative sign ensures that high similarity lowers energy.
    
    Args:
        vectors: Shape (N, d) - normalized vector memory
        
    Returns:
        float: Vector coherence energy
    """
    # Compute similarity matrix
    S = cosine_similarity_matrix(vectors)
    
    # E_V = -(1/2) Σ_{i,j} S_ij
    # For symmetric matrix, sum over all entries and divide by 2
    energy = -0.5 * np.sum(S)
    
    return energy


def energy_graph(vectors: np.ndarray, adjacency: np.ndarray) -> float:
    """
    Compute graph-structural energy E_G.
    
    E_G = -(1/2) Σ_{i,j} G_ij σ(v_i, v_j)
    
    This measures how well graph structure aligns with vector similarity.
    Strong edges between similar vectors lower energy.
    
    Args:
        vectors: Shape (N, d) - normalized vector memory
        adjacency: Shape (N, N) - symmetric adjacency matrix G
        
    Returns:
        float: Graph-structural energy
    """
    # Compute similarity matrix
    S = cosine_similarity_matrix(vectors)
    
    # E_G = -(1/2) Σ_{i,j} G_ij S_ij
    # Element-wise product and sum
    energy = -0.5 * np.sum(adjacency * S)
    
    return energy


def compute_total_energy(vectors: np.ndarray, adjacency: np.ndarray,
                        cache: np.ndarray, cache_indices: np.ndarray) -> dict:
    """
    Compute total system energy and components.
    
    E_t = E_C + E_V + E_G
    
    Args:
        vectors: Shape (N, d) - normalized vector memory
        adjacency: Shape (N, N) - symmetric adjacency matrix
        cache: Shape (k, d) - cache entries
        cache_indices: Shape (k,) - associated vector indices
        
    Returns:
        dict: Energy components with keys 'E_C', 'E_V', 'E_G', 'E_total'
    """
    E_C = energy_cache(vectors, cache, cache_indices)
    E_V = energy_vector(vectors)
    E_G = energy_graph(vectors, adjacency)
    E_total = E_C + E_V + E_G
    
    return {
        'E_C': E_C,
        'E_V': E_V,
        'E_G': E_G,
        'E_total': E_total
    }
