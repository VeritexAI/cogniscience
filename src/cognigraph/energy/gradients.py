"""
Gradient computations for the Cognitive Engine.

Implements all partial derivatives from Appendix A of the paper:

∂E_C/∂v_i = v_i - c_i

∂E_V/∂v_i = -(1/||v_i||) Σ_j (I - u_i u_i^T) u_j
           where u_i = v_i/||v_i||

∂E_G/∂G_ij = -σ(v_i, v_j)

∂E_G/∂v_i = -(1/||v_i||) Σ_j G_ij (I - u_i u_i^T) u_j
"""

import numpy as np
from cognigraph.energy.similarity import cosine_similarity_matrix


def gradient_cache_wrt_vectors(vectors: np.ndarray, cache: np.ndarray,
                               cache_indices: np.ndarray) -> np.ndarray:
    """
    Compute ∂E_C/∂v_i for each vector.
    
    From Appendix A.1:
    ∂E_C/∂v_i = v_i - c_i (when v_i is matched to c_i)
    
    Args:
        vectors: Shape (N, d) - normalized vectors
        cache: Shape (k, d) - cache entries
        cache_indices: Shape (k,) - associated vector indices
        
    Returns:
        np.ndarray: Shape (N, d) - gradient for each vector
    """
    N, d = vectors.shape
    grad = np.zeros((N, d))
    
    if len(cache) == 0:
        return grad
    
    for i, c_i in enumerate(cache):
        idx = cache_indices[i]
        
        if idx >= 0 and idx < N:
            # ∂E_C/∂v_idx = v_idx - c_i
            grad[idx] += vectors[idx] - c_i
        else:
            # Find nearest vector
            similarities = vectors @ c_i
            idx = np.argmax(similarities)
            grad[idx] += vectors[idx] - c_i
    
    return grad


def tangent_space_projection(vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    """
    Project target vectors onto tangent space of unit sphere at each point.
    
    For vector v_i on unit sphere, tangent space projection is:
    P_i = I - u_i u_i^T where u_i = v_i/||v_i||
    
    Result: (I - u_i u_i^T) @ target_i for each i
    
    Args:
        vectors: Shape (N, d) - normalized vectors (points on sphere)
        target_vectors: Shape (N, d) - vectors to project
        
    Returns:
        np.ndarray: Shape (N, d) - projected vectors
    """
    # For normalized vectors, u_i = v_i
    # Compute u_i u_i^T @ target for each i
    # This is the component along u_i
    radial_component = np.sum(vectors * target_vectors, axis=1, keepdims=True) * vectors
    
    # Tangent component = total - radial
    tangent_component = target_vectors - radial_component
    
    return tangent_component


def gradient_vector_field_wrt_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Compute ∂E_V/∂v_i for each vector.
    
    From Appendix A.2:
    ∂E_V/∂v_i = -(1/||v_i||) Σ_j (I - u_i u_i^T) u_j
    
    For normalized vectors (||v_i|| = 1), this simplifies to:
    ∂E_V/∂v_i = -Σ_j (I - v_i v_i^T) v_j
    
    Args:
        vectors: Shape (N, d) - normalized vectors
        
    Returns:
        np.ndarray: Shape (N, d) - gradient for each vector
    """
    N, d = vectors.shape
    
    # Sum of all vectors: Σ_j v_j
    sum_vectors = np.sum(vectors, axis=0)  # Shape (d,)
    
    # For each v_i, compute (I - v_i v_i^T) @ sum_vectors
    # This projects sum_vectors onto tangent space at v_i
    target = np.tile(sum_vectors, (N, 1))  # Shape (N, d)
    projected = tangent_space_projection(vectors, target)
    
    # Negative sign from energy formula
    grad = -projected
    
    return grad


def gradient_graph_wrt_adjacency(vectors: np.ndarray) -> np.ndarray:
    """
    Compute ∂E_G/∂G_ij for each edge.
    
    From Appendix A.3:
    ∂E_G/∂G_ij = -σ(v_i, v_j)
    
    Args:
        vectors: Shape (N, d) - normalized vectors
        
    Returns:
        np.ndarray: Shape (N, N) - gradient for adjacency matrix
    """
    # Similarity matrix
    S = cosine_similarity_matrix(vectors)
    
    # ∂E_G/∂G = -S
    return -S


def gradient_graph_wrt_vectors(vectors: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """
    Compute ∂E_G/∂v_i for each vector (contribution from graph).
    
    From Appendix A.3:
    ∂E_G/∂v_i = -(1/||v_i||) Σ_j G_ij (I - v_i v_i^T) v_j
    
    For normalized vectors:
    ∂E_G/∂v_i = -Σ_j G_ij (I - v_i v_i^T) v_j
    
    Args:
        vectors: Shape (N, d) - normalized vectors
        adjacency: Shape (N, N) - symmetric adjacency matrix
        
    Returns:
        np.ndarray: Shape (N, d) - gradient for each vector
    """
    N, d = vectors.shape
    
    # Weighted sum for each node: Σ_j G_ij v_j
    weighted_neighbors = adjacency @ vectors  # Shape (N, d)
    
    # Project onto tangent space at each v_i
    projected = tangent_space_projection(vectors, weighted_neighbors)
    
    # Negative sign from energy formula
    grad = -projected
    
    return grad


def compute_gradients(vectors: np.ndarray, adjacency: np.ndarray,
                     cache: np.ndarray, cache_indices: np.ndarray,
                     weights: dict = None) -> dict:
    """
    Compute all gradients for the Cognitive Engine.
    
    Returns gradients:
    - grad_V: ∂E_t/∂V (from E_C, E_V, and E_G)
    - grad_G: ∂E_t/∂G (from E_G only)
    
    Args:
        vectors: Shape (N, d) - normalized vectors
        adjacency: Shape (N, N) - symmetric adjacency matrix
        cache: Shape (k, d) - cache entries
        cache_indices: Shape (k,) - associated vector indices
        weights: Optional dict with keys 'eta1', 'eta2', 'eta3' for weighting terms
        
    Returns:
        dict: Gradients with keys 'grad_V' and 'grad_G'
    """
    if weights is None:
        weights = {'eta1': 1.0, 'eta2': 1.0, 'eta3': 1.0}
    
    # Gradient of E_C with respect to V
    grad_C = gradient_cache_wrt_vectors(vectors, cache, cache_indices)
    
    # Gradient of E_V with respect to V
    grad_V_field = gradient_vector_field_wrt_vectors(vectors)
    
    # Gradient of E_G with respect to V
    grad_G_vectors = gradient_graph_wrt_vectors(vectors, adjacency)
    
    # Combined gradient for vectors
    # Note: In the paper, eta1 is for cache, eta2 for vector field + graph
    # We separate them here for flexibility
    grad_V = grad_C + grad_V_field + grad_G_vectors
    
    # Gradient of E_G with respect to G
    grad_G = gradient_graph_wrt_adjacency(vectors)
    
    return {
        'grad_V': grad_V,
        'grad_G': grad_G,
        'grad_V_cache': grad_C,
        'grad_V_field': grad_V_field,
        'grad_V_graph': grad_G_vectors
    }
