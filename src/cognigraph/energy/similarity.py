"""
Similarity metrics for the Cognitive Engine.

Implements cosine similarity as defined in the paper:
σ(v_i, v_j) = v_i^T v_j / (||v_i|| ||v_j||) = u_i^T u_j

For normalized vectors on the unit sphere, this simplifies to dot product.
"""

import numpy as np


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for all vector pairs.
    
    For normalized vectors (||v_i|| = 1), this is simply the dot product matrix.
    
    Args:
        vectors: Shape (N, d) - N normalized vectors of dimension d
        
    Returns:
        np.ndarray: Shape (N, N) - similarity matrix where S_ij = σ(v_i, v_j)
    """
    # For unit vectors: v_i^T v_j
    return vectors @ vectors.T


def cosine_similarity_pairwise(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: Shape (d,) - first vector
        v2: Shape (d,) - second vector
        
    Returns:
        float: Cosine similarity in [-1, 1]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def cosine_similarity_with_cache(vectors: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Compute similarity between vector memory and cache entries.
    
    Args:
        vectors: Shape (N, d) - vector memory (normalized)
        cache: Shape (k, d) - cache entries (may not be normalized)
        
    Returns:
        np.ndarray: Shape (N, k) - similarity matrix where S_ij = σ(v_i, c_j)
    """
    # Normalize cache for similarity computation
    cache_norms = np.linalg.norm(cache, axis=1, keepdims=True)
    cache_norms = np.maximum(cache_norms, 1e-10)  # Avoid division by zero
    cache_normalized = cache / cache_norms
    
    # Compute similarities: (N, d) @ (d, k) = (N, k)
    return vectors @ cache_normalized.T
