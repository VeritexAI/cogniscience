"""
Update dynamics for the Cognitive Engine.

Implements gradient descent updates from Section 5:

Vector update:
Δv_i = η₁(c_i - v_i) + η₂ Σ_j (1 + G_ij) ∂σ/∂v_i

Graph update (Hebbian + decay):
ΔG_ij = η₃ σ(v_i, v_j) - λG_ij

With normalization:
v_i ← v_i/||v_i||
G_ij ← tanh(G_ij)
"""

import numpy as np
from typing import Tuple


def apply_updates(vectors: np.ndarray, adjacency: np.ndarray,
                 grad_V: np.ndarray, grad_G: np.ndarray,
                 eta1: float, eta2: float, eta3: float,
                 lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply gradient descent updates to vectors and graph.
    
    Updates:
    V ← V - η * ∇_V E_t
    G ← G - η * ∇_G E_t - λG (decay term)
    
    Then normalize vectors and squash graph weights.
    
    Args:
        vectors: Shape (N, d) - current vectors
        adjacency: Shape (N, N) - current adjacency matrix
        grad_V: Shape (N, d) - gradient wrt vectors
        grad_G: Shape (N, N) - gradient wrt adjacency
        eta1: Learning rate for cache-driven updates
        eta2: Learning rate for vector field + graph updates
        eta3: Learning rate for graph updates
        lam: Decay coefficient for graph edges
        
    Returns:
        Tuple of updated (vectors, adjacency)
    """
    # Update vectors with gradient descent
    # Note: The gradients already combine cache, field, and graph terms
    # For now, use eta2 as the overall rate (can refine later)
    new_vectors = vectors - eta2 * grad_V
    
    # Normalize vectors to unit sphere
    norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    new_vectors = new_vectors / (norms + 1e-8)
    
    # Update graph with Hebbian reinforcement + decay
    # ΔG = -η₃ * ∇_G E_G - λG
    # Since ∇_G E_G = -σ(v_i, v_j), we have:
    # ΔG = η₃ * σ(v_i, v_j) - λG
    # This is: -η₃ * grad_G - λG
    new_adjacency = adjacency - eta3 * grad_G - lam * adjacency
    
    # Enforce symmetry
    new_adjacency = (new_adjacency + new_adjacency.T) / 2
    
    # Squash with tanh to keep bounded
    new_adjacency = np.tanh(new_adjacency)
    
    # Zero diagonal
    np.fill_diagonal(new_adjacency, 0)
    
    return new_vectors, new_adjacency


def apply_vector_update_only(vectors: np.ndarray, grad_V: np.ndarray,
                             eta: float) -> np.ndarray:
    """
    Apply update to vectors only (useful for certain experiments).
    
    Args:
        vectors: Shape (N, d) - current vectors
        grad_V: Shape (N, d) - gradient wrt vectors
        eta: Learning rate
        
    Returns:
        np.ndarray: Updated normalized vectors
    """
    new_vectors = vectors - eta * grad_V
    
    # Normalize
    norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    new_vectors = new_vectors / (norms + 1e-8)
    
    return new_vectors


def apply_graph_update_only(adjacency: np.ndarray, grad_G: np.ndarray,
                            eta: float, lam: float) -> np.ndarray:
    """
    Apply update to graph only (useful for certain experiments).
    
    Args:
        adjacency: Shape (N, N) - current adjacency
        grad_G: Shape (N, N) - gradient wrt adjacency
        eta: Learning rate
        lam: Decay coefficient
        
    Returns:
        np.ndarray: Updated adjacency matrix
    """
    new_adjacency = adjacency - eta * grad_G - lam * adjacency
    
    # Enforce symmetry and squash
    new_adjacency = (new_adjacency + new_adjacency.T) / 2
    new_adjacency = np.tanh(new_adjacency)
    np.fill_diagonal(new_adjacency, 0)
    
    return new_adjacency


def compute_gradient_norm(grad_V: np.ndarray, grad_G: np.ndarray) -> float:
    """
    Compute total gradient magnitude for convergence monitoring.
    
    ||∇E_t|| = sqrt(||∇_V E||² + ||∇_G E||²)
    
    Args:
        grad_V: Shape (N, d) - gradient wrt vectors
        grad_G: Shape (N, N) - gradient wrt adjacency
        
    Returns:
        float: Total gradient norm
    """
    norm_V = np.linalg.norm(grad_V)
    norm_G = np.linalg.norm(grad_G)
    
    return np.sqrt(norm_V**2 + norm_G**2)
