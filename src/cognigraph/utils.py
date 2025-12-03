"""
Utility functions for the Cognitive Engine.

Includes metrics computation, normalization helpers, and analysis tools.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import silhouette_score


def compute_metrics(vectors: np.ndarray, adjacency: np.ndarray,
                   grad_norm: float) -> Dict[str, float]:
    """
    Compute system metrics for monitoring and analysis.
    
    Metrics include:
    - Graph density: proportion of non-zero edges
    - Mean degree: average node connectivity
    - Cluster coherence: mean intra-cluster similarity (if clusters identified)
    - Drift index: measure of vector stability
    
    Args:
        vectors: Shape (N, d) - current vectors
        adjacency: Shape (N, N) - current adjacency
        grad_norm: Current gradient norm
        
    Returns:
        dict: Computed metrics
    """
    N = len(vectors)
    
    # Graph density
    threshold = 0.01
    mask = np.abs(adjacency) > threshold
    np.fill_diagonal(mask, False)
    num_edges = np.sum(mask) / 2
    max_edges = N * (N - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0.0
    
    # Mean degree
    degrees = np.sum(np.abs(adjacency), axis=1)
    mean_degree = np.mean(degrees)
    
    # Vector spread (how dispersed are vectors)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, 0)
    mean_similarity = np.mean(np.abs(similarities))
    
    metrics = {
        'density': density,
        'mean_degree': mean_degree,
        'mean_similarity': mean_similarity,
        'gradient_norm': grad_norm
    }
    
    return metrics


def estimate_cluster_coherence(vectors: np.ndarray, adjacency: np.ndarray,
                               threshold: float = 0.1) -> float:
    """
    Estimate cluster coherence based on graph structure.
    
    Uses graph-based clustering: nodes with strong connections form clusters.
    Coherence is the mean similarity within clusters.
    
    Args:
        vectors: Shape (N, d) - current vectors
        adjacency: Shape (N, N) - current adjacency
        threshold: Minimum edge strength for cluster membership
        
    Returns:
        float: Mean intra-cluster similarity
    """
    N = len(vectors)
    
    # Simple clustering: assign labels based on strongest connections
    # For each node, find its most connected neighbors
    clusters = np.arange(N)  # Start with each node in its own cluster
    
    # Merge based on strong edges
    strong_edges = adjacency > threshold
    
    # For simplicity, use connected components
    # (proper implementation would use community detection)
    for i in range(N):
        neighbors = np.where(strong_edges[i])[0]
        if len(neighbors) > 0:
            # Assign to cluster of strongest neighbor
            strongest = neighbors[np.argmax(adjacency[i, neighbors])]
            clusters[i] = min(clusters[i], clusters[strongest])
    
    # Compute intra-cluster similarity
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) == N:
        # No clustering occurred
        return 0.0
    
    coherences = []
    for cluster_id in unique_clusters:
        members = np.where(clusters == cluster_id)[0]
        if len(members) > 1:
            cluster_vectors = vectors[members]
            sim_matrix = cluster_vectors @ cluster_vectors.T
            # Mean of off-diagonal elements
            mask = ~np.eye(len(members), dtype=bool)
            coherences.append(np.mean(sim_matrix[mask]))
    
    return np.mean(coherences) if coherences else 0.0


def compute_drift_index(vectors_before: np.ndarray, vectors_after: np.ndarray) -> float:
    """
    Compute drift index: average change in vector positions.
    
    drift = (1/N) Σ ||v_i(t+1) - v_i(t)||
    
    Args:
        vectors_before: Shape (N, d) - vectors at time t
        vectors_after: Shape (N, d) - vectors at time t+1
        
    Returns:
        float: Mean drift magnitude
    """
    diffs = vectors_after - vectors_before
    norms = np.linalg.norm(diffs, axis=1)
    return np.mean(norms)


def analyze_edge_distribution(adjacency: np.ndarray, num_bins: int = 20) -> Dict:
    """
    Analyze distribution of edge strengths.
    
    Returns histogram and statistics of edge weights,
    useful for visualizing pruning and heavy-tail formation.
    
    Args:
        adjacency: Shape (N, N) - adjacency matrix
        num_bins: Number of histogram bins
        
    Returns:
        dict: Statistics including histogram, mean, std, etc.
    """
    # Get upper triangle values (symmetric matrix)
    N = len(adjacency)
    upper_tri = adjacency[np.triu_indices(N, k=1)]
    
    # Filter near-zero values
    significant = upper_tri[np.abs(upper_tri) > 1e-6]
    
    if len(significant) == 0:
        return {
            'hist': (np.zeros(num_bins), np.zeros(num_bins + 1)),
            'mean': 0.0,
            'std': 0.0,
            'max': 0.0,
            'num_significant': 0
        }
    
    hist, bin_edges = np.histogram(significant, bins=num_bins)
    
    return {
        'hist': (hist, bin_edges),
        'mean': np.mean(significant),
        'std': np.std(significant),
        'max': np.max(np.abs(significant)),
        'num_significant': len(significant)
    }
