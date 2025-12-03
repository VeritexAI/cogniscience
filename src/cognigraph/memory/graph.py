"""
Graph Memory: Dynamic relational structure encoding associations.

The graph is represented as a symmetric adjacency matrix G where G_ij represents
the strength of association between nodes i and j. Edges evolve through Hebbian
reinforcement and decay, bounded by tanh squashing.
"""

import numpy as np


class GraphMemory:
    """
    Relational memory represented as symmetric adjacency matrix.
    
    Attributes:
        adjacency (np.ndarray): Shape (N, N) - symmetric edge weights
        N (int): Number of nodes
    """
    
    def __init__(self, N: int, init_strength: float = 0.1, random_seed: int = None):
        """
        Initialize graph with weak random connections.
        
        Args:
            N: Number of nodes
            init_strength: Maximum initial edge strength (default 0.1)
            random_seed: Optional seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.N = N
        
        # Initialize with small random symmetric weights
        random_weights = np.random.uniform(-init_strength, init_strength, (N, N))
        # Make symmetric
        self.adjacency = (random_weights + random_weights.T) / 2
        # Zero diagonal (no self-loops)
        np.fill_diagonal(self.adjacency, 0)
    
    def update(self, delta: np.ndarray):
        """
        Apply update to adjacency matrix and maintain symmetry.
        
        Args:
            delta: Shape (N, N) - update to add to adjacency
        """
        self.adjacency += delta
        self._enforce_symmetry()
        self._squash_weights()
    
    def _enforce_symmetry(self):
        """Ensure adjacency matrix remains symmetric."""
        self.adjacency = (self.adjacency + self.adjacency.T) / 2
    
    def _squash_weights(self):
        """
        Apply tanh squashing to keep edge weights bounded in [-1, 1].
        This prevents unbounded growth and maintains numerical stability.
        """
        self.adjacency = np.tanh(self.adjacency)
        # Ensure diagonal remains zero
        np.fill_diagonal(self.adjacency, 0)
    
    def get_adjacency(self) -> np.ndarray:
        """
        Get current adjacency matrix.
        
        Returns:
            np.ndarray: Shape (N, N) symmetric adjacency matrix
        """
        return self.adjacency.copy()
    
    def set_adjacency(self, adjacency: np.ndarray):
        """
        Set adjacency matrix.
        
        Args:
            adjacency: Shape (N, N) - new adjacency matrix
        """
        assert adjacency.shape == (self.N, self.N), f"Expected shape {(self.N, self.N)}, got {adjacency.shape}"
        self.adjacency = adjacency.copy()
        self._enforce_symmetry()
        self._squash_weights()
    
    def get_density(self, threshold: float = 0.01) -> float:
        """
        Calculate graph density (proportion of non-zero edges).
        
        Args:
            threshold: Minimum absolute weight to consider edge as present
            
        Returns:
            float: Density in [0, 1]
        """
        # Count edges above threshold (excluding diagonal)
        mask = np.abs(self.adjacency) > threshold
        np.fill_diagonal(mask, False)
        num_edges = np.sum(mask) / 2  # Divide by 2 for symmetric matrix
        max_edges = self.N * (self.N - 1) / 2
        return num_edges / max_edges if max_edges > 0 else 0.0
    
    def get_degree_distribution(self) -> np.ndarray:
        """
        Get degree (sum of absolute edge weights) for each node.
        
        Returns:
            np.ndarray: Shape (N,) - degree of each node
        """
        return np.sum(np.abs(self.adjacency), axis=1)
    
    def __repr__(self):
        density = self.get_density()
        return f"GraphMemory(N={self.N}, density={density:.3f})"
