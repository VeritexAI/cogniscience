"""
Vector Memory: Long-term semantic embeddings on the unit sphere.

Each vector v_i represents a concept/memory in a normalized d-dimensional space.
Vectors are constrained to the unit sphere: ||v_i|| = 1 for all i.
"""

import numpy as np


class VectorMemory:
    """
    Long-term semantic memory represented as normalized vectors on unit sphere.
    
    Attributes:
        vectors (np.ndarray): Shape (N, d) - N normalized d-dimensional vectors
        N (int): Number of concept nodes
        d (int): Dimensionality of each vector
    """
    
    def __init__(self, N: int, d: int, random_seed: int = None):
        """
        Initialize vector memory with random points on unit sphere.
        
        Args:
            N: Number of nodes/concepts
            d: Dimensionality of each vector
            random_seed: Optional seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.N = N
        self.d = d
        
        # Initialize with random vectors on unit sphere
        # Sample from normal distribution and normalize
        self.vectors = np.random.randn(N, d)
        self._normalize()
    
    def _normalize(self, epsilon: float = 1e-8):
        """
        Normalize all vectors to unit length.
        
        Args:
            epsilon: Small constant to prevent division by zero
        """
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norms + epsilon)
    
    def update(self, delta: np.ndarray):
        """
        Apply update to vectors and re-normalize.
        
        Args:
            delta: Shape (N, d) - update to add to vectors
        """
        self.vectors += delta
        self._normalize()
    
    def get_vectors(self) -> np.ndarray:
        """
        Get current vector state.
        
        Returns:
            np.ndarray: Shape (N, d) normalized vectors
        """
        return self.vectors.copy()
    
    def set_vectors(self, vectors: np.ndarray):
        """
        Set vectors and normalize.
        
        Args:
            vectors: Shape (N, d) - new vectors
        """
        assert vectors.shape == (self.N, self.d), f"Expected shape {(self.N, self.d)}, got {vectors.shape}"
        self.vectors = vectors.copy()
        self._normalize()
    
    def get_unit_vectors(self) -> np.ndarray:
        """
        Get normalized unit vectors (same as get_vectors since always normalized).
        
        Returns:
            np.ndarray: Shape (N, d) unit vectors
        """
        return self.vectors
    
    def __repr__(self):
        return f"VectorMemory(N={self.N}, d={self.d})"
