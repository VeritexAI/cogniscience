"""
Cache Memory: Volatile short-term working context.

The cache holds recent experiences/inputs that are being actively processed.
Unlike vector memory, cache is not normalized and has high volatility (can change
rapidly). Cache inputs drive consolidation into long-term vector memory.
"""

import numpy as np
from typing import Optional, List


class CacheMemory:
    """
    Short-term working memory for active context.
    
    Attributes:
        cache (np.ndarray): Shape (k, d) - k cache entries of dimension d
        k (int): Number of cache slots (can vary)
        d (int): Dimensionality matching vector memory
        active_indices (List[int]): Indices of vector nodes currently in cache
    """
    
    def __init__(self, d: int):
        """
        Initialize empty cache.
        
        Args:
            d: Dimensionality matching vector memory
        """
        self.d = d
        self.cache = np.empty((0, d))
        self.active_indices = []
    
    def add_input(self, embedding: np.ndarray, node_index: Optional[int] = None):
        """
        Add new input to cache.
        
        Args:
            embedding: Shape (d,) - embedding vector to cache
            node_index: Optional index of associated node in vector memory
        """
        assert embedding.shape == (self.d,), f"Expected shape ({self.d},), got {embedding.shape}"
        
        # Add to cache
        self.cache = np.vstack([self.cache, embedding.reshape(1, -1)])
        
        # Track associated node if provided
        if node_index is not None:
            self.active_indices.append(node_index)
        else:
            self.active_indices.append(-1)  # -1 indicates no association
    
    def set_cache(self, embeddings: np.ndarray, indices: Optional[List[int]] = None):
        """
        Replace entire cache with new inputs.
        
        Args:
            embeddings: Shape (k, d) - new cache contents
            indices: Optional list of associated node indices
        """
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.d, \
            f"Expected shape (k, {self.d}), got {embeddings.shape}"
        
        self.cache = embeddings.copy()
        
        if indices is not None:
            assert len(indices) == len(embeddings), "Number of indices must match cache size"
            self.active_indices = indices.copy()
        else:
            self.active_indices = [-1] * len(embeddings)
    
    def clear(self):
        """Clear all cache contents."""
        self.cache = np.empty((0, self.d))
        self.active_indices = []
    
    def get_cache(self) -> np.ndarray:
        """
        Get current cache contents.
        
        Returns:
            np.ndarray: Shape (k, d) - current cache
        """
        return self.cache.copy()
    
    def get_size(self) -> int:
        """
        Get number of items in cache.
        
        Returns:
            int: Current cache size k
        """
        return len(self.cache)
    
    def get_active_indices(self) -> List[int]:
        """
        Get indices of nodes associated with cache entries.
        
        Returns:
            List[int]: Node indices (-1 for unassociated entries)
        """
        return self.active_indices.copy()
    
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return len(self.cache) == 0
    
    def __repr__(self):
        return f"CacheMemory(k={len(self.cache)}, d={self.d})"
    
    def __len__(self):
        return len(self.cache)
