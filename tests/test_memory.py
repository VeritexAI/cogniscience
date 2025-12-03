"""
Unit tests for memory classes.

Tests VectorMemory, GraphMemory, and CacheMemory classes.
"""

import numpy as np
import pytest
from cognigraph.memory import VectorMemory, GraphMemory, CacheMemory


class TestVectorMemory:
    """Test VectorMemory class."""
    
    def test_initialization(self):
        """Test vector memory initialization."""
        vm = VectorMemory(N=10, d=5, random_seed=42)
        
        assert vm.N == 10
        assert vm.d == 5
        assert vm.vectors.shape == (10, 5)
    
    def test_vectors_normalized(self):
        """Test that vectors are normalized to unit length."""
        vm = VectorMemory(N=10, d=5, random_seed=42)
        
        norms = np.linalg.norm(vm.vectors, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_update_maintains_normalization(self):
        """Test that updates maintain normalization."""
        vm = VectorMemory(N=5, d=3, random_seed=42)
        
        # Apply random update
        delta = np.random.randn(5, 3) * 0.1
        vm.update(delta)
        
        # Should still be normalized
        norms = np.linalg.norm(vm.vectors, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_set_vectors(self):
        """Test setting vectors."""
        vm = VectorMemory(N=3, d=2, random_seed=42)
        
        new_vectors = np.array([[1, 1], [2, 0], [0, 3]], dtype=float)
        vm.set_vectors(new_vectors)
        
        # Should be normalized
        norms = np.linalg.norm(vm.vectors, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_reproducibility(self):
        """Test that random seed produces reproducible results."""
        vm1 = VectorMemory(N=5, d=3, random_seed=123)
        vm2 = VectorMemory(N=5, d=3, random_seed=123)
        
        assert np.allclose(vm1.vectors, vm2.vectors)


class TestGraphMemory:
    """Test GraphMemory class."""
    
    def test_initialization(self):
        """Test graph memory initialization."""
        gm = GraphMemory(N=10, random_seed=42)
        
        assert gm.N == 10
        assert gm.adjacency.shape == (10, 10)
    
    def test_symmetric_adjacency(self):
        """Test that adjacency matrix is symmetric."""
        gm = GraphMemory(N=5, random_seed=42)
        
        assert np.allclose(gm.adjacency, gm.adjacency.T)
    
    def test_zero_diagonal(self):
        """Test that diagonal is zero (no self-loops)."""
        gm = GraphMemory(N=5, random_seed=42)
        
        assert np.allclose(np.diag(gm.adjacency), 0)
    
    def test_initial_weights_small(self):
        """Test that initial weights are small."""
        gm = GraphMemory(N=10, init_strength=0.1, random_seed=42)
        
        assert np.all(np.abs(gm.adjacency) <= 0.1)
    
    def test_update_maintains_symmetry(self):
        """Test that updates maintain symmetry."""
        gm = GraphMemory(N=5, random_seed=42)
        
        # Apply asymmetric update
        delta = np.random.randn(5, 5) * 0.1
        gm.update(delta)
        
        # Should still be symmetric
        assert np.allclose(gm.adjacency, gm.adjacency.T)
    
    def test_squashing_bounds_weights(self):
        """Test that tanh squashing keeps weights in [-1, 1]."""
        gm = GraphMemory(N=5, random_seed=42)
        
        # Apply large update
        delta = np.random.randn(5, 5) * 10
        gm.update(delta)
        
        # Should be bounded
        assert np.all(gm.adjacency >= -1.0)
        assert np.all(gm.adjacency <= 1.0)
    
    def test_density_calculation(self):
        """Test density calculation."""
        gm = GraphMemory(N=5, random_seed=42)
        
        density = gm.get_density(threshold=0.01)
        
        assert 0 <= density <= 1.0
    
    def test_degree_distribution(self):
        """Test degree distribution."""
        gm = GraphMemory(N=5, random_seed=42)
        
        degrees = gm.get_degree_distribution()
        
        assert degrees.shape == (5,)
        assert np.all(degrees >= 0)


class TestCacheMemory:
    """Test CacheMemory class."""
    
    def test_initialization(self):
        """Test cache memory initialization."""
        cm = CacheMemory(d=5)
        
        assert cm.d == 5
        assert cm.is_empty()
    
    def test_add_input(self):
        """Test adding input to cache."""
        cm = CacheMemory(d=3)
        
        embedding = np.array([1, 2, 3], dtype=float)
        cm.add_input(embedding, node_index=0)
        
        assert cm.get_size() == 1
        assert cm.get_active_indices() == [0]
    
    def test_multiple_inputs(self):
        """Test adding multiple inputs."""
        cm = CacheMemory(d=2)
        
        cm.add_input(np.array([1, 0], dtype=float), node_index=0)
        cm.add_input(np.array([0, 1], dtype=float), node_index=1)
        
        assert cm.get_size() == 2
        cache = cm.get_cache()
        assert cache.shape == (2, 2)
    
    def test_set_cache(self):
        """Test setting entire cache."""
        cm = CacheMemory(d=3)
        
        embeddings = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        indices = [0, 2]
        cm.set_cache(embeddings, indices)
        
        assert cm.get_size() == 2
        assert cm.get_active_indices() == [0, 2]
    
    def test_clear(self):
        """Test clearing cache."""
        cm = CacheMemory(d=3)
        
        cm.add_input(np.array([1, 2, 3], dtype=float))
        cm.clear()
        
        assert cm.is_empty()
        assert cm.get_size() == 0
    
    def test_unassociated_indices(self):
        """Test cache entries without associated nodes."""
        cm = CacheMemory(d=2)
        
        cm.add_input(np.array([1, 0], dtype=float))  # No node_index
        
        assert cm.get_active_indices() == [-1]
    
    def test_get_cache_copy(self):
        """Test that get_cache returns a copy."""
        cm = CacheMemory(d=2)
        
        embedding = np.array([1, 2], dtype=float)
        cm.add_input(embedding)
        
        cache = cm.get_cache()
        cache[0, 0] = 999  # Modify copy
        
        # Original should be unchanged
        assert cm.get_cache()[0, 0] != 999


class TestMemoryIntegration:
    """Test integration between memory classes."""
    
    def test_compatible_dimensions(self):
        """Test that memory classes work with same dimensions."""
        N, d = 10, 5
        
        vm = VectorMemory(N, d, random_seed=42)
        gm = GraphMemory(N, random_seed=42)
        cm = CacheMemory(d)
        
        # Vector memory and cache have compatible dimensions
        assert vm.d == cm.d
        
        # Vector memory and graph have compatible node count
        assert vm.N == gm.N
    
    def test_cache_to_vector_association(self):
        """Test association between cache and vector memory."""
        N, d = 5, 3
        
        vm = VectorMemory(N, d, random_seed=42)
        cm = CacheMemory(d)
        
        # Add cache entries associated with specific nodes
        for i in range(3):
            # Perturb vector slightly
            perturbed = vm.vectors[i] + np.random.randn(d) * 0.1
            cm.add_input(perturbed, node_index=i)
        
        assert cm.get_size() == 3
        indices = cm.get_active_indices()
        assert all(0 <= idx < N for idx in indices)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
