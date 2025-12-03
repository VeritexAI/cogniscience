"""
Unit tests for energy functions.

Verifies that E_C, E_V, E_G are computed correctly according to paper formulas.
"""

import numpy as np
import pytest
from cognigraph.energy.functions import energy_cache, energy_vector, energy_graph, compute_total_energy
from cognigraph.energy.similarity import cosine_similarity_matrix


class TestSimilarity:
    """Test similarity computations."""
    
    def test_cosine_similarity_identity(self):
        """Test that identical normalized vectors have similarity 1."""
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        S = cosine_similarity_matrix(v)
        
        # Diagonal should be all 1s
        assert np.allclose(np.diag(S), 1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test that orthogonal vectors have similarity 0."""
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        S = cosine_similarity_matrix(v)
        
        # Off-diagonal should be all 0s for orthogonal vectors
        S_off = S.copy()
        np.fill_diagonal(S_off, 0)
        assert np.allclose(S_off, 0.0)
    
    def test_cosine_similarity_symmetric(self):
        """Test that similarity matrix is symmetric."""
        np.random.seed(42)
        v = np.random.randn(5, 3)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        
        S = cosine_similarity_matrix(v)
        assert np.allclose(S, S.T)
    
    def test_cosine_similarity_bounded(self):
        """Test that similarities are in [-1, 1]."""
        np.random.seed(42)
        v = np.random.randn(10, 5)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        
        S = cosine_similarity_matrix(v)
        # Allow small numerical tolerance
        assert np.all(S >= -1.0 - 1e-10) and np.all(S <= 1.0 + 1e-10)


class TestEnergyCacheVector:
    """Test cache-vector alignment energy E_C."""
    
    def test_zero_energy_perfect_match(self):
        """Test E_C = 0 when cache matches vectors exactly."""
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cache = np.array([[1, 0, 0]], dtype=float)  # Matches first vector
        cache_indices = np.array([0])
        
        E_C = energy_cache(vectors, cache, cache_indices)
        assert np.isclose(E_C, 0.0)
    
    def test_positive_energy_mismatch(self):
        """Test E_C > 0 when cache doesn't match vectors."""
        vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        cache = np.array([[0.5, 0.5, 0]], dtype=float)  # Doesn't match either
        cache_indices = np.array([0])
        
        E_C = energy_cache(vectors, cache, cache_indices)
        assert E_C > 0
    
    def test_empty_cache_zero_energy(self):
        """Test E_C = 0 when cache is empty."""
        vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        cache = np.empty((0, 3))
        cache_indices = np.array([])
        
        E_C = energy_cache(vectors, cache, cache_indices)
        assert np.isclose(E_C, 0.0)
    
    def test_energy_formula(self):
        """Test E_C = (1/2) ||c - v||^2."""
        vectors = np.array([[1, 0, 0]], dtype=float)
        cache = np.array([[0.8, 0.2, 0.1]], dtype=float)
        cache_indices = np.array([0])
        
        # Manual computation
        diff = cache[0] - vectors[0]
        expected = 0.5 * np.dot(diff, diff)
        
        E_C = energy_cache(vectors, cache, cache_indices)
        assert np.isclose(E_C, expected)


class TestEnergyVector:
    """Test vector-field coherence energy E_V."""
    
    def test_negative_energy(self):
        """Test that E_V is negative (similarity lowers energy)."""
        np.random.seed(42)
        vectors = np.random.randn(5, 3)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        E_V = energy_vector(vectors)
        assert E_V < 0  # Should be negative due to negative sign in formula
    
    def test_orthogonal_vectors(self):
        """Test E_V for orthogonal vectors."""
        # Orthogonal vectors have zero similarity except diagonal
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        
        E_V = energy_vector(vectors)
        # E_V = -(1/2) * Σ σ_ij
        # For orthogonal: only diagonal is 1, so sum = 3
        expected = -0.5 * 3
        assert np.isclose(E_V, expected)
    
    def test_identical_vectors_high_negative(self):
        """Test that identical vectors give large negative energy."""
        # All vectors pointing same direction
        vectors = np.tile([1, 0, 0], (5, 1)).astype(float)
        
        E_V = energy_vector(vectors)
        # All pairs have similarity 1, so sum = 5*5 = 25
        expected = -0.5 * 25
        assert np.isclose(E_V, expected)
    
    def test_formula_consistency(self):
        """Test E_V = -(1/2) Σ σ(v_i, v_j)."""
        np.random.seed(123)
        vectors = np.random.randn(4, 3)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Manual computation
        S = cosine_similarity_matrix(vectors)
        expected = -0.5 * np.sum(S)
        
        E_V = energy_vector(vectors)
        assert np.isclose(E_V, expected)


class TestEnergyGraph:
    """Test graph-structural energy E_G."""
    
    def test_zero_adjacency_zero_energy(self):
        """Test E_G = 0 when graph has no edges."""
        vectors = np.random.randn(5, 3)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        adjacency = np.zeros((5, 5))
        
        E_G = energy_graph(vectors, adjacency)
        assert np.isclose(E_G, 0.0)
    
    def test_negative_energy_positive_edges(self):
        """Test E_G < 0 when edges connect similar vectors."""
        # Two similar vectors with edge between them
        vectors = np.array([[1, 0, 0], [0.9, 0.1, 0]], dtype=float)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Strong edge between them
        adjacency = np.array([[0, 0.5], [0.5, 0]], dtype=float)
        
        E_G = energy_graph(vectors, adjacency)
        assert E_G < 0  # Should be negative
    
    def test_formula_consistency(self):
        """Test E_G = -(1/2) Σ G_ij σ(v_i, v_j)."""
        np.random.seed(456)
        vectors = np.random.randn(4, 3)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        adjacency = np.random.uniform(-0.1, 0.1, (4, 4))
        adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
        np.fill_diagonal(adjacency, 0)
        
        # Manual computation
        S = cosine_similarity_matrix(vectors)
        expected = -0.5 * np.sum(adjacency * S)
        
        E_G = energy_graph(vectors, adjacency)
        assert np.isclose(E_G, expected)
    
    def test_hebbian_interpretation(self):
        """Test that strong edges between similar nodes lower energy."""
        # Create two similar and two dissimilar vectors
        v1 = np.array([1, 0, 0], dtype=float)
        v2 = np.array([0.9, 0.1, 0], dtype=float)
        v3 = np.array([0, 1, 0], dtype=float)
        v4 = np.array([0, 0.9, 0.1], dtype=float)
        
        vectors = np.vstack([v1, v2, v3, v4])
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Edges between similar pairs
        adjacency = np.zeros((4, 4))
        adjacency[0, 1] = adjacency[1, 0] = 0.5  # Similar pair
        adjacency[2, 3] = adjacency[3, 2] = 0.5  # Similar pair
        
        E_G_aligned = energy_graph(vectors, adjacency)
        
        # Edges between dissimilar pairs
        adjacency_bad = np.zeros((4, 4))
        adjacency_bad[0, 2] = adjacency_bad[2, 0] = 0.5  # Dissimilar
        adjacency_bad[1, 3] = adjacency_bad[3, 1] = 0.5  # Dissimilar
        
        E_G_misaligned = energy_graph(vectors, adjacency_bad)
        
        # Aligned edges should give lower (more negative) energy
        assert E_G_aligned < E_G_misaligned


class TestTotalEnergy:
    """Test total energy computation."""
    
    def test_energy_components_sum(self):
        """Test E_total = E_C + E_V + E_G."""
        np.random.seed(789)
        N, d = 5, 3
        
        vectors = np.random.randn(N, d)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        adjacency = np.random.uniform(-0.1, 0.1, (N, N))
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 0)
        
        cache = np.random.randn(2, d)
        cache_indices = np.array([0, 2])
        
        # Compute components separately
        E_C = energy_cache(vectors, cache, cache_indices)
        E_V = energy_vector(vectors)
        E_G = energy_graph(vectors, adjacency)
        expected_total = E_C + E_V + E_G
        
        # Compute via function
        result = compute_total_energy(vectors, adjacency, cache, cache_indices)
        
        assert np.isclose(result['E_C'], E_C)
        assert np.isclose(result['E_V'], E_V)
        assert np.isclose(result['E_G'], E_G)
        assert np.isclose(result['E_total'], expected_total)
    
    def test_energy_dict_keys(self):
        """Test that energy dict has all required keys."""
        vectors = np.eye(3)
        adjacency = np.zeros((3, 3))
        cache = np.empty((0, 3))
        cache_indices = np.array([])
        
        result = compute_total_energy(vectors, adjacency, cache, cache_indices)
        
        assert 'E_C' in result
        assert 'E_V' in result
        assert 'E_G' in result
        assert 'E_total' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
