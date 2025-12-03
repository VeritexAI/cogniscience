"""
Tests for hierarchical concepts and recency-enhanced dreaming features.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock

from src.cognigraph.memory.semantic_memory import MemoryNode
from src.cognigraph.knowledge.concepts import ConceptNode, ConceptExtractor
from src.cognigraph.dynamics.dreaming import (
    pre_dream_rumination,
    activation_weighted_noise,
    controlled_dreaming_step_recency,
    compute_temporal_gradient,
    compute_combined_gradient
)


class TestTemporalActivation:
    """Tests for temporal activation tracking."""
    
    def test_memory_node_has_activation_fields(self):
        """Test that MemoryNode has activation tracking fields."""
        node = MemoryNode(
            node_index=0,
            db_id=None,
            text="test",
            timestamp=time.time()
        )
        
        assert hasattr(node, 'activation')
        assert hasattr(node, 'last_accessed')
        assert hasattr(node, 'access_history')
        assert node.activation == 1.0  # New nodes start fully activated
        assert node.last_accessed == 0.0
        assert isinstance(node.access_history, list)
    
    def test_pre_dream_rumination(self):
        """Test that pre-dream rumination boosts top-k memories."""
        activations = np.array([0.3, 0.8, 0.5, 0.9, 0.2])
        boosted = pre_dream_rumination(activations, top_k=2, rumination_boost=0.2)
        
        # Top 2 should be indices 3 (0.9) and 1 (0.8)
        assert boosted[3] == 1.0  # 0.9 + 0.2 capped at 1.0
        assert boosted[1] == 1.0  # 0.8 + 0.2 capped at 1.0
        assert boosted[2] == 0.5  # Unchanged
        assert boosted[0] == 0.3  # Unchanged
        assert boosted[4] == 0.2  # Unchanged


class TestActivationWeightedNoise:
    """Tests for activation-weighted noise generation."""
    
    def test_vector_noise_shape(self):
        """Test that vector noise has correct shape."""
        N, d = 10, 64
        activations = np.random.rand(N)
        rs = np.random.RandomState(42)
        
        noise = activation_weighted_noise(
            (N, d), activations, sigma_n=0.01, recency_bias=0.5, random_state=rs
        )
        
        assert noise.shape == (N, d)
        assert isinstance(noise, np.ndarray)
    
    def test_adjacency_noise_shape(self):
        """Test that adjacency noise has correct shape."""
        N = 10
        activations = np.random.rand(N)
        rs = np.random.RandomState(42)
        
        noise = activation_weighted_noise(
            (N, N), activations, sigma_n=0.01, recency_bias=0.5, random_state=rs
        )
        
        assert noise.shape == (N, N)
        assert isinstance(noise, np.ndarray)
    
    def test_recency_bias_affects_noise(self):
        """Test that recency bias weights noise by activation."""
        N, d = 10, 64
        activations = np.zeros(N)
        activations[5] = 1.0  # Only one highly activated node
        rs = np.random.RandomState(42)
        
        # High recency bias
        noise_biased = activation_weighted_noise(
            (N, d), activations, sigma_n=0.01, recency_bias=1.0, random_state=rs
        )
        
        # Node 5 should have larger noise magnitude
        mag_5 = np.linalg.norm(noise_biased[5])
        mag_0 = np.linalg.norm(noise_biased[0])
        assert mag_5 > mag_0


class TestRecencyDreamingStep:
    """Tests for recency-enhanced dreaming step."""
    
    def test_dreaming_step_updates_state(self):
        """Test that dreaming step updates vectors and adjacency."""
        N, d = 10, 64
        vectors = np.random.randn(N, d)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        adjacency = np.random.randn(N, N) * 0.1
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 0)
        
        grad_V = np.random.randn(N, d) * 0.01
        grad_G = np.random.randn(N, N) * 0.01
        activations = np.random.rand(N)
        
        new_V, new_G = controlled_dreaming_step_recency(
            vectors, adjacency, grad_V, grad_G, activations,
            eta=0.01, sigma_n=0.002, lam=0.001, recency_bias=0.3
        )
        
        assert new_V.shape == (N, d)
        assert new_G.shape == (N, N)
        
        # Vectors should be normalized
        norms = np.linalg.norm(new_V, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(N), decimal=5)
        
        # Adjacency should be symmetric
        np.testing.assert_array_almost_equal(new_G, new_G.T, decimal=10)
        
        # Diagonal should be zero
        assert np.all(np.diag(new_G) == 0)


class TestTemporalGradients:
    """Tests for temporal proximity gradients."""
    
    def test_temporal_gradient_shape(self):
        """Test that temporal gradient has correct shape."""
        N = 10
        last_accessed = np.array([time.time() - i*60 for i in range(N)])
        adjacency = np.random.randn(N, N) * 0.1
        
        grad_T = compute_temporal_gradient(
            last_accessed, adjacency, window_seconds=300
        )
        
        assert grad_T.shape == (N, N)
        assert np.all(np.diag(grad_T) == 0)  # Diagonal is zero
    
    def test_temporal_proximity_decay(self):
        """Test that temporal proximity decays with time difference."""
        N = 5
        current_time = time.time()
        last_accessed = np.array([
            current_time,
            current_time - 60,    # 1 min ago
            current_time - 300,   # 5 min ago
            current_time - 600,   # 10 min ago
            current_time - 3600   # 1 hour ago
        ])
        adjacency = np.zeros((N, N))
        
        grad_T = compute_temporal_gradient(
            last_accessed, adjacency, window_seconds=300, temporal_weight=0.3
        )
        
        # Memories closer in time should have stronger negative gradient
        # (negative because we want to encourage connections)
        proximity_0_1 = abs(grad_T[0, 1])
        proximity_0_4 = abs(grad_T[0, 4])
        
        assert proximity_0_1 > proximity_0_4  # Recent pair > distant pair
    
    def test_combined_gradient_blends_semantic_temporal(self):
        """Test that combined gradient blends semantic and temporal."""
        N = 10
        grad_semantic = np.random.randn(N, N) * 0.1
        last_accessed = np.array([time.time() - i*60 for i in range(N)])
        adjacency = np.random.randn(N, N) * 0.1
        
        grad_combined = compute_combined_gradient(
            grad_semantic, last_accessed, adjacency,
            temporal_weight=0.3
        )
        
        assert grad_combined.shape == (N, N)
        # Combined should be between semantic and temporal components
        assert not np.array_equal(grad_combined, grad_semantic)


class TestConceptExtractor:
    """Tests for concept extraction."""
    
    def test_concept_extractor_initialization(self):
        """Test ConceptExtractor initializes correctly."""
        extractor = ConceptExtractor(min_cluster_size=3)
        
        assert extractor.min_cluster_size == 3
        assert len(extractor.concepts) == 0
        assert len(extractor.memory_to_concept) == 0
    
    def test_concept_node_has_required_fields(self):
        """Test that ConceptNode has all required fields."""
        centroid = np.random.randn(64)
        concept = ConceptNode(
            concept_id=0,
            name="Test_Concept",
            description="Test description",
            centroid=centroid,
            member_indices=[0, 1, 2]
        )
        
        assert concept.concept_id == 0
        assert concept.name == "Test_Concept"
        assert len(concept.member_indices) == 3
        assert hasattr(concept, 'activation')
        assert hasattr(concept, 'coherence')
        assert hasattr(concept, 'keywords')
    
    def test_extract_concepts_with_synthetic_clusters(self):
        """Test concept extraction with synthetic clustered data."""
        extractor = ConceptExtractor(min_cluster_size=3)
        
        # Create 3 synthetic clusters
        np.random.seed(42)
        cluster_centers = [
            np.random.randn(64),
            np.random.randn(64),
            np.random.randn(64)
        ]
        
        embeddings = []
        texts = []
        for i, center in enumerate(cluster_centers):
            for j in range(4):
                embeddings.append(center + np.random.randn(64) * 0.1)
                texts.append(f"Memory about topic {i}, instance {j}")
        
        embeddings = np.array(embeddings)
        node_indices = list(range(12))
        
        concepts = extractor.extract_concepts(embeddings, texts, node_indices)
        
        # Should extract ~3 concepts (may vary with fallback clustering)
        assert len(concepts) >= 2
        assert len(concepts) <= 4
        
        # Each concept should have >= min_cluster_size members
        for c in concepts:
            assert len(c.member_indices) >= extractor.min_cluster_size
            assert c.coherence >= 0.0
            assert c.coherence <= 1.0
    
    def test_concept_activation_update(self):
        """Test that concept activation updates from members."""
        centroid = np.random.randn(64)
        concept = ConceptNode(
            concept_id=0,
            name="Test",
            description="Test",
            centroid=centroid,
            member_indices=[0, 1, 2]
        )
        
        member_activations = [0.8, 0.6, 0.4]
        concept.update_activation(member_activations)
        
        # Should be mean of member activations
        expected_mean = np.mean(member_activations)
        assert abs(concept.activation - expected_mean) < 1e-6
    
    def test_get_concept_summary(self):
        """Test concept summary statistics."""
        extractor = ConceptExtractor(min_cluster_size=3)
        
        # Initially empty
        summary = extractor.get_concept_summary()
        assert summary['total_concepts'] == 0
        assert summary['total_edges'] == 0
        
        # Add a mock concept
        centroid = np.random.randn(64)
        concept = ConceptNode(
            concept_id=0,
            name="Test",
            description="Test",
            centroid=centroid,
            member_indices=[0, 1, 2]
        )
        concept.activation = 0.8
        concept.coherence = 0.9
        
        extractor.concepts[0] = concept
        
        summary = extractor.get_concept_summary()
        assert summary['total_concepts'] == 1
        assert summary['avg_activation'] == 0.8
        assert summary['avg_coherence'] == 0.9
        assert summary['avg_members'] == 3.0


class TestConceptCoherence:
    """Tests for concept coherence computation."""
    
    def test_coherence_for_tight_cluster(self):
        """Test that tight clusters have high coherence."""
        center = np.random.randn(64)
        # Very tight cluster (small noise)
        embeddings = np.array([center + np.random.randn(64) * 0.01 for _ in range(5)])
        
        concept = ConceptNode(
            concept_id=0,
            name="Test",
            description="Test",
            centroid=center,
            member_indices=[0, 1, 2, 3, 4]
        )
        
        coherence = concept.compute_coherence(embeddings)
        
        # Tight cluster should have high coherence
        assert coherence > 0.9
    
    def test_coherence_for_loose_cluster(self):
        """Test that loose clusters have lower coherence."""
        center = np.random.randn(64)
        # Loose cluster (large noise)
        embeddings = np.array([center + np.random.randn(64) * 2.0 for _ in range(5)])
        
        concept = ConceptNode(
            concept_id=0,
            name="Test",
            description="Test",
            centroid=center,
            member_indices=[0, 1, 2, 3, 4]
        )
        
        coherence = concept.compute_coherence(embeddings)
        
        # Loose cluster should have lower coherence
        assert coherence < 0.9
    
    def test_coherence_single_member(self):
        """Test coherence for single-member concept."""
        embedding = np.random.randn(1, 64)
        
        concept = ConceptNode(
            concept_id=0,
            name="Test",
            description="Test",
            centroid=embedding[0],
            member_indices=[0]
        )
        
        coherence = concept.compute_coherence(embedding)
        
        # Single member should have perfect coherence
        assert coherence == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
