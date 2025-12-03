"""
Concept extraction layer for hierarchical abstraction.

Extracts abstract concepts from memory clusters using HDBSCAN,
maintaining a concept graph that sits above the memory-level graph
from the paper. This enables reasoning about abstract themes while
preserving the original gradient descent dynamics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import time

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. Concept extraction will use fallback clustering.")


@dataclass
class ConceptNode:
    """
    An abstract concept extracted from memory clusters.
    
    Concepts are higher-level abstractions that emerge from grouping
    semantically related memories. They form a concept graph separate
    from (but linked to) the memory-level graph.
    
    Attributes:
        concept_id: Unique identifier
        name: Human-readable concept name (extracted or generated)
        description: Brief description of what this concept represents
        centroid: Mean embedding of member memories
        member_indices: Node indices of memories belonging to this concept
        activation: Aggregate activation (mean of member activations)
        coherence: Cluster coherence score (0-1)
        created_at: Timestamp of concept creation
        last_updated: Timestamp of last update
        keywords: Key terms associated with this concept
        related_concepts: IDs of related concepts with weights
    """
    concept_id: int
    name: str
    description: str
    centroid: np.ndarray
    member_indices: List[int]
    activation: float = 1.0
    coherence: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    keywords: List[str] = field(default_factory=list)
    related_concepts: Dict[int, float] = field(default_factory=dict)  # concept_id -> weight
    
    def update_activation(self, member_activations: List[float]) -> None:
        """Update concept activation from member memory activations."""
        if member_activations:
            self.activation = float(np.mean(member_activations))
        self.last_updated = time.time()
    
    def compute_coherence(self, member_embeddings: np.ndarray) -> float:
        """
        Compute cluster coherence as mean pairwise similarity.
        
        High coherence means members are tightly clustered.
        """
        if len(member_embeddings) < 2:
            self.coherence = 1.0
            return self.coherence
        
        # Normalize embeddings
        norms = np.linalg.norm(member_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = member_embeddings / norms
        
        # Compute pairwise similarities
        similarities = normalized @ normalized.T
        
        # Get upper triangle (excluding diagonal)
        n = len(member_embeddings)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_indices]
        
        self.coherence = float(np.mean(pairwise_sims))
        return self.coherence


class ConceptExtractor:
    """
    Extracts and manages abstract concepts from memory clusters.
    
    Uses HDBSCAN for clustering memory embeddings, then extracts
    concept names and maintains concept-concept relationships.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.0,
        metric: str = 'euclidean'
    ):
        """
        Initialize concept extractor.
        
        Args:
            min_cluster_size: Minimum memories to form a concept
            min_samples: HDBSCAN min_samples parameter
            cluster_selection_epsilon: HDBSCAN cluster selection threshold
            metric: Distance metric for clustering
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        
        # Concept storage
        self.concepts: Dict[int, ConceptNode] = {}
        self.next_concept_id = 0
        
        # Mapping from memory index to concept
        self.memory_to_concept: Dict[int, int] = {}  # node_idx -> concept_id
        
        # Concept graph edges: (concept_i, concept_j) -> weight
        self.concept_edges: Dict[Tuple[int, int], float] = {}
        
        # Statistics
        self.stats = {
            'extractions': 0,
            'concepts_created': 0,
            'concepts_merged': 0,
            'noise_memories': 0
        }
    
    def extract_concepts(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        node_indices: List[int],
        activations: Optional[np.ndarray] = None
    ) -> List[ConceptNode]:
        """
        Extract concepts from a set of memory embeddings.
        
        Args:
            embeddings: Memory embeddings (N x D)
            texts: Corresponding text for each memory
            node_indices: Node indices in cognitive engine
            activations: Optional activation levels for each memory
            
        Returns:
            List of extracted ConceptNodes
        """
        self.stats['extractions'] += 1
        
        if len(embeddings) < self.min_cluster_size:
            return []
        
        if activations is None:
            activations = np.ones(len(embeddings))
        
        # Run clustering
        labels = self._cluster_embeddings(embeddings)
        
        # Extract concepts from clusters
        new_concepts = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                noise_count = np.sum(labels == -1)
                self.stats['noise_memories'] += noise_count
                continue
            
            # Get cluster members
            mask = labels == label
            member_indices = [node_indices[i] for i in np.where(mask)[0]]
            member_embeddings = embeddings[mask]
            member_texts = [texts[i] for i in np.where(mask)[0]]
            member_activations = activations[mask]
            
            # Skip small clusters
            if len(member_indices) < self.min_cluster_size:
                continue
            
            # Create concept
            concept = self._create_concept(
                member_indices=member_indices,
                member_embeddings=member_embeddings,
                member_texts=member_texts,
                member_activations=member_activations
            )
            
            new_concepts.append(concept)
            self.stats['concepts_created'] += 1
        
        # Update concept relationships
        self._update_concept_relationships(new_concepts)
        
        return new_concepts
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN or fallback.
        
        Returns:
            Array of cluster labels (-1 for noise)
        """
        if HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric
            )
            labels = clusterer.fit_predict(embeddings)
        else:
            # Fallback: simple similarity-based clustering
            labels = self._fallback_clustering(embeddings)
        
        return labels
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Simple fallback clustering when HDBSCAN unavailable.
        
        Uses greedy similarity-based grouping.
        """
        n = len(embeddings)
        labels = np.full(n, -1)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        
        # Compute pairwise similarities
        similarities = normalized @ normalized.T
        
        current_label = 0
        threshold = 0.7  # Similarity threshold
        
        for i in range(n):
            if labels[i] != -1:
                continue
            
            # Start new cluster
            labels[i] = current_label
            cluster_members = [i]
            
            # Add similar points
            for j in range(i + 1, n):
                if labels[j] == -1 and similarities[i, j] > threshold:
                    labels[j] = current_label
                    cluster_members.append(j)
            
            # Only keep if cluster is big enough
            if len(cluster_members) >= self.min_cluster_size:
                current_label += 1
            else:
                for m in cluster_members:
                    labels[m] = -1
        
        return labels
    
    def _create_concept(
        self,
        member_indices: List[int],
        member_embeddings: np.ndarray,
        member_texts: List[str],
        member_activations: np.ndarray
    ) -> ConceptNode:
        """
        Create a ConceptNode from cluster members.
        """
        concept_id = self.next_concept_id
        self.next_concept_id += 1
        
        # Compute centroid
        centroid = np.mean(member_embeddings, axis=0)
        
        # Extract keywords from texts
        keywords = self._extract_keywords(member_texts)
        
        # Generate name and description
        name = self._generate_concept_name(keywords, member_texts)
        description = self._generate_concept_description(keywords, len(member_indices))
        
        concept = ConceptNode(
            concept_id=concept_id,
            name=name,
            description=description,
            centroid=centroid,
            member_indices=member_indices,
            keywords=keywords
        )
        
        # Compute coherence
        concept.compute_coherence(member_embeddings)
        
        # Update activation from members
        concept.update_activation(member_activations.tolist())
        
        # Store concept and update mappings
        self.concepts[concept_id] = concept
        for idx in member_indices:
            self.memory_to_concept[idx] = concept_id
        
        return concept
    
    def _extract_keywords(self, texts: List[str], top_k: int = 5) -> List[str]:
        """
        Extract key terms from a collection of texts.
        
        Simple TF-based extraction (can be enhanced with TF-IDF).
        """
        # Tokenize and count
        word_counts: Dict[str, int] = {}
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'this', 'that', 'these', 'those', 'it', 'its',
            'i', 'you', 'he', 'she', 'we', 'they', 'who', 'what',
            'which', 'when', 'where', 'why', 'how', 'all', 'each',
            'not', 'no', 'yes', 'if', 'then', 'than', 'more', 'most',
            'some', 'any', 'other', 'such', 'only', 'also', 'just',
            'about', 'over', 'under', 'again', 'further', 'once'
        }
        
        for text in texts:
            # Simple tokenization
            words = text.lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 2 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_k]]
    
    def _generate_concept_name(self, keywords: List[str], texts: List[str]) -> str:
        """
        Generate a human-readable concept name from keywords.
        """
        if not keywords:
            return f"Concept_{self.next_concept_id}"
        
        # Use top 2-3 keywords
        name_keywords = keywords[:min(3, len(keywords))]
        return "_".join(name_keywords).title()
    
    def _generate_concept_description(self, keywords: List[str], member_count: int) -> str:
        """
        Generate a brief description of the concept.
        """
        if keywords:
            keyword_str = ", ".join(keywords[:5])
            return f"Cluster of {member_count} memories related to: {keyword_str}"
        return f"Cluster of {member_count} related memories"
    
    def _update_concept_relationships(self, new_concepts: List[ConceptNode]) -> None:
        """
        Update concept-concept edges based on centroid similarity.
        """
        all_concepts = list(self.concepts.values())
        
        for i, concept_i in enumerate(all_concepts):
            for concept_j in all_concepts[i+1:]:
                # Compute similarity between centroids
                norm_i = np.linalg.norm(concept_i.centroid)
                norm_j = np.linalg.norm(concept_j.centroid)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = float(
                        np.dot(concept_i.centroid, concept_j.centroid) / (norm_i * norm_j)
                    )
                else:
                    similarity = 0.0
                
                # Store edge if similarity above threshold
                if similarity > 0.3:
                    edge_key = (concept_i.concept_id, concept_j.concept_id)
                    self.concept_edges[edge_key] = similarity
                    
                    # Update related_concepts in both directions
                    concept_i.related_concepts[concept_j.concept_id] = similarity
                    concept_j.related_concepts[concept_i.concept_id] = similarity
    
    def get_concept_for_memory(self, node_idx: int) -> Optional[ConceptNode]:
        """Get the concept a memory belongs to, if any."""
        concept_id = self.memory_to_concept.get(node_idx)
        if concept_id is not None:
            return self.concepts.get(concept_id)
        return None
    
    def get_related_concepts(self, concept_id: int, min_weight: float = 0.3) -> List[Tuple[ConceptNode, float]]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept_id: ID of the source concept
            min_weight: Minimum relationship weight
            
        Returns:
            List of (ConceptNode, weight) tuples sorted by weight
        """
        if concept_id not in self.concepts:
            return []
        
        concept = self.concepts[concept_id]
        related = []
        
        for related_id, weight in concept.related_concepts.items():
            if weight >= min_weight and related_id in self.concepts:
                related.append((self.concepts[related_id], weight))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def update_all_activations(self, get_memory_activation) -> None:
        """
        Update all concept activations from member memory activations.
        
        Args:
            get_memory_activation: Function that takes node_idx and returns activation
        """
        for concept in self.concepts.values():
            member_activations = [
                get_memory_activation(idx) 
                for idx in concept.member_indices
            ]
            concept.update_activation(member_activations)
    
    def get_high_activation_concepts(self, threshold: float = 0.5) -> List[ConceptNode]:
        """Get concepts with activation above threshold."""
        return [c for c in self.concepts.values() if c.activation >= threshold]
    
    def get_concept_summary(self) -> Dict:
        """Get summary statistics about extracted concepts."""
        if not self.concepts:
            return {
                'total_concepts': 0,
                'total_edges': 0,
                'avg_coherence': 0.0,
                'avg_activation': 0.0,
                'avg_members': 0.0
            }
        
        coherences = [c.coherence for c in self.concepts.values()]
        activations = [c.activation for c in self.concepts.values()]
        member_counts = [len(c.member_indices) for c in self.concepts.values()]
        
        return {
            'total_concepts': len(self.concepts),
            'total_edges': len(self.concept_edges),
            'avg_coherence': float(np.mean(coherences)),
            'avg_activation': float(np.mean(activations)),
            'avg_members': float(np.mean(member_counts)),
            'stats': self.stats.copy()
        }
    
    def to_networkx(self):
        """
        Export concept graph to NetworkX for visualization.
        
        Returns:
            NetworkX graph (or None if NetworkX unavailable)
        """
        try:
            import networkx as nx
        except ImportError:
            return None
        
        G = nx.Graph()
        
        # Add concept nodes
        for concept in self.concepts.values():
            G.add_node(
                concept.concept_id,
                name=concept.name,
                description=concept.description,
                activation=concept.activation,
                coherence=concept.coherence,
                member_count=len(concept.member_indices),
                keywords=concept.keywords
            )
        
        # Add edges
        for (c1, c2), weight in self.concept_edges.items():
            G.add_edge(c1, c2, weight=weight)
        
        return G
