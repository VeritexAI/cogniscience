"""
Cache input generators for the Cognitive Engine.

Provides structured semantic sequences that create thematic clusters
and temporal patterns for system learning and consolidation.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


class SemanticClusterGenerator:
    """
    Generate cache inputs with thematic clustering.
    
    Creates semantic themes (clusters) and samples cache inputs from them
    with temporal patterns. This simulates structured learning experiences
    where related concepts appear together in time.
    
    Attributes:
        N: Number of nodes in the system
        d: Embedding dimensionality
        num_clusters: Number of semantic themes
        cluster_centers: Shape (num_clusters, d) - theme centroids
        cluster_assignments: Shape (N,) - which cluster each node belongs to
    """
    
    def __init__(self, N: int, d: int, num_clusters: int = 5,
                 spread: float = 0.3, random_seed: Optional[int] = None):
        """
        Initialize semantic cluster generator.
        
        Args:
            N: Number of nodes in the system
            d: Embedding dimensionality
            num_clusters: Number of distinct semantic themes
            spread: Cluster spread (0-1, smaller = tighter clusters)
            random_seed: Optional seed for reproducibility
        """
        self.N = N
        self.d = d
        self.num_clusters = num_clusters
        self.spread = spread
        
        self.rng = np.random.RandomState(random_seed)
        
        # Generate cluster centers on unit sphere
        self.cluster_centers = self.rng.randn(num_clusters, d)
        self.cluster_centers /= np.linalg.norm(self.cluster_centers, axis=1, keepdims=True)
        
        # Assign nodes to clusters
        nodes_per_cluster = N // num_clusters
        self.cluster_assignments = np.repeat(np.arange(num_clusters), nodes_per_cluster)
        # Handle remainder
        if len(self.cluster_assignments) < N:
            extra = N - len(self.cluster_assignments)
            self.cluster_assignments = np.concatenate([
                self.cluster_assignments,
                self.rng.choice(num_clusters, extra)
            ])
        
        self.rng.shuffle(self.cluster_assignments)
    
    def sample_from_cluster(self, cluster_id: int, num_samples: int = 1) -> Tuple[np.ndarray, List[int]]:
        """
        Sample cache inputs from a specific cluster.
        
        Args:
            cluster_id: Which cluster to sample from
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (embeddings, node_indices)
        """
        # Get nodes in this cluster
        cluster_nodes = np.where(self.cluster_assignments == cluster_id)[0]
        
        if len(cluster_nodes) == 0:
            # Return empty if cluster has no nodes
            return np.empty((0, self.d)), []
        
        # Sample nodes
        sampled_indices = self.rng.choice(
            cluster_nodes, 
            size=min(num_samples, len(cluster_nodes)),
            replace=False
        )
        
        # Generate embeddings near cluster center
        embeddings = []
        for idx in sampled_indices:
            # Start from cluster center
            emb = self.cluster_centers[cluster_id].copy()
            
            # Add noise
            noise = self.rng.randn(self.d) * self.spread
            emb += noise
            
            # Normalize
            emb /= np.linalg.norm(emb)
            
            embeddings.append(emb)
        
        return np.array(embeddings), sampled_indices.tolist()
    
    def temporal_sequence(self, num_steps: int, 
                         cluster_focus_duration: int = 50,
                         samples_per_step: int = 3) -> Callable:
        """
        Create a temporal sequence generator with cluster focus periods.
        
        The generator cycles through clusters, spending `cluster_focus_duration`
        steps on each before moving to the next. This simulates learning
        episodes where attention focuses on related concepts.
        
        Args:
            num_steps: Total number of steps (not used in generator, for reference)
            cluster_focus_duration: Steps to spend on each cluster
            samples_per_step: Cache samples per step
            
        Returns:
            Callable generator function(step, engine) -> (embeddings, indices)
        """
        def generator(step: int, engine=None):
            # Determine current cluster based on step
            cluster_id = (step // cluster_focus_duration) % self.num_clusters
            
            # Sample from current cluster
            embeddings, indices = self.sample_from_cluster(cluster_id, samples_per_step)
            
            return embeddings, indices
        
        return generator
    
    def random_walk_sequence(self, transition_prob: float = 0.1,
                            samples_per_step: int = 3) -> Callable:
        """
        Create a random walk generator that transitions between clusters.
        
        Stays in current cluster with probability (1 - transition_prob),
        or jumps to a random cluster with probability transition_prob.
        
        Args:
            transition_prob: Probability of switching clusters each step
            samples_per_step: Cache samples per step
            
        Returns:
            Callable generator function(step, engine) -> (embeddings, indices)
        """
        current_cluster = [0]  # Mutable to maintain state
        
        def generator(step: int, engine=None):
            # Random walk: maybe switch clusters
            if self.rng.random() < transition_prob:
                current_cluster[0] = self.rng.randint(0, self.num_clusters)
            
            # Sample from current cluster
            embeddings, indices = self.sample_from_cluster(
                current_cluster[0], samples_per_step
            )
            
            return embeddings, indices
        
        return generator
    
    def mixed_sequence(self, cluster_prob: float = 0.7,
                      samples_per_step: int = 3) -> Callable:
        """
        Create a mixed generator with both focused and exploratory samples.
        
        Each step, samples are drawn from:
        - Current focus cluster with probability cluster_prob
        - Random clusters with probability (1 - cluster_prob)
        
        Args:
            cluster_prob: Probability of sampling from focus cluster
            samples_per_step: Cache samples per step
            
        Returns:
            Callable generator function(step, engine) -> (embeddings, indices)
        """
        current_cluster = [0]
        
        def generator(step: int, engine=None):
            embeddings_list = []
            indices_list = []
            
            # Update focus cluster periodically
            if step % 100 == 0:
                current_cluster[0] = self.rng.randint(0, self.num_clusters)
            
            for _ in range(samples_per_step):
                if self.rng.random() < cluster_prob:
                    # Sample from focus cluster
                    cluster_id = current_cluster[0]
                else:
                    # Sample from random cluster
                    cluster_id = self.rng.randint(0, self.num_clusters)
                
                emb, idx = self.sample_from_cluster(cluster_id, num_samples=1)
                if len(emb) > 0:
                    embeddings_list.append(emb[0])
                    indices_list.extend(idx)
            
            if embeddings_list:
                return np.array(embeddings_list), indices_list
            else:
                return np.empty((0, self.d)), []
        
        return generator


class AdaptiveGenerator:
    """
    Adaptive cache generator that responds to system state.
    
    Samples based on current energy, gradient norms, or other metrics
    to create feedback-driven learning patterns.
    """
    
    def __init__(self, N: int, d: int, random_seed: Optional[int] = None):
        """
        Initialize adaptive generator.
        
        Args:
            N: Number of nodes
            d: Embedding dimensionality
            random_seed: Optional seed
        """
        self.N = N
        self.d = d
        self.rng = np.random.RandomState(random_seed)
    
    def high_energy_sampling(self, samples_per_step: int = 3) -> Callable:
        """
        Sample from high-energy regions (vectors with high gradient norms).
        
        This focuses learning on areas where the system is changing most rapidly.
        
        Args:
            samples_per_step: Cache samples per step
            
        Returns:
            Callable generator function(step, engine) -> (embeddings, indices)
        """
        def generator(step: int, engine):
            if engine is None or step == 0:
                # Random sampling for first step
                indices = self.rng.choice(self.N, samples_per_step, replace=False)
                vectors = engine.vector_memory.get_vectors() if engine else np.random.randn(self.N, self.d)
                embeddings = vectors[indices]
                return embeddings, indices.tolist()
            
            # Get current state and compute gradients
            vectors = engine.vector_memory.get_vectors()
            adjacency = engine.graph_memory.get_adjacency()
            cache = engine.cache_memory.get_cache()
            cache_indices = np.array(engine.cache_memory.get_active_indices())
            
            from cognigraph.energy import compute_gradients
            grads = compute_gradients(vectors, adjacency, cache, cache_indices)
            
            # Find nodes with highest gradient magnitudes
            grad_norms = np.linalg.norm(grads['grad_V'], axis=1)
            high_energy_nodes = np.argsort(grad_norms)[-samples_per_step:]
            
            # Sample near these high-energy nodes
            embeddings = []
            for idx in high_energy_nodes:
                # Perturb slightly
                emb = vectors[idx] + self.rng.randn(self.d) * 0.1
                emb /= np.linalg.norm(emb)
                embeddings.append(emb)
            
            return np.array(embeddings), high_energy_nodes.tolist()
        
        return generator
    
    def exploration_exploitation(self, exploration_rate: float = 0.3,
                                samples_per_step: int = 3) -> Callable:
        """
        Balance exploration (random) and exploitation (high-gradient).
        
        Args:
            exploration_rate: Fraction of samples from random exploration
            samples_per_step: Cache samples per step
            
        Returns:
            Callable generator function(step, engine) -> (embeddings, indices)
        """
        def generator(step: int, engine):
            if engine is None:
                indices = self.rng.choice(self.N, samples_per_step, replace=False)
                return np.random.randn(samples_per_step, self.d), indices.tolist()
            
            vectors = engine.vector_memory.get_vectors()
            
            num_explore = int(samples_per_step * exploration_rate)
            num_exploit = samples_per_step - num_explore
            
            # Exploration: random samples
            explore_indices = self.rng.choice(self.N, num_explore, replace=False)
            explore_embs = vectors[explore_indices]
            
            # Exploitation: based on gradient (if past first step)
            if step > 0 and num_exploit > 0:
                adjacency = engine.graph_memory.get_adjacency()
                cache = engine.cache_memory.get_cache()
                cache_indices = np.array(engine.cache_memory.get_active_indices())
                
                from cognigraph.energy import compute_gradients
                grads = compute_gradients(vectors, adjacency, cache, cache_indices)
                grad_norms = np.linalg.norm(grads['grad_V'], axis=1)
                exploit_indices = np.argsort(grad_norms)[-num_exploit:]
                exploit_embs = vectors[exploit_indices]
            else:
                exploit_indices = np.array([], dtype=int)
                exploit_embs = np.empty((0, self.d))
            
            # Combine
            all_embeddings = np.vstack([explore_embs, exploit_embs]) if num_exploit > 0 else explore_embs
            all_indices = np.concatenate([explore_indices, exploit_indices]).tolist()
            
            return all_embeddings, all_indices
        
        return generator


def create_default_generator(N: int, d: int, mode: str = 'temporal',
                            random_seed: Optional[int] = None) -> Callable:
    """
    Create a default cache input generator.
    
    Args:
        N: Number of nodes
        d: Embedding dimensionality
        mode: Generator mode - 'temporal', 'random_walk', 'mixed', or 'adaptive'
        random_seed: Optional seed for reproducibility
        
    Returns:
        Callable generator function
    """
    if mode == 'temporal':
        gen = SemanticClusterGenerator(N, d, num_clusters=max(5, N // 200), 
                                      random_seed=random_seed)
        return gen.temporal_sequence(num_steps=5000, cluster_focus_duration=50)
    
    elif mode == 'random_walk':
        gen = SemanticClusterGenerator(N, d, num_clusters=max(5, N // 200),
                                      random_seed=random_seed)
        return gen.random_walk_sequence(transition_prob=0.1)
    
    elif mode == 'mixed':
        gen = SemanticClusterGenerator(N, d, num_clusters=max(5, N // 200),
                                      random_seed=random_seed)
        return gen.mixed_sequence(cluster_prob=0.7)
    
    elif mode == 'adaptive':
        gen = AdaptiveGenerator(N, d, random_seed=random_seed)
        return gen.exploration_exploitation(exploration_rate=0.3)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'temporal', 'random_walk', 'mixed', 'adaptive'")
