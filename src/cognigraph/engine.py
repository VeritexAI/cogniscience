"""
Cognitive Engine: Main orchestrator for the energy-based memory system.

Integrates Cache, Vector, and Graph memory strata with energy minimization
dynamics to create a self-organizing knowledge system.
"""

import numpy as np
from typing import Dict, List, Optional
from cognigraph.memory import CacheMemory, VectorMemory, GraphMemory
from cognigraph.energy import compute_total_energy, compute_gradients
from cognigraph.dynamics import apply_updates
from cognigraph.dynamics.dreaming import controlled_dreaming_step, detect_creative_bridges
from cognigraph.utils import compute_metrics, compute_drift_index


class CognitiveEngine:
    """
    Main Cognitive Engine implementing energy-based hybrid memory system.
    
    The engine evolves by minimizing total energy E_t = E_C + E_V + E_G through
    gradient descent, producing Hebbian learning, consolidation, and creative exploration.
    
    Attributes:
        vector_memory: Long-term semantic embeddings (N vectors of dimension d)
        graph_memory: Relational structure (N×N adjacency matrix)
        cache_memory: Short-term working context
        N: Number of nodes/concepts
        d: Embedding dimensionality
        eta1, eta2, eta3: Learning rates
        lam: Decay coefficient
    """
    
    def __init__(self, N: int, d: int, 
                 eta1: float = 0.02, eta2: float = 0.01, eta3: float = 0.01,
                 lam: float = 0.005, random_seed: Optional[int] = None):
        """
        Initialize Cognitive Engine.
        
        Args:
            N: Number of nodes/concepts
            d: Embedding dimensionality
            eta1: Learning rate for cache consolidation
            eta2: Learning rate for vector updates
            eta3: Learning rate for graph updates
            lam: Decay coefficient for edges
            random_seed: Optional seed for reproducibility
        """
        self.N = N
        self.d = d
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.lam = lam
        
        # Initialize memory strata
        self.vector_memory = VectorMemory(N, d, random_seed=random_seed)
        self.graph_memory = GraphMemory(N, init_strength=0.1, random_seed=random_seed)
        self.cache_memory = CacheMemory(d)
        
        # History tracking
        self.energy_history = []
        self.metrics_history = []
        self.time_step = 0
        
        # Random state for dreaming
        self.random_state = np.random.RandomState(random_seed)
    
    def set_cache(self, cache_inputs: np.ndarray, indices: Optional[List[int]] = None):
        """
        Set cache contents.
        
        Args:
            cache_inputs: Shape (k, d) - new cache entries
            indices: Optional list of associated vector indices
        """
        self.cache_memory.set_cache(cache_inputs, indices)
    
    def compute_energy(self) -> Dict[str, float]:
        """
        Compute current system energy.
        
        Returns:
            dict: Energy components (E_C, E_V, E_G, E_total)
        """
        vectors = self.vector_memory.get_vectors()
        adjacency = self.graph_memory.get_adjacency()
        cache = self.cache_memory.get_cache()
        cache_indices = np.array(self.cache_memory.get_active_indices())
        
        return compute_total_energy(vectors, adjacency, cache, cache_indices)
    
    def step(self, record_history: bool = True) -> Dict:
        """
        Perform one step of energy minimization.
        
        Computes gradients, applies updates, and optionally records metrics.
        
        Args:
            record_history: Whether to save energy and metrics
            
        Returns:
            dict: Step results including energy, gradients, and metrics
        """
        # Get current state
        vectors = self.vector_memory.get_vectors()
        adjacency = self.graph_memory.get_adjacency()
        cache = self.cache_memory.get_cache()
        cache_indices = np.array(self.cache_memory.get_active_indices())
        
        # Compute energy before update
        energy_dict = compute_total_energy(vectors, adjacency, cache, cache_indices)
        
        # Compute gradients
        grads = compute_gradients(vectors, adjacency, cache, cache_indices)
        grad_V = grads['grad_V']
        grad_G = grads['grad_G']
        
        # Compute gradient norm
        grad_norm = np.sqrt(np.linalg.norm(grad_V)**2 + np.linalg.norm(grad_G)**2)
        
        # Apply updates
        new_vectors, new_adjacency = apply_updates(
            vectors, adjacency, grad_V, grad_G,
            self.eta1, self.eta2, self.eta3, self.lam
        )
        
        # Update memory
        self.vector_memory.set_vectors(new_vectors)
        self.graph_memory.set_adjacency(new_adjacency)
        
        # Compute metrics
        metrics = compute_metrics(new_vectors, new_adjacency, grad_norm)
        
        # Record history
        if record_history:
            self.energy_history.append(energy_dict['E_total'])
            self.metrics_history.append(metrics)
        
        self.time_step += 1
        
        return {
            'step': self.time_step,
            'energy': energy_dict,
            'gradient_norm': grad_norm,
            'metrics': metrics
        }
    
    def run_simulation(self, num_steps: int, cache_generator=None,
                      verbose: bool = True, log_interval: int = 100) -> List[Dict]:
        """
        Run simulation for multiple steps.
        
        Args:
            num_steps: Number of steps to simulate
            cache_generator: Optional function that generates cache inputs per step
            verbose: Whether to print progress
            log_interval: Print progress every N steps
            
        Returns:
            List of step results
        """
        results = []
        
        for step in range(num_steps):
            # Generate cache input if generator provided
            if cache_generator is not None:
                cache_inputs, indices = cache_generator(step, self)
                self.set_cache(cache_inputs, indices)
            
            # Perform step
            step_result = self.step(record_history=True)
            results.append(step_result)
            
            # Log progress
            if verbose and (step + 1) % log_interval == 0:
                energy = step_result['energy']['E_total']
                grad_norm = step_result['gradient_norm']
                density = step_result['metrics']['density']
                print(f"Step {step + 1}/{num_steps}: "
                      f"E_total={energy:.4f}, "
                      f"||∇E||={grad_norm:.6f}, "
                      f"density={density:.3f}")
        
        return results
    
    def dream(self, num_steps: int, sigma_n: float = 0.002,
             verbose: bool = True) -> List[Dict]:
        """
        Run dreaming phase with stochastic exploration.
        
        During dreaming, no cache input is provided and noise is added
        to enable creative recombination.
        
        Args:
            num_steps: Number of dreaming steps
            sigma_n: Noise amplitude (default 0.002 from paper)
            verbose: Whether to print progress
            
        Returns:
            List of step results including detected bridges
        """
        # Clear cache for dreaming
        self.cache_memory.clear()
        
        results = []
        adjacency_before = self.graph_memory.get_adjacency()
        
        for step in range(num_steps):
            # Get current state
            vectors = self.vector_memory.get_vectors()
            adjacency = self.graph_memory.get_adjacency()
            cache = np.empty((0, self.d))  # Empty cache during dreaming
            cache_indices = np.array([])
            
            # Compute energy and gradients
            energy_dict = compute_total_energy(vectors, adjacency, cache, cache_indices)
            grads = compute_gradients(vectors, adjacency, cache, cache_indices)
            
            # Dreaming step with noise
            new_vectors, new_adjacency = controlled_dreaming_step(
                vectors, adjacency, grads['grad_V'], grads['grad_G'],
                self.eta2, sigma_n, self.lam, self.random_state
            )
            
            # Update memory
            self.vector_memory.set_vectors(new_vectors)
            self.graph_memory.set_adjacency(new_adjacency)
            
            # Record
            grad_norm = np.sqrt(np.linalg.norm(grads['grad_V'])**2 + 
                              np.linalg.norm(grads['grad_G'])**2)
            metrics = compute_metrics(new_vectors, new_adjacency, grad_norm)
            
            self.energy_history.append(energy_dict['E_total'])
            self.metrics_history.append(metrics)
            self.time_step += 1
            
            results.append({
                'step': self.time_step,
                'energy': energy_dict,
                'gradient_norm': grad_norm,
                'metrics': metrics
            })
        
        # Detect creative bridges
        adjacency_after = self.graph_memory.get_adjacency()
        bridges = detect_creative_bridges(adjacency_before, adjacency_after, threshold=0.05)
        
        if verbose:
            print(f"Dreaming complete: {num_steps} steps")
            print(f"Creative bridges formed: {len(bridges)}")
            if bridges:
                print(f"Top bridge: nodes {bridges[0][0]}-{bridges[0][1]}, "
                      f"strength {bridges[0][2]:.4f}")
        
        return results, bridges
    
    def get_state(self) -> Dict:
        """
        Get current system state.
        
        Returns:
            dict: Complete state including vectors, adjacency, cache, and history
        """
        return {
            'vectors': self.vector_memory.get_vectors(),
            'adjacency': self.graph_memory.get_adjacency(),
            'cache': self.cache_memory.get_cache(),
            'cache_indices': self.cache_memory.get_active_indices(),
            'energy_history': self.energy_history.copy(),
            'metrics_history': self.metrics_history.copy(),
            'time_step': self.time_step,
            'parameters': {
                'N': self.N,
                'd': self.d,
                'eta1': self.eta1,
                'eta2': self.eta2,
                'eta3': self.eta3,
                'lam': self.lam
            }
        }
    
    def __repr__(self):
        return (f"CognitiveEngine(N={self.N}, d={self.d}, "
                f"step={self.time_step}, "
                f"density={self.graph_memory.get_density():.3f})")
