"""
Stochastic exploration (dreaming) for the Cognitive Engine.

Implements the dreaming phase from Section 6.7:
ẋ = -η∇E_t + σ_n ξ(t)

During low-input periods, controlled noise enables creative exploration
while remaining within bounded energy fluctuations (±2%).
"""

import numpy as np
from typing import Tuple, List


def add_exploration_noise(vectors: np.ndarray, adjacency: np.ndarray,
                         sigma_n: float, random_state: np.random.RandomState = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add stochastic noise for creative exploration.
    
    Adds Gaussian noise to both vectors and graph, then renormalizes.
    The noise amplitude σ_n controls exploration intensity.
    
    Args:
        vectors: Shape (N, d) - current vectors
        adjacency: Shape (N, N) - current adjacency
        sigma_n: Noise amplitude (typically 0.001-0.01)
        random_state: Optional random state for reproducibility
        
    Returns:
        Tuple of (noisy_vectors, noisy_adjacency)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    N, d = vectors.shape
    
    # Add noise to vectors
    noise_V = random_state.randn(N, d) * sigma_n
    noisy_vectors = vectors + noise_V
    
    # Renormalize vectors
    norms = np.linalg.norm(noisy_vectors, axis=1, keepdims=True)
    noisy_vectors = noisy_vectors / (norms + 1e-8)
    
    # Add noise to adjacency
    noise_G = random_state.randn(N, N) * sigma_n
    noisy_adjacency = adjacency + noise_G
    
    # Enforce symmetry and squash
    noisy_adjacency = (noisy_adjacency + noisy_adjacency.T) / 2
    noisy_adjacency = np.tanh(noisy_adjacency)
    np.fill_diagonal(noisy_adjacency, 0)
    
    return noisy_vectors, noisy_adjacency


def detect_creative_bridges(adjacency_before: np.ndarray, adjacency_after: np.ndarray,
                           threshold: float = 0.05) -> List[Tuple[int, int, float]]:
    """
    Detect new edges formed during dreaming phase.
    
    "Creative bridges" are weak connections that emerge between previously
    disconnected or weakly connected nodes during stochastic exploration.
    
    Args:
        adjacency_before: Shape (N, N) - adjacency before dreaming
        adjacency_after: Shape (N, N) - adjacency after dreaming
        threshold: Minimum edge strength to consider as a bridge
        
    Returns:
        List of (i, j, strength) tuples for new bridges
    """
    # Compute change in adjacency
    delta = adjacency_after - adjacency_before
    
    # Find significant positive changes (new connections)
    N = len(adjacency_before)
    bridges = []
    
    for i in range(N):
        for j in range(i + 1, N):  # Only upper triangle for symmetric matrix
            # Check if edge strengthened significantly
            if delta[i, j] > threshold and abs(adjacency_before[i, j]) < threshold:
                bridges.append((i, j, adjacency_after[i, j]))
    
    # Sort by strength
    bridges.sort(key=lambda x: x[2], reverse=True)
    
    return bridges


def controlled_dreaming_step(vectors: np.ndarray, adjacency: np.ndarray,
                             grad_V: np.ndarray, grad_G: np.ndarray,
                             eta: float, sigma_n: float, lam: float,
                             random_state: np.random.RandomState = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one step of controlled dreaming.
    
    Combines gradient descent with exploration noise:
    V ← V - η∇_V E_t + σ_n ξ_V(t)
    G ← G - η∇_G E_t - λG + σ_n ξ_G(t)
    
    Args:
        vectors: Shape (N, d) - current vectors
        adjacency: Shape (N, N) - current adjacency
        grad_V: Shape (N, d) - gradient wrt vectors
        grad_G: Shape (N, N) - gradient wrt adjacency
        eta: Learning rate
        sigma_n: Noise amplitude
        lam: Decay coefficient
        random_state: Optional random state
        
    Returns:
        Tuple of (updated_vectors, updated_adjacency)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    N, d = vectors.shape
    
    # Gradient update
    new_vectors = vectors - eta * grad_V
    new_adjacency = adjacency - eta * grad_G - lam * adjacency
    
    # Add exploration noise
    noise_V = random_state.randn(N, d) * sigma_n
    noise_G = random_state.randn(N, N) * sigma_n
    
    new_vectors += noise_V
    new_adjacency += noise_G
    
    # Normalize and constrain
    norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    new_vectors = new_vectors / (norms + 1e-8)
    
    new_adjacency = (new_adjacency + new_adjacency.T) / 2
    new_adjacency = np.tanh(new_adjacency)
    np.fill_diagonal(new_adjacency, 0)
    
    return new_vectors, new_adjacency


def measure_energy_fluctuation(energy_history: List[float], window: int = 50) -> float:
    """
    Measure energy fluctuation amplitude during dreaming.
    
    Computes relative std deviation over recent window:
    fluctuation = std(E_recent) / mean(E_recent)
    
    Args:
        energy_history: List of energy values over time
        window: Number of recent steps to consider
        
    Returns:
        float: Relative fluctuation amplitude
    """
    if len(energy_history) < window:
        return 0.0
    
    recent_energies = energy_history[-window:]
    mean_energy = np.mean(recent_energies)
    std_energy = np.std(recent_energies)
    
    if abs(mean_energy) < 1e-10:
        return 0.0
    
    return std_energy / abs(mean_energy)
