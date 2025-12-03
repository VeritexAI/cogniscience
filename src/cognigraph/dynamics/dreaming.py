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


# =========================================================================
# Recency-Enhanced Dreaming
# =========================================================================

def pre_dream_rumination(
    activations: np.ndarray,
    top_k: int = 10,
    rumination_boost: float = 0.2
) -> np.ndarray:
    """
    Pre-dream rumination phase that boosts highly activated memories.
    
    Before dreaming, this simulates the natural tendency to "dwell on"
    recent experiences, giving them extra activation to increase their
    influence during the dream phase.
    
    Args:
        activations: Current activation levels (N,)
        top_k: Number of top-activated memories to boost
        rumination_boost: Activation boost amount (added to top-k)
        
    Returns:
        Boosted activation array (N,)
    """
    boosted = activations.copy()
    
    # Find top-k activated memories
    if len(activations) > top_k:
        top_indices = np.argpartition(activations, -top_k)[-top_k:]
    else:
        top_indices = np.arange(len(activations))
    
    # Apply rumination boost (capped at 1.0)
    boosted[top_indices] = np.minimum(1.0, boosted[top_indices] + rumination_boost)
    
    return boosted


def activation_weighted_noise(
    shape: Tuple,
    activations: np.ndarray,
    sigma_n: float,
    recency_bias: float = 0.3,
    random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Generate activation-weighted noise for recency-biased exploration.
    
    Instead of uniform noise, higher-activated memories receive more
    noise, causing the dream to focus on recently accessed content.
    
    Args:
        shape: Shape of noise array (N, d) for vectors or (N, N) for adjacency
        activations: Activation levels per node (N,)
        sigma_n: Base noise amplitude
        recency_bias: How much to weight by activation (0 = uniform, 1 = fully weighted)
        random_state: Random state for reproducibility
        
    Returns:
        Weighted noise array of given shape
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    N = len(activations)
    base_noise = random_state.randn(*shape) * sigma_n
    
    # Compute per-node weight: blend of uniform and activation-based
    weights = (1 - recency_bias) + recency_bias * activations
    
    # Apply weights depending on shape
    if len(shape) == 2 and shape[0] == N and shape[1] != N:
        # Vector noise (N, d) - weight by row
        weighted_noise = base_noise * weights[:, np.newaxis]
    elif len(shape) == 2 and shape[0] == N and shape[1] == N:
        # Adjacency noise (N, N) - weight by geometric mean of row/col activations
        weight_matrix = np.sqrt(np.outer(weights, weights))
        weighted_noise = base_noise * weight_matrix
    else:
        weighted_noise = base_noise
    
    return weighted_noise


def controlled_dreaming_step_recency(
    vectors: np.ndarray,
    adjacency: np.ndarray,
    grad_V: np.ndarray,
    grad_G: np.ndarray,
    activations: np.ndarray,
    eta: float,
    sigma_n: float,
    lam: float,
    recency_bias: float = 0.3,
    random_state: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one step of recency-enhanced dreaming.
    
    Like controlled_dreaming_step but with activation-weighted noise
    that focuses exploration on recently accessed memories.
    
    Args:
        vectors: Shape (N, d) - current vectors
        adjacency: Shape (N, N) - current adjacency
        grad_V: Shape (N, d) - gradient wrt vectors
        grad_G: Shape (N, N) - gradient wrt adjacency
        activations: Shape (N,) - activation levels per node
        eta: Learning rate
        sigma_n: Noise amplitude
        lam: Decay coefficient
        recency_bias: How much to weight noise by activation (0-1)
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
    
    # Add activation-weighted exploration noise
    noise_V = activation_weighted_noise(
        (N, d), activations, sigma_n, recency_bias, random_state
    )
    noise_G = activation_weighted_noise(
        (N, N), activations, sigma_n, recency_bias, random_state
    )
    
    new_vectors += noise_V
    new_adjacency += noise_G
    
    # Normalize and constrain
    norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    new_vectors = new_vectors / (norms + 1e-8)
    
    new_adjacency = (new_adjacency + new_adjacency.T) / 2
    new_adjacency = np.tanh(new_adjacency)
    np.fill_diagonal(new_adjacency, 0)
    
    return new_vectors, new_adjacency


def compute_temporal_gradient(
    last_accessed: np.ndarray,
    adjacency: np.ndarray,
    window_seconds: float = 3600.0,
    temporal_weight: float = 0.3
) -> np.ndarray:
    """
    Compute temporal proximity gradient for graph consolidation.
    
    Encourages connections between memories accessed close in time,
    simulating the tendency to associate concurrent experiences.
    
    grad_G_temporal[i,j] = -temporal_weight * proximity(i,j)
    
    Where proximity is Gaussian decay based on access time difference.
    
    Args:
        last_accessed: Array of last access timestamps (N,)
        adjacency: Current adjacency matrix (N, N)
        window_seconds: Time window for proximity calculation
        temporal_weight: Weight of temporal gradient (α in the plan)
        
    Returns:
        Temporal gradient (N, N) - negative where we want stronger connections
    """
    N = len(last_accessed)
    
    # Compute pairwise time differences
    time_diff = np.abs(last_accessed[:, np.newaxis] - last_accessed[np.newaxis, :])
    
    # Gaussian proximity decay
    proximity = np.exp(-(time_diff ** 2) / (2 * window_seconds ** 2))
    
    # Zero diagonal
    np.fill_diagonal(proximity, 0)
    
    # Gradient: negative proximity encourages stronger connections
    grad_G_temporal = -temporal_weight * proximity
    
    return grad_G_temporal


def compute_combined_gradient(
    grad_G_semantic: np.ndarray,
    last_accessed: np.ndarray,
    adjacency: np.ndarray,
    temporal_weight: float = 0.3,
    window_seconds: float = 3600.0
) -> np.ndarray:
    """
    Combine semantic and temporal gradients for dream-phase consolidation.
    
    grad_combined = (1-α) * grad_semantic + α * grad_temporal
    
    Args:
        grad_G_semantic: Semantic gradient from energy function (N, N)
        last_accessed: Last access timestamps (N,)
        adjacency: Current adjacency (N, N)
        temporal_weight: α - weight of temporal component (0.3 default)
        window_seconds: Time window for temporal proximity
        
    Returns:
        Combined gradient (N, N)
    """
    grad_G_temporal = compute_temporal_gradient(
        last_accessed, adjacency, window_seconds, temporal_weight=1.0
    )
    
    # Blend gradients
    grad_combined = (1 - temporal_weight) * grad_G_semantic + temporal_weight * grad_G_temporal
    
    return grad_combined
