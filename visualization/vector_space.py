"""
Vector space visualization for the Cognitive Engine.

Visualizes semantic embeddings in 2D/3D using dimensionality reduction
techniques (PCA, t-SNE) to show cluster formation and semantic organization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, List
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_vector_projection_2d(vectors: np.ndarray,
                               method: str = 'pca',
                               title: str = "Vector Space Projection (2D)",
                               figsize: tuple = (10, 8),
                               labels: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
    """
    Project vectors to 2D and visualize.
    
    Args:
        vectors: Shape (N, d) - vectors to project
        method: 'pca' or 'tsne' for dimensionality reduction
        title: Plot title
        figsize: Figure size
        labels: Optional cluster labels for coloring
        save_path: Optional path to save figure
    """
    N, d = vectors.shape
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(vectors)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"PCA (explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == 'tsne':
        perplexity = min(30, N - 1)  # t-SNE requires perplexity < N
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords_2d = reducer.fit_transform(vectors)
        subtitle = "t-SNE projection"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by labels if provided, otherwise use index
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                      c=[color], label=f'Cluster {label}',
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.legend(loc='best', fontsize=9)
    else:
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=np.arange(N), cmap='viridis',
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Node Index')
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, coords_2d


def plot_vector_projection_3d(vectors: np.ndarray,
                               method: str = 'pca',
                               title: str = "Vector Space Projection (3D)",
                               figsize: tuple = (12, 9),
                               labels: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
    """
    Project vectors to 3D and visualize.
    
    Args:
        vectors: Shape (N, d) - vectors to project
        method: 'pca' or 'tsne' for dimensionality reduction
        title: Plot title
        figsize: Figure size
        labels: Optional cluster labels for coloring
        save_path: Optional path to save figure
    """
    N, d = vectors.shape
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=3, random_state=42)
        coords_3d = reducer.fit_transform(vectors)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"PCA (explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}, {explained_var[2]:.2%})"
    elif method == 'tsne':
        perplexity = min(30, N - 1)
        reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        coords_3d = reducer.fit_transform(vectors)
        subtitle = "t-SNE projection"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by labels if provided
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                      c=[color], label=f'Cluster {label}',
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.legend(loc='best', fontsize=9)
    else:
        scatter = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                           c=np.arange(N), cmap='viridis',
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Node Index', pad=0.1)
    
    ax.set_xlabel(f'{method.upper()} 1', fontsize=11)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=11)
    ax.set_zlabel(f'{method.upper()} 3', fontsize=11)
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, coords_3d


def plot_similarity_matrix(vectors: np.ndarray,
                           title: str = "Vector Similarity Matrix",
                           figsize: tuple = (10, 8),
                           save_path: Optional[str] = None):
    """
    Visualize pairwise similarity matrix as heatmap.
    
    Args:
        vectors: Shape (N, d) - vectors
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Compute similarity matrix
    similarity = vectors @ vectors.T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_vector_evolution(vector_snapshots: List[np.ndarray],
                          timestamps: List[int],
                          method: str = 'pca',
                          figsize: tuple = (16, 4),
                          save_path: Optional[str] = None):
    """
    Plot vector space evolution over time.
    
    Args:
        vector_snapshots: List of vector arrays at different times
        timestamps: Corresponding time steps
        method: Dimensionality reduction method
        figsize: Figure size
        save_path: Optional path to save figure
    """
    num_snapshots = len(vector_snapshots)
    
    fig, axes = plt.subplots(1, num_snapshots, figsize=figsize)
    if num_snapshots == 1:
        axes = [axes]
    
    for idx, (vectors, t) in enumerate(zip(vector_snapshots, timestamps)):
        ax = axes[idx]
        N, d = vectors.shape
        
        # Project to 2D
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(vectors)
        else:
            perplexity = min(30, N - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords_2d = reducer.fit_transform(vectors)
        
        # Plot
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                  c=np.arange(N), cmap='viridis',
                  s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        ax.set_title(f't = {t}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=9)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=9)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Vector Space Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_vector_interactive_3d(vectors: np.ndarray,
                               method: str = 'pca',
                               labels: Optional[np.ndarray] = None,
                               title: str = "Interactive 3D Vector Space"):
    """
    Create interactive 3D vector space visualization.
    
    Args:
        vectors: Shape (N, d) - vectors
        method: Dimensionality reduction method
        labels: Optional cluster labels
        title: Plot title
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    N, d = vectors.shape
    
    # Project to 3D
    if method == 'pca':
        reducer = PCA(n_components=3, random_state=42)
        coords_3d = reducer.fit_transform(vectors)
    else:
        perplexity = min(30, N - 1)
        reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        coords_3d = reducer.fit_transform(vectors)
    
    # Create trace
    if labels is not None:
        # Color by cluster
        scatter = go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=labels,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster"),
                line=dict(width=0.5, color='white')
            ),
            text=[f'Node {i}<br>Cluster: {labels[i]}' for i in range(N)],
            hoverinfo='text'
        )
    else:
        scatter = go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=np.arange(N),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Node Index"),
                line=dict(width=0.5, color='white')
            ),
            text=[f'Node {i}' for i in range(N)],
            hoverinfo='text'
        )
    
    fig = go.Figure(data=[scatter])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            zaxis_title=f'{method.upper()} 3',
        ),
        height=700
    )
    
    return fig
