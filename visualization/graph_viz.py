"""
Graph visualization for the Cognitive Engine.

Visualizes the dynamic graph structure, showing edge weights, clusters,
and topology evolution over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_graph_structure(adjacency: np.ndarray,
                         threshold: float = 0.1,
                         title: str = "Graph Structure",
                         figsize: tuple = (12, 10),
                         node_labels: Optional[List] = None,
                         save_path: Optional[str] = None):
    """
    Plot graph structure with edge weights (matplotlib + networkx).
    
    Args:
        adjacency: Shape (N, N) - adjacency matrix
        threshold: Minimum edge weight to display
        title: Plot title
        figsize: Figure size
        node_labels: Optional labels for nodes
        save_path: Optional path to save figure
    """
    N = len(adjacency)
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(N):
        G.add_node(i)
    
    # Add edges above threshold
    edges_with_weights = []
    for i in range(N):
        for j in range(i+1, N):  # Upper triangle only
            if abs(adjacency[i, j]) > threshold:
                G.add_edge(i, j, weight=adjacency[i, j])
                edges_with_weights.append((i, j, adjacency[i, j]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    if N <= 100:
        pos = nx.spring_layout(G, k=2/np.sqrt(N), iterations=50, seed=42)
    else:
        # For large graphs, use faster layout
        pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes
    node_sizes = [100 + 50 * G.degree(i) for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color='#2E86AB', alpha=0.7, ax=ax)
    
    # Draw edges with varying thickness based on weight
    if edges_with_weights:
        weights = [abs(w) for _, _, w in edges_with_weights]
        max_weight = max(weights) if weights else 1.0
        
        for i, j, w in edges_with_weights:
            width = 3 * (abs(w) / max_weight)
            color = '#6A994E' if w > 0 else '#C73E1D'
            nx.draw_networkx_edges(G, pos, [(i, j)], 
                                  width=width, alpha=0.5,
                                  edge_color=color, ax=ax)
    
    # Labels for small graphs
    if node_labels and N <= 50:
        nx.draw_networkx_labels(G, pos, labels={i: node_labels[i] for i in range(N)},
                               font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    if edges_with_weights:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#6A994E', linewidth=2, label='Positive weight'),
            Line2D([0], [0], color='#C73E1D', linewidth=2, label='Negative weight')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add stats
    density = len(edges_with_weights) / (N * (N-1) / 2)
    stats_text = f"Nodes: {N}\nEdges (>{threshold}): {len(edges_with_weights)}\nDensity: {density:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_edge_distribution(adjacency: np.ndarray,
                           title: str = "Edge Weight Distribution",
                           figsize: tuple = (10, 6),
                           save_path: Optional[str] = None):
    """
    Plot distribution of edge weights (histogram).
    
    Args:
        adjacency: Shape (N, N) - adjacency matrix
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Get upper triangle (no diagonal)
    N = len(adjacency)
    upper_tri = adjacency[np.triu_indices(N, k=1)]
    
    # Filter near-zero
    significant = upper_tri[np.abs(upper_tri) > 1e-6]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(significant, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Edge Weight', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Edge Weight Histogram', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax1.legend()
    
    # Log-scale to show heavy tail
    ax2.hist(np.abs(significant), bins=50, color='#6A994E', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('|Edge Weight|', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Magnitude Distribution (Heavy Tail)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Stats
    stats_text = f"Mean: {np.mean(significant):.4f}\nStd: {np.std(significant):.4f}\nMax: {np.max(np.abs(significant)):.4f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_degree_distribution(adjacency: np.ndarray,
                             title: str = "Degree Distribution",
                             figsize: tuple = (10, 6),
                             threshold: float = 0.01,
                             save_path: Optional[str] = None):
    """
    Plot node degree distribution.
    
    Args:
        adjacency: Shape (N, N) - adjacency matrix
        threshold: Minimum edge weight to count
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Count edges above threshold for each node
    binary_adj = (np.abs(adjacency) > threshold).astype(int)
    np.fill_diagonal(binary_adj, 0)
    degrees = np.sum(binary_adj, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(degrees, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Node Degree', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Degree Histogram', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=np.mean(degrees), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(degrees):.1f}')
    ax1.legend()
    
    # Cumulative distribution
    sorted_degrees = np.sort(degrees)
    cumulative = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
    ax2.plot(sorted_degrees, cumulative, linewidth=2, color='#C73E1D')
    ax2.set_xlabel('Node Degree', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Degree Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_graph_evolution(adjacency_snapshots: List[np.ndarray],
                        timestamps: List[int],
                        threshold: float = 0.1,
                        figsize: tuple = (16, 4),
                        save_path: Optional[str] = None):
    """
    Plot graph evolution over time (multiple snapshots).
    
    Args:
        adjacency_snapshots: List of adjacency matrices at different times
        timestamps: Corresponding time steps
        threshold: Minimum edge weight to display
        figsize: Figure size
        save_path: Optional path to save figure
    """
    num_snapshots = len(adjacency_snapshots)
    
    fig, axes = plt.subplots(1, num_snapshots, figsize=figsize)
    if num_snapshots == 1:
        axes = [axes]
    
    for idx, (adj, t) in enumerate(zip(adjacency_snapshots, timestamps)):
        ax = axes[idx]
        N = len(adj)
        
        # Create graph
        G = nx.Graph()
        for i in range(N):
            G.add_node(i)
        
        for i in range(N):
            for j in range(i+1, N):
                if abs(adj[i, j]) > threshold:
                    G.add_edge(i, j, weight=adj[i, j])
        
        # Use consistent layout (spring with same seed)
        pos = nx.spring_layout(G, k=2/np.sqrt(N), iterations=30, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='#2E86AB', alpha=0.7, ax=ax)
        
        edges = list(G.edges(data=True))
        if edges:
            weights = [abs(e[2]['weight']) for e in edges]
            max_w = max(weights) if weights else 1.0
            
            for i, j, data in edges:
                w = data['weight']
                width = 2 * (abs(w) / max_w)
                color = '#6A994E' if w > 0 else '#C73E1D'
                nx.draw_networkx_edges(G, pos, [(i, j)], width=width,
                                      alpha=0.4, edge_color=color, ax=ax)
        
        density = len(edges) / (N * (N-1) / 2) if N > 1 else 0
        ax.set_title(f't = {t}\nEdges: {len(edges)}\nDensity: {density:.2f}',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Graph Structure Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_graph_interactive(adjacency: np.ndarray,
                           vectors: Optional[np.ndarray] = None,
                           threshold: float = 0.1,
                           title: str = "Interactive Graph Structure"):
    """
    Create interactive graph visualization with plotly.
    
    Args:
        adjacency: Shape (N, N) - adjacency matrix
        vectors: Optional shape (N, d) - node embeddings for layout
        threshold: Minimum edge weight to display
        title: Plot title
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    N = len(adjacency)
    
    # Create networkx graph for layout
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    
    edges = []
    edge_weights = []
    for i in range(N):
        for j in range(i+1, N):
            if abs(adjacency[i, j]) > threshold:
                G.add_edge(i, j, weight=adjacency[i, j])
                edges.append((i, j))
                edge_weights.append(adjacency[i, j])
    
    # Layout
    if vectors is not None and vectors.shape[1] >= 2:
        # Use first 2 dimensions of vectors for layout
        pos = {i: (vectors[i, 0], vectors[i, 1]) for i in range(N)}
    else:
        pos = nx.spring_layout(G, k=2/np.sqrt(N), iterations=50, seed=42)
    
    # Create edge traces
    edge_traces = []
    for (i, j), w in zip(edges, edge_weights):
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        
        color = '#6A994E' if w > 0 else '#C73E1D'
        width = 1 + 3 * (abs(w) / max(np.abs(edge_weights)))
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = [pos[i][0] for i in range(N)]
    node_y = [pos[i][1] for i in range(N)]
    node_degrees = [G.degree(i) for i in range(N)]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=[10 + 2*d for d in node_degrees],
            color=node_degrees,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Degree"),
            line=dict(width=1, color='white')
        ),
        text=[f'Node {i}<br>Degree: {G.degree(i)}' for i in range(N)],
        hoverinfo='text',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700
    )
    
    return fig
