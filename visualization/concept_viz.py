"""
Concept graph visualization for hierarchical abstraction.

Visualizes the concept layer extracted from memory clusters,
including concept relationships, activation heatmaps, and
the dual-level (memory + concept) graph structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List, Dict, Tuple
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_concept_graph(
    concept_extractor,
    title: str = "Concept Graph",
    figsize: Tuple[int, int] = (14, 10),
    min_edge_weight: float = 0.3,
    show_keywords: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot the concept-level graph showing abstract concepts and their relationships.
    
    Args:
        concept_extractor: ConceptExtractor instance with extracted concepts
        title: Plot title
        figsize: Figure size
        min_edge_weight: Minimum edge weight to display
        show_keywords: Whether to show concept keywords
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    G = concept_extractor.to_networkx()
    if G is None or G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No concepts extracted yet", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter edges by weight
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get('weight', 0) < min_edge_weight]
    G.remove_edges_from(edges_to_remove)
    
    # Layout
    if G.number_of_nodes() <= 20:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Node properties based on activation and member count
    activations = [G.nodes[n].get('activation', 0.5) for n in G.nodes()]
    member_counts = [G.nodes[n].get('member_count', 1) for n in G.nodes()]
    
    # Scale node sizes by member count
    max_members = max(member_counts) if member_counts else 1
    node_sizes = [300 + 500 * (m / max_members) for m in member_counts]
    
    # Color by activation (red = low, green = high)
    cmap = plt.cm.RdYlGn
    node_colors = [cmap(a) for a in activations]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=activations,
        cmap=cmap,
        vmin=0, vmax=1,
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges with weight-based thickness
    edges = list(G.edges(data=True))
    if edges:
        max_weight = max(d.get('weight', 0.1) for _, _, d in edges)
        for u, v, d in edges:
            weight = d.get('weight', 0.1)
            width = 1 + 4 * (weight / max_weight)
            alpha = 0.3 + 0.5 * (weight / max_weight)
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                width=width,
                alpha=alpha,
                edge_color='#4A90A4',
                ax=ax
            )
    
    # Labels: show concept names
    labels = {n: G.nodes[n].get('name', f'C{n}')[:20] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    # Add keywords if requested
    if show_keywords:
        for n in G.nodes():
            keywords = G.nodes[n].get('keywords', [])[:3]
            if keywords:
                x, y = pos[n]
                keyword_text = ', '.join(keywords)
                ax.annotate(
                    keyword_text,
                    (x, y - 0.08),
                    ha='center', va='top',
                    fontsize=7, color='gray',
                    style='italic'
                )
    
    # Colorbar for activation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Activation Level', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add summary stats
    summary = concept_extractor.get_concept_summary()
    stats_text = (
        f"Concepts: {summary['total_concepts']}\n"
        f"Edges: {summary['total_edges']}\n"
        f"Avg Coherence: {summary['avg_coherence']:.2f}\n"
        f"Avg Activation: {summary['avg_activation']:.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_activation_heatmap(
    activations: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Memory Activation Heatmap",
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
):
    """
    Plot activation levels as a heatmap.
    
    Args:
        activations: Activation array (N,)
        labels: Optional text labels for each memory
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    N = len(activations)
    
    # Reshape for heatmap
    n_cols = min(50, N)
    n_rows = (N + n_cols - 1) // n_cols
    padded = np.zeros(n_rows * n_cols)
    padded[:N] = activations
    heatmap_data = padded.reshape(n_rows, n_cols)
    
    # Create custom colormap (cool blue to hot red)
    colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', 
              '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    cmap = LinearSegmentedColormap.from_list('activation', colors)
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xlabel('Memory Index (mod 50)', fontsize=11)
    ax.set_ylabel('Row', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Activation Level', fontsize=10)
    
    # Stats annotation
    nonzero = activations[activations > 0]
    if len(nonzero) > 0:
        stats_text = (
            f"Active: {len(nonzero)}/{N}\n"
            f"Mean: {np.mean(nonzero):.3f}\n"
            f"Max: {np.max(activations):.3f}"
        )
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dual_level_graph(
    memory_adjacency: np.ndarray,
    concept_extractor,
    memory_activations: np.ndarray,
    memory_labels: Optional[List[str]] = None,
    title: str = "Dual-Level Graph: Memories and Concepts",
    figsize: Tuple[int, int] = (16, 12),
    memory_threshold: float = 0.1,
    save_path: Optional[str] = None
):
    """
    Plot both memory-level and concept-level graphs with connections between layers.
    
    Args:
        memory_adjacency: Memory-level adjacency matrix (N, N)
        concept_extractor: ConceptExtractor with concepts
        memory_activations: Activation levels for memories (N,)
        memory_labels: Optional text labels for memories
        title: Plot title
        figsize: Figure size
        memory_threshold: Threshold for displaying memory edges
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, (ax_mem, ax_concept) = plt.subplots(1, 2, figsize=figsize)
    
    N = len(memory_adjacency)
    
    # === Memory Level (Left Panel) ===
    G_mem = nx.Graph()
    for i in range(N):
        G_mem.add_node(i)
    
    for i in range(N):
        for j in range(i+1, N):
            if abs(memory_adjacency[i, j]) > memory_threshold:
                G_mem.add_edge(i, j, weight=memory_adjacency[i, j])
    
    pos_mem = nx.spring_layout(G_mem, k=2/np.sqrt(max(N, 1)), iterations=50, seed=42)
    
    # Color memories by concept membership
    concept_colors = {}
    color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for concept in concept_extractor.concepts.values():
        color = color_palette[concept.concept_id % 20]
        for idx in concept.member_indices:
            concept_colors[idx] = color
    
    node_colors_mem = [concept_colors.get(i, (0.7, 0.7, 0.7, 0.7)) for i in range(N)]
    
    # Size by activation
    node_sizes_mem = [50 + 200 * memory_activations[i] for i in range(N)]
    
    nx.draw_networkx_nodes(G_mem, pos_mem, node_size=node_sizes_mem,
                          node_color=node_colors_mem, alpha=0.7, ax=ax_mem)
    
    # Draw edges
    edges = list(G_mem.edges(data=True))
    if edges:
        max_w = max(abs(d['weight']) for _, _, d in edges)
        for u, v, d in edges:
            w = d['weight']
            width = 1 + 2 * (abs(w) / max_w)
            color = '#6A994E' if w > 0 else '#C73E1D'
            nx.draw_networkx_edges(G_mem, pos_mem, [(u, v)],
                                  width=width, alpha=0.3,
                                  edge_color=color, ax=ax_mem)
    
    ax_mem.set_title("Memory Level\n(colored by concept)", fontsize=12, fontweight='bold')
    ax_mem.axis('off')
    
    # === Concept Level (Right Panel) ===
    G_concept = concept_extractor.to_networkx()
    
    if G_concept and G_concept.number_of_nodes() > 0:
        pos_concept = nx.spring_layout(G_concept, k=2, iterations=50, seed=42)
        
        # Node properties
        concept_activations = [G_concept.nodes[n].get('activation', 0.5) for n in G_concept.nodes()]
        member_counts = [G_concept.nodes[n].get('member_count', 1) for n in G_concept.nodes()]
        max_members = max(member_counts) if member_counts else 1
        
        node_sizes_concept = [300 + 600 * (m / max_members) for m in member_counts]
        node_colors_concept = [color_palette[n % 20] for n in G_concept.nodes()]
        
        nx.draw_networkx_nodes(G_concept, pos_concept, 
                              node_size=node_sizes_concept,
                              node_color=node_colors_concept,
                              alpha=0.8, ax=ax_concept)
        
        # Edges
        edges_c = list(G_concept.edges(data=True))
        if edges_c:
            max_w = max(d.get('weight', 0.1) for _, _, d in edges_c)
            for u, v, d in edges_c:
                w = d.get('weight', 0.1)
                width = 1 + 4 * (w / max_w)
                nx.draw_networkx_edges(G_concept, pos_concept, [(u, v)],
                                      width=width, alpha=0.5,
                                      edge_color='#4A90A4', ax=ax_concept)
        
        # Labels
        labels = {n: G_concept.nodes[n].get('name', f'C{n}')[:15] for n in G_concept.nodes()}
        nx.draw_networkx_labels(G_concept, pos_concept, labels, 
                               font_size=9, font_weight='bold', ax=ax_concept)
    else:
        ax_concept.text(0.5, 0.5, "No concepts extracted", 
                       ha='center', va='center', fontsize=14)
    
    ax_concept.set_title("Concept Level\n(size = member count)", fontsize=12, fontweight='bold')
    ax_concept.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_concept_evolution(
    concept_snapshots: List[Dict],
    timestamps: List[float],
    title: str = "Concept Evolution Over Time",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot how concepts evolve over time.
    
    Args:
        concept_snapshots: List of concept summary dicts over time
        timestamps: Corresponding timestamps
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract metrics over time
    n_concepts = [s.get('total_concepts', 0) for s in concept_snapshots]
    n_edges = [s.get('total_edges', 0) for s in concept_snapshots]
    avg_coherence = [s.get('avg_coherence', 0) for s in concept_snapshots]
    avg_activation = [s.get('avg_activation', 0) for s in concept_snapshots]
    
    # Number of concepts
    axes[0, 0].plot(timestamps, n_concepts, 'o-', linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Number of Concepts', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of edges
    axes[0, 1].plot(timestamps, n_edges, 's-', linewidth=2, color='#6A994E')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Concept-Concept Edges', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average coherence
    axes[1, 0].plot(timestamps, avg_coherence, '^-', linewidth=2, color='#F18F01')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Coherence')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Average Concept Coherence', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average activation
    axes[1, 1].plot(timestamps, avg_activation, 'd-', linewidth=2, color='#C73E1D')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Activation')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Average Concept Activation', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_concept_interactive(concept_extractor, title: str = "Interactive Concept Graph"):
    """
    Create interactive concept graph with Plotly.
    
    Args:
        concept_extractor: ConceptExtractor with concepts
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    G = concept_extractor.to_networkx()
    if G is None or G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No concepts extracted yet", 
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Edge traces
    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = d.get('weight', 0.5)
        
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=1 + 4*weight, color='rgba(74, 144, 164, 0.5)'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Node trace
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_activations = [G.nodes[n].get('activation', 0.5) for n in G.nodes()]
    member_counts = [G.nodes[n].get('member_count', 1) for n in G.nodes()]
    names = [G.nodes[n].get('name', f'C{n}') for n in G.nodes()]
    keywords = [', '.join(G.nodes[n].get('keywords', [])[:5]) for n in G.nodes()]
    
    hover_text = [
        f"<b>{name}</b><br>"
        f"Members: {mc}<br>"
        f"Activation: {act:.2f}<br>"
        f"Keywords: {kw}"
        for name, mc, act, kw in zip(names, member_counts, node_activations, keywords)
    ]
    
    max_members = max(member_counts) if member_counts else 1
    node_sizes = [15 + 35 * (m / max_members) for m in member_counts]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_activations,
            colorscale='RdYlGn',
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="Activation"),
            line=dict(width=2, color='white')
        ),
        text=names,
        textposition='top center',
        textfont=dict(size=10),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    )
    
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
