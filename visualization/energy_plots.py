"""
Energy visualization for the Cognitive Engine.

Provides functions to plot energy evolution, component breakdowns,
and convergence metrics over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_energy_evolution(energy_history: List[float], 
                          title: str = "Energy Evolution",
                          figsize: tuple = (10, 6),
                          save_path: Optional[str] = None):
    """
    Plot total energy over time (matplotlib).
    
    Args:
        energy_history: List of energy values over time
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = np.arange(len(energy_history))
    
    ax.plot(steps, energy_history, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Total Energy $E_t$', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotate initial and final
    ax.annotate(f'Initial: {energy_history[0]:.2f}',
                xy=(0, energy_history[0]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.annotate(f'Final: {energy_history[-1]:.2f}',
                xy=(len(energy_history)-1, energy_history[-1]),
                xytext=(-80, -20), textcoords='offset points',
                fontsize=10, color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_energy_components(results: List[Dict],
                           title: str = "Energy Components",
                           figsize: tuple = (12, 6),
                           save_path: Optional[str] = None):
    """
    Plot energy components (E_C, E_V, E_G) over time.
    
    Args:
        results: List of step results with energy dicts
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    steps = [r['step'] for r in results]
    E_C = [r['energy']['E_C'] for r in results]
    E_V = [r['energy']['E_V'] for r in results]
    E_G = [r['energy']['E_G'] for r in results]
    E_total = [r['energy']['E_total'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Stacked area plot of components
    ax1.fill_between(steps, 0, E_C, alpha=0.7, label='$E_C$ (Cache)', color='#F18F01')
    ax1.fill_between(steps, E_C, np.array(E_C) + np.array(E_V), 
                     alpha=0.7, label='$E_V$ (Vector)', color='#C73E1D')
    ax1.fill_between(steps, np.array(E_C) + np.array(E_V), E_total,
                     alpha=0.7, label='$E_G$ (Graph)', color='#6A994E')
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Energy Components (Stacked)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Individual components
    ax2.plot(steps, E_C, label='$E_C$ (Cache)', linewidth=2, color='#F18F01')
    ax2.plot(steps, E_V, label='$E_V$ (Vector)', linewidth=2, color='#C73E1D')
    ax2.plot(steps, E_G, label='$E_G$ (Graph)', linewidth=2, color='#6A994E')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Components (Separate)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gradient_evolution(results: List[Dict],
                            title: str = "Gradient Norm Evolution",
                            figsize: tuple = (10, 6),
                            save_path: Optional[str] = None):
    """
    Plot gradient norm over time to show convergence.
    
    Args:
        results: List of step results with gradient_norm
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    steps = [r['step'] for r in results]
    grad_norms = [r['gradient_norm'] for r in results]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(steps, grad_norms, linewidth=2, color='#A23B72')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('$||\\nabla E_t||$ (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add convergence threshold line
    threshold = 1e-3
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
               label=f'Convergence threshold ({threshold})')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_dashboard(results: List[Dict],
                           figsize: tuple = (14, 10),
                           save_path: Optional[str] = None):
    """
    Create comprehensive dashboard with multiple metrics.
    
    Args:
        results: List of step results
        figsize: Figure size
        save_path: Optional path to save figure
    """
    steps = [r['step'] for r in results]
    energies = [r['energy']['E_total'] for r in results]
    grad_norms = [r['gradient_norm'] for r in results]
    densities = [r['metrics']['density'] for r in results]
    mean_degrees = [r['metrics']['mean_degree'] for r in results]
    mean_sims = [r['metrics']['mean_similarity'] for r in results]
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Energy evolution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, energies, linewidth=2, color='#2E86AB')
    ax1.set_ylabel('Total Energy', fontsize=11)
    ax1.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradient norm (log scale)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(steps, grad_norms, linewidth=2, color='#A23B72')
    ax2.set_ylabel('$||\\nabla E_t||$', fontsize=11)
    ax2.set_title('Gradient Norm (Convergence)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Graph density
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(steps, densities, linewidth=2, color='#6A994E')
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Graph Density', fontsize=11, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    
    # 4. Mean degree
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(steps, mean_degrees, linewidth=2, color='#F18F01')
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Mean Degree', fontsize=11)
    ax4.set_title('Average Node Connectivity', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Mean similarity
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(steps, mean_sims, linewidth=2, color='#C73E1D')
    ax5.set_xlabel('Time Step', fontsize=11)
    ax5.set_ylabel('Mean Similarity', fontsize=11)
    ax5.set_title('Vector Coherence', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Cognitive Engine Metrics Dashboard', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Interactive plotly versions
def plot_energy_interactive(energy_history: List[float],
                            title: str = "Energy Evolution (Interactive)"):
    """
    Create interactive energy plot with plotly.
    
    Args:
        energy_history: List of energy values
        title: Plot title
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    steps = np.arange(len(energy_history))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=energy_history,
        mode='lines',
        name='Energy',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='Step: %{x}<br>Energy: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Step',
        yaxis_title='Total Energy E_t',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_metrics_interactive(results: List[Dict],
                             title: str = "Cognitive Engine Metrics"):
    """
    Create interactive metrics dashboard with plotly.
    
    Args:
        results: List of step results
        title: Dashboard title
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    steps = [r['step'] for r in results]
    energies = [r['energy']['E_total'] for r in results]
    E_C = [r['energy']['E_C'] for r in results]
    E_V = [r['energy']['E_V'] for r in results]
    E_G = [r['energy']['E_G'] for r in results]
    grad_norms = [r['gradient_norm'] for r in results]
    densities = [r['metrics']['density'] for r in results]
    mean_degrees = [r['metrics']['mean_degree'] for r in results]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Total Energy', 'Energy Components',
                       'Gradient Norm', 'Graph Density',
                       'Mean Degree', 'Convergence Check'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Total energy
    fig.add_trace(go.Scatter(x=steps, y=energies, name='E_total',
                            line=dict(color='#2E86AB')), row=1, col=1)
    
    # Energy components
    fig.add_trace(go.Scatter(x=steps, y=E_C, name='E_C', line=dict(color='#F18F01')), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=E_V, name='E_V', line=dict(color='#C73E1D')), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=E_G, name='E_G', line=dict(color='#6A994E')), row=1, col=2)
    
    # Gradient norm (log scale)
    fig.add_trace(go.Scatter(x=steps, y=grad_norms, name='||∇E||',
                            line=dict(color='#A23B72')), row=2, col=1)
    
    # Graph density
    fig.add_trace(go.Scatter(x=steps, y=densities, name='Density',
                            line=dict(color='#6A994E')), row=2, col=2)
    
    # Mean degree
    fig.add_trace(go.Scatter(x=steps, y=mean_degrees, name='Mean Degree',
                            line=dict(color='#F18F01')), row=3, col=1)
    
    # Energy decrease rate
    energy_diffs = np.diff(energies)
    fig.add_trace(go.Scatter(x=steps[1:], y=energy_diffs, name='dE/dt',
                            line=dict(color='#C73E1D')), row=3, col=2)
    
    # Update axes
    fig.update_yaxes(type='log', row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        height=900,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig
