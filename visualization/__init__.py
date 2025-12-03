"""
Visualization tools for the Cognitive Engine.

Provides comprehensive visualization of energy dynamics, graph structure,
and vector space organization using both static (matplotlib) and
interactive (plotly) visualizations.
"""

# Energy visualizations
from visualization.energy_plots import (
    plot_energy_evolution,
    plot_energy_components,
    plot_gradient_evolution,
    plot_metrics_dashboard,
    plot_energy_interactive,
    plot_metrics_interactive
)

# Graph visualizations
from visualization.graph_viz import (
    plot_graph_structure,
    plot_edge_distribution,
    plot_degree_distribution,
    plot_graph_evolution,
    plot_graph_interactive
)

# Vector space visualizations
from visualization.vector_space import (
    plot_vector_projection_2d,
    plot_vector_projection_3d,
    plot_similarity_matrix,
    plot_vector_evolution,
    plot_vector_interactive_3d
)

__all__ = [
    # Energy
    'plot_energy_evolution',
    'plot_energy_components',
    'plot_gradient_evolution',
    'plot_metrics_dashboard',
    'plot_energy_interactive',
    'plot_metrics_interactive',
    # Graph
    'plot_graph_structure',
    'plot_edge_distribution',
    'plot_degree_distribution',
    'plot_graph_evolution',
    'plot_graph_interactive',
    # Vector space
    'plot_vector_projection_2d',
    'plot_vector_projection_3d',
    'plot_similarity_matrix',
    'plot_vector_evolution',
    'plot_vector_interactive_3d',
]
