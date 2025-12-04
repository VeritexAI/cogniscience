"""Update dynamics and stochastic exploration for the Cognitive Engine."""

from cognigraph.dynamics.dreaming import (
    add_exploration_noise,
    detect_creative_bridges,
    controlled_dreaming_step,
    controlled_dreaming_step_recency,
    pre_dream_rumination,
    activation_weighted_noise,
    compute_temporal_gradient,
    compute_combined_gradient,
    measure_energy_fluctuation
)
from cognigraph.dynamics.updates import apply_updates

__all__ = [
    "apply_updates",
    "add_exploration_noise",
    "detect_creative_bridges",
    "controlled_dreaming_step",
    "controlled_dreaming_step_recency",
    "pre_dream_rumination",
    "activation_weighted_noise",
    "compute_temporal_gradient",
    "compute_combined_gradient",
    "measure_energy_fluctuation"
]
