"""Update dynamics and stochastic exploration for the Cognitive Engine."""

from cognigraph.dynamics.dreaming import add_exploration_noise, detect_creative_bridges
from cognigraph.dynamics.updates import apply_updates

__all__ = ["apply_updates", "add_exploration_noise", "detect_creative_bridges"]
