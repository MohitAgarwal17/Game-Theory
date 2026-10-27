"""Analysis module — metrics, comparison, and visualization."""

from src.analysis.comparator import MechanismComparator
from src.analysis.metrics import (
    MetricsCalculator,
    SimulationRecord,
    SafetyMetrics,
    EfficiencyMetrics,
    FairnessMetrics,
    GameTheoreticMetrics,
)
from src.analysis.plotter import (
    plot_payoff_matrix,
    plot_best_response,
    plot_repeated_game_history,
    plot_mechanism_comparison,
)

__all__ = [
    "MechanismComparator",
    "MetricsCalculator",
    "SimulationRecord",
    "SafetyMetrics",
    "EfficiencyMetrics",
    "FairnessMetrics",
    "GameTheoreticMetrics",
    "plot_payoff_matrix",
    "plot_best_response",
    "plot_repeated_game_history",
    "plot_mechanism_comparison",
]

