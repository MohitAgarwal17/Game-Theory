"""
Metrics module for evaluating intersection management strategies.

Computes safety, efficiency, fairness, and game-theoretic metrics
to enable rigorous comparison between different approaches
(Nash, Stackelberg, Auction, Correlated Equilibrium, etc.).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimulationRecord:
    """
    Record of a single intersection interaction for metric computation.

    Attributes
    ----------
    vehicle_ids : list[str]
        IDs of participating vehicles.
    actions : list[int]
        Action chosen by each vehicle (0=Go, 1=Yield).
    payoffs : list[float]
        Payoff received by each vehicle.
    wait_times : list[float]
        Time each vehicle waited (seconds).
    collision : bool
        Whether a collision occurred.
    near_miss : bool
        Whether a near-miss occurred.
    mechanism : str
        Which mechanism was used (nash, stackelberg, auction, etc.).
    """
    vehicle_ids: list[str]
    actions: list[int]
    payoffs: list[float]
    wait_times: list[float]
    collision: bool = False
    near_miss: bool = False
    mechanism: str = ""


@dataclass
class SafetyMetrics:
    """Safety-related metrics."""
    collision_rate: float
    near_miss_rate: float
    total_collisions: int
    total_near_misses: int
    total_interactions: int
    avg_time_to_collision: float = float("inf")


@dataclass
class EfficiencyMetrics:
    """Efficiency-related metrics."""
    avg_wait_time: float
    max_wait_time: float
    min_wait_time: float
    throughput: float  # vehicles per unit time
    total_delay: float
    avg_payoff: float


@dataclass
class FairnessMetrics:
    """Fairness-related metrics."""
    jain_fairness_index: float
    gini_coefficient: float
    max_min_ratio: float
    std_wait_time: float
    coefficient_of_variation: float


@dataclass
class GameTheoreticMetrics:
    """Game-theory-specific metrics."""
    price_of_anarchy: float
    price_of_stability: float
    social_welfare: float
    avg_regret: float
    social_optimum: float


class MetricsCalculator:
    """
    Computes all metric categories from simulation records.

    Examples
    --------
    >>> calc = MetricsCalculator()
    >>> records = [SimulationRecord(...), ...]
    >>> safety = calc.compute_safety(records)
    >>> efficiency = calc.compute_efficiency(records)
    >>> fairness = calc.compute_fairness(records)
    """

    def compute_safety(self, records: list[SimulationRecord]) -> SafetyMetrics:
        """
        Compute safety metrics from simulation records.

        Parameters
        ----------
        records : list[SimulationRecord]
            Simulation interaction records.

        Returns
        -------
        SafetyMetrics
            Computed safety metrics.
        """
        n = len(records)
        if n == 0:
            return SafetyMetrics(0, 0, 0, 0, 0)

        collisions = sum(1 for r in records if r.collision)
        near_misses = sum(1 for r in records if r.near_miss)

        return SafetyMetrics(
            collision_rate=collisions / n,
            near_miss_rate=near_misses / n,
            total_collisions=collisions,
            total_near_misses=near_misses,
            total_interactions=n,
        )

    def compute_efficiency(
        self, records: list[SimulationRecord]
    ) -> EfficiencyMetrics:
        """
        Compute efficiency metrics from simulation records.

        Parameters
        ----------
        records : list[SimulationRecord]
            Simulation interaction records.

        Returns
        -------
        EfficiencyMetrics
            Computed efficiency metrics.
        """
        if not records:
            return EfficiencyMetrics(0, 0, 0, 0, 0, 0)

        all_waits = []
        all_payoffs = []
        for r in records:
            all_waits.extend(r.wait_times)
            all_payoffs.extend(r.payoffs)

        if not all_waits:
            all_waits = [0.0]

        total_time = max(sum(all_waits), 1.0)
        total_vehicles = len(all_waits)

        return EfficiencyMetrics(
            avg_wait_time=np.mean(all_waits),
            max_wait_time=max(all_waits),
            min_wait_time=min(all_waits),
            throughput=total_vehicles / total_time if total_time > 0 else 0,
            total_delay=sum(all_waits),
            avg_payoff=np.mean(all_payoffs) if all_payoffs else 0,
        )

    def compute_fairness(
        self, records: list[SimulationRecord]
    ) -> FairnessMetrics:
        """
        Compute fairness metrics from simulation records.

        Parameters
        ----------
        records : list[SimulationRecord]
            Simulation interaction records.

        Returns
        -------
        FairnessMetrics
            Computed fairness metrics.
        """
        all_waits = []
        for r in records:
            all_waits.extend(r.wait_times)

        if not all_waits or len(all_waits) < 2:
            return FairnessMetrics(1.0, 0.0, 1.0, 0.0, 0.0)

        waits = np.array(all_waits)

        # Jain's Fairness Index: (Σxi)² / (n · Σxi²)
        n = len(waits)
        sum_x = waits.sum()
        sum_x2 = (waits ** 2).sum()
        jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0

        # Gini Coefficient
        gini = self._gini_coefficient(waits)

        # Max-Min ratio
        min_wait = waits.min()
        max_wait = waits.max()
        max_min = max_wait / min_wait if min_wait > 0 else float("inf")

        # Standard deviation and CV
        std = float(np.std(waits))
        mean = float(np.mean(waits))
        cv = std / mean if mean > 0 else 0.0

        return FairnessMetrics(
            jain_fairness_index=float(jain),
            gini_coefficient=gini,
            max_min_ratio=float(max_min),
            std_wait_time=std,
            coefficient_of_variation=cv,
        )

    def compute_game_theoretic(
        self,
        records: list[SimulationRecord],
        social_optimum: float | None = None,
    ) -> GameTheoreticMetrics:
        """
        Compute game-theoretic metrics.

        Parameters
        ----------
        records : list[SimulationRecord]
            Simulation interaction records.
        social_optimum : float, optional
            The social optimum welfare for PoA/PoS computation.

        Returns
        -------
        GameTheoreticMetrics
            Computed game-theoretic metrics.
        """
        if not records:
            return GameTheoreticMetrics(1.0, 1.0, 0.0, 0.0, 0.0)

        # Social welfare per interaction
        welfares = [sum(r.payoffs) for r in records]
        avg_welfare = np.mean(welfares)
        worst_welfare = min(welfares)
        best_welfare = max(welfares)

        if social_optimum is None:
            social_optimum = best_welfare

        # Price of Anarchy and Stability
        poa = (social_optimum / worst_welfare
               if worst_welfare > 0 else float("inf"))
        pos = (social_optimum / best_welfare
               if best_welfare > 0 else float("inf"))

        # Average regret (difference from best possible payoff)
        all_payoffs = [p for r in records for p in r.payoffs]
        best_payoff = max(all_payoffs) if all_payoffs else 0
        avg_regret = best_payoff - np.mean(all_payoffs) if all_payoffs else 0

        return GameTheoreticMetrics(
            price_of_anarchy=poa,
            price_of_stability=pos,
            social_welfare=float(avg_welfare),
            avg_regret=float(avg_regret),
            social_optimum=float(social_optimum),
        )

    def compute_all(
        self,
        records: list[SimulationRecord],
        social_optimum: float | None = None,
    ) -> dict:
        """
        Compute all metric categories at once.

        Returns
        -------
        dict
            Contains 'safety', 'efficiency', 'fairness', 'game_theoretic'.
        """
        return {
            "safety": self.compute_safety(records),
            "efficiency": self.compute_efficiency(records),
            "fairness": self.compute_fairness(records),
            "game_theoretic": self.compute_game_theoretic(
                records, social_optimum
            ),
        }

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """Compute the Gini coefficient for an array of values."""
        if len(values) == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        indices = np.arange(1, n + 1)
        numerator = 2 * np.sum(indices * sorted_vals)
        denominator = n * np.sum(sorted_vals)
        if denominator == 0:
            return 0.0
        return float((numerator / denominator) - (n + 1) / n)
