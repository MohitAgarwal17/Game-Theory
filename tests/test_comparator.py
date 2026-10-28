"""
Unit tests for Mechanism Comparator and Evaluation Metrics.
"""

import pytest
from src.games.normal_form import NormalFormGame
from src.analysis.comparator import MechanismComparator
from src.analysis.metrics import MetricsCalculator, SimulationRecord
from src.utils.payoff_matrices import chicken_game


def test_metrics_calculator():
    """Test standard evaluation metrics calculation using MetricsCalculator."""
    calc = MetricsCalculator()

    # Create dummy records
    records = [
        SimulationRecord(
            vehicle_ids=["V1", "V2"],
            actions=[0, 1],
            payoffs=[10.0, 2.0],
            wait_times=[0.0, 2.0],
            collision=False,
            near_miss=False,
            mechanism="stackelberg",
        ),
        SimulationRecord(
            vehicle_ids=["V3", "V4"],
            actions=[1, 0],
            payoffs=[2.0, 10.0],
            wait_times=[2.0, 0.0],
            collision=False,
            near_miss=False,
            mechanism="stackelberg",
        ),
    ]

    all_metrics = calc.compute_all(records, social_optimum=12.0)

    assert all_metrics["safety"].collision_rate == 0.0
    assert all_metrics["safety"].total_collisions == 0
    assert all_metrics["efficiency"].avg_wait_time == 1.0
    assert all_metrics["fairness"].jain_fairness_index > 0.0
    assert all_metrics["game_theoretic"].social_welfare == 12.0
    assert all_metrics["game_theoretic"].price_of_anarchy == 1.0


def test_comparator_all_mechanisms():
    """Test MechanismComparator generates complete benchmark dictionary."""
    comparator = MechanismComparator()
    payoff_a, payoff_b = chicken_game()
    game = NormalFormGame(payoff_a, payoff_b)

    results = comparator.compare_all(game)

    expected_mechanisms = [
        "Nash Equilibrium",
        "Nash (Worst)",
        "Stackelberg",
        "Correlated Equilibrium",
        "Traffic Light",
        "Social Optimum",
    ]

    for mech in expected_mechanisms:
        assert mech in results
        assert "social_welfare" in results[mech]
        assert "payoff_a" in results[mech]
        assert "payoff_b" in results[mech]

    assert "_price_of_anarchy" in results
