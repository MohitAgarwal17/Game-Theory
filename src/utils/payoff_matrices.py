"""
Predefined payoff matrices for intersection game scenarios.

Each matrix models a different strategic interaction between vehicles
at an intersection. Payoffs reflect real-world consequences:
  - Collision: catastrophic negative utility
  - Proceeding: time saved
  - Yielding: slight delay but safe
  - Deadlock: mutual waiting
"""

import numpy as np


def chicken_game(collision=-100, proceed=10, yield_to=2, deadlock=0):
    """
    Classic Chicken (Hawk-Dove) game for two vehicles at an intersection.

    Two vehicles approach an intersection simultaneously. Each must decide
    to Go (proceed) or Yield (wait). If both Go, they collide.

    Actions: {Go, Yield}
    
    Payoff matrix (row = Vehicle A, col = Vehicle B):
                    Go          Yield
        Go      (coll,coll)  (proc, yld)
        Yield   (yld, proc)  (dead, dead)

    Parameters
    ----------
    collision : float
        Payoff when both vehicles Go (crash).
    proceed : float
        Payoff for the vehicle that proceeds while other yields.
    yield_to : float
        Payoff for the vehicle that yields while other proceeds.
    deadlock : float
        Payoff when both vehicles yield (wasted time).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Payoff matrices for player A and player B.
    """
    payoff_a = np.array([
        [collision, proceed],
        [yield_to, deadlock],
    ], dtype=float)

    payoff_b = np.array([
        [collision, yield_to],
        [proceed, deadlock],
    ], dtype=float)

    return payoff_a, payoff_b


def asymmetric_intersection(priority_bonus=3):
    """
    Asymmetric intersection game where Vehicle A has right-of-way priority.

    Vehicle A is on a main road, Vehicle B is on a side road.
    A gets a bonus for proceeding due to implicit right-of-way.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Payoff matrices for player A (main road) and player B (side road).
    """
    payoff_a = np.array([
        [-100, 10 + priority_bonus],
        [2, 0],
    ], dtype=float)

    payoff_b = np.array([
        [-100, 2],
        [10, 0],
    ], dtype=float)

    return payoff_a, payoff_b


def coordination_game():
    """
    Pure coordination game: both vehicles benefit from choosing the same
    conventional rule (e.g., right-hand priority).

    Actions: {Follow Convention, Ignore Convention}
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Payoff matrices for both players.
    """
    payoff_a = np.array([
        [10, -50],
        [-50, 5],
    ], dtype=float)

    payoff_b = np.array([
        [10, -50],
        [-50, 5],
    ], dtype=float)

    return payoff_a, payoff_b


def three_strategy_intersection():
    """
    Extended intersection game with three strategies: Go, Slow, Yield.
    
    'Slow' means approaching the intersection cautiously — allows
    last-second braking but sacrifices some time.

    Actions: {Go, Slow, Yield}

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        3×3 payoff matrices for both players.
    """
    payoff_a = np.array([
        [-100, -20,  10],   # Go vs Go=-100, Go vs Slow=-20, Go vs Yield=10
        [ -20,   5,   7],   # Slow vs Go=-20, Slow vs Slow=5, Slow vs Yield=7
        [   2,   3,   0],   # Yield vs Go=2, Yield vs Slow=3, Yield vs Yield=0
    ], dtype=float)

    payoff_b = np.array([
        [-100, -20,   2],
        [ -20,   5,   3],
        [  10,   7,   0],
    ], dtype=float)

    return payoff_a, payoff_b


def emergency_vehicle_game(emergency_penalty=-200):
    """
    Game where Vehicle A is an emergency vehicle (ambulance/fire truck).
    Blocking an emergency vehicle incurs a severe social penalty.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Payoff matrices where Vehicle A has emergency priority.
    """
    payoff_a = np.array([
        [-100, 20],    # Emergency vehicle strongly prefers Go
        [0, -5],       # Yielding is very costly for emergency vehicle
    ], dtype=float)

    payoff_b = np.array([
        [emergency_penalty, 0],   # Blocking emergency = huge penalty
        [2, 0],
    ], dtype=float)

    return payoff_a, payoff_b


def bayesian_type_payoffs():
    """
    Type-dependent payoff matrices for Bayesian games.
    
    Two driver types:
    - Aggressive (θ_A): Higher value for proceeding, lower collision cost
      (overconfident about avoiding crashes)
    - Cautious (θ_C): Lower value for proceeding, very high collision cost
      (risk-averse)

    Returns
    -------
    dict
        Payoff matrices keyed by type pair:
        {('aggressive', 'aggressive'): (A_payoff, B_payoff), ...}
    """
    types = {
        ("aggressive", "aggressive"): (
            np.array([[-80, 15], [1, -2]], dtype=float),
            np.array([[-80, 1], [15, -2]], dtype=float),
        ),
        ("aggressive", "cautious"): (
            np.array([[-80, 15], [1, -2]], dtype=float),
            np.array([[-120, 2], [8, 0]], dtype=float),
        ),
        ("cautious", "aggressive"): (
            np.array([[-120, 8], [2, 0]], dtype=float),
            np.array([[-80, 1], [15, -2]], dtype=float),
        ),
        ("cautious", "cautious"): (
            np.array([[-120, 8], [2, 0]], dtype=float),
            np.array([[-120, 2], [8, 0]], dtype=float),
        ),
    }
    return types


def prisoners_dilemma():
    """
    Classic Prisoner's Dilemma adapted for intersections.
    Actions: {Go (Defect), Yield (Cooperate)}
    Defecting is strictly dominant, leading to suboptimal mutual Go.
    """
    payoff_a = np.array([
        [-5, 5],     # Go: vs Go=-5, vs Yield=5
        [-10, 0],    # Yield: vs Go=-10, vs Yield=0
    ], dtype=float)

    payoff_b = np.array([
        [-5, -10],
        [5, 0],
    ], dtype=float)

    return payoff_a, payoff_b


# Aliases
asymmetric_chicken_game = asymmetric_intersection
bayesian_intersection_game = bayesian_type_payoffs


def get_all_scenarios():
    """
    Return a dictionary of all predefined intersection game scenarios.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Scenario name → (payoff_A, payoff_B) mapping.
    """
    return {
        "Chicken Game (Symmetric)": chicken_game(),
        "Asymmetric Intersection": asymmetric_intersection(),
        "Coordination Game": coordination_game(),
        "Three-Strategy Intersection": three_strategy_intersection(),
        "Emergency Vehicle": emergency_vehicle_game(),
        "Prisoner's Dilemma": prisoners_dilemma(),
    }

