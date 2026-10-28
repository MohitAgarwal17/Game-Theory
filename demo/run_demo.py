"""
End-to-End Demonstration and Experiment Runner.

Executes all game-theoretic models, computes equilibria, runs simulation
scenarios, and generates figures and summary tables into results/.
"""

import os
import sys
import json

# Ensure stdout/stderr handle UTF-8 cleanly on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure writable matplotlib cache directory
mpl_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplcache"))
os.makedirs(mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_cache)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.games.normal_form import NormalFormGame
from src.games.extensive_form import ExtensiveFormGame
from src.games.bayesian_game import BayesianGame
from src.games.repeated_game import RepeatedGame, StrategyType
from src.solvers.nash_solver import NashSolver
from src.solvers.stackelberg_solver import StackelbergSolver
from src.solvers.bayesian_solver import BayesianSolver
from src.solvers.correlated_solver import CorrelatedSolver
from src.mechanisms.auction import VickreyAuction, FirstPriceAuction, compare_auction_mechanisms
from src.mechanisms.voting import PriorityVotingMechanism, VehicleVoter
from src.analysis.plotter import (
    plot_payoff_matrix,
    plot_best_response,
    plot_repeated_game_history,
    plot_mechanism_comparison,
)
from src.analysis.comparator import MechanismComparator
from src.simulation.intersection import Intersection
from src.simulation.vehicle import Vehicle
from src.simulation.traffic_manager import TrafficManager
from src.simulation.renderer import IntersectionRenderer
from src.utils.payoff_matrices import (
    chicken_game,
    asymmetric_intersection,
    coordination_game,
    three_strategy_intersection,
)


def run_all_experiments():
    print("=" * 70)
    print("  GAME-THEORETIC MODELING OF AUTONOMOUS VEHICLES AT INTERSECTIONS")
    print("  Team: Mohit Agarwal (2427030564) | Ineesh Khindria (242703063) | Akshit Goel (2427030638)")
    print("=" * 70)

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. NORMAL-FORM GAMES & NASH EQUILIBRIUM
    # ------------------------------------------------------------------------
    print("\n[1/6] Solving Normal-Form Intersection Games (Nash Equilibrium)...")
    game_chicken = NormalFormGame(
        *chicken_game(),
        strategy_labels_a=["Go", "Yield"],
        strategy_labels_b=["Go", "Yield"],
        player_names=("Vehicle A", "Vehicle B"),
    )
    nash_solver = NashSolver()
    equilibria = nash_solver.solve(game_chicken)

    print(f"  * Found {len(equilibria)} Nash Equilibria for Chicken Game:")
    for i, eq in enumerate(equilibria, 1):
        ptype = "Pure" if eq.is_pure else "Mixed"
        print(f"    - EQ {i} ({ptype}): {eq.strategy_label(game_chicken)} | Payoffs: ({eq.payoff_a:.2f}, {eq.payoff_b:.2f}) | Pareto Optimal: {eq.is_pareto_optimal}")

    # Generate and save plots
    plot_payoff_matrix(game_chicken, filepath="results/figures/chicken_payoff_matrix.png", title="Chicken Game Payoff Heatmap")
    plot_best_response(game_chicken, filepath="results/figures/chicken_best_response.png")
    print("  [OK] Saved chicken_payoff_matrix.png & chicken_best_response.png")

    # ------------------------------------------------------------------------
    # 2. STACKELBERG LEADER-FOLLOWER GAME
    # ------------------------------------------------------------------------
    print("\n[2/6] Solving Extensive-Form Stackelberg Games...")
    stack_solver = StackelbergSolver()
    stack_eq = stack_solver.solve_from_normal_form(game_chicken)
    print(f"  * Stackelberg Equilibrium:")
    print(f"    - Leader Action: {'Go' if stack_eq.leader_action == 0 else 'Yield'}")
    print(f"    - Leader Payoff: {stack_eq.leader_payoff} | Follower Payoff: {stack_eq.follower_payoff}")
    print(f"    - Outcome: {stack_eq.outcome}")

    advantage = stack_solver.compare_leader_advantage(game_chicken)
    print(f"    - First-Mover Advantage: A as leader gets {advantage['a_leads']['leader_payoff']}, B as leader gets {advantage['b_leads']['leader_payoff']}")

    # ------------------------------------------------------------------------
    # 3. BAYESIAN GAMES (MIXED TRAFFIC: AV vs. HUMAN)
    # ------------------------------------------------------------------------
    print("\n[3/6] Solving Bayesian Games (AV vs. Unknown Driver Types)...")
    bayesian_game = BayesianGame.create_av_vs_human(prob_aggressive=0.35)
    bayesian_solver = BayesianSolver()
    b_result = bayesian_solver.solve(bayesian_game)
    print(f"  * Found {b_result.num_equilibria} Bayesian Nash Equilibrium:")
    print(f"    - AV Yield Probability: {b_result.av_yield_rate:.1%}")
    print(f"    - Expected Collision Rate: {b_result.expected_collision_rate:.2%}")

    # ------------------------------------------------------------------------
    # 4. MECHANISM DESIGN & AUCTION COMPARISON
    # ------------------------------------------------------------------------
    print("\n[4/6] Running Mechanism Design (Vickrey vs. First-Price Auction)...")
    auction_comp = compare_auction_mechanisms(n_vehicles=4, n_trials=500)
    print(f"  * Auction Results (500 trials, 4 vehicles):")
    print(f"    - Vickrey Second-Price: Efficiency={auction_comp['vickrey']['avg_efficiency']:.1%}, Revenue={auction_comp['vickrey']['avg_revenue']:.2f}")
    print(f"    - First-Price Sealed:   Efficiency={auction_comp['first_price']['avg_efficiency']:.1%}, Revenue={auction_comp['first_price']['avg_revenue']:.2f}")

    # Save to data CSV
    df_auction = pd.DataFrame(auction_comp).T
    df_auction.to_csv("results/data/auction_comparison.csv")
    print("  [OK] Saved results/data/auction_comparison.csv")

    # ------------------------------------------------------------------------
    # 5. REPEATED GAMES & COOPERATION EVOLUTION
    # ------------------------------------------------------------------------
    print("\n[5/6] Simulating Repeated Intersection Game (100 rounds)...")
    rep_game = RepeatedGame(
        stage_game=game_chicken,
        num_rounds=100,
        discount_factor=0.95,
        strategy_a=StrategyType.TIT_FOR_TAT,
        strategy_b=StrategyType.TIT_FOR_TAT,
    )
    rep_res = rep_game.play()
    print(f"  * Tit-for-Tat vs. Tit-for-Tat:")
    print(f"    - Cooperation Rate A: {rep_res['cooperation_rate_a']:.1%}")
    print(f"    - Cooperation Rate B: {rep_res['cooperation_rate_b']:.1%}")
    print(f"    - Total Discounted Payoff A: {rep_res['total_payoff_a']:.2f} | B: {rep_res['total_payoff_b']:.2f}")

    plot_repeated_game_history(
        rep_res["history"],
        "Tit-for-Tat",
        "Tit-for-Tat",
        filepath="results/figures/repeated_game_convergence.png",
    )
    print("  [OK] Saved results/figures/repeated_game_convergence.png")

    # ------------------------------------------------------------------------
    # 6. FULL MULTI-MECHANISM COMPARISON & SIMULATION CAPTURE
    # ------------------------------------------------------------------------
    print("\n[6/6] Running Full Mechanism Comparison & Simulation Frame Generation...")
    comparator = MechanismComparator()
    comp_results = comparator.compare_all(game_chicken)
    plot_mechanism_comparison(comp_results, filepath="results/figures/mechanism_comparison.png")
    print("  [OK] Saved results/figures/mechanism_comparison.png")

    # Save comparison data JSON
    with open("results/data/mechanism_comparison.json", "w") as f:
        clean_res = {
            k: {
                sub_k: (float(sub_v) if isinstance(sub_v, (np.floating, float, int, np.integer)) else str(sub_v))
                for sub_k, sub_v in v.items()
                if not isinstance(sub_v, (np.ndarray, list))
            }
            for k, v in comp_results.items()
        }
        json.dump(clean_res, f, indent=2)
    print("  [OK] Saved results/data/mechanism_comparison.json")

    # Headless simulation run & screenshot capture
    print("  * Generating simulation snapshot screenshot...")
    renderer = IntersectionRenderer(headless=True)
    env = Intersection()
    env.add_vehicle(Vehicle("AV_North", "north", is_av=True, urgency=7.5))
    env.add_vehicle(Vehicle("AV_South", "south", is_av=True, urgency=4.2))
    env.add_vehicle(Vehicle("HDV_East", "east", is_av=False, driver_type="aggressive", urgency=8.0))
    env.add_vehicle(Vehicle("EMG_West", "west", is_av=True, is_emergency=True, urgency=10.0))

    tm = TrafficManager(mode="auction")

    for step in range(35):
        tm.coordinate(env, dt=0.5)
        env.step(dt=0.5)

    renderer.render(env, tm)
    renderer.save_screenshot("results/figures/simulation_screenshot.png")
    renderer.close()
    print("  [OK] Saved results/figures/simulation_screenshot.png")

    print("\n" + "=" * 70)
    print("  ALL DEMO EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("  Results available in results/figures/ and results/data/")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
