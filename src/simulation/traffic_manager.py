"""
Traffic Manager and Game-Theoretic Coordinator for Intersection Simulation.

Orchestrates vehicle decisions using:
1. Nash Simultaneous Game (NormalFormGame + NashSolver)
2. Stackelberg Leader-Follower (StackelbergSolver)
3. Bayesian Incomplete Information (BayesianGame + BayesianSolver)
4. Vickrey Auction (VickreyAuction)
5. Priority-Weighted Voting (PriorityVotingMechanism)
6. Correlated Equilibrium / Traffic Signal (CorrelatedEquilibriumSolver)
"""

import random
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.simulation.vehicle import Vehicle
from src.simulation.intersection import Intersection
from src.games.normal_form import NormalFormGame
from src.solvers.nash_solver import NashSolver
from src.solvers.stackelberg_solver import StackelbergSolver
from src.solvers.bayesian_solver import BayesianSolver
from src.solvers.correlated_solver import CorrelatedEquilibriumSolver
from src.mechanisms.auction import VickreyAuction, AuctionBid
from src.mechanisms.voting import PriorityVotingMechanism, VehicleVoter
from src.utils.payoff_matrices import (
    chicken_game,
    asymmetric_chicken_game,
    bayesian_intersection_game,
)
from src.utils.config import INTERSECTION, PAYOFFS


class TrafficManager:
    """
    Game-Theoretic Intersection Traffic Manager.

    Parameters
    ----------
    mode : str
        Decision mechanism:
        - 'nash': Simultaneous move (Chicken game)
        - 'stackelberg': Leader-follower priority
        - 'bayesian': AV vs unknown driver types
        - 'auction': Vickrey sealed-bid urgency auction
        - 'voting': Priority-weighted social choice
        - 'correlated': Correlated equilibrium traffic light
    """

    def __init__(self, mode: str = "nash"):
        self.mode = mode.lower()
        self.nash_solver = NashSolver()
        self.stackelberg_solver = StackelbergSolver()
        self.bayesian_solver = BayesianSolver()
        self.correlated_solver = CorrelatedEquilibriumSolver()
        self.vickrey_auction = VickreyAuction()
        self.voting = PriorityVotingMechanism()

        # Cache negotiation states
        self.negotiation_done = False
        self.active_pair: Optional[Tuple[str, str]] = None
        self.active_winner: Optional[str] = None
        self.current_signal_phase = "NS"  # 'NS' or 'EW'
        self.signal_timer = 0.0
        self.signal_period = 80.0  # steps per phase

    def set_mode(self, mode: str):
        """Change negotiation controller mode."""
        self.mode = mode.lower()
        self.negotiation_done = False
        self.active_pair = None
        self.active_winner = None

    def _enforce_car_following(self, active_vehicles: List[Vehicle]):
        """
        Enforce safety distance between vehicles traveling in the same lane.
        Trailing vehicles must yield/brake if approaching a lead vehicle.
        """
        by_direction: Dict[str, List[Vehicle]] = {
            "north": [], "south": [], "east": [], "west": []
        }
        for v in active_vehicles:
            if v.direction in by_direction:
                by_direction[v.direction].append(v)

        # Sort each lane so the vehicle closest to / farthest along the path is first
        for direction, vehicles in by_direction.items():
            if len(vehicles) < 2:
                continue

            if direction == "north":
                # Moving southward: y increases. Larger y is ahead.
                vehicles.sort(key=lambda v: v.y, reverse=True)
                for i in range(1, len(vehicles)):
                    lead = vehicles[i - 1]
                    trail = vehicles[i]
                    gap = lead.y - trail.y
                    if gap < 45.0:
                        trail.set_action("yield")
            elif direction == "south":
                # Moving northward: y decreases. Smaller y is ahead.
                vehicles.sort(key=lambda v: v.y)
                for i in range(1, len(vehicles)):
                    lead = vehicles[i - 1]
                    trail = vehicles[i]
                    gap = trail.y - lead.y
                    if gap < 45.0:
                        trail.set_action("yield")
            elif direction == "east":
                # Moving westward: x decreases. Smaller x is ahead.
                vehicles.sort(key=lambda v: v.x)
                for i in range(1, len(vehicles)):
                    lead = vehicles[i - 1]
                    trail = vehicles[i]
                    gap = trail.x - lead.x
                    if gap < 45.0:
                        trail.set_action("yield")
            elif direction == "west":
                # Moving eastward: x increases. Larger x is ahead.
                vehicles.sort(key=lambda v: v.x, reverse=True)
                for i in range(1, len(vehicles)):
                    lead = vehicles[i - 1]
                    trail = vehicles[i]
                    gap = lead.x - trail.x
                    if gap < 45.0:
                        trail.set_action("yield")

    def coordinate(self, intersection: Intersection, dt: float = 1.0):
        """
        Evaluate and dispatch actions to vehicles in the intersection environment.
        """
        active_vehicles = [
            v for v in intersection.vehicles
            if v.state not in ("passed", "collided")
        ]

        if not active_vehicles:
            return

        # Check vehicles nearing the decision zone
        decision_zone_radius = INTERSECTION["intersection_size"] / 2 + 100
        approaching = [
            v for v in active_vehicles
            if v.distance_to_center <= decision_zone_radius and not v.is_in_intersection
        ]

        if self.mode == "nash":
            self._handle_nash(approaching)
        elif self.mode == "stackelberg":
            self._handle_stackelberg(approaching)
        elif self.mode == "bayesian":
            self._handle_bayesian(approaching)
        elif self.mode == "auction":
            self._handle_auction(approaching)
        elif self.mode == "voting":
            self._handle_voting(approaching)
        elif self.mode == "correlated":
            self._handle_correlated_signal(active_vehicles, dt)
        else:
            # Default fallback: safe stop
            for v in approaching:
                v.set_action("slow")

        # Crucial safety layer: prevent rear-end collisions in the same lane
        self._enforce_car_following(active_vehicles)

    def _handle_nash(self, approaching: List[Vehicle]):
        """
        Solve normal-form Chicken game between two approaching vehicles.
        Negotiates pure or mixed strategy equilibrium and maintains action
        until crossing resolution rather than re-rolling mixed probabilities every frame.
        """
        if len(approaching) < 2:
            self.active_pair = None
            for v in approaching:
                v.set_action("go")
            return

        v1, v2 = approaching[0], approaching[1]
        pair = (v1.vehicle_id, v2.vehicle_id)

        # If already negotiated for this current pair and both are still active, keep actions
        if self.active_pair == pair and v1.action in ("go", "yield") and v2.action in ("go", "yield"):
            if v1.action == "yield" and (v2.has_passed or v2.state == "crossing"):
                # If leader is crossing or has passed, yielding car can proceed safely
                if v2.has_passed or v2.distance_to_center > 40:
                    v1.set_action("go")
            elif v2.action == "yield" and (v1.has_passed or v1.state == "crossing"):
                if v1.has_passed or v1.distance_to_center > 40:
                    v2.set_action("go")
            return

        self.active_pair = pair
        game = chicken_game()
        equilibria = self.nash_solver.find_all_equilibria(game)

        if equilibria.get("pure"):
            # Select pure Nash equilibrium (Go, Yield) or (Yield, Go)
            pure_ne = random.choice(equilibria["pure"])
            labels = ["go", "yield"]
            act_a = labels[pure_ne[0]]
            act_b = labels[pure_ne[1]]
            v1.set_action(act_a)
            v2.set_action(act_b)
        elif equilibria.get("mixed"):
            sigma_a, sigma_b = equilibria["mixed"][0]
            act_a = "go" if random.random() < sigma_a[0] else "yield"
            act_b = "go" if random.random() < sigma_b[0] else "yield"
            v1.set_action(act_a)
            v2.set_action(act_b)
        else:
            v1.set_action("slow")
            v2.set_action("slow")

    def _handle_stackelberg(self, approaching: List[Vehicle]):
        """
        Solve Stackelberg leader-follower game.
        Vehicle closer to intersection center (or earlier arrival) is Leader.
        """
        if not approaching:
            return
        if len(approaching) == 1:
            approaching[0].set_action("go")
            return

        # Sort by distance: closest is Leader
        sorted_v = sorted(approaching, key=lambda v: v.distance_to_center)
        leader, follower = sorted_v[0], sorted_v[1]

        # Leader commits to Go; Follower rationally yields
        leader.set_action("go")
        follower.set_action("yield")

        for other in sorted_v[2:]:
            other.set_action("yield")

    def _handle_bayesian(self, approaching: List[Vehicle]):
        """
        Bayesian game: AV faces unknown human driver types (aggressive vs cautious).
        Computes Bayesian Nash Equilibrium and sets conservative safe response.
        """
        if len(approaching) < 2:
            for v in approaching:
                v.set_action("go")
            return

        av = next((v for v in approaching if v.is_av), approaching[0])
        other = next((v for v in approaching if v != av), approaching[1])

        # Prior: p(aggressive)=0.35, p(cautious)=0.65
        p_agg = 0.35
        # Aggressive drivers Go with high probability (0.85)
        # Cautious drivers Yield with high probability (0.80)
        prob_other_goes = p_agg * 0.85 + (1 - p_agg) * 0.20

        # AV calculates expected utility of Go vs Yield
        # u(Go) = prob_other_goes * (-100) + (1 - prob_other_goes) * (10)
        # u(Yield) = prob_other_goes * (2) + (1 - prob_other_goes) * (0)
        u_go = prob_other_goes * PAYOFFS["collision"] + (1 - prob_other_goes) * PAYOFFS["proceed"]
        u_yield = prob_other_goes * PAYOFFS["yield_to_other"] + (1 - prob_other_goes) * PAYOFFS["deadlock"]

        if u_go > u_yield:
            av.set_action("go")
            other.set_action("yield" if other.driver_type == "cautious" else "go")
        else:
            av.set_action("yield")
            other.set_action("go")

    def _handle_auction(self, approaching: List[Vehicle]):
        """
        Vickrey second-price sealed-bid auction for right-of-way.
        Highest bidder wins and proceeds; others yield in priority order.
        Maintains winner priority until they clear the intersection.
        """
        if not approaching:
            return

        # If previous winner is still active and approaching/crossing, retain priority
        if self.active_winner:
            matching = [v for v in approaching if v.vehicle_id == self.active_winner]
            if matching and not matching[0].has_passed:
                for v in approaching:
                    if v.vehicle_id == self.active_winner:
                        v.set_action("go")
                    else:
                        v.set_action("yield")
                return
            else:
                self.active_winner = None

        if len(approaching) == 1:
            self.active_winner = approaching[0].vehicle_id
            approaching[0].set_action("go")
            return

        bids = [
            AuctionBid(
                vehicle_id=v.vehicle_id,
                true_value=v.urgency,
                bid=v.urgency,  # dominant strategy is truth-telling
                direction=v.direction,
                is_emergency=v.is_emergency,
            )
            for v in approaching
        ]

        result = self.vickrey_auction.run(bids)
        winner_id = result.winner
        self.active_winner = winner_id

        for v in approaching:
            if v.vehicle_id == winner_id:
                v.set_action("go")
            else:
                v.set_action("yield")

    def _handle_voting(self, approaching: List[Vehicle]):
        """
        Priority-weighted Borda Count voting mechanism.
        Vehicles vote on traversal priority sequence.
        Maintains winning voter's right-of-way until they clear the intersection.
        """
        if not approaching:
            return

        # If previous winner is still active and approaching/crossing, retain priority
        if self.active_winner:
            matching = [v for v in approaching if v.vehicle_id == self.active_winner]
            if matching and not matching[0].has_passed:
                for v in approaching:
                    if v.vehicle_id == self.active_winner:
                        v.set_action("go")
                    else:
                        v.set_action("yield")
                return
            else:
                self.active_winner = None

        if len(approaching) == 1:
            self.active_winner = approaching[0].vehicle_id
            approaching[0].set_action("go")
            return

        candidates = [v.vehicle_id for v in approaching]
        voters = []

        for v in approaching:
            pref = [v.vehicle_id] + [c for c in candidates if c != v.vehicle_id]
            voter = VehicleVoter(
                vehicle_id=v.vehicle_id,
                direction=v.direction,
                preferences=pref,
                is_emergency=v.is_emergency,
                waiting_time=v.waiting_time,
            )
            voter.compute_priority_weight()
            voters.append(voter)

        res = self.voting.borda_count(voters, candidates, weighted=True)
        winner_id = res.winner
        self.active_winner = winner_id

        for v in approaching:
            if v.vehicle_id == winner_id:
                v.set_action("go")
            else:
                v.set_action("yield")

    def _handle_correlated_signal(self, active_vehicles: List[Vehicle], dt: float):
        """
        Correlated Equilibrium implementation via timed phase coordination.
        Ensures 100% collision-free crossing with clearance interval (yellow/all-red)
        preventing newly green traffic from entering before perpendicular traffic clears.
        """
        self.signal_timer += dt
        if self.signal_timer >= self.signal_period:
            self.signal_timer = 0.0
            self.current_signal_phase = "EW" if self.current_signal_phase == "NS" else "NS"

        # Check if any vehicle from conflicting perpendicular direction is still inside the intersection box
        ns_in_box = any(
            v.is_in_intersection for v in active_vehicles
            if v.direction in ("north", "south") and not v.has_passed
        )
        ew_in_box = any(
            v.is_in_intersection for v in active_vehicles
            if v.direction in ("east", "west") and not v.has_passed
        )

        for v in active_vehicles:
            if v.is_in_intersection or v.has_passed:
                # Once already inside intersection, proceed to clear the junction
                v.set_action("go")
                continue

            if self.current_signal_phase == "NS":
                # Green for NS, but hold if EW vehicle is still clearing the box
                if v.direction in ("north", "south") and not ew_in_box:
                    v.set_action("go")
                else:
                    v.set_action("yield")
            else:  # EW phase
                # Green for EW, but hold if NS vehicle is still clearing the box
                if v.direction in ("east", "west") and not ns_in_box:
                    v.set_action("go")
                else:
                    v.set_action("yield")
