"""
Intersection environment and conflict manager.

Simulates a 4-way intersection, tracks vehicle collisions, near-misses,
waiting times, throughput, and fairness metrics under various game-theoretic controllers.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.simulation.vehicle import Vehicle
from src.utils.config import INTERSECTION, VEHICLE, PAYOFFS


class Intersection:
    """
    4-way urban intersection simulation environment.

    Parameters
    ----------
    center_x : float
        Pixel center X of the intersection.
    center_y : float
        Pixel center Y of the intersection.
    collision_radius : float
        Threshold distance in pixels for a crash.
    near_miss_radius : float
        Threshold distance for a near-miss warning.
    """

    def __init__(
        self,
        center_x: float = 375.0,
        center_y: float = 375.0,
        collision_radius: float = 24.0,
        near_miss_radius: float = 45.0,
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.collision_radius = collision_radius
        self.near_miss_radius = near_miss_radius

        self.vehicles: List[Vehicle] = []
        self.completed_vehicles: List[Vehicle] = []
        self.collisions: List[Tuple[str, str, float]] = []  # (v1_id, v2_id, time)
        self.near_misses: List[Tuple[str, str, float]] = []

        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.total_throughput: int = 0

    def add_vehicle(self, vehicle: Vehicle):
        """Add vehicle agent to the intersection."""
        vehicle.center_x = self.center_x
        vehicle.center_y = self.center_y
        self.vehicles.append(vehicle)

    def reset(self):
        """Reset environment state."""
        self.vehicles.clear()
        self.completed_vehicles.clear()
        self.collisions.clear()
        self.near_misses.clear()
        self.sim_time = 0.0
        self.step_count = 0
        self.total_throughput = 0

    def check_conflicts(self):
        """Check for collisions and near misses between all pairs in or near intersection."""
        n = len(self.vehicles)
        for i in range(n):
            v1 = self.vehicles[i]
            if v1.state in ("passed", "collided"):
                continue

            for j in range(i + 1, n):
                v2 = self.vehicles[j]
                if v2.state in ("passed", "collided"):
                    continue

                dist = math.hypot(v1.x - v2.x, v1.y - v2.y)

                if dist < self.collision_radius:
                    # Collision detected!
                    v1.state = "collided"
                    v2.state = "collided"
                    v1.payoff += PAYOFFS["collision"]
                    v2.payoff += PAYOFFS["collision"]
                    self.collisions.append((v1.vehicle_id, v2.vehicle_id, self.sim_time))
                elif dist < self.near_miss_radius:
                    if (
                        v1.distance_to_center < INTERSECTION["intersection_size"]
                        and v2.distance_to_center < INTERSECTION["intersection_size"]
                    ):
                        v1.near_miss_count += 1
                        v2.near_miss_count += 1
                        self.near_misses.append((v1.vehicle_id, v2.vehicle_id, self.sim_time))

    def step(self, dt: float = 1.0):
        """Advance simulation by dt seconds."""
        self.sim_time += dt
        self.step_count += 1

        # Update all active vehicles
        for v in self.vehicles:
            v.update(dt)

        # Check for conflicts / collisions
        self.check_conflicts()

        # Archive vehicles that passed safely
        for v in list(self.vehicles):
            if v.has_passed and v not in self.completed_vehicles:
                self.completed_vehicles.append(v)
                self.total_throughput += 1
                v.payoff += PAYOFFS["proceed"]

    def is_finished(self) -> bool:
        """True if all vehicles have either passed or collided."""
        if not self.vehicles:
            return True
        return all(v.state in ("passed", "collided") for v in self.vehicles)

    def get_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive safety, efficiency, and fairness metrics."""
        all_v = self.vehicles + [v for v in self.completed_vehicles if v not in self.vehicles]
        n_total = len(all_v)
        if n_total == 0:
            return {
                "collision_rate": 0.0,
                "near_miss_rate": 0.0,
                "avg_waiting_time": 0.0,
                "throughput": 0,
                "jains_fairness_index": 1.0,
                "social_welfare": 0.0,
            }

        n_collisions = len(self.collisions)
        collision_rate = n_collisions / max(1, n_total)
        near_miss_rate = len(self.near_misses) / max(1, n_total)

        wait_times = [v.waiting_time for v in all_v]
        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
        max_wait = float(np.max(wait_times)) if wait_times else 0.0
        min_wait = float(np.min(wait_times)) if wait_times else 0.0

        # Jain's Fairness Index: (sum x_i)^2 / (n * sum x_i^2)
        # Using delays + small epsilon to avoid div by zero
        delays = np.array(wait_times, dtype=float) + 0.1
        jains_index = float((np.sum(delays) ** 2) / (n_total * np.sum(delays ** 2)))

        social_welfare = float(sum(v.payoff for v in all_v))

        return {
            "total_vehicles": n_total,
            "completed_vehicles": len(self.completed_vehicles),
            "collision_count": n_collisions,
            "collision_rate": collision_rate,
            "near_miss_count": len(self.near_misses),
            "near_miss_rate": near_miss_rate,
            "avg_waiting_time": avg_wait,
            "max_waiting_time": max_wait,
            "min_waiting_time": min_wait,
            "jains_fairness_index": jains_index,
            "social_welfare": social_welfare,
            "sim_duration": self.sim_time,
        }
