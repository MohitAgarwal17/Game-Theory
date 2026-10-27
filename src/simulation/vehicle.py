"""
Vehicle agent class for intersection simulation.

Models both Autonomous Vehicles (AVs) and Human-Driven Vehicles (HDVs)
with strategic decision-making capabilities (Nash, Stackelberg, Bayesian, Auction).
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np

from src.utils.config import VEHICLE, INTERSECTION, COLORS, DRIVER_TYPES


class Vehicle:
    """
    Vehicle agent that interacts at the intersection.

    Parameters
    ----------
    vehicle_id : str
        Unique identifier.
    direction : str
        Approach direction: 'north', 'south', 'east', 'west'.
    is_av : bool
        True if vehicle is autonomous; False if human-driven.
    driver_type : str
        'cautious', 'aggressive', or 'rational_av'.
    urgency : float
        Private urgency value for auction mechanisms (1.0 to 10.0).
    is_emergency : bool
        True if emergency vehicle.
    """

    def __init__(
        self,
        vehicle_id: str,
        direction: str,
        is_av: bool = True,
        driver_type: str = "rational_av",
        urgency: float = 5.0,
        is_emergency: bool = False,
        center_x: float = 375.0,
        center_y: float = 375.0,
    ):
        self.vehicle_id = vehicle_id
        self.direction = direction.lower()
        self.is_av = is_av
        self.driver_type = driver_type
        self.urgency = urgency
        self.is_emergency = is_emergency

        self.center_x = center_x
        self.center_y = center_y

        self.length = VEHICLE["length"]
        self.width = VEHICLE["width"]
        self.max_speed = VEHICLE["max_speed"]
        self.accel = VEHICLE["acceleration"]
        self.decel = VEHICLE["deceleration"]

        # Physics state
        self.speed = 0.0
        self.target_speed = self.max_speed
        self.action = "go"  # 'go', 'yield', 'slow'
        self.state = "approaching"  # 'approaching', 'negotiating', 'yielding', 'crossing', 'passed', 'collided'

        # Metrics tracking
        self.waiting_time = 0.0
        self.total_travel_time = 0.0
        self.payoff = 0.0
        self.near_miss_count = 0

        # Initialize spatial position based on approach direction
        self._init_position()

    def _init_position(self):
        """Set starting coordinates based on approach lane."""
        offset = INTERSECTION["approach_length"]
        lane_offset = INTERSECTION["lane_width"] / 2

        if self.direction == "north":
            # Approaching from North heading South
            self.x = self.center_x - lane_offset
            self.y = self.center_y - offset
            self.vx = 0.0
            self.vy = self.max_speed * 0.8
        elif self.direction == "south":
            # Approaching from South heading North
            self.x = self.center_x + lane_offset
            self.y = self.center_y + offset
            self.vx = 0.0
            self.vy = -self.max_speed * 0.8
        elif self.direction == "east":
            # Approaching from East heading West
            self.x = self.center_x + offset
            self.y = self.center_y - lane_offset
            self.vx = -self.max_speed * 0.8
            self.vy = 0.0
        elif self.direction == "west":
            # Approaching from West heading East
            self.x = self.center_x - offset
            self.y = self.center_y + lane_offset
            self.vx = self.max_speed * 0.8
            self.vy = 0.0
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        self.speed = math.hypot(self.vx, self.vy)

    @property
    def distance_to_center(self) -> float:
        """Distance from vehicle center to intersection center point."""
        return math.hypot(self.x - self.center_x, self.y - self.center_y)

    @property
    def distance_to_stop_line(self) -> float:
        """Distance to the entrance stop line."""
        stop_dist = INTERSECTION["intersection_size"] / 2
        d = self.distance_to_center - stop_dist
        return max(0.0, d)

    @property
    def is_in_intersection(self) -> bool:
        """True if the vehicle bounding box is inside the intersection box."""
        half_box = INTERSECTION["intersection_size"] / 2
        return (
            abs(self.x - self.center_x) <= half_box
            and abs(self.y - self.center_y) <= half_box
        )

    @property
    def has_passed(self) -> bool:
        """True if vehicle has completely exited the intersection."""
        half_box = INTERSECTION["intersection_size"] / 2 + 30
        if self.direction == "north":
            return self.y > self.center_y + half_box
        elif self.direction == "south":
            return self.y < self.center_y - half_box
        elif self.direction == "east":
            return self.x < self.center_x - half_box
        elif self.direction == "west":
            return self.x > self.center_x + half_box
        return False

    def set_action(self, action: str):
        """Set strategic decision ('go', 'yield', 'slow')."""
        self.action = action.lower()
        if self.action == "go":
            self.target_speed = self.max_speed
            if self.state == "yielding":
                self.state = "crossing"
        elif self.action == "yield":
            self.target_speed = 0.0
            if not self.is_in_intersection:
                self.state = "yielding"
        elif self.action == "slow":
            self.target_speed = self.max_speed * 0.4
            self.state = "negotiating"

    def update(self, dt: float = 1.0):
        """Update vehicle physics, position, and metrics for one frame/step."""
        self.total_travel_time += dt

        if self.state == "collided":
            self.speed = 0.0
            return

        if self.has_passed:
            self.state = "passed"
            self.target_speed = self.max_speed

        # Speed adjustment toward target_speed
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + self.accel * dt)
        elif self.speed > self.target_speed:
            self.speed = max(self.target_speed, self.speed - self.decel * dt)

        # Track waiting time if nearly stopped at intersection
        if self.speed < 0.2 and not self.has_passed:
            self.waiting_time += dt

        # Update position along heading
        if self.direction == "north":
            self.y += self.speed * dt
        elif self.direction == "south":
            self.y -= self.speed * dt
        elif self.direction == "east":
            self.x -= self.speed * dt
        elif self.direction == "west":
            self.x += self.speed * dt

        # State transition into intersection
        if self.is_in_intersection and self.state not in ("collided", "passed"):
            self.state = "crossing"

    def get_color(self) -> Tuple[int, int, int]:
        """Color representation for renderer."""
        if self.state == "collided":
            return COLORS["vehicle_collision"]
        if self.is_emergency:
            return (255, 30, 80)
        if self.state == "yielding":
            return COLORS["vehicle_yielding"]
        if self.is_av:
            return COLORS["vehicle_av"]
        return COLORS["vehicle_human"]
