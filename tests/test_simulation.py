"""
Unit tests for Intersection Simulation and Traffic Manager.
"""

import pytest
from src.simulation.vehicle import Vehicle
from src.simulation.intersection import Intersection
from src.simulation.traffic_manager import TrafficManager
from src.simulation.renderer import IntersectionRenderer


def test_vehicle_initialization_and_physics():
    """Test vehicle geometry, initialization, and physics steps."""
    v = Vehicle("V1", direction="north", is_av=True)
    initial_y = v.y
    v.set_action("go")
    v.update(dt=1.0)

    # Approaching from North heading South => y should increase
    assert v.y > initial_y
    assert v.speed > 0


def test_intersection_simulation_step():
    """Test stepping the intersection with vehicles."""
    env = Intersection()
    v1 = Vehicle("V1", direction="north", is_av=True)
    v2 = Vehicle("V2", direction="east", is_av=True)
    env.add_vehicle(v1)
    env.add_vehicle(v2)

    mgr = TrafficManager(mode="stackelberg")

    for _ in range(50):
        mgr.coordinate(env, dt=0.5)
        env.step(dt=0.5)

    metrics = env.get_metrics()
    assert metrics["total_vehicles"] == 2
    assert "collision_rate" in metrics
    assert "jains_fairness_index" in metrics


def test_traffic_manager_modes():
    """Verify all traffic manager modes run without errors."""
    modes = ["nash", "stackelberg", "bayesian", "auction", "voting", "correlated"]
    for mode in modes:
        env = Intersection()
        env.add_vehicle(Vehicle("A", "north", is_av=True, urgency=6.0))
        env.add_vehicle(Vehicle("B", "west", is_av=True, urgency=8.0))
        mgr = TrafficManager(mode=mode)
        mgr.coordinate(env, dt=1.0)
        env.step(dt=1.0)


def test_renderer_headless():
    """Test that renderer initializes in headless mode without crashing."""
    renderer = IntersectionRenderer(headless=True)
    env = Intersection()
    mgr = TrafficManager()
    renderer.render(env, mgr)
    renderer.close()
