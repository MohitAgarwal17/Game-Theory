"""
Pygame Renderer for Intersection Simulation.

Provides rich 2D visualization of vehicles, road geometries, signal phases,
and real-time game-theoretic metrics overlay.
Supports both interactive display and headless image/frame export.
"""

import os
import math
from typing import Optional, Tuple
import pygame

from src.simulation.intersection import Intersection
from src.simulation.traffic_manager import TrafficManager
from src.simulation.vehicle import Vehicle
from src.utils.config import INTERSECTION, VEHICLE, COLORS, WINDOW


class IntersectionRenderer:
    """
    Visualizer for 2D intersection simulation.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    headless : bool
        If True, runs without opening an interactive desktop window.
    """

    def __init__(
        self,
        width: int = WINDOW["width"],
        height: int = WINDOW["height"],
        headless: bool = False,
    ):
        self.width = width
        self.height = height
        self.headless = headless

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.font.init()

        if self.headless:
            self.screen = pygame.Surface((self.width, self.height))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(WINDOW["title"])

        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Helvetica, Arial, sans-serif", 20, bold=True)
        self.font_medium = pygame.font.SysFont("Helvetica, Arial, sans-serif", 15, bold=True)
        self.font_small = pygame.font.SysFont("Helvetica, Arial, sans-serif", 13)

        self.center_x = 380
        self.center_y = self.height // 2

    def draw_roads(self, current_phase: str = "NS"):
        """Draw road geometry, lane markings, and stop lines."""
        # Background
        self.screen.fill(COLORS["background"])

        road_w = INTERSECTION["road_width"]
        half_w = road_w // 2

        # Draw Roads (cross)
        road_rect_h = pygame.Rect(0, self.center_y - half_w, 760, road_w)
        road_rect_v = pygame.Rect(self.center_x - half_w, 0, road_w, self.height)

        pygame.draw.rect(self.screen, COLORS["road"], road_rect_h)
        pygame.draw.rect(self.screen, COLORS["road"], road_rect_v)

        # Intersection Box
        box_size = INTERSECTION["intersection_size"]
        half_box = box_size // 2
        box_rect = pygame.Rect(
            self.center_x - half_box,
            self.center_y - half_box,
            box_size,
            box_size,
        )
        pygame.draw.rect(self.screen, COLORS["intersection_zone"], box_rect)
        pygame.draw.rect(self.screen, (70, 70, 95), box_rect, 2)

        # Lane Center Markings (dashed lines)
        dash_len = 14
        gap = 10

        # Horizontal road center line (East-West)
        for x in range(0, 760, dash_len + gap):
            if abs(x - self.center_x) > half_box:
                pygame.draw.line(
                    self.screen,
                    COLORS["road_line"],
                    (x, self.center_y),
                    (min(x + dash_len, 760), self.center_y),
                    2,
                )

        # Vertical road center line (North-South)
        for y in range(0, self.height, dash_len + gap):
            if abs(y - self.center_y) > half_box:
                pygame.draw.line(
                    self.screen,
                    COLORS["road_line"],
                    (self.center_x, y),
                    (self.center_x, min(y + dash_len, self.height)),
                    2,
                )

        # Stop Lines (White solid lines)
        stop_color = (240, 240, 240)
        # North stop line
        pygame.draw.line(
            self.screen,
            stop_color,
            (self.center_x - half_w, self.center_y - half_box),
            (self.center_x, self.center_y - half_box),
            4,
        )
        # South stop line
        pygame.draw.line(
            self.screen,
            stop_color,
            (self.center_x, self.center_y + half_box),
            (self.center_x + half_w, self.center_y + half_box),
            4,
        )
        # West stop line
        pygame.draw.line(
            self.screen,
            stop_color,
            (self.center_x - half_box, self.center_y),
            (self.center_x - half_box, self.center_y + half_w),
            4,
        )
        # East stop line
        pygame.draw.line(
            self.screen,
            stop_color,
            (self.center_x + half_box, self.center_y - half_w),
            (self.center_x + half_box, self.center_y),
            4,
        )

        # Signal indicators (small glowing circles near stop lines)
        ns_color = COLORS["success"] if current_phase == "NS" else COLORS["danger"]
        ew_color = COLORS["success"] if current_phase == "EW" else COLORS["danger"]
        pygame.draw.circle(self.screen, ns_color, (self.center_x - half_w - 10, self.center_y - half_box), 6)
        pygame.draw.circle(self.screen, ns_color, (self.center_x + half_w + 10, self.center_y + half_box), 6)
        pygame.draw.circle(self.screen, ew_color, (self.center_x - half_box, self.center_y + half_w + 10), 6)
        pygame.draw.circle(self.screen, ew_color, (self.center_x + half_box, self.center_y - half_w - 10), 6)

    def draw_vehicle(self, vehicle: Vehicle):
        """Draw vehicle body, heading, and state indicators."""
        color = vehicle.get_color()
        w, l = vehicle.width, vehicle.length

        # Horizontal vs vertical dimensions based on direction
        if vehicle.direction in ("north", "south"):
            rect = pygame.Rect(vehicle.x - w / 2, vehicle.y - l / 2, w, l)
        else:
            rect = pygame.Rect(vehicle.x - l / 2, vehicle.y - w / 2, l, w)

        # Body
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=4)

        # Status badge / ID
        badge = self.font_small.render(vehicle.vehicle_id, True, (255, 255, 255))
        self.screen.blit(badge, (rect.x - 4, rect.y - 14))

        # Action tag
        act_color = COLORS["success"] if vehicle.action == "go" else COLORS["warning"]
        act_text = self.font_small.render(vehicle.action.upper(), True, act_color)
        self.screen.blit(act_text, (rect.x, rect.bottom + 2))

    def draw_hud(self, traffic_mgr: TrafficManager, metrics: dict):
        """Draw dashboard control HUD and live telemetry on side panel."""
        panel_x = 770
        panel_w = self.width - panel_x - 10
        panel_rect = pygame.Rect(panel_x, 10, panel_w, self.height - 20)

        pygame.draw.rect(self.screen, COLORS["panel_bg"], panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["panel_border"], panel_rect, 2, border_radius=8)

        y = 25
        title = self.font_large.render("Game Theory AV HUD", True, COLORS["text_accent"])
        self.screen.blit(title, (panel_x + 15, y))
        y += 35

        # Current Mode
        mode_label = f"Mechanism: {traffic_mgr.mode.upper()}"
        mode_surf = self.font_medium.render(mode_label, True, (255, 255, 120))
        self.screen.blit(mode_surf, (panel_x + 15, y))
        y += 28

        # Divider
        pygame.draw.line(self.screen, COLORS["panel_border"], (panel_x + 10, y), (panel_x + panel_w - 10, y))
        y += 15

        # Key Metrics
        metrics_list = [
            ("Throughput", f"{metrics.get('completed_vehicles', 0)} / {metrics.get('total_vehicles', 0)}"),
            ("Collisions", f"{metrics.get('collision_count', 0)}", COLORS["danger"] if metrics.get('collision_count', 0) > 0 else COLORS["success"]),
            ("Near Misses", f"{metrics.get('near_miss_count', 0)}", COLORS["warning"] if metrics.get('near_miss_count', 0) > 0 else COLORS["text_primary"]),
            ("Avg Waiting", f"{metrics.get('avg_waiting_time', 0.0):.1f} s"),
            ("Jain's Fairness", f"{metrics.get('jains_fairness_index', 1.0):.3f}"),
            ("Social Welfare", f"{metrics.get('social_welfare', 0.0):.1f}"),
            ("Sim Duration", f"{metrics.get('sim_duration', 0.0):.1f} s"),
        ]

        for item in metrics_list:
            label, val = item[0], item[1]
            val_col = item[2] if len(item) > 2 else COLORS["text_primary"]
            lbl_surf = self.font_small.render(label + ":", True, COLORS["text_secondary"])
            val_surf = self.font_small.render(val, True, val_col)
            self.screen.blit(lbl_surf, (panel_x + 15, y))
            self.screen.blit(val_surf, (panel_x + 130, y))
            y += 22

        y += 10
        pygame.draw.line(self.screen, COLORS["panel_border"], (panel_x + 10, y), (panel_x + panel_w - 10, y))
        y += 15

        # Interactive controls guide
        guide_header = self.font_medium.render("Keyboard Hotkeys:", True, COLORS["text_accent"])
        self.screen.blit(guide_header, (panel_x + 15, y))
        y += 25

        hotkeys = [
            ("[N]", "Nash Chicken Game"),
            ("[S]", "Stackelberg Priority"),
            ("[B]", "Bayesian Mixed Traffic"),
            ("[A]", "Vickrey Auction"),
            ("[V]", "Priority Voting"),
            ("[C]", "Correlated Signal"),
            ("[R]", "Reset Scenario"),
            ("[Space]", "Pause / Resume"),
        ]

        for key, desc in hotkeys:
            k_surf = self.font_small.render(key, True, (0, 200, 255))
            d_surf = self.font_small.render(desc, True, COLORS["text_secondary"])
            self.screen.blit(k_surf, (panel_x + 15, y))
            self.screen.blit(d_surf, (panel_x + 75, y))
            y += 20

    def render(self, intersection: Intersection, traffic_mgr: TrafficManager):
        """Draw full simulation scene and refresh display."""
        self.draw_roads(current_phase=traffic_mgr.current_signal_phase)

        for v in intersection.vehicles:
            self.draw_vehicle(v)

        metrics = intersection.get_metrics()
        self.draw_hud(traffic_mgr, metrics)

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(WINDOW["fps"])

    def save_screenshot(self, filepath: str):
        """Save current frame to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pygame.image.save(self.screen, filepath)

    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()
