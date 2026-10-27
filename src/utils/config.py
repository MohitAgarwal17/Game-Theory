"""
Configuration constants for the intersection game simulation.

Contains all tunable parameters for games, simulation physics,
rendering, and analysis.
"""

# ============================================================================
# GAME PARAMETERS
# ============================================================================

# Default payoff values for intersection games
PAYOFFS = {
    "collision": -100,       # Both go simultaneously → crash
    "proceed": 10,           # One proceeds while other yields → good
    "yield_to_other": 2,     # Yield while other proceeds → slight delay but safe
    "deadlock": 0,           # Both yield → wasted time
    "near_miss": -20,        # Close call but no collision
}

# Bayesian game type probabilities
DRIVER_TYPES = {
    "aggressive": {
        "probability": 0.3,
        "go_bias": 0.8,        # Tendency to choose Go
        "description": "Prioritizes speed, will Go unless collision is certain",
    },
    "cautious": {
        "probability": 0.7,
        "go_bias": 0.2,        # Tendency to choose Go
        "description": "Prioritizes safety, will Yield if any doubt",
    },
}

# Repeated game parameters
REPEATED_GAME = {
    "num_rounds": 100,
    "discount_factor": 0.95,   # δ: importance of future payoffs
    "memory_length": 5,        # How many past rounds to remember
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Intersection geometry (pixels)
INTERSECTION = {
    "road_width": 80,          # Width of each road (pixels)
    "lane_width": 40,          # Width of each lane
    "intersection_size": 160,  # Size of the intersection box
    "approach_length": 300,    # Length of approach road visible
}

# Vehicle physics
VEHICLE = {
    "length": 30,              # Vehicle length (pixels)
    "width": 18,               # Vehicle width (pixels)
    "max_speed": 5.0,          # Maximum speed (pixels/frame)
    "acceleration": 0.15,      # Acceleration rate
    "deceleration": 0.3,       # Braking rate
    "stop_distance": 60,       # Distance from intersection center to stop
    "collision_radius": 20,    # Collision detection radius
}

# ============================================================================
# RENDERING PARAMETERS
# ============================================================================

WINDOW = {
    "width": 1000,
    "height": 750,
    "fps": 60,
    "title": "AV Intersection Game Theory Simulator",
}

# Color palette (dark theme)
COLORS = {
    "background": (18, 18, 24),
    "road": (45, 45, 55),
    "road_line": (200, 200, 80),
    "road_center_line": (255, 255, 100),
    "intersection_zone": (55, 55, 70),
    "grass": (30, 50, 30),
    "vehicle_av": (0, 180, 255),       # Cyan for autonomous vehicles
    "vehicle_human": (255, 120, 50),   # Orange for human-driven
    "vehicle_yielding": (100, 200, 100),
    "vehicle_collision": (255, 50, 50),
    "text_primary": (240, 240, 240),
    "text_secondary": (160, 160, 180),
    "text_accent": (0, 200, 255),
    "panel_bg": (25, 25, 35),
    "panel_border": (60, 60, 80),
    "button_normal": (50, 50, 70),
    "button_hover": (70, 70, 100),
    "button_active": (0, 150, 255),
    "success": (50, 200, 100),
    "warning": (255, 200, 50),
    "danger": (255, 60, 60),
    "info": (80, 160, 255),
}

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

ANALYSIS = {
    "num_simulations": 1000,   # Monte Carlo runs per scenario
    "confidence_level": 0.95,  # For confidence intervals
    "figure_dpi": 150,         # Plot resolution
    "figure_size": (10, 6),    # Default figure size (inches)
}

# ============================================================================
# STRATEGY LABELS
# ============================================================================

STRATEGIES = {
    "go": "Go (Proceed)",
    "yield": "Yield (Wait)",
    "slow": "Slow (Cautious Approach)",
}

DIRECTIONS = {
    "north": "North (↑)",
    "south": "South (↓)",
    "east": "East (→)",
    "west": "West (←)",
}
