# 🚗 Game-Theoretic Modeling of Autonomous Vehicles at Intersections

A comprehensive simulation and analysis platform that applies game theory to autonomous vehicle decision-making at intersections.

## Team Members

| # | Name | Registration Number |
|---|------|---------------------|
| 1 | Mohit Agarwal | 2427030564 |
| 2 | Ineesh Khindria | 242703063 |
| 3 | Akshit Goel | 2427030638 |

## Features

- **Nash Equilibrium Solver** — Pure and mixed strategy NE for intersection games
- **Stackelberg Equilibrium** — Leader-follower sequential decision-making
- **Bayesian Games** — AV vs. unknown human driver types
- **Repeated Games** — Strategy evolution with Tit-for-Tat, Grim Trigger, etc.
- **Vickrey Auction** — Incentive-compatible right-of-way allocation
- **Correlated Equilibrium** — Traffic light as a correlating device
- **Pygame Simulation** — Interactive 2D intersection visualization
- **Web Dashboard** — Analysis results and interactive payoff editor

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo/run_demo.py

# Run the simulation
python -m src.simulation.renderer

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
src/
├── games/          # Game representations (normal-form, extensive, Bayesian, repeated)
├── solvers/        # Equilibrium solvers (Nash, Stackelberg, Bayesian NE, correlated)
├── mechanisms/     # Mechanism design (Vickrey auction, priority voting)
├── simulation/     # Pygame-based intersection simulator
├── analysis/       # Metrics, comparison, and plotting
└── utils/          # Payoff matrices and configuration
```

## License

Academic project — all rights reserved.
