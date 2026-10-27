"""
Plotting module for generating publication-quality analysis figures.

Generates visualizations for:
- Payoff matrix heatmaps
- Best-response diagrams
- Nash equilibrium visualization
- Repeated game evolution
- Auction comparison charts
- Strategy tournament results
- Bayesian sensitivity analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from src.games.normal_form import NormalFormGame


# Global style configuration
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def save_or_show(fig, filepath: str | None, dpi: int = 150):
    """Save figure to file or display."""
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_payoff_matrix(
    game: NormalFormGame,
    filepath: str | None = None,
    title: str = "Payoff Matrix",
):
    """
    Create an annotated heatmap of the payoff matrix.

    Shows both players' payoffs in each cell with color indicating
    Player A's payoff.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#00d4ff")

    for idx, (matrix, name) in enumerate([
        (game.payoff_a, game.player_names[0]),
        (game.payoff_b, game.player_names[1]),
    ]):
        ax = axes[idx]
        sns.heatmap(
            matrix, annot=True, fmt=".0f", cmap="RdYlGn",
            xticklabels=game.strategy_labels_b,
            yticklabels=game.strategy_labels_a,
            center=0, ax=ax, linewidths=2, linecolor="#1a1a2e",
            cbar_kws={"label": "Payoff"},
        )
        ax.set_title(f"{name}'s Payoffs", fontsize=13, color="#00d4ff")
        ax.set_xlabel(game.player_names[1])
        ax.set_ylabel(game.player_names[0])

    fig.tight_layout()
    save_or_show(fig, filepath)
    return fig


def plot_best_response(
    game: NormalFormGame,
    filepath: str | None = None,
):
    """
    Plot best-response correspondences for a 2×2 game.

    Shows how each player's optimal mixing probability changes
    as a function of the opponent's mixing probability.
    """
    if game.shape != (2, 2):
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Player A's expected payoff from Go as function of q (B's prob of Go)
    q_range = np.linspace(0, 1, 200)
    p_range = np.linspace(0, 1, 200)

    # Best response for A: Go if E[Go] > E[Yield]
    # E[Go] = q*u(0,0) + (1-q)*u(0,1)
    # E[Yield] = q*u(1,0) + (1-q)*u(1,1)
    br_a_q = []
    for q in q_range:
        e_go = q * game.payoff_a[0, 0] + (1 - q) * game.payoff_a[0, 1]
        e_yield = q * game.payoff_a[1, 0] + (1 - q) * game.payoff_a[1, 1]
        if e_go > e_yield + 1e-9:
            br_a_q.append(1.0)  # Go with prob 1
        elif e_yield > e_go + 1e-9:
            br_a_q.append(0.0)  # Yield with prob 1
        else:
            br_a_q.append(0.5)  # Indifferent

    # Best response for B: Go if E[Go] > E[Yield]
    br_b_p = []
    for p in p_range:
        e_go = p * game.payoff_b[0, 0] + (1 - p) * game.payoff_b[1, 0]
        e_yield = p * game.payoff_b[0, 1] + (1 - p) * game.payoff_b[1, 1]
        if e_go > e_yield + 1e-9:
            br_b_p.append(1.0)
        elif e_yield > e_go + 1e-9:
            br_b_p.append(0.0)
        else:
            br_b_p.append(0.5)

    ax.plot(q_range, br_a_q, color="#00d4ff", linewidth=2.5,
            label=f"{game.player_names[0]}'s BR (p vs q)")
    ax.plot(br_b_p, p_range, color="#ff6b6b", linewidth=2.5,
            label=f"{game.player_names[1]}'s BR (q vs p)")

    ax.set_xlabel(f"q (Prob. {game.player_names[1]} plays {game.strategy_labels_b[0]})",
                  fontsize=12)
    ax.set_ylabel(f"p (Prob. {game.player_names[0]} plays {game.strategy_labels_a[0]})",
                  fontsize=12)
    ax.set_title("Best Response Correspondences", fontsize=14,
                 fontweight="bold", color="#00d4ff")
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    save_or_show(fig, filepath)
    return fig


def plot_repeated_game_history(
    history: list,
    strategy_a: str,
    strategy_b: str,
    filepath: str | None = None,
):
    """
    Plot the evolution of actions and cumulative payoffs over rounds
    in a repeated game.
    """
    rounds = [r.round_num for r in history]
    actions_a = [r.action_a for r in history]
    actions_b = [r.action_b for r in history]
    payoffs_a = np.cumsum([r.payoff_a for r in history])
    payoffs_b = np.cumsum([r.payoff_b for r in history])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Repeated Game: {strategy_a} vs {strategy_b}",
                 fontsize=16, fontweight="bold", color="#00d4ff")

    # Actions over time
    ax1 = axes[0]
    ax1.step(rounds, actions_a, color="#00d4ff", linewidth=1.5,
             alpha=0.8, label="Player A", where="mid")
    ax1.step(rounds, actions_b, color="#ff6b6b", linewidth=1.5,
             alpha=0.8, label="Player B", where="mid")
    ax1.set_ylabel("Action (0=Go, 1=Yield)")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Go", "Yield"])
    ax1.legend(facecolor="#1a1a2e", edgecolor="#444")
    ax1.set_title("Actions Over Rounds", color="#e0e0e0")

    # Cumulative payoffs
    ax2 = axes[1]
    ax2.plot(rounds, payoffs_a, color="#00d4ff", linewidth=2,
             label="Player A")
    ax2.plot(rounds, payoffs_b, color="#ff6b6b", linewidth=2,
             label="Player B")
    ax2.fill_between(rounds, payoffs_a, alpha=0.15, color="#00d4ff")
    ax2.fill_between(rounds, payoffs_b, alpha=0.15, color="#ff6b6b")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Payoff")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444")
    ax2.set_title("Cumulative Payoffs", color="#e0e0e0")

    fig.tight_layout()
    save_or_show(fig, filepath)
    return fig


def plot_tournament_results(
    tournament: dict,
    filepath: str | None = None,
):
    """
    Plot tournament results as a bar chart of total scores.
    """
    ranking = tournament["ranking"]
    names = [r[0] for r in ranking]
    scores = [r[1] for r in ranking]

    # Color gradient from best to worst
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="#1a1a2e",
                   linewidth=1.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    ax.set_xlabel("Total Tournament Score")
    ax.set_title("Repeated Game Strategy Tournament",
                 fontsize=14, fontweight="bold", color="#00d4ff")

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", fontsize=10, color="#e0e0e0")

    ax.invert_yaxis()
    fig.tight_layout()
    save_or_show(fig, filepath)
    return fig


def plot_mechanism_comparison(
    comparison_data: dict[str, dict],
    filepath: str | None = None,
):
    """
    Create a grouped bar chart comparing different intersection mechanisms.

    Parameters
    ----------
    comparison_data : dict
        Mechanism name → {metric_name: value} mapping.
    """
    mechanisms = list(comparison_data.keys())
    metrics = list(next(iter(comparison_data.values())).keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(mechanisms)
    colors = ["#00d4ff", "#ff6b6b", "#50fa7b", "#ffb86c", "#bd93f9"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mech in enumerate(mechanisms):
        values = [comparison_data[mech].get(m, 0) for m in metrics]
        offset = (i - len(mechanisms) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=mech,
               color=colors[i % len(colors)], edgecolor="#1a1a2e",
               linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics],
                       rotation=20, ha="right")
    ax.set_title("Mechanism Comparison", fontsize=14,
                 fontweight="bold", color="#00d4ff")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444")

    fig.tight_layout()
    save_or_show(fig, filepath)
    return fig


def plot_bayesian_sensitivity(
    sensitivity_data: dict,
    filepath: str | None = None,
):
    """
    Plot Bayesian NE sensitivity analysis: how AV behavior changes
    with beliefs about aggressive driver probability.
    """
    prob = sensitivity_data["prob_aggressive"]
    yield_rates = sensitivity_data["av_yield_rates"]
    collision_rates = sensitivity_data["collision_rates"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#00d4ff"
    color2 = "#ff6b6b"

    ax1.plot(prob, yield_rates, color=color1, linewidth=2.5,
             marker="o", markersize=4, label="AV Yield Rate")
    ax1.set_xlabel("P(Aggressive Driver)", fontsize=12)
    ax1.set_ylabel("AV Yield Rate", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(prob, collision_rates, color=color2, linewidth=2.5,
             marker="s", markersize=4, label="Collision Rate")
    ax2.set_ylabel("Expected Collision Rate", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Bayesian NE Sensitivity Analysis",
                 fontsize=14, fontweight="bold", color="#00d4ff")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#1a1a2e", edgecolor="#444", loc="center right")

    fig.tight_layout()
    save_or_show(fig, filepath)
    return fig


def plot_correlated_equilibrium(
    distribution: np.ndarray,
    game: NormalFormGame,
    filepath: str | None = None,
):
    """
    Visualize the correlated equilibrium probability distribution
    as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        distribution, annot=True, fmt=".4f", cmap="YlOrRd",
        xticklabels=game.strategy_labels_b,
        yticklabels=game.strategy_labels_a,
        ax=ax, linewidths=2, linecolor="#1a1a2e",
        cbar_kws={"label": "Probability"},
    )
    ax.set_xlabel(game.player_names[1])
    ax.set_ylabel(game.player_names[0])
    ax.set_title("Correlated Equilibrium Distribution",
                 fontsize=14, fontweight="bold", color="#00d4ff")

    fig.tight_layout()
    save_or_show(fig, filepath)
    return fig
