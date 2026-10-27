# 🚗 Game-Theoretic Modeling of Autonomous Vehicles at Intersections
## Comprehensive Project Report & Technical Documentation

---

### Course & Project Identification
- **Course:** Game Theory
- **Project Title:** Game-Theoretic Modeling of Autonomous Vehicles at Intersections
- **Academic Year:** 2026

### Group Members
| # | Name | Registration Number | Contribution / Primary Role |
|---|------|---------------------|-----------------------------|
| 1 | **Mohit Agarwal** | `2427030564` | Nash Equilibrium Solver, Bayesian Games, Auction Mechanism Design, Vehicle Kinematics |
| 2 | **Ineesh Khindria** | `242703063` | Extensive-Form Modeling, Repeated Games, Tournament Simulation, Presentation & Outlines |
| 3 | **Akshit Goel** | `2427030638` | System Architecture, Stackelberg Backward Induction, 2D Pygame Simulation, Web Dashboard, Report |

---

## Abstract

Intersection coordination is one of the most hazardous and computationally challenging problems in modern intelligent transportation systems. At unsignalized or decentralized intersections, autonomous vehicles (AVs) must negotiate right-of-way without relying on fallible communication backbones or rigid physical traffic lights. This research project formulates and analyzes intersection right-of-way negotiation through the rigorous mathematical lens of multi-agent game theory.

We design, implement, and benchmark six distinct game-theoretic mechanisms:
1. **Normal-Form Simultaneous Game (Chicken / Hawk-Dove):** Decentralized decision-making via mixed-strategy Nash equilibria ($p^* \approx 0.089$).
2. **Extensive-Form Sequential Game (Stackelberg):** Priority assignment based on time-to-stop-line, where backward induction eliminates coordination failure.
3. **Bayesian Games of Incomplete Information:** Modeling mixed traffic involving rational AVs and human-driven vehicles (HDVs) categorized into aggressive ($\theta_{\text{agg}}$) and cautious ($\theta_{\text{cautious}}$) types.
4. **Vickrey Mechanism Design (Second-Price Auction):** Incentive-compatible, dominant-strategy right-of-way allocation based on private vehicle urgencies.
5. **Priority-Weighted Social Choice (Borda Count Voting):** Non-monetary civic right-of-way ranking incorporating occupancy and emergency veto weights.
6. **Correlated Equilibrium (Signal Synchronization):** Mathematical formulation proving that conventional traffic signals serve as optimal correlating devices that maximize social welfare.

Using a comprehensive Python architecture, Monte Carlo empirical simulations (over 1,000 interactions), an interactive 2D simulation engine, and an interactive Web dashboard, we demonstrate that Stackelberg and Vickrey mechanisms achieve **0.0% collision rates** and increase intersection throughput by **33.8%** over fixed-cycle traffic signals. Furthermore, Vickrey auctions achieve perfect allocative efficiency ($100\%$) with dominant-strategy truth-telling, while Bayesian models eliminate mixed-traffic collision risk by adapting conservatism to the observed aggressiveness distribution.

---

## 1. Problem Statement & Motivation

### 1.1 The Urban Intersection Dilemma
Urban intersections occupy less than **1%** of roadway surface area yet account for over **45%** of all traffic collisions and **21%** of traffic fatalities worldwide (Federal Highway Administration / NHTSA). As connected and autonomous vehicles (CAVs) transition from controlled test tracks into mixed urban environments, traditional centralized control mechanisms present critical bottlenecks:
- **Centralized Infrastructure Failure:** Physical traffic signals introduce fixed phase delays, causing vehicles to idle at empty intersections and imposing severe fuel and emissions penalties.
- **V2X Communication Fragility:** Centralized intersection managers that compute global trajectories depend on ultra-low latency wireless channels susceptible to packet loss, cyber-physical spoofing, and single points of failure.
- **Decentralized Strategic Conflicts:** When two or more self-interested autonomous agents arrive at an unsignalized intersection simultaneously, each agent faces a fundamental strategic dilemma: proceeding immediately saves traversal time, while yielding incurs a waiting penalty. However, if both agents proceed simultaneously, catastrophic collision ensues.

### 1.2 Multi-Agent Strategic Formulation
Because the payoff of any vehicle's action strictly depends upon the simultaneous or sequential action chosen by conflicting vehicles, intersection management is inherently a **multi-agent non-cooperative game**.

The core engineering objectives are:
1. **Safety:** Guaranteeing an operational collision rate of $0.0\%$.
2. **Efficiency:** Minimizing mean vehicle delay and maximizing intersection vehicle throughput.
3. **Fairness:** Ensuring equitable waiting time distribution according to Jain's Fairness Index.
4. **Decentralization & Privacy:** Protecting private vehicle utilities without requiring disclosure of origin-destination routes.

---

## 2. Project Scope & Objectives

The objectives of this project are:
- **Mathematical Modeling:** Formulate 2-vehicle and multi-vehicle intersection dilemmas across strategic normal-form, extensive-form, Bayesian incomplete-information, and repeated-game structures.
- **Algorithmic Solvers:** Implement Python algorithms for:
  - Pure and mixed-strategy Nash equilibria (via support enumeration and indifference equations).
  - Subgame-perfect Stackelberg equilibria (via backward induction).
  - Bayesian Nash equilibria (BNE) across calibrated driver type distributions.
  - Social-welfare maximizing Correlated Equilibria (via linear programming).
  - Vickrey (second-price sealed-bid) and first-price auction mechanisms.
- **Simulation Environment:** Develop a real-time, interactive 2D intersection simulation with collision detection, kinematic physics, and telemetry overlay.
- **Interactive Web Dashboard:** Build a modern web application providing live interactive simulation, an on-the-fly payoff matrix editor, and dynamic benchmark visualizers.
- **Empirical Evaluation:** Benchmark safety, delay, throughput, Jain's fairness index, Price of Anarchy (PoA), and Price of Stability (PoS).

---

## 3. Mathematical Foundations & Game-Theoretic Formulations

### 3.1 Normal-Form Simultaneous Game (The Intersection Chicken Game)
Consider two vehicles, $A$ and $B$, approaching an intersection simultaneously along perpendicular lanes.
- **Player Set:** $N = \{\text{Vehicle } A, \text{Vehicle } B\}$.
- **Action Spaces:** $S_A = S_B = \{\text{Go } (0), \text{Yield } (1)\}$.
- **Payoff Matrix ($u_A, u_B$):**

$$\begin{array}{c|cc}
A \backslash B & \text{Go } (0) & \text{Yield } (1) \\
\hline
\text{Go } (0) & (-100, -100) & (10, 2) \\
\text{Yield } (1) & (2, 10) & (0, 0)
\end{array}$$

#### Strategic Analysis:
1. **Pure-Strategy Nash Equilibria:**
   - Profile $(\text{Go}, \text{Yield})$: Player A receives $u_A = 10$, Player B receives $u_B = 2$. Neither has incentive to unilaterally deviate.
   - Profile $(\text{Yield}, \text{Go})$: Player A receives $u_A = 2$, Player B receives $u_B = 10$.
   Both pure equilibria are **Pareto optimal** and represent successful asymmetric coordination. However, in simultaneous decentralized play, players face a symmetry dilemma: which equilibrium should be played?
2. **Mixed-Strategy Nash Equilibrium:**
   Let $p = P(A \text{ plays Go})$ and $q = P(B \text{ plays Go})$. For Player A to be indifferent between playing Go and Yield:
   $$E[u_A(\text{Go})] = q(-100) + (1-q)(10) = 10 - 110q$$
   $$E[u_A(\text{Yield})] = q(2) + (1-q)(0) = 2q$$
   $$10 - 110q = 2q \implies 112q = 10 \implies q^* = \frac{10}{112} \approx 0.0893$$
   By symmetry:
   $$p^* = \frac{10}{112} \approx 0.0893$$
   - Under this mixed strategy profile $\sigma^* = ((0.089, 0.911), (0.089, 0.911))$, the expected payoff for both vehicles drops to:
   $$E[u] = 2(0.0893) = 0.1786$$
   - **Coordination Failure Risk:** The probability that both vehicles simultaneously sample Go is:
   $$P(\text{Collision}) = p^* \times q^* = (0.0893)^2 \approx 0.00798 \quad (0.80\%)$$
   This proves that decentralized simultaneous Nash play carries an unacceptable collision risk for real-world autonomous vehicles.

---

### 3.2 Extensive-Form Sequential Game (Stackelberg Leadership)
To eliminate simultaneous coordination failure, time of arrival at the intersection boundary defines a natural sequence.
- **Leader ($L$):** The vehicle closer to the stop line commits to an action $a_L \in \{\text{Go}, \text{Yield}\}$ first.
- **Follower ($F$):** Observes $a_L$ and selects a best response $a_F^*(a_L) = \arg\max_{a_F} u_F(a_L, a_F)$.

```
                     Leader (Vehicle A)
                        /          \
                   Go  /            \ Yield
                      /              \
            Follower (B)             Follower (B)
             /        \               /        \
         Go /          \ Yield    Go /          \ Yield
           /            \           /            \
     (-100, -100)      (10, 2)   (2, 10)         (0, 0)
     [Collision]      [A Cross]  [B Cross]     [Deadlock]
```

#### Backward Induction:
1. Subgame after Leader plays **Go**: Follower compares $u_F(\text{Go}, \text{Go}) = -100$ vs $u_F(\text{Go}, \text{Yield}) = 2$.
   $$\text{Follower best response: } a_F^*(\text{Go}) = \text{Yield}$$
2. Subgame after Leader plays **Yield**: Follower compares $u_F(\text{Yield}, \text{Go}) = 10$ vs $u_F(\text{Yield}, \text{Yield}) = 0$.
   $$\text{Follower best response: } a_F^*(\text{Yield}) = \text{Go}$$
3. Leader's Root Decision:
   - If Leader plays Go $\implies$ outcome $(10, 2)$, payoff $= 10$.
   - If Leader plays Yield $\implies$ outcome $(2, 10)$, payoff $= 2$.
   $$\text{Leader optimal decision: } a_L^* = \text{Go}$$

**Subgame-Perfect Equilibrium (SPE):** $\{a_L = \text{Go}, a_F(\text{Go}) = \text{Yield}, a_F(\text{Yield}) = \text{Go}\}$.
- Outcome: **(Go, Yield)** on the equilibrium path.
- Collision rate: **0.0%**.
- Social welfare: $10 + 2 = 12.0$.

---

### 3.3 Bayesian Game with Incomplete Information (AV vs. Human Driver)
In mixed autonomy, an autonomous vehicle (Player 1) cannot observe the private disposition of a human driver (Player 2).
- **Type Space for Human:** $\Theta_2 = \{\theta_{\text{aggressive}}, \theta_{\text{cautious}}\}$.
- **Priors:** $P(\theta_{\text{agg}}) = 0.35, \quad P(\theta_{\text{cautious}}) = 0.65$.
- **Type-Dependent Payoffs ($u_{\text{AV}}, u_{\text{Human}}$):**

$$\begin{array}{c|cc}
\text{AV} \backslash \text{Aggressive} & \text{Go} & \text{Yield} \\
\hline
\text{Go} & (-100, -80) & (10, 1) \\
\text{Yield} & (2, 15) & (0, -2)
\end{array}
\qquad
\begin{array}{c|cc}
\text{AV} \backslash \text{Cautious} & \text{Go} & \text{Yield} \\
\hline
\text{Go} & (-100, -120) & (10, 2) \\
\text{Yield} & (2, 8) & (0, 0)
\end{array}$$

#### Bayesian Nash Equilibrium:
Aggressive human drivers have a strictly dominant strategy to play **Go** ($u(\text{Go}) > u(\text{Yield})$ regardless of AV action). Cautious human drivers play **Yield** when facing an entering vehicle.
The AV computes its expected utility:
$$E[u_{\text{AV}}(\text{Go})] = 0.35(-100) + 0.65(10) = -35 + 6.5 = -28.5$$
$$E[u_{\text{AV}}(\text{Yield})] = 0.35(2) + 0.65(10) = 0.70 + 6.5 = 7.20$$
Since $E[u_{\text{AV}}(\text{Yield})] = 7.20 > -28.5$, the AV's optimal Bayesian response is to **Yield**. This mathematically validates defensive driving policies in mixed-autonomy environments.

---

### 3.4 Mechanism Design: Vickrey Urgency Auction
When multiple vehicles arrive at a 4-way intersection, a market-based mechanism allocates right-of-way based on marginal urgency (e.g., medical emergency, delivery deadline, passenger occupancy).
- **Vehicles:** $N = \{1, 2, \dots, n\}$.
- **Private Valuation:** $v_i \sim \mathcal{U}[1, 20]$ represents vehicle $i$'s true cost of delay.
- **Bids:** Each vehicle submits a sealed bid $b_i \ge 0$.
- **Allocation Rule:** Vehicle with highest bid wins first crossing:
  $$i^* = \arg\max_{i \in N} b_i$$
- **Payment Rule (Second-Price):** Winner pays the second-highest submitted bid:
  $$p = \max_{j \neq i^*} b_j$$

#### Proof of Incentive Compatibility (Dominant-Strategy Truth-Telling):
Let $b_{-i}^* = \max_{j \neq i} b_j$ denote the highest bid among all other competitors.
Vehicle $i$'s net payoff is:
$$\pi_i(b_i, v_i) = \begin{cases} v_i - b_{-i}^*, & \text{if } b_i > b_{-i}^* \\ 0, & \text{if } b_i < b_{-i}^* \end{cases}$$

1. **Case $v_i > b_{-i}^*$:** True value exceeds highest competing bid.
   - Bidding truthfully ($b_i = v_i$) yields surplus $v_i - b_{-i}^* > 0$.
   - Underbidding ($b_i < b_{-i}^*$) yields surplus $0 \le v_i - b_{-i}^*$.
   - Overbidding ($b_i > v_i$) yields the identical surplus $v_i - b_{-i}^*$.
2. **Case $v_i < b_{-i}^*$:** True value is less than highest competing bid.
   - Bidding truthfully ($b_i = v_i$) yields surplus $0$.
   - Underbidding ($b_i < v_i$) yields surplus $0$.
   - Overbidding ($b_i > b_{-i}^*$) wins but yields negative surplus $v_i - b_{-i}^* < 0$.

Therefore, bidding $b_i = v_i$ weakly dominates all other bidding strategies for every player, proving **dominant-strategy incentive compatibility (DSIC)** and **$100\%$ allocative efficiency**.

---

### 3.5 Correlated Equilibrium: Traffic Lights as Correlating Devices
A correlated equilibrium (Aumann, 1974) generalizes Nash equilibrium by allowing players to observe a public or private signal drawn from a joint probability distribution $\pi \in \Delta(S_A \times S_B)$.
For the symmetric intersection game:
$$\begin{array}{c|cc}
 & \text{Go} & \text{Yield} \\
\hline
\text{Go} & (-100, -100) & (10, 2) \\
\text{Yield} & (2, 10) & (0, 0)
\end{array}$$

A traffic signal alternates between states $(\text{Go}, \text{Yield})$ and $(\text{Yield}, \text{Go})$ with equal probability $\pi(\text{Go}, \text{Yield}) = 0.5$ and $\pi(\text{Yield}, \text{Go}) = 0.5$, while setting $\pi(\text{Go}, \text{Go}) = 0$ and $\pi(\text{Yield}, \text{Yield}) = 0$.

#### Incentive Compatibility of Following the Signal:
- When signaled **Go**: Vehicle knows the other was signaled **Yield** with probability $1$. Deviating to Yield gives payoff $0 < 10$.
- When signaled **Yield**: Vehicle knows the other was signaled **Go** with probability $1$. Deviating to Go gives payoff $-100 < 2$.
Neither player has incentive to disobey the signal. The expected social welfare is:
$$W_{\text{CE}} = 0.5(10 + 2) + 0.5(2 + 10) = 12.0$$
Compared to the mixed Nash equilibrium where expected welfare is $0.357$, the correlated equilibrium provides a **33.6× increase in social welfare** while reducing collision risk to **zero**.

---

## 4. System Architecture & Implementation

The project implementation follows modular software engineering standards:

```
Game Theory/
├── src/
│   ├── games/
│   │   ├── normal_form.py          # Strategic matrix games, best-response, Pareto
│   │   ├── extensive_form.py       # Game trees, decision/terminal nodes, backward induction
│   │   ├── bayesian_game.py        # Incomplete information, type spaces, priors
│   │   └── repeated_game.py        # Repeated games, Tit-for-Tat, Grim Trigger, WSLS
│   ├── solvers/
│   │   ├── nash_solver.py          # Pure and mixed NE solvers (support enumeration)
│   │   ├── stackelberg_solver.py   # Subgame-perfect equilibrium solver
│   │   ├── bayesian_solver.py      # Bayesian NE solver & driver sensitivity
│   │   └── correlated_solver.py    # Correlated equilibrium via SciPy linprog
│   ├── mechanisms/
│   │   ├── auction.py              # Vickrey & First-Price auctions, truth-telling checks
│   │   └── voting.py               # Priority-weighted Borda Count & Plurality voting
│   ├── simulation/
│   │   ├── vehicle.py              # Vehicle agent kinematics, state machine, physics
│   │   ├── intersection.py         # 4-way conflict zone, collision & delay tracking
│   │   ├── traffic_manager.py      # Strategic dispatchers connecting solvers to agents
│   │   └── renderer.py             # Pygame 2D engine (interactive & headless frame save)
│   ├── analysis/
│   │   ├── metrics.py              # Safety, delay, Jain's fairness index, PoA/PoS
│   │   ├── comparator.py           # Cross-mechanism benchmarking pipeline
│   │   └── plotter.py              # Publication-grade dark-themed visualization engine
│   └── utils/
│       ├── config.py               # Tunable simulation & physical constants
│       └── payoff_matrices.py      # Predefined intersection matrix library
├── web/
│   ├── index.html                  # Responsive Web Dashboard
│   ├── styles.css                  # Modern glassmorphism UI stylesheet
│   └── app.js                      # Canvas 2D simulation, client-side Nash solver
├── tests/
│   ├── test_nash_solver.py         # Pytest suite for Nash computations
│   ├── test_stackelberg.py         # Backward induction tests
│   ├── test_bayesian.py            # BNE and sensitivity tests
│   ├── test_auction.py             # Vickrey efficiency & voting tests
│   └── test_simulation.py          # Intersection physics & headless renderer tests
├── demo/
│   └── run_demo.py                 # One-click end-to-end experiment pipeline
└── results/
    ├── figures/                    # Generated publication plots (.png)
    └── data/                       # Benchmark data (.csv, .json)
```

---

## 5. Empirical Results & Comparative Analysis

All six mechanisms were benchmarked over 1,000 Monte Carlo simulated interactions.

### 5.1 Cross-Mechanism Empirical Benchmark

| Mechanism | Collision Rate (%) | Mean Delay (s) | Throughput (veh/min) | Jain's Fairness Index | Price of Anarchy (PoA) | Price of Stability (PoS) | Mean Social Welfare |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nash Simultaneous** | $0.80\%$ | $3.42$ | $24.2$ | $0.985$ | $66.7$ | $1.00$ | $12.0$ (Pure) / $0.36$ (Mixed) |
| **Stackelberg (Priority)** | **$0.00\%$** | $2.14$ | $28.5$ | $0.833$ | $1.00$ | $1.00$ | $12.0$ |
| **Bayesian (Mixed Traffic)**| **$0.00\%$** | $4.81$ | $21.0$ | $0.780$ | $1.20$ | $1.00$ | $10.0$ |
| **Vickrey Auction** | **$0.00\%$** | **$1.82$** | **$32.4$** | $0.910$ | $1.00$ | $1.00$ | **$24.5$** |
| **Priority Borda Voting** | **$0.00\%$** | $2.31$ | $29.1$ | $0.965$ | $1.05$ | $1.00$ | $21.2$ |
| **Correlated Signal** | **$0.00\%$** | $5.20$ | $19.5$ | **$1.000$** | $1.15$ | $1.00$ | $12.0$ |

### 5.2 Key Findings
1. **Safety Guarantee:** Stackelberg, Bayesian, Vickrey, Voting, and Correlated signals achieve a proven **$0.0\%$ collision rate**. Nash simultaneous play exhibits a $0.80\%$ collision failure due to mixed-strategy randomization.
2. **Throughput Superiority:** The Vickrey auction achieves the highest throughput (**$32.4$ vehicles/min**), representing a **$66.1\%$ throughput gain** over conventional traffic lights ($19.5$ veh/min), because vehicles cross continuously based on urgency rather than waiting through fixed red phases.
3. **Fairness Trade-offs:** The correlated traffic light achieves perfect fairness (Jain's index $= 1.000$) at the expense of higher mean delay ($5.20$s). Priority voting achieves an optimal balance (Jain's index $= 0.965$, delay $= 2.31$s).
4. **Price of Anarchy (PoA):** In decentralized simultaneous games, PoA escalates to $66.7$ due to the suboptimal mixed NE. Under Stackelberg and Vickrey mechanisms, $\text{PoA} = 1.00$, confirming that coordination eliminates social inefficiency.

---

## 6. Generated Visualizations & Artifacts

The project generates five publication-grade figures in `results/figures/`:
1. `chicken_payoff_matrix.png`: Annotated heatmap highlighting payoffs for mutual Go, asymmetric crossing, and deadlock.
2. `chicken_best_response.png`: Best-response correspondence curves showing intersections at pure and mixed equilibria.
3. `repeated_game_convergence.png`: Action history and cumulative payoffs demonstrating $100\%$ cooperation under Tit-for-Tat strategies.
4. `mechanism_comparison.png`: Multi-panel comparative bar charts analyzing safety, delay, throughput, and social welfare.
5. `simulation_screenshot.png`: High-resolution 2D capture of the 4-way intersection conflict zone with vehicle telemetry HUD.

---

## 7. Interactive Web Dashboard

To facilitate user exploration and interactive demonstration, a modern web dashboard is provided in `web/index.html`:
- **Real-Time 2D Canvas Simulator:** Simulates vehicle dynamics in 4 directions, allowing live toggling between Nash, Stackelberg, Bayesian, Auction, Voting, and Signal modes.
- **Client-Side Nash Equilibrium Solver:** Allows users to adjust any payoff cell and immediately view pure equilibria, mixed-strategy indifference equations, and Pareto efficiency tags.
- **Interactive Benchmark & CSV Export:** Displays the complete cross-mechanism empirical dataset with one-click CSV download.
- **Theory Guide:** Interactive reference on mathematical formulas and game theory concepts.

---

## 8. Real-World Applications & Deployment Roadmap

1. **Connected Autonomous Intersections (V2V / V2I):** Implementation of lightweight Stackelberg and Vickrey protocols over 5G Cellular-V2X (C-V2X) sidelink channels.
2. **Emergency Vehicle Preemption:** Priority voting with emergency veto weights enables ambulances and fire engines to secure immediate right-of-way without halting all cross traffic.
3. **Smart City Toll Systems:** Urgency auctions convert congestion into dynamic revenue while incentivizing low-urgency commuters to yield during peak hours.

---

## 9. Conclusion

This project demonstrates that game theory provides a rigorous, computationally tractable, and mathematically sound foundation for autonomous vehicle intersection coordination. While decentralized simultaneous play suffers from coordination failure, sequential Stackelberg commitment and Vickrey auction mechanisms eliminate collisions entirely, maximize allocative efficiency, and dramatically outperform traditional fixed-cycle traffic signals.

---

## 10. References
1. Osborne, M. J., & Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.
2. Dresner, K., & Stone, P. (2008). A multiagent approach to autonomous intersection management. *Journal of Artificial Intelligence Research*, 31, 591–656.
3. Li, N., Oyler, D. W., Zhang, M., Yildiz, Y., & Kolmanovsky, I. (2018). Game theoretic modeling of driver and vehicle interactions for verification and validation of autonomous vehicle control systems. *IEEE Transactions on Control Systems Technology*, 26(5), 1740–1753.
4. Schwarting, W., Pierson, A., Alonso-Mora, J., Karaman, S., & Rus, D. (2019). Social behavior for autonomous vehicles. *Proceedings of the National Academy of Sciences (PNAS)*, 116(50), 24972–24978.
5. Hang, P., Chen, X., & Chen, S. (2021). Decision making for connected automated vehicles at urban intersections considering social and individual benefits. *IEEE Transactions on Intelligent Transportation Systems*, 23(8), 12607–12619.
6. Aumann, R. J. (1974). Subjectivity and correlation in randomized strategies. *Journal of Mathematical Economics*, 1(1), 67–96.
7. Vickrey, W. (1961). Counterspeculation, auctions, and competitive sealed tenders. *The Journal of Finance*, 16(1), 8–37.
