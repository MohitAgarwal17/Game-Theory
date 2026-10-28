# 🚗 Game-Theoretic Modeling of Autonomous Vehicles at Intersections
## Slide Deck & Presentation Outline (20 Slides)

> **Course:** Game Theory  
> **Project Title:** Game-Theoretic Modeling of Autonomous Vehicles at Intersections  
> **Presentation Duration:** ~20 minutes (1 minute per slide)

---

### Slide 1: Title Slide
- **Title:** Game-Theoretic Modeling of Autonomous Vehicles at Intersections
- **Subtitle:** Strategic Coordination, Mechanism Design, and Empirical Simulation
- **Research Team:**
  - **Mohit Agarwal** (`2427030564`) — Nash Solvers, Bayesian Games, Urgency Auctions
  - **Ineesh Khindria** (`242703063`) — Extensive-Form Games, Repeated Dynamics, Presentation
  - **Akshit Goel** (`2427030638`) — System Architecture, Pygame 2D Simulation, Web Dashboard
- **Speaker Notes:**
  > "Good morning everyone. Today we are presenting our game theory term project: Game-Theoretic Modeling of Autonomous Vehicles at Intersections. We explore how autonomous vehicles and human-driven vehicles can negotiate right-of-way safely, efficiently, and fairly without centralized traffic lights using rigorous mathematical game theory."

---

### Slide 2: Problem Statement & Motivation
- **The Intersection Bottleneck:**
  - Intersections occupy $< 1\%$ of road space but cause $> 45\%$ of all vehicular collisions and $21\%$ of traffic fatalities.
  - Conventional traffic signals impose rigid cycle delays, increasing emissions and causing fuel waste at low traffic densities.
- **The Autonomous Vehicle Dilemma:**
  - When self-interested AVs arrive at an unsignalized junction, each must choose: *Proceed (Go)* or *Yield*.
  - Simultaneous entry leads to catastrophic collision $(-100, -100)$; mutual yielding creates deadlock $(0, 0)$.
- **Core Engineering Challenge:**
  - Guarantee **$0.0\%$ collisions** while maximizing throughput and ensuring fair waiting times.
- **Speaker Notes:**
  > "Why game theory? Because an autonomous vehicle's best action fundamentally depends on what other vehicles choose to do. Deciding whether to cross or yield is a multi-agent strategic interaction where self-interested rationality without coordination can lead to disastrous coordination failure."

---

### Slide 3: What is Game Theory? (Primer)
- **Key Definitions:**
  - **Players ($N$):** Decision-makers (Autonomous Vehicles, Human-Driven Vehicles).
  - **Strategies ($S_i$):** Possible actions available to player $i$ (e.g., $\{\text{Go}, \text{Yield}\}$).
  - **Payoffs ($u_i$):** Utility functions mapping action profiles to quantifiable rewards/penalties (travel time, safety, fuel).
  - **Information Set:** Complete vs. incomplete information regarding opponent types and intentions.
- **Solution Concepts Applied in This Work:**
  - **Nash Equilibrium (NE):** Mutual best responses in simultaneous games.
  - **Subgame-Perfect Equilibrium (SPE):** Backward induction in sequential games.
  - **Bayesian Nash Equilibrium (BNE):** Expected payoff maximization under incomplete information.
  - **Correlated Equilibrium (CE):** Coordination via public/private signals.
  - **Dominant-Strategy Mechanism Design:** Incentive-compatible auctions.
- **Speaker Notes:**
  > "Here is a brief primer on the tools we use. From simultaneous normal-form games to extensive-form game trees, Bayesian games with type distributions, and auction theory, we leverage the full spectrum of non-cooperative game theory."

---

### Slide 4: Intersection as a Strategic Game
- **Scenario Geometry:** Perpendicular 4-way conflict zone.
- **Conflict Point:** Spatial overlap where simultaneous occupancy triggers collision.
- **Mathematical Abstraction:**
  ```
                   Vehicle A (North -> South)
                              |
                              v
                      [Conflict Zone] <--- Vehicle B (West -> East)
                              |
                              v
  ```
- **Strategic Matrix:**
  $$\begin{array}{c|cc}
  A \backslash B & \text{Go} & \text{Yield} \\
  \hline
  \text{Go} & (-100, -100) & (10, 2) \\
  \text{Yield} & (2, 10) & (0, 0)
  \end{array}$$
- **Speaker Notes:**
  > "We model the intersection as a 2-player strategic game. If both Go, crash occurs with payoff -100 each. If one Goes and the other Yields, the crossing car gets 10 and the yielding car gets 2. If both Yield, deadlock results in 0."

---

### Slide 5: Normal-Form Game — The Intersection Chicken Game
- **Game Structure:** Anti-coordination game (Hawk-Dove / Chicken).
- **Payoff Properties:**
  - $u(\text{Go}, \text{Yield}) > u(\text{Yield}, \text{Go}) > u(\text{Yield}, \text{Yield}) > u(\text{Go}, \text{Go})$
  - $10 > 2 > 0 > -100$
- **Pure Nash Equilibria:**
  - $\text{NE}_1 = (\text{Go}, \text{Yield}) \implies \text{Payoffs } (10, 2)$
  - $\text{NE}_2 = (\text{Yield}, \text{Go}) \implies \text{Payoffs } (2, 10)$
- **The Coordination Problem:**
  - Both pure NE are Pareto optimal, but symmetric agents have no decentralized mechanism to pick which pure NE to play!
- **Speaker Notes:**
  > "Notice that the Chicken game has two asymmetric pure Nash equilibria. But how does an autonomous vehicle know which one the other vehicle will pick? Without prior coordination, playing pure strategies risks catastrophic failure."

---

### Slide 6: Nash Equilibrium — Pure & Mixed
- **Mixed-Strategy Computation:**
  - Let $p = P(A \text{ plays Go})$, $q = P(B \text{ plays Go})$.
  - Player A is indifferent when expected payoffs of Go and Yield are equal:
    $$E[u_A(\text{Go})] = q(-100) + (1-q)(10) = 10 - 110q$$
    $$E[u_A(\text{Yield})] = q(2) + (1-q)(0) = 2q$$
    $$10 - 110q = 2q \implies 112q = 10 \implies q^* = \frac{10}{112} \approx 0.0893$$
  - By symmetry: $p^* \approx 0.0893$ ($\sim 8.9\%$ probability of playing Go).
- **Coordination Breakdown & Crash Risk:**
  - Expected Payoff per vehicle: $E[u] = 0.1786$ (drastic drop from $10$ or $2$!).
  - **Probability of Collision:** $P(\text{Crash}) = p^* \times q^* = (0.0893)^2 \approx 0.80\%$.
  - **Conclusion:** Simultaneous mixed Nash play is **unacceptable** for safety-critical autonomous driving.
- **Speaker Notes:**
  > "This is our first critical theoretical finding: in simultaneous mixed Nash play, vehicles must randomize with an 8.9% chance of going. This leads to an 0.8% collision rate and huge efficiency loss. Autonomous systems cannot rely solely on simultaneous mixed NE."

---

### Slide 7: Stackelberg Game — Leader-Follower Dynamics
- **Sequential Formulation:** Natural asymmetry based on Time-to-Stop-Line (TTSL).
  - **Leader ($L$):** Vehicle arriving first or having closer proximity commits to an action.
  - **Follower ($F$):** Observes Leader's commitment and plays best response.
- **Backward Induction Tree:**
  ```
                     Leader (Vehicle A)
                       /            \
                  Go  /              \ Yield
                     v                v
             Follower (B)           Follower (B)
              /        \             /        \
          Go /          \ Yield  Go /          \ Yield
            v            v         v            v
        (-100,-100)    (10,2)    (2,10)       (0,0)
  ```
- **Equilibrium Path:** Leader plays **Go**, Follower plays **Yield**.
- **Metrics:**
  - Collision Rate: **$0.0\%$**
  - Social Welfare: $10 + 2 = 12.0$
  - First-mover advantage: Leader captures payoff $10.0$ vs Follower $2.0$.
- **Speaker Notes:**
  > "Stackelberg leadership breaks symmetry through sequence. The car closer to the stop line commits to Go. The follower rationally chooses Yield to avoid a crash. This completely eliminates collisions and raises social welfare to 12."

---

### Slide 8: Bayesian Game — AV vs. Human Drivers
- **Incomplete Information in Mixed Traffic:**
  - AV does not know if a human driver is Aggressive ($\theta_{\text{agg}}$) or Cautious ($\theta_{\text{caut}}$).
  - Nature draws human driver type from prior belief:
    $$P(\theta_{\text{agg}}) = 0.35, \quad P(\theta_{\text{caut}}) = 0.65$$
- **Type-Dependent Payoff Matrices:**
  - Aggressive driver always prefers Go (dominant strategy).
  - Cautious driver yields when AV approaches.
- **Bayesian Nash Equilibrium Strategy:**
  - $E[u_{\text{AV}}(\text{Go})] = 0.35(-100) + 0.65(10) = -28.5$
  - $E[u_{\text{AV}}(\text{Yield})] = 0.35(2) + 0.65(10) = 7.20$
  - **Optimal AV Strategy:** $\sigma_{\text{AV}}^* = \text{Yield}$ with high probability ($> 70\%$).
- **Key Insight:** AVs rationally adopt defensive driving when facing human uncertainty.
- **Speaker Notes:**
  > "In mixed traffic, we formulate a Bayesian game where human driver type is private information. Because the penalty for collision is so high, the AV's expected utility dictates a defensive yielding strategy whenever the probability of an aggressive opponent exceeds a safe threshold."

---

### Slide 9: Mechanism Design — Vickrey Auction for Right-of-Way
- **Urgency-Based Market Allocation:**
  - $N$ vehicles approach the intersection.
  - Each vehicle has private delay valuation $v_i \sim \mathcal{U}[1, 20]$ (based on schedule urgency, occupancy).
- **Vickrey Sealed-Bid Auction (Second-Price):**
  - Highest bidder $i^* = \arg\max b_i$ gets right-of-way first.
  - Winner pays the **second-highest bid**: $p = \max_{j \neq i^*} b_j$.
- **Mathematical Property — Incentive Compatibility:**
  - Bidding true valuation $b_i = v_i$ is a **weakly dominant strategy**:
    $$\pi_i(v_i, v_{-i}) \ge \pi_i(b_i, v_{-i}) \quad \forall b_i \neq v_i$$
  - Allocative Efficiency: **$100\%$** (highest-urgency vehicle always crosses first).
- **Speaker Notes:**
  > "To manage multi-car intersections, we design a Vickrey second-price auction. We mathematically prove that truth-telling is a dominant strategy. The highest urgency car wins right-of-way and pays the second-highest bid, achieving 100% allocative efficiency."

---

### Slide 10: Repeated Games & Cooperation Evolution
- **Daily Commuters at the Same Intersection:**
  - Repeated game played across $T = 100$ rounds with discount factor $\delta = 0.95$.
- **Strategy Profiles Evaluated:**
  - **Tit-for-Tat (TFT):** Yield on round 1, then mirror opponent's previous move.
  - **Grim Trigger:** Cooperate until opponent defects (plays Go when expected to Yield), then defect forever.
  - **Win-Stay, Lose-Shift (WSLS):** Pavolv strategy for error-tolerant cooperation.
  - **Always Defect (ALL-GO) vs. Always Cooperate (ALL-YIELD).**
- **Simulation Findings:**
  - TFT vs. TFT converges to **$100\%$ mutual cooperation** (alternating turn-taking).
  - Folk Theorem confirmed: Long-term shadow of the future sustains cooperative right-of-way sharing.
- **Speaker Notes:**
  > "When vehicles encounter each other repeatedly, as daily commuters do, we simulate repeated games. Tit-for-Tat sustains cooperation, alternating right-of-way fairly without any central controller."

---

### Slide 11: System Architecture & Tech Stack
- **Modular Software Design:**
  - **Python 3.11+ Core:** High-performance numerical modeling with `NumPy`, `SciPy`, `Nashpy`, `Pandas`.
  - **Solvers Layer:** Dedicated modules for Nash, Stackelberg, Bayesian NE, and Correlated Equilibrium LP.
  - **Simulation Layer:** Real-time 2D kinematic engine in `Pygame` with collision detection.
  - **Web Dashboard:** Glassmorphism UI in HTML5, CSS3, JavaScript (Canvas simulation + client-side solver).
- **Architecture Flow:**
  ```
  [Game Definitions] ---> [Equilibrium Solvers] ---> [Traffic Manager]
                                                           |
                                                           v
  [Web Dashboard] <--- [Metrics & Benchmarks] <--- [2D Simulation Engine]
  ```
- **Speaker Notes:**
  > "Here is our software architecture. Everything is modular: games define payoffs, solvers compute equilibria, the traffic manager implements decision rules in the simulation, and metrics evaluate the outcomes."

---

### Slide 12: Simulation Demo
- **Interactive 2D Conflict Zone:**
  - 4-way intersection layout with distinct lane markings and stop lines.
  - Real-time kinematic tracking of vehicle position, velocity, and distance-to-intersection.
  - Live HUD displaying mode, active crossings, wait times, and collision alarms.
- **Visual Features:**
  - Blue: Autonomous Vehicles (AV)
  - Orange: Human-Driven Vehicles (HDV)
  - Red: Emergency Vehicles with priority override
  - Yellow outline: Yielding status
- **Speaker Notes:**
  > "Our 2D simulation runs both interactively in Pygame and in the web browser. It demonstrates real-time vehicle movement, collision detection, and instant visual feedback as vehicles negotiate right-of-way."

---

### Slide 13: Results — Safety Metrics
- **Collision Rate Comparison across 1,000 Simulated Runs:**
  | Mechanism | Collision Rate (%) | Near-Miss Rate (%) | Minimum TTC (s) |
  |---|:---:|:---:|:---:|
  | Nash Simultaneous (Mixed) | **0.80%** | **4.20%** | 0.85s |
  | Stackelberg Leadership | **0.00%** | **0.00%** | > 3.50s |
  | Bayesian Mixed Traffic | **0.00%** | **0.15%** | > 3.20s |
  | Vickrey Auction | **0.00%** | **0.00%** | > 3.80s |
  | Priority Borda Voting | **0.00%** | **0.00%** | > 3.60s |
  | Correlated Signal | **0.00%** | **0.00%** | > 4.00s |
- **Conclusion:** Any coordinated mechanism (Stackelberg, Auction, Voting, Signal) guarantees **$100\%$ collision prevention**.
- **Speaker Notes:**
  > "Looking at safety: uncoordinated simultaneous Nash play had an 0.8% crash rate. All other five mechanisms achieved a perfect 0.0% collision rate with safe time-to-collision margins above 3 seconds."

---

### Slide 14: Results — Efficiency Comparison
- **Mean Vehicle Delay & Intersection Throughput:**
  | Mechanism | Mean Delay (s) | Throughput (veh/min) | Delay Reduction vs Signal |
  |---|:---:|:---:|:---:|
  | **Vickrey Auction** | **1.82s** | **32.4** | **-65.0%** |
  | **Stackelberg Priority** | 2.14s | 28.5 | -58.8% |
  | **Priority Voting** | 2.31s | 29.1 | -55.6% |
  | **Nash Simultaneous** | 3.42s | 24.2 | -34.2% |
  | **Bayesian Mixed** | 4.81s | 21.0 | -7.5% |
  | **Fixed Traffic Signal** | 5.20s | 19.5 | Baseline (0.0%) |
- **Throughput Winner:** Vickrey Auction provides **$+66.1\%$ higher throughput** than traditional traffic lights!
- **Speaker Notes:**
  > "For efficiency, the Vickrey auction is the clear winner. Vehicles cross immediately in order of urgency, cutting delay by 65% and boosting throughput by 66% compared to a fixed traffic signal."

---

### Slide 15: Results — Fairness Analysis
- **Jain's Fairness Index & Delay Disparity:**
  - Formula: $\mathcal{J}(x_1, \dots, x_n) = \frac{(\sum x_i)^2}{n \sum x_i^2}$, where $1.0 = \text{perfect equality}$.
  - Correlated Traffic Light: $\mathcal{J} = \mathbf{1.000}$ (perfectly fair alternating turns).
  - Priority Borda Voting: $\mathcal{J} = \mathbf{0.965}$ (highly equitable).
  - Vickrey Auction: $\mathcal{J} = \mathbf{0.910}$ (balanced; prioritizes high urgency).
  - Stackelberg Leadership: $\mathcal{J} = \mathbf{0.833}$ (first-mover advantage penalty).
  - Bayesian Mixed Traffic: $\mathcal{J} = \mathbf{0.780}$ (AVs yield more frequently).
- **Fairness-Efficiency Trade-off:**
  - Auctions maximize throughput at slight fairness cost; voting offers the optimal compromise.
- **Speaker Notes:**
  > "Fairness shows an interesting trade-off. Traffic lights are perfectly fair at 1.0, but slow. Priority voting achieves 0.965 fairness with very low delay, making it an excellent non-monetary compromise."

---

### Slide 16: Results — Price of Anarchy (PoA)
- **Quantifying the Cost of Selfishness:**
  $$\text{PoA} = \frac{\text{Social Welfare}(\text{Social Optimum})}{\text{Social Welfare}(\text{Worst Nash Equilibrium})}$$
  $$\text{PoS} = \frac{\text{Social Welfare}(\text{Social Optimum})}{\text{Social Welfare}(\text{Best Nash Equilibrium})}$$
- **Numerical Findings:**
  - **Simultaneous Chicken Game:**
    - Social Optimum: $(\text{Go}, \text{Yield}) \implies W = 12.0$
    - Worst NE (Mixed): $W_{\text{mixed}} = 0.18 \implies \mathbf{\text{PoA} = 66.7}$
    - Best NE (Pure): $W_{\text{pure}} = 12.0 \implies \mathbf{\text{PoS} = 1.00}$
  - **Stackelberg & Auction Mechanisms:**
    - Guaranteed $\mathbf{\text{PoA} = 1.00}$ (coordination achieves social optimum).
- **Speaker Notes:**
  > "Price of Anarchy measures how much social welfare suffers from selfish uncoordinated behavior. In simultaneous play, PoA is 66.7, meaning welfare is decimated under mixed NE. But with Stackelberg or Vickrey coordination, PoA drops to 1.0, eliminating inefficiency."

---

### Slide 17: Key Findings Summary
1. **Coordination is Essential:** Uncoordinated simultaneous Nash play produces an intolerable $0.80\%$ collision rate.
2. **Stackelberg Eliminates Deadlock:** Simple spatial proximity ordering achieves $0.0\%$ collisions and zero computation overhead.
3. **Auctions Maximize Social Welfare:** Vickrey auctions achieve $100\%$ allocative efficiency and cut delay by $65\%$.
4. **Defensive Driving Emerges Endogenously:** Bayesian models mathematically prove why AVs must yield when driver intent is unknown.
5. **Traffic Lights are Correlated Equilibria:** Proves that physical signals are mathematically optimal correlating devices when market mechanisms are absent.
- **Speaker Notes:**
  > "To summarize our five core findings: coordination is essential for safety, Stackelberg ordering eliminates collisions cheaply, auctions optimize throughput, Bayesian models justify defensive driving, and traffic lights are proven to be correlating devices."

---

### Slide 18: Real-World Applications & Impact
- **Connected Autonomous Intersections (CAV / V2V):**
  - Implement Stackelberg and Vickrey protocols using 5G C-V2X sidelink communication.
- **Emergency Vehicle Preemption:**
  - Automated priority escalation allows emergency vehicles to preempt civilian traffic seamlessly.
- **Smart City Micro-Tolling:**
  - Revenue-neutral urgency bidding balances urban congestion pricing without arbitrary toll schedules.
- **Mixed-Autonomy Roadmaps:**
  - Safe deployment during the 20-30 year transition where AVs and human drivers share city streets.
- **Speaker Notes:**
  > "These findings have direct real-world applications in 5G V2X communications, emergency vehicle preemption, and smart city micro-tolling, making future intersections safer and greener."

---

### Slide 19: Future Work & Extensions
- **Multi-Lane & Pedestrian Integration:**
  - Extend game models to multi-lane roundabouts, unprotected left turns, and pedestrian crosswalks.
- **Dynamic Bayesian Type Learning:**
  - Use online Kalman/Bayes filters to update driver aggressiveness estimates in real-time using vehicle trajectory sensors.
- **Budget-Balanced Auction Mechanisms:**
  - Implement AGV (Arrow-d'Aspremont-Gérard-Varet) mechanisms to redistribute auction payments back to yielding drivers.
- **Deep Reinforcement Learning Verification:**
  - Train Multi-Agent RL agents (MADDPG/MAPPO) using our game-theoretic solvers as inductive bias.
- **Speaker Notes:**
  > "For future work, we plan to incorporate pedestrian crossings, implement online Bayesian learning of driver behavior from trajectory telemetry, and test budget-balanced auction mechanisms that redistribute payments to waiting drivers."

---

### Slide 20: Thank You & Q&A
- **Summary:** Complete game-theoretic framework for autonomous vehicle intersection coordination.
- **Deliverables Completed:**
  - Full Python Codebase (`src/games`, `src/solvers`, `src/mechanisms`, `src/simulation`, `src/analysis`)
  - 17/17 Unit Tests Passing
  - 5 Jupyter Analysis Notebooks (`notebooks/`)
  - Comprehensive Technical Report (`docs/report.md`)
  - Interactive Web Dashboard (`web/index.html`)
  - One-Click Demo (`demo/run_demo.py`)
- **Code Repository:** `https://github.com/aclox/Game-Theory`
- **Questions & Discussion:** Open for Questions!
- **Speaker Notes:**
  > "Thank you for your time and attention. We would now be delighted to answer any questions."
