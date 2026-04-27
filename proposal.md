# Proposal: Skill-Stratified Co-Evolutionary Game Balancing in Monopoly

---

## 1. Why does this problem matter?

Game balancing is one of the most expensive and subjective steps in game development. Developers rely on human playtesters — sessions that are slow, costly, and biased toward a narrow range of player skill. A board that feels fair to expert players may be frustrating for beginners; a board tuned for casual play may offer no strategic depth for experienced players. Yet no automated system currently evaluates a game's quality *across the full spectrum of player skill simultaneously*.

The cost of creating interactive worlds is collapsing — vibe-coded environments, procedurally generated levels, and AI-assisted game design are producing worlds faster than humans can evaluate them. The bottleneck has shifted from creation to assessment. An automated system that can evaluate and optimize game parameters end-to-end, in simulation, replaces a slow and expensive beta-testing pipeline with a fast, principled, reproducible one.

---

## 2. Why is this hard?

- **Quality is multi-dimensional and skill-dependent.** Fairness, engagement, game length, and skill expression are all valid quality criteria — and they trade off against each other differently at different skill levels. A single scalar reward cannot capture this.
- **The parameter-outcome relationship is non-differentiable.** Monopoly outcomes emerge from stochastic multi-agent interactions over hundreds of turns. You cannot backpropagate through a game. Optimization must be black-box.
- **Skill is a spectrum, not a binary.** A board balanced for a random agent and an expert agent may still be broken for intermediate players. The optimization must account for the full population.
- **Luck and skill are entangled.** Monopoly is deliberately luck-heavy. Disentangling how much of an outcome is due to player skill vs. dice rolls requires careful metric design and large simulation samples.

---

## 3. What has been tried and why it fails

| Method | Approach | Limitation |
|---|---|---|
| RL agents for Monopoly (prior work) | Train one agent to win | Optimizes playing, not balancing |
| PCGRL (Khalifa et al., 2020) | RL as level designer for tile games | Single fixed evaluator agent; not multi-skill |
| RuleSmith (2026) | LLM self-play + Bayesian optimization | LLM players only; asymmetric role games; not economic board games |
| Human playtesting | Manual beta testing | Slow, expensive, narrow skill coverage |

No prior work optimizes game parameters against a *population* of agents at different skill levels simultaneously. The skill-stratified evaluation loop is entirely absent from the literature.

---

## 4. Proposed method

We propose a co-evolutionary balancing framework for Monopoly: board parameters co-evolve with agent behavior across skill levels until the board satisfies multi-objective quality criteria for the full player population.

**The loop:**
1. Sample a set of board parameters θ (property prices, rent values, tax amounts, jail mechanics, community chest distributions)
2. Run tournaments on the parameterized board using a population of agents at K skill levels: random (beginner), rule-based heuristic (intermediate), trained DQN (expert)
3. Compute quality metrics from tournament outcomes
4. Use Bayesian optimization (or CMA-ES) to propose a new θ that improves the quality metrics
5. Repeat until convergence

**Quality metrics (multi-objective reward):**
- **Fairness:** win rate variance across starting positions and player order — does position 1 have an unfair advantage?
- **Skill expression:** does the DQN agent win significantly more than the random agent? (if not, the board is pure luck; if always, there's no fun for beginners)
- **Engagement:** mean game length — boards that end too fast or drag indefinitely are penalized
- **Competitiveness:** wealth Gini coefficient at game end — do all players stay competitive, or does one player snowball immediately?

**Skill population:**
- Random agent: buys everything, random decisions
- Rule-based agent: classic heuristics (buy color sets, build houses early, avoid mortgaging)
- DQN agent: trained against self-play on standard Monopoly board

**Key hypothesis:** No single board configuration maximizes quality across all skill levels simultaneously. The optimizer will discover Pareto-optimal boards that best balance competing criteria — and the resulting boards will differ meaningfully from the original Hasbro design.

---

## 5. Connection to co-evolutionary framework

This is the direct game-design inversion of the "What if Eye...?" framework (Tiwary et al., Science Advances 2025). That work co-evolves eye morphology and agent behavior under fixed environmental pressures to understand vision. We co-evolve world parameters (board design) and agent behavior (skill-stratified policies) under fixed quality pressures to understand game balance. The agent is the measurement instrument; the world is the optimizable variable. Both frameworks answer the same class of question: *what world produces the target behavior?*

---

## 6. Feasibility

**Environment:** Open-source Monopoly simulators exist (gym-style Gymnasium environments, PyMonopoly). A single game runs in milliseconds in Python. 10,000 games per parameter evaluation is feasible on CPU.

**Agents:** Random and rule-based agents require no training. The DQN agent can be trained on standard Monopoly in a few hours on a single GPU using stable-baselines3.

**Optimizer:** Bayesian optimization (BoTorch) or CMA-ES over ~10-20 continuous board parameters. Both are well-supported, low-compute, and sample-efficient for black-box optimization.

**Compute budget:** Entire pipeline runnable on a single GPU machine. No large-scale pretraining required.

---

## 7. Non-goals

- We are not proposing a new RL algorithm
- We are not building a new Monopoly environment from scratch
- We are not targeting human-level Monopoly play
- We are not claiming the optimized board is the "best" Monopoly — we are demonstrating the automated balancing pipeline works end-to-end

---

## 8. Broader impact

The framework generalizes beyond Monopoly to any parameterized game or interactive world — platformer levels, card game decks, RPG economy systems, robotics training environments. The core contribution is the skill-stratified evaluation loop as a general-purpose automated playtesting primitive. As AI-generated worlds become ubiquitous, this kind of automated quality assessment becomes essential infrastructure.