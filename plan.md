Week 1-2: Build the environment

Fork gamescomputersplay/monopoly — it's the most simulation-ready
Expose board parameters (property prices, rents, tax, jail fine, starting money) as a config dict so you can swap them programmatically
Wrap the game loop in a PettingZoo multi-agent API (standard for multi-agent RL) — reset(), step(), observe(), done()
Verify you can run 10K games fast in pure Python — target under a minute on CPU. If too slow, vectorize the game loop with numpy

Week 3: Build the agent population

Random agent — already exists in most simulators
Rule-based agent — buy everything you land on, build houses as soon as you have a color set, don't mortgage unless bankrupt. ~50 lines of code
DQN agent — train with stable-baselines3 PPO or DQN against the rule-based agent for a few hours on CPU. This is your "expert"

Week 4: Define and validate quality metrics

Implement fairness (win rate variance by starting position), skill expression (DQN win rate vs random), game length, wealth Gini coefficient
Run baseline tournaments on the standard Hasbro board and verify the metrics behave sensibly — e.g. DQN should beat random reliably

Week 5-6: Optimization loop

Define your parameter space — ~10-15 continuous parameters
Plug in BoTorch Bayesian optimization or CMA-ES as the outer loop
Each evaluation: sample θ, run 1000 tournament games across all skill pairings, compute multi-objective quality score, return to optimizer
Run until convergence, analyze what the optimizer discovers about the board

Week 7: Analysis and writeup

Compare optimized board vs standard Hasbro board on all metrics
Visualize what parameters changed and why — does the optimizer reduce first-mover advantage? flatten rent curves? shorten game length?
This is your main result: the pipeline works end-to-end and produces interpretable, actionable design insights