"""Game-design optimisation package.

Components:
  strategy_pool    -- curated + sampled ParametricPlayer strategies
  design_space     -- board-parameter vector ↔ GameConfig encoding
  simulate         -- runs games, collects per-game stats (incl. money transfer)
  objectives       -- fairness / length / draw / money-transfer scoring
  search           -- random search + GA over the design space
"""
