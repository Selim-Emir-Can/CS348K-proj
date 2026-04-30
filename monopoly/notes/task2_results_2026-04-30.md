# Task 2 (LLM-driven GA) results — 2026-04-30

## Summary

Ran the LLM-driven board-design GA: pop=8 × generations=5 = 32 evaluations
(elitism=2 carries 2 individuals unchanged each generation, so the
inner-loop budget is `pop + (generations-1)*(pop-elitism)` = 32). Each
candidate evaluated by 5 all-LLM (Qwen2.5-1.5B) 2p games. 2-player only.

Wall time: **4:57:50** end-to-end (mean 558 s / candidate; some sparse
boards finished in <2 min, dense boards took ~12 min).

## Convergence

| gen | n_evals | best score | rounds | fairness | xfer/r |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 0.565 | 30.6 | 0.20 | 93.4 |
| 1 | 6 | 0.526 | 33.0 | 0.20 | 100.5 |
| **2** | 6 | **0.465** | 43.6 | 0.20 | 90.6 |
| 3 | 6 | 0.465 | 43.6 | 0.20 | 90.6 (carried) |
| 4 | 6 | 0.513 | 40.0 | 0.20 | 84.7 |

The GA found its winner at generation 2 (overall iter=15), held it
through gen 3 via elitism, regressed slightly in gen 4 (noise on top of
a real optimum given n_seeds=5 per candidate). Composite-score drop
from generation 0 to generation 2 is ~18%, comparable in magnitude to
what the rule-based GA achieved at much higher eval budget.

## Headline comparison vs rule-based GA winner

| | rounds | fairness | xfer/r | composite |
|---|---:|---:|---:|---:|
| Rule-based GA-2p winner | 61.0 | 0.24 | 73.2 | 0.728 |
| **LLM-GA-2p winner**    | **43.6** | **0.20** | **90.6** | **0.465** |

Composite scores aren't directly comparable (different evaluators, vastly
different game counts) but the per-component metrics are: under LLM
play, the LLM-GA winner's board is shorter (-28%), fairer (-17%), and
more interactive (+24%) than the rule-based GA winner.

## Why this is interesting (for the report)

The LLM-driven GA selected a board that pushes games into a *more
extreme* corner of the design space than the rule-based GA did. Two
possible explanations, both worth surfacing:

1. **The LLM player is more aggressive than the average rule-based
   strategy** — it buys more often when affordable, builds less
   carefully, plays more "all-or-nothing". A board that exaggerates
   cost/rent multipliers will turn the LLM's aggression into faster
   bankruptcies, hence shorter games.
2. **The LLM-GA's evaluator is noisier (n_seeds=5 vs rule-based's 1000
   games per candidate)** so the optimiser is partly fitting noise. The
   "winner" board may be a board that *happens* to give big rent swings
   on the 5 seeds we evaluated — true expected score might be closer to
   the rule-based winner's.

A small follow-up to disambiguate: re-evaluate the LLM-GA winner at
n_seeds=20 (matches Task 1) and see if the score holds. ~30 min.

## Files

- `logs/optimizer_llm/llm_ga_2p/evals.jsonl`     — 32 evaluations
- `logs/optimizer_llm/llm_ga_2p/best_design.json` — winner vec + metrics
- `logs/optimizer_llm/llm_ga_2p/decisions/eval_<NNN>/seed_<S>.jsonl`
  — every LLM decision in every game (~5 KB/game × 5 games × 32 evals
  ≈ 800 KB total; useful for retroactive analysis of *why* the winner
  scored well)
- `logs/optimizer_llm/llm_ga_2p/meta.json` — config

## Open questions for the report

- Does the LLM-GA winner remain a strong design when played by the
  rule-based pool? Cross-eval like the rule-based GA winner was tested
  by all-LLM seats. Direction: take `best_design.json` from this run,
  evaluate it under `scripts/optimize_board.py`-style 10-matchup
  rule-based eval, compare its metrics to the rule-based GA winner.
- Do LLM-GA convergence curves overlay sensibly on rule-based GA
  curves? A combined plot in the report would show whether the search
  algorithms converged at similar rates relative to budget.
- Decision-log analysis on the GA winner: did the LLM behave
  differently on this board vs. on default? Buy-rate by group, by
  cash bucket, etc. — same analyser as Task 1.
