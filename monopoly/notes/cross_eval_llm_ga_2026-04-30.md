# Cross-eval: LLM-driven GA winner under rule-based pool — 2026-04-30

## Setup

Re-evaluated the LLM-driven GA-2p winner (`logs/optimizer_llm/llm_ga_2p/best_design.json`,
score 0.465 under its native LLM evaluator) under the rule-based
30-strategy pool's standard protocol: 1{,}000 games × 10 strategy
matchups, both 2p and 3p.

Command:
```
python scripts/cross_eval.py \
    --runs logs/optimizer_llm/llm_ga_2p/evals.jsonl \
    --n-games 1000 \
    --out logs/optimizer/cross_eval_llm_ga_winner.json
```

## Results

| design                       | n_p | score | fair  | max_f | rounds | draw  | xfer/r |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity_default             | 2   | 1.463 | 0.454 | 0.940 | 103.9  | 0.074 | 49.7   |
| identity_default             | 3   | 1.230 | 0.412 | 0.680 | 110.8  | 0.176 | 99.4   |
| ga_2p_mask_best (rule-based) | 2   | 0.817 | 0.237 | 0.960 | 62.5   | 0.013 | 74.7   |
| ga_2p_mask_best (rule-based) | 3   | 0.724 | 0.251 | 0.570 | 66.8   | 0.012 | 142.4  |
| ga_3p_mask_best (rule-based) | 2   | 0.923 | 0.294 | 0.990 | 63.8   | 0.004 | 66.0   |
| ga_3p_mask_best (rule-based) | 3   | 0.787 | 0.361 | 0.610 | 64.4   | 0.011 | 127.0  |
| **llm_ga_2p_winner**         | 2   | **1.045** | **0.379** | **0.990** | **45.9** | 0.003 | **82.4** |
| **llm_ga_2p_winner**         | 3   | **1.015** | **0.404** | **0.640** | **47.1** | 0.008 | **160.6** |

## Three findings

**1. Partial generalisation.** The LLM-driven winner is a real
improvement over the default board on both player counts (1.045 vs
1.463 at 2p; 1.015 vs 1.230 at 3p) but loses to the rule-based GA's
own 2p winner (0.817 < 1.045) when scored on the rule-based protocol.
Both search algorithms found optima *inside* their evaluator's
distribution.

**2. Length transfers; fairness does not.** Rounds-to-completion is
nearly identical between evaluators (43.6 LLM eval / 45.9 pool eval),
which makes sense: game length is mostly a function of the board's
mechanical incentives, not which strategy is playing. Fairness, by
contrast, exploded from 0.20 (LLM, two identical seats — measures
seat-position bias only) to 0.379 (pool, 30 different strategy
archetypes — measures real strategic asymmetry). The LLM evaluator
masked exploitable asymmetries because it played the same way from
every seat.

**3. Money transfer overshoots target.** The LLM-GA winner's
xfer/round is 82.4 (2p) and 160.6 (3p), well above the
target=100. Under the LLM evaluator the 2p value was 90.6, close
to target; under the pool the same board generates much more money
flow because diverse strategies trade and pay rent more than the
LLM did. This is again the "evaluator distribution mismatch" effect.

## Implication for the report

A single-personality evaluator can drive useful GA convergence on
metrics that depend on the board's mechanics (length, money flow),
but it systematically under-counts fairness problems that only show
up when diverse strategies probe the design space. For design
problems where fairness is a target, evaluator diversity is
non-optional even when the chosen agent class is otherwise strong.
