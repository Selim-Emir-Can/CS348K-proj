# Beta-Testing Monopoly: Closed-Loop Game-Design Optimisation

_Selim Emir Can (selimcan@stanford.edu) · CS348K project · 2026_

> **Final report:**
> [`monopoly/report/report_with_playtest_shortlist.pdf`](monopoly/report/report_with_playtest_shortlist.pdf)
> (source `report_with_playtest_shortlist.tex`). Canonical numbers are in
> [`monopoly/RESULTS.md`](monopoly/RESULTS.md); project context is in
> [`monopoly/context.md`](monopoly/context.md).

This repository automates the beta-testing loop that game and system designers
usually run by hand. We treat Monopoly as a parameterised multi-agent system
and build a closed-loop optimiser: a diverse rule-based agent pool plays on
candidate boards while a genetic algorithm searches a parameterised
environment space to minimise a composite of fairness, game length,
decisiveness, and inter-agent interactivity. Applied to Monopoly, the
optimiser cuts the combined score ~47% below the default board, and we
validate the winners with an independent LLM agent class and a human
playtest pilot, surfacing transferable principles for automating
beta-testing of any multi-agent system.

**The full write-up** (with convergence curves, ablation matrices,
cross-evaluation at _n_=1000, per-archetype win-rate shifts, the LLM
cross-class verdict, and the human playtest pilot) is in
[`monopoly/report/report_with_playtest_shortlist.tex`](monopoly/report/report_with_playtest_shortlist.tex)
(compiled PDF:
[`report_with_playtest_shortlist.pdf`](monopoly/report/report_with_playtest_shortlist.pdf)).

---

## Pipeline at a glance

```
30-strategy agent pool          66-dim design space
(10 named archetypes +          (22 cost mults + 22 rent mults
 20 random samples from a        + 22-bit keep-mask that structurally
 17-dim parametric ruleset)       shortens the board from 40 cells)
           │                            │
           └───────────┬────────────────┘
                       ▼
            candidate evaluation
            (10 matchups × 10 games
             with shared seeds for CRN
             variance reduction)
                       │
                       ▼
            composite score
            (fairness / length / draw / money-transfer)
                       │
                       ▼
            genetic-algorithm / random-search outer loop
```

Everything is deterministic: the same run-name produces byte-identical
JSONL history across fresh Python processes (popcount enforcement uses
SHA-256, not Python's randomised `hash()`).

---

## Repository layout

```
monopoly/
├── agents.py                   ParametricPlayer + legacy RandomPlayer / DQNPlayer
├── player_settings.py          17-dim ParametricPlayerSettings dataclass
├── monopoly_env.py             PettingZoo env wrapper (RL pipeline, optional)
├── config.py                   GameConfig (↔ YAML) + player class/settings registry
├── monopoly/core/              Upstream Monopoly simulator
│   ├── game.py                 Core game loop
│   ├── player.py               Turn logic, buying/building/trading/jail (patched to support shrunk boards)
│   └── board.py                Board structure + landmark-index lookups
├── optimizer/                  Design-space optimisation package
│   ├── simulate.py             Per-game stat collection + money-transfer tracking
│   ├── design_space.py         66-dim vector ↔ GameConfig encoder/decoder
│   ├── strategy_pool.py        30-strategy pool + evaluation matchup sampler
│   ├── objectives.py           Fairness / length / draw / money-transfer
│   └── search.py               Random search + Genetic Algorithm
└── scripts/                    CLI drivers
    ├── build_strategy_pool.py  One-shot: build and save the 30-strategy pool
    ├── optimize_board.py       Run a search (random or GA) on the design space
    ├── cross_eval.py           High-confidence (n=1000) re-evaluation of winners
    ├── strategy_heatmap.py     30×30 strategy win-rate matrix on a chosen design
    ├── eval_default.py         Default-board reference baseline
    ├── report_runs.py          Convergence and Pareto plots from run histories
    ├── render_board.py         Canonical board render (shrunk layout)
    ├── render_board_legacy.py  Canonical board render (40-cell layout)
    └── render_all_boards*.py   Batch renderers for a full ablation matrix
```

---

## Quickstart

Built and tested with Python 3.10 on Windows (WSL bash) with:

```
pettingzoo  gymnasium  numpy  matplotlib  pyyaml  tqdm
```

RL-pipeline scripts additionally need `stable-baselines3`, `sb3-contrib`,
`torch`, `wandb`; the optimisation pipeline does not.

All commands assume you are inside the `monopoly/` subdirectory
(`cd monopoly` from the repo root):

```cmd
:: 0. Build the diverse strategy pool (once)
set PYTHONPATH=. && python scripts/build_strategy_pool.py

:: 1. Default-board baseline
set PYTHONPATH=. && python scripts/eval_default.py --n-players 2 --out logs/optimizer/default_2p.json
set PYTHONPATH=. && python scripts/eval_default.py --n-players 3 --out logs/optimizer/default_3p.json

:: 2. Genetic-algorithm search (combined objective, 2p and 3p)
set PYTHONPATH=. && python scripts/optimize_board.py --search ga --generations 20 --n-players 2 --run-name ga_2p
set PYTHONPATH=. && python scripts/optimize_board.py --search ga --generations 20 --n-players 3 --run-name ga_3p

:: 3. Random-search baseline at matched budget
set PYTHONPATH=. && python scripts/optimize_board.py --search random --iters 362 --n-players 2 --run-name random_2p
set PYTHONPATH=. && python scripts/optimize_board.py --search random --iters 362 --n-players 3 --run-name random_3p

:: 4. Single-objective ablations (4 per player-count)
set PYTHONPATH=. && python scripts/optimize_board.py --search ga --generations 20 --n-players 2 --w-fair 1 --w-fmax 0 --w-len 0 --w-draw 0 --w-money 0 --run-name abl_fair_2p
set PYTHONPATH=. && python scripts/optimize_board.py --search ga --generations 20 --n-players 2 --w-fair 0 --w-fmax 0 --w-len 1 --w-draw 0 --w-money 0 --run-name abl_len_2p
set PYTHONPATH=. && python scripts/optimize_board.py --search ga --generations 20 --n-players 2 --w-fair 0 --w-fmax 0 --w-len 0 --w-draw 1 --w-money 0 --run-name abl_draw_2p
set PYTHONPATH=. && python scripts/optimize_board.py --search ga --generations 20 --n-players 2 --w-fair 0 --w-fmax 0 --w-len 0 --w-draw 0 --w-money 1 --run-name abl_money_2p
:: (and similarly for 3p)

:: 5. High-confidence cross-evaluation (n=1000)
set PYTHONPATH=. && python scripts/cross_eval.py --runs logs/optimizer/ga_2p.jsonl logs/optimizer/ga_3p.jsonl --identity --n-games 1000 --out logs/optimizer/cross_eval.json

:: 6. 30×30 strategy heatmaps (mean-|W-0.5| before/after diff)
set PYTHONPATH=. && python scripts/strategy_heatmap.py --runs logs/optimizer/ga_2p.jsonl --identity-baseline --n-players 2 --n-games 20 --out logs/optimizer/heatmap_ga2p

:: 7. Convergence / Pareto plots
set PYTHONPATH=. && python scripts/report_runs.py logs/optimizer/*.jsonl --out-dir logs/optimizer/reports

:: 8. Board renders (both shrunk and canonical 40-cell layout)
set PYTHONPATH=. && python scripts/render_all_boards.py        --out-dir ../report/figures/boards
set PYTHONPATH=. && python scripts/render_all_boards_legacy.py --out-dir ../report/figures/boards_legacy
```

Total wall-clock for the full experiment matrix: ~45-60 min on a single CPU.
Every output file has an accompanying `.meta.json` with all seeds and CLI
args so any reported number can be reproduced from a single meta file.

---

## What the optimiser actually finds

At _n_=1000 games per cell (cross-count-comparable composite, lower = better;
transfer/turn = $/round ÷ players, target 50):

| Design            | 2p score | 3p score | Fairness F̄ (2p) | Rounds (2p) | Draws (2p) | Transfer/turn (2p) |
|-------------------|---------:|---------:|-----------------:|------------:|-----------:|-------------------:|
| Default Monopoly  |    1.463 |    1.329 |            0.454 |       103.9 |      7.4 % |               24.8 |
| GA-2p winner      | **0.774** |    0.729 |            0.215 |        62.2 |      1.7 % |               36.7 |
| GA-3p winner      |    0.897 | **0.602** |            0.287 |        56.2 |      0.5 % |               34.7 |

> **Validated across agent classes and humans.** The agent-internal
> winners were re-checked with an independent LLM agent class
> (Qwen2.5-1.5B, cross-class agreement on every metric) and an _N_=10
> two-player human playtest pilot, where all four trained effects held in
> the same direction and magnitude and humans showed an even larger
> default-board failure than the simulators predicted. The full
> multi-subject playtest remains deferred (shortlist in the report
> appendix).

The composite score drops ~47% below the default in both regimes. Games
end in ~62 rounds instead of ~104. Draw rate falls by 80%+. The board
change also reshuffles which archetypes win: slow, cash-conserving
strategies move from middling to top-tier, notable because environment
tweaks usually can't shift strategy-level skill asymmetry.

Single-objective ablations confirm each term of the composite is doing
useful work: every one drives its own metric to (or near) its bound,
but always by degrading at least one other metric, so the combined
objective is a genuine multi-objective trade-off.

---

## Monopoly simulator (upstream, unchanged defaults)

This repository is forked from a detailed Monopoly simulator whose
original documentation on rules, parameters, and game mechanics is
preserved below. The optimiser uses that simulator as its inner loop.

<details>
<summary>Upstream simulator docs</summary>

The Monopoly Simulator does exactly what it says: it simulates playing a
Monopoly game with several players. It handles player movements on the
board, property purchases, rent payments, and actions related to
Community Chest and Chance cards. The resulting data includes the
winning (or, more precisely, "not losing" or "survival") rates for
players, game length, and other metrics.

The simulator allows for assigning different behavior rules to each
player, such as "don't buy things if you have less than 200 dollars" or
"never build hotels." Pitting a player with specific behaviors against
regular players allows for testing whether such strategies are
beneficial.

### Implemented rules

Based on Hasbro's official manual, with parameter tweaks possible.

### Default player behavior

- Buy whatever you land on.
- Build at the first opportunity.
- Unmortgage property as soon as possible.
- Get out of jail on doubles; do not pay the fine until you have to.
- Maintain a certain cash threshold below which the player won't buy,
  improve, or unmortgage property.
- Trade 1-on-1 with the goal of completing the player's monopoly.
  Players who give cheaper property should provide compensation equal to
  the difference in the official price. Don't agree to a trade if the
  properties are too unequal.

These defaults are overridden per-player by the `ParametricPlayerSettings`
dataclass in `player_settings.py` (17 configurable knobs) which powers the
30-strategy pool described above.

</details>

---

## Validation

The project's central claim, that agent feedback is _directionally_
predictive of real human play even when not literally accurate, was
validated with two independent checks beyond the rule-based pool:

- **LLM cross-class check (done).** `agents.LLMPlayer` runs a local
  Qwen2.5-1.5B-Instruct with a structured, echo-validated prompt for each
  buy decision. Both the rule-based pool and all-LLM seats agree on the
  direction of every metric, and the pool-driven board beats an
  LLM-driven board on 4 of 5 metrics under both evaluator classes, at
  zero per-decision LLM cost.
- **Human playtest (done).** An _N_=10 two-player pilot (default vs.
  GA-2p winner, matched seeds) confirmed all four trained effects in the
  same direction and magnitude; humans showed an even larger
  default-board failure than the simulators predicted. The full
  multi-subject study is specified as a falsification plan in the report
  appendix and remains deferred.

## Citation

If you use this codebase, please cite:

```
Selim Emir Can. "Beta-Testing Monopoly: Closed-Loop Game-Design
Optimisation over a Diverse Strategy Pool." CS348K project, 2026.
https://github.com/Selim-Emir-Can/CS348K-proj
```
