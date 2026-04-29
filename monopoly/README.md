# Beta-Testing Monopoly: Closed-Loop Game-Design Optimisation

_Selim Emir Can (emirc) and Alaz Cig · CS348K project · 2026_

This repository automates the beta-testing loop that game and system designers
usually run by hand. We treat Monopoly as a parameterised multi-agent system
and build a closed-loop optimiser: a diverse rule-based agent pool plays on
candidate boards while a genetic algorithm searches a parameterised
environment space to minimise a composite of fairness, game length,
decisiveness, and inter-agent interactivity. Applied to Monopoly, the
optimiser cuts the combined score ~40% below the default board and surfaces
transferable principles for automating beta-testing of any multi-agent system.

**The full write-up** (with convergence curves, ablation matrices,
cross-evaluation at _n_=1000, per-strategy heatmaps, and qualitative board
renders) is in [`../report/report.tex`](../report/report.tex).

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

From inside `monopoly/`:

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

At _n_=1000 games per cell (Wilson 95% CI of ±3pp on individual win rates):

| Design            | 2p score | 3p score | Mean fairness (2p) | Rounds (2p) | Draws (2p) | Transfer/round (2p) |
|-------------------|---------:|---------:|-------------------:|------------:|-----------:|---------------------:|
| Default Monopoly  |    1.463 |    1.230 |              0.454 |       103.9 |      7.4 % |                 $50 |
| GA-2p winner      |    **0.82** |    0.72 |              0.24  |        62.5 |      1.3 % |                 $75 |
| GA-3p winner      |    0.92  |    **0.79** |              0.25  |        63.8 |      0.4 % |                 $66 |

> **Pending human validation.** All numbers above are internal to the
> agent loop. Whether these designs play the way the agents predict for
> real human players is the subject of an ongoing validation effort
> (simplified-board sanity check + human playtest + LLM-agent
> cross-check). See `notes/kayvon_meeting_2026-04-24.md` for the
> validation plan and `report/report.tex` §7 for the formal write-up.

The composite score drops ~40% below the default in both regimes. Games
end in ~62 rounds instead of ~104. Draw rate falls by 80%+. Per-strategy
structural asymmetry (mean `|W-0.5|` over the 30×30 matchup matrix) also
drops from 0.21 to 0.17 on 2p and 0.26 to 0.21 on 3p — notable because
environment tweaks usually can't shift strategy-level skill asymmetry.

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

## Roadmap

The current codebase covers the agent-internal optimisation loop. The
project's central claim (agent feedback is _directionally_ predictive of
real human play, even when not literally accurate) is being validated in
three phases. See `notes/kayvon_meeting_2026-04-24.md` and §7 of the
report for full detail.

- **Phase A: simplified-board sanity check.** A 4×4 (16-cell) variant
  of Monopoly with 8 colour-group properties across 4 groups lives at
  `configs/mini/`; see `optimizer/simulate.py` for usage. Pre-declared
  knobs to validate on this board: salary level, removing one colour
  group, doubling rent on one expensive group.
- **Phase B: human playtest.** 3-5 testers playing 5-10 games per
  board. Default vs. GA-optimised winner. Falsification criterion
  pre-declared in `report/report.tex` §7.2.
- **Phase C: LLM-agent cross-check.** `agents.LLMPlayer` calls a local
  Qwen2.5-0.5B-Instruct (or any OpenAI-compatible endpoint) for each
  buy decision. Used as a third independent signal alongside rule-based
  agents and humans.

## Citation

If you use this codebase, please cite:

```
Selim Emir Can and Alaz Cig. "Beta-Testing Monopoly: Closed-Loop
Game-Design Optimisation over a Diverse Strategy Pool." CS348K project,
2026. https://github.com/Selim-Emir-Can/CS348K-proj
```
