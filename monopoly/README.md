# Monopoly: agent-driven closed-loop game-design optimisation

_Selim Emir Can (selimcan@stanford.edu) · CS348K project · 2026_

This directory holds the CS348K final project: a closed-loop workflow that
treats Monopoly as a parameterised multi-agent system and uses a diverse pool
of imperfect agents as design probes. A genetic algorithm searches a 66-dim
board-design space (per-property cost/rent multipliers + a 22-bit keep-mask
that shrinks the lap) against a composite objective (fairness, game length,
draw rate, inter-player money transfer), evaluated against a fixed pool of 30
rule-based strategies. The winning boards are then re-checked by an independent
agent class (an all-LLM seat pool, Qwen2.5-1.5B) and by an N=10 human playtest
pilot.

## Where the writeup lives

- **Final report (PDF):** [`report/report_with_playtest_shortlist.pdf`](report/report_with_playtest_shortlist.pdf)
  (source: [`report/report_with_playtest_shortlist.tex`](report/report_with_playtest_shortlist.tex);
  build with `lualatex`, since `microtype` needs scalable fonts).
- **Presentation slides (PPTX):** [`report/slides_final.pptx`](report/slides_final.pptx)
  (LaTeX/Beamer mirror: [`report/slides_final.tex`](report/slides_final.tex)).
- **Project overview & quickstart:** the repository-root
  [`../README.md`](../README.md) has the full pipeline diagram, run commands,
  and headline results.

## Code map

```
monopoly/
├── agents.py              # all player classes, including LLMPlayer
├── config.py              # GameConfig + YAML round-trip
├── player_settings.py     # 17-knob ParametricPlayerSettings dataclass
├── monopoly/core/         # vendored Monopoly game engine (board, player, game)
├── optimizer/             # design-space optimisation package
│   ├── design_space.py    #   66-dim vector <-> GameConfig encoder
│   ├── simulate.py        #   per-game stat collection (run_single_game)
│   ├── strategy_pool.py   #   30-strategy pool + evaluation-matchup sampler
│   ├── objectives.py      #   fairness / length / draw / money-transfer
│   └── search.py          #   random search + genetic algorithm
├── scripts/               # CLI drivers (optimise, cross-eval, heatmaps,
│                          #   LLM eval, plots, board renders)
├── prompts/               # canonical record of the LLM player prompt
├── report/                # final report (LaTeX + PDF) and slides
├── notes/                 # design notes and revert points
└── human_players_test/    # human playtest pilot logs
```

## Reproducing the experiments

Run from this directory with the project's Python environment (see the root
README for the full command list and dependencies). All optimiser outputs are
deterministic given a seed, and each run writes a `.meta.json` recording every
seed and CLI argument so any reported number can be reproduced from a single
meta file.
