# Final Report — pinned snapshot

**Pinned: 2026-05-06.** This folder is a self-contained copy of the
final report at the v3 results state. All figures are baked in;
opening `report.tex` in Overleaf or compiling with `pdflatex` from
this directory will reproduce the report exactly as cited in
[`../CHECKPOINT_1.md`](../CHECKPOINT_1.md).

## Contents

```
final_report/
├── README.md            this file
├── report.tex           LaTeX source (CS348K format)
└── figures/             every PNG cited by report.tex
    ├── convergence_2p.png, convergence_3p.png
    ├── cost_multipliers_2p.png, rent_multipliers_2p.png, keep_mask_2p.png
    ├── ga_2p_pareto.png, ga_3p_pareto.png
    ├── heatmap_ga2p.png, heatmap_ga2p.diff.png
    └── llm/
        ├── fig_v1_vs_v2_hallucination.png
        ├── fig_cross_class_agreement.png
        ├── fig_llm_buy_rate_slices.png
        ├── fig_llm_ga_convergence.png
        ├── fig_llm_ga_score_distribution.png
        ├── fig_cross_evaluator_gap.png
        └── fig_fairness_asymmetry.png
```

## Provenance

- Rule-based GA + ablations + cross-eval + heatmap + Pareto figures:
  `logs/optimizer_v3/` (overnight run, 2026-05-06).
  Trigger: `scripts/rerun_strategy_experiments_overnight.bat`.
  Protocol: `--pop 30 --generations 30 --elitism 2 --n-games 200`,
  `--base-seed 42 --search-seed 0 --matchup-seed 1234`,
  `--max-turns 200 --removal-direction cheapest`,
  weights `(1.0, 0.5, 0.5, 0.3, 0.3)`, targets $R^*=60$, $T^*=100$.
- LLM cross-class probe (Phase C) figures:
  `logs/llm_eval/{2p_v2, 3p_v2}/` (April 29).
- LLM-driven GA (Task 2) figures:
  `logs/optimizer_llm/llm_ga_2p/` (April 30).
- LLM-GA cross-eval (the headline diagnostic):
  `logs/optimizer/cross_eval_llm_ga_winner.json` (April 30).

## Headline numbers

- Default board: composite score $1.463$ at 2p, $1.229$ at 3p.
- GA-2p winner under 2p eval: $0.773$ (a $47\%$ improvement;
  $\overline{F} = 0.215$).
- GA-3p winner under 3p eval: $0.641$ (a $48\%$ improvement;
  $\overline{F} = 0.280$).
- GA outperforms random search at matched budget by $\sim 13\%$ in
  both regimes.
- LLM probe: 0 first-pass hallucinations across $2{,}288$ calls
  under structured ECHO validation.
- LLM-driven GA winner cross-eval: fairness gap $+0.18$ at 2p
  ($0.20$ LLM eval vs $0.379$ pool eval), $+0.35$ at 3p ($0.05$ vs
  $0.404$). Gap widens with player count — the central diagnostic
  of the project.

Full numerical specification in [`../RESULTS.md`](../RESULTS.md).

## How to compile

In Overleaf, upload this folder as a new project and set
`report.tex` as the main document. Or locally with MiKTeX or
TeXLive: `pdflatex report.tex` (twice for cross-references).
