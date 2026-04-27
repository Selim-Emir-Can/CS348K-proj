# Project report

`report.tex` is a self-contained LaTeX article. All figures it cites are
already copied into `figures/` so the report directory is self-contained
and can be uploaded to Overleaf as-is.

```
report/
├── report.tex
├── README.md
└── figures/
    ├── convergence_2p.png
    ├── convergence_3p.png
    ├── ga_2p_pareto.png
    ├── ga_3p_pareto.png
    ├── heatmap_ga2p.png
    ├── heatmap_ga2p.diff.png
    ├── heatmap_ga3p.png
    └── heatmap_ga3p.diff.png
```

If you re-run the optimisation pipeline and want the report to reflect
the new figures, re-copy them into `figures/` (see the bottom of this
README) — `report.tex` doesn't watch the source directory.

## Compile

### Local (MikTeX / TeX Live)
```
cd report
pdflatex report.tex
pdflatex report.tex     # 2nd pass for refs
```

### Overleaf
1. Create a new project; upload `report.tex` and the entire `figures/` folder.
2. Compile with pdfLaTeX.

## Re-generating figures (only needed if data changed)

Step 1 — regenerate the source PNGs in `monopoly/logs/optimizer/`:
```cmd
:: From monopoly/
set PYTHONPATH=. && python scripts/report_runs.py ^
   logs/optimizer/random_2p.jsonl logs/optimizer/ga_2p.jsonl logs/optimizer/abl_*_2p.jsonl ^
   --out-dir logs/optimizer/reports_2p

set PYTHONPATH=. && python scripts/report_runs.py ^
   logs/optimizer/random_3p.jsonl logs/optimizer/ga_3p.jsonl logs/optimizer/abl_*_3p.jsonl ^
   --out-dir logs/optimizer/reports_3p

set PYTHONPATH=. && python scripts/strategy_heatmap.py ^
   --runs logs/optimizer/ga_2p.jsonl --identity-baseline ^
   --n-players 2 --n-games 20 --out logs/optimizer/heatmap_ga2p

set PYTHONPATH=. && python scripts/strategy_heatmap.py ^
   --runs logs/optimizer/ga_3p.jsonl --identity-baseline ^
   --n-players 3 --n-games 20 --out logs/optimizer/heatmap_ga3p
```

Step 2 — copy the regenerated PNGs into `report/figures/` (run from project root):
```cmd
copy monopoly\logs\optimizer\reports_2p\convergence.png report\figures\convergence_2p.png
copy monopoly\logs\optimizer\reports_3p\convergence.png report\figures\convergence_3p.png
copy monopoly\logs\optimizer\heatmap_ga2p.png      report\figures\
copy monopoly\logs\optimizer\heatmap_ga2p.diff.png report\figures\
copy monopoly\logs\optimizer\heatmap_ga3p.png      report\figures\
copy monopoly\logs\optimizer\heatmap_ga3p.diff.png report\figures\
copy monopoly\logs\optimizer\reports_2p\ga_2p_pareto.png report\figures\
copy monopoly\logs\optimizer\reports_3p\ga_3p_pareto.png report\figures\
```

## What's in the report

- **Sections 1-2** (Introduction + Problem definition): addresses grading
  criterion 1 (problem specificity).
- **Sections 3-4** (Approach + Evaluation): addresses criterion 2
  (evaluation thoughtfulness) — convergence, ablations, cross-evaluation,
  heatmaps.
- **Section 5** (Generalisation): eight transferable principles for
  multi-agent system design.
- **Section 6** (Limitations): explicit caveats with mitigations.
- **Appendix**: explicit cross-reference of report sections to the two
  grading criteria.
