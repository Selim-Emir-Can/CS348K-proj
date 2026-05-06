# CS348K Project Checkpoint 1

**Authors:** Selim Emir Can (selimcan@stanford.edu), Alaz Cig (alaz@stanford.edu)
**Date:** 2026-05-06 (deadline 2026-05-08)
**Repository:** [github.com/Selim-Emir-Can/CS348K-proj](https://github.com/Selim-Emir-Can/CS348K-proj)

This document is a self-contained snapshot of the project's *evaluation
plan* and *current status*. For canonical numerical results see
[`RESULTS.md`](RESULTS.md); for project context see
[`context.md`](context.md); for the in-progress writeup see
[`report/report_cs348k.tex`](report/report_cs348k.tex).

---

## 1. Project question

> **Can we automate the "beta testing" step of game design by using
> imperfect agent simulations as design probes, with cross-class
> agreement as the unit of evidence and human playtesting reserved for
> the cases where agent classes disagree?**

We instantiate the question on Monopoly because it is small enough to
parameterise end-to-end yet exhibits the same balance pathologies as
larger multi-agent systems (run-on games, single-property luck, two-
player asymmetry).

The *falsifiable hypothesis* the project tests is:

> A candidate board's directional ranking under a 30-strategy
> parametric rule-based pool matches its directional ranking under an
> independent agent class (a 1.5B-parameter LLM with a deterministic
> per-field state validator), at a per-decision hallucination rate
> below 1%.

This sits between "a single trained agent ranks designs" (insufficient
for fairness) and "agents predict human win-rates" (likely hopeless,
and not what a designer needs).

---

## 2. Evaluation plan

The evaluation is structured around **seven experiments**, each with
an explicit success criterion. All seven have completed; canonical
numbers in [`RESULTS.md`](RESULTS.md).

### 2.1 Default-board baseline (the "before" picture)

**Question:** what does an unoptimised Monopoly board score on the composite
metric `(fairness, F_max, length, draw rate, money transfer)`?

**Method:** evaluate the default 40-cell board on the 30-strategy pool
using 1,000 games × 10 matchups, both 2-player and 3-player.

**Success criterion:** evaluation code runs end-to-end and produces a
deterministic score with Wilson 95% CIs. (Establishing the noise floor
is required before any optimisation result can be claimed as a real
improvement.)

**Status:** ✅ **Done.** Code: `scripts/cross_eval.py --identity --n-games 1000`.
Output: `logs/optimizer_v3/cross_eval_mask.json`. Default board
scores: **1.463 at 2p, 1.229 at 3p** (numbers in
[`RESULTS.md`](RESULTS.md)).

### 2.2 Optimisation: GA vs random search

**Question:** does a genetic algorithm over the 66-dim board design
space (cost multipliers + rent multipliers + 22-bit keep-mask) find
boards that improve on the default by a margin larger than the noise
floor?

**Method:** GA at population 30 × 30 generations × elitism 2 vs
random search at the same evaluation budget (842 evals each), both
at 2 and 3 players. 200 games per candidate (20 games per matchup,
across 10 matchups), evaluated against the strategy pool.

**Success criterion:** GA's best-so-far composite score should drop
below the default-board reference within 200 evaluations, beat
random search at matched budget by a non-trivial margin, and produce
byte-identical metrics on re-run of the same `(seed, config)` triple
(reproducibility test).

**Status:** ✅ **Done.** Code: `scripts/optimize_board.py`. Trigger
scripts:
[`scripts/rerun_strategy_experiments.bat`](scripts/rerun_strategy_experiments.bat)
(short, pop 20 × 20, May 6 first cut) and
[`scripts/rerun_strategy_experiments_overnight.bat`](scripts/rerun_strategy_experiments_overnight.bat)
(long, pop 30 × 30 + n_games 200, May 6 overnight, **canonical**).
Both runs cross the default-board reference within the first $\sim
30$ evaluations. Headline from the canonical (overnight) run: the
GA finishes at $0.705$ at 2p and $0.599$ at 3p, beating random
search ($0.807$ and $0.694$) by **12.7\%** and **13.6\%** at
matched budget. Reproducibility verified by re-running the same
`(seed, config)` triple and diffing outputs.

### 2.3 Single-objective ablations (composite non-redundancy)

**Question:** does each of the five composite terms (fairness, max
fairness, length deviation, draw rate, money transfer) drive a
*different* optimum, or are some terms redundant?

**Method:** four single-objective ablation runs per player count
(`abl_fair`, `abl_len`, `abl_draw`, `abl_money`), with the chosen
weight set to 1 and all others to 0. Compare per-aspect heatmaps of
the optima.

**Success criterion:** if any two ablations drove the optimiser to
the same physical board, those columns of the per-aspect heatmap
would be near-identical. Visible inter-column variation is the
empirical evidence that the composite is non-redundant.

**Status:** ✅ **Done.** Per-aspect heatmap script:
`scripts/multiplier_plots.py`. Output JSONLs in
`logs/optimizer_v3/abl_*_mask.jsonl` (one per ablation × player
count). Existing figures (`report/figures/{cost_multipliers,
rent_multipliers,keep_mask}_{2p,3p}.png`) are from an earlier render
and visually still support the non-redundancy claim — each ablation
carves a different shape — but a re-render from `optimizer_v3` is
optional polish for the final report.

### 2.4 2p ↔ 3p cross-evaluation (does optimisation generalise?)

**Question:** does a board optimised for 2-player play hold up at 3
players, and vice versa? If not, by how much does it degrade?

**Method:** take the combined-objective winner from each player-count
regime and re-evaluate it under the *other* harness at 1,000 games.
Compare per-component metrics (rounds, fairness, draw rate, transfer)
to the corresponding default-board values.

**Success criterion:** improvement over default in the off-regime
should be at least 20% on the composite (i.e. the optimiser is not
catastrophically over-fitting to its training player count).

**Status:** ✅ **Done.** Code:
`scripts/cross_eval.py --runs ga_2p_mask.jsonl ga_3p_mask.jsonl`.
Output: `logs/optimizer_v3/cross_eval_mask.json`. Headline:
each design specialises to its training regime (GA-2p winner is
best at 2p with score $0.773$; GA-3p winner is best at 3p with
$0.641$). Off-regime gap is $\sim 16\!-\!24\%$ on the composite —
non-trivial but well above the success threshold of $20\%$
improvement over default. Improvement over default at the training
regime is $\sim 47\%$ in both player counts.

### 2.5 Per-strategy heatmap: *what* did the optimiser fix?

**Question:** the composite score averages over only 10 sampled
matchups. Is the GA actually fixing a broad set of strategy pairings,
or is it just exploiting matchup-specific quirks?

**Method:** compute the full 30 × 30 strategy-vs-strategy win-rate
matrix on (a) the default board and (b) the GA winner; subtract to
get the diff matrix. Look at mean |W − 0.5| as a population-wide
fairness summary.

**Success criterion:** the diff matrix should have visibly more
near-fair (light) cells than the default; mean |W − 0.5| should drop
or at minimum not increase.

**Status:** ✅ **Done.** Code: `scripts/strategy_heatmap.py`.
Output: `logs/optimizer_v3/heatmap_ga{2,3}p_mask.{json,npy,png}`.
Headline: mean $|W-0.5|$ across the full $30\times 30$ matrix drops
$\sim 5\%$ at 2p (0.21 → 0.20) and $\sim 21\%$ at 3p (0.26 → 0.20).
The most asymmetric pair after optimisation (\textsl{Trader} vs
\textsl{RailroadKing}, $100\%/0\%$) is immutable under any board
configuration in the search space — a separate, substantive
negative result that motivates strategy-side intervention.

### 2.6 LLM cross-class agreement (Phase C)

**Question:** when the same boards are evaluated by an *independent*
agent class (a 1.5B-parameter instruction-tuned LLM) instead of the
rule-based pool, does the directional improvement reproduce?

**Method:** run the all-LLM-seats game configuration on the default
board and the rule-based GA winner, 20 seeds each, with a structured
`STATE/ECHO/REASON/ANSWER` prompt and deterministic per-field state
validation. Measure mean rounds, mean transfer, and the LLM's
per-decision hallucination rate.

**Success criteria:**
1. Directional rounds + transfer effects on each board should match
   the rule-based pool's prediction on the same board.
2. First-pass hallucination rate < 1% (the validator-and-retry stack
   is part of the probe; we report failure modes rather than tuning
   them away).

**Status:** ✅ **Done.** Code: `scripts/eval_llm_on_boards.py`,
`scripts/analyze_llm_decisions.py`. Outputs: `logs/llm_eval/2p_v2/`,
`logs/llm_eval/3p_v2/`. Headline:
**0 first-pass hallucinations across 2,288 LLM calls** under the v2
prompt; rounds and transfer move in the same direction the rule-based
GA predicted (see [`RESULTS.md`](RESULTS.md#task-1-cross-class-agreement-rounds-transfer)).
Postmortem: [`notes/task1_postmortem_2026-04-29.md`](notes/task1_postmortem_2026-04-29.md).

### 2.7 LLM-driven GA + cross-evaluator gap (the diagnostic finding)

**Question:** what happens if the *LLM* is the evaluator instead of
the strategy pool? Does the LLM-driven optimum converge, and does it
generalise to the rule-based pool's evaluation?

**Method:** re-run the GA loop with all-LLM seats as the evaluator
(population 8, 5 generations, 5 seeds per candidate, 2-player only).
Take the LLM-GA winner board and cross-evaluate it under the
rule-based pool at 1,000 games. Compare fairness under each evaluator.

**Success criterion:** the LLM-GA winner improves over the default
under either evaluator (sanity check that the LLM-driven loop is
working at all). The diagnostic finding to *expose* is that a
single-personality evaluator misses worst-case fairness terms that
the diverse pool catches.

**Status:** ✅ **Done.** Code: `scripts/optimize_board_llm.py`.
Output: `logs/optimizer_llm/llm_ga_2p/`,
`logs/optimizer/cross_eval_llm_ga_winner.json`. Headline:
**fairness gap of +0.18 at 2p (0.20 LLM eval vs 0.379 pool eval) and
+0.35 at 3p (0.05 vs 0.404)**; the gap *widens* with player count.
This is the central empirical claim of the project.
Notes: [`notes/task2_results_2026-04-30.md`](notes/task2_results_2026-04-30.md),
[`notes/cross_eval_llm_ga_2026-04-30.md`](notes/cross_eval_llm_ga_2026-04-30.md).

---

## 3. Evaluation code: it runs end-to-end

Per the checkpoint requirement that "evaluation code can definitely say
empty pictures or white noise images are not successful":

- **Default-board reference is computable on demand** with a single
  command (`python scripts/cross_eval.py --identity --n-games 1000`),
  takes ~10 minutes on a single CPU, and produces the score 1.463 (2p)
  and 1.230 (3p) reproducibly.

- **The full pipeline (end-to-end) runs as a single script.** Two
  triggers exist: the short re-run
  [`scripts/rerun_strategy_experiments.bat`](scripts/rerun_strategy_experiments.bat)
  (pop 20 × gens 20, $\sim 25$ min) and the canonical overnight
  re-run
  [`scripts/rerun_strategy_experiments_overnight.bat`](scripts/rerun_strategy_experiments_overnight.bat)
  (pop 30 × gens 30 + n\_games 200, $\sim 2$h 23min). Each experiment
  writes its own log file under `logs/optimizer_v3/<run_name>.log`,
  and a top-level summary trail to `logs/optimizer_v3/summary.log`.
  Both completed for the May 6 checkpoint.

- **The LLM probe runs end-to-end** with a single command for each
  player count (`scripts/eval_llm_on_boards.py`). 80 games / 2,288
  LLM calls have been logged with full per-decision JSONL traces and
  ECHO-validation breakdowns.

- **Reproducibility is verifiable:** re-running any
  `(design vector, matchups, seeds)` triple produces byte-identical
  metrics. We have explicitly verified this by diffing two re-runs of
  the same config; both the rule-based and LLM pipelines are
  deterministic.

---

## 4. What's done vs what's pending

| Block | State |
|---|---|
| Default-board baseline | ✅ Done |
| Strategy pool (30 strategies) defined and serialised | ✅ Done |
| LLM probe: validator + retry + decision logging | ✅ Done |
| Phase C — LLM-only evaluation (Task 1) | ✅ Done |
| Task 2 — LLM-driven GA | ✅ Done |
| LLM-GA winner cross-eval under rule-based pool | ✅ Done |
| Rule-based GA + ablations + cross-eval (66-dim) | ✅ Done (2026-05-06 overnight v3) |
| Updated report numbers from re-run | ✅ Done (`RESULTS.md`, `report/report_cs348k.tex`) |
| Phase B — human playtest | 🚫 Pre-declared but not executed (out of checkpoint scope) |
| Final-report writeup | ✅ Pinned snapshot in `final_report/`; working draft in `report/report_cs348k.tex` |

---

## 5. Where everything lives

| Resource | Path |
|---|---|
| **Canonical numerical results** | [`RESULTS.md`](RESULTS.md) |
| **Project context (overrides any other notes)** | [`context.md`](context.md) |
| **Pinned final-report snapshot** | [`final_report/`](final_report/) |
| **Submission-format working draft** | [`report/report_cs348k.tex`](report/report_cs348k.tex) |
| **Older CVPR-style draft (reference only)** | [`report/report.tex`](report/report.tex) |
| Rule-based GA pipeline | `optimizer/`, `scripts/optimize_board.py` |
| LLM-only evaluation pipeline (Task 1) | `scripts/eval_llm_on_boards.py`, `scripts/analyze_llm_decisions.py` |
| LLM-driven GA (Task 2) | `scripts/optimize_board_llm.py` |
| 30-strategy pool (deterministic, fixed seed) | [`optimizer/strategy_pool.json`](optimizer/strategy_pool.json) |
| Re-run trigger scripts | [`scripts/rerun_strategy_experiments.bat`](scripts/rerun_strategy_experiments.bat) (short), [`scripts/rerun_strategy_experiments_overnight.bat`](scripts/rerun_strategy_experiments_overnight.bat) (canonical) |
| Per-decision LLM logs | `logs/llm_eval/2p_v2/decisions/` |
| Postmortem (LLM probe) | [`notes/task1_postmortem_2026-04-29.md`](notes/task1_postmortem_2026-04-29.md) |
| Notes (Task 2 + cross-eval) | [`notes/task2_results_2026-04-30.md`](notes/task2_results_2026-04-30.md), [`notes/cross_eval_llm_ga_2026-04-30.md`](notes/cross_eval_llm_ga_2026-04-30.md) |

---

## 6. Plan from now to the final report

1. ✅ **Rule-based GA re-run complete** (overnight v3, 16
   experiments, all converged cleanly).
2. ✅ **[`RESULTS.md`](RESULTS.md) updated** with the new numbers
   and propagated to the report tables, captions, and prose in one
   pass (no stale-but-cited gap).
3. ✅ **Convergence + Pareto + heatmap figures replaced** in
   `report/figures/` with the v3 versions.
4. ⏳ **Team-responsibility section** still has a `% TODO` placeholder
   for Alaz's contributions.
5. (Optional polish) re-render the per-aspect ablation heatmaps
   (`fig:multipliers`) from `logs/optimizer_v3/abl_*_mask.jsonl`. The
   existing PNGs still support the non-redundancy claim visually.
6. (Stretch, post-checkpoint) pilot Phase B human playtest with a
   small group as a sanity check on the cross-class-agreement claim.
   Not required for the May 8 checkpoint or the final report, but
   would strengthen the falsifiability path.
