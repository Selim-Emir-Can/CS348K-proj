# Plan v2 — Apr 26 forward

Supersedes: `plan.md` (week-1-2 sketch from the original DQN + BoTorch
proposal era; retained for history).

**Status.** Advisor approved the current report for full marks at the
2026-04-24 meeting. Everything below is upside: workshop-paper-quality
framing and evidence on top of an already-passing project.

**Anchor paper.** Isaksen, Gopstein, Togelius, Nealen (2018), *Exploring
Game Space of Minimal Action Games via Parameter Tuning and Survival
Analysis*, IEEE Transactions on Games 10(2). Recommended by the advisor
in the 2026-04-26 meeting. Provides the framework the project has been
improvising toward.

## Reframe

The contribution moves from *"we built a board optimizer"* to *"we built
a game-space exploration framework for an economic multi-agent board
game, demonstrating the four canonical operations: 1-D parameter sweep,
multi-objective optimization, novelty search, and cross-source validation
against humans."*

Player-model adaptation made explicit in the report: Isaksen models human
dexterity via timing-precision sigma; Monopoly has no dexterity dimension,
so we substitute strategic-archetype diversity (30 ParametricPlayers + 1
LLM) as the player population. This is a contribution, not a regression.

## In scope this round (executable now)

### Code (parallelizable, ~5 days total)

1. **Survival-analysis hazard plots (three curves per board).** Apply the
   Isaksen survival-analysis treatment to monopoly outcome data. Three
   complementary hazards, one figure per hazard, five curves per figure
   (default, GA-2p winner, GA-3p winner, salary x2, Drop Brown).
   Reproducible from `scripts/hazard_curves.py` with a fixed seed.
   Reuses outcome data already collected by the simulator.

   - **(1a) Game-end hazard `h(t)`.** Rate at which games end at turn
     `t`, given still in progress at turn `t`. Pacing dynamics. Direct
     analog of Isaksen Fig 9-13.
   - **(1b) Cash-level bankruptcy hazard `h(c)`.** Cross-sectional
     Markov hazard: across all `(player, turn)` pairs in the
     simulation, condition on current cash bin `c`, compute the rate
     of bankruptcy on the very next turn. Player-experience analog:
     "is this board cruel or forgiving when you're losing?"
   - **(1c) Time-to-first-monopoly hazard `h(t)`.** Rate at which the
     first colour-set monopoly forms at turn `t`, given no monopoly
     has formed yet. Tells you whether monopolies form on a timer
     (constant hazard) or accumulate (increasing hazard). Reuses the
     already-pre-declared "time to first monopoly" Phase A statistic.

2. **Novelty-search demo.** Reuse GA infrastructure with a swapped
   objective: maximize minimum pairwise Euclidean distance in normalized
   66-dim design space across `k=3` candidate boards, subject to combined
   score `< 1.2` (filters degenerates). Render the three boards in legacy
   and shrunk layouts; report a per-metric profile table. Direct analog
   of Isaksen Fig 21-22 / "Frisbee Bird." Reproducible from
   `scripts/novelty_search.py`.

3. **LLM character induction by environment.** Hold the LLM completely
   fixed — single model (Qwen2.5-1.5B-Instruct, already in use per
   Phase A), single system prompt with no personality scaffolding,
   same instance in every seat. Vary only the board. The experimental
   question: *does the same instrument exhibit different strategic
   character in different environments, and is that character visible
   in its reasoning?* This is the agent-as-mirror move spelled out
   directly: the world is the variable, the agent is the measurement
   instrument.

   Setup:
   - Self-play: same LLM in both seats per game (clean attribution —
     any reasoning shift is induced by the game state, not opponent
     strategy).
   - Per-decision reasoning emission: the LLM is prompted to verbalize
     its reasoning *before* committing each action (chain-of-thought
     style: "given my cash, owned properties, opponent state, and the
     decision in front of me, my reasoning is: …").
   - 12 self-play games × 5 hazard boards (default, GA-2p winner,
     GA-3p winner, salary x2, Drop Brown) = 60 games.
   - Reproducible from `scripts/llm_character.py`. Reasoning logs
     dumped to JSONL alongside action logs.

   Compute budget: per-decision reasoning generation multiplies LLM
   call cost vs Phase A's classification-only setup. Estimated 8-12
   min/game (vs Phase A's 85 sec/game), so 60 games ≈ 8-12 hr
   wall-clock. One overnight run on the available 1.5B-capable GPU.
   If overruns, fall back to 6 games/board (≈ 4-6 hr) or 0.5B Qwen
   (faster but less interpretable text).

   Two complementary analyses, both novel relative to the existing
   report:
   - **(3a) Per-board reasoning corpus.** Aggregate all per-decision
     reasoning text per board. Both qualitative read (sample 20
     reasoning blocks per board, characterize the dominant strategic
     framing in plain language) and lightweight quantitative features
     (word/concept frequency over a small dictionary: "cash,"
     "risk," "monopoly," "trade," "rent," "bankrupt," "accumulate";
     average reasoning length; sentiment proxy). Output: one
     paragraph + one figure (small-multiple bar charts of concept
     frequencies, one panel per board).
   - **(3b) Cross-board character divergence.** The headline
     analysis. Same LLM, same prompt, different boards → what shifts
     in how it talks about the game? Compute pairwise divergence
     (KL or simple L1 over the concept-frequency distributions)
     between board reasoning corpora; identify the board pair with
     maximum divergence. Headline framing target: *"On board X the
     LLM reasons primarily about acquisition; on board Y the same
     model reasons primarily about risk-management. The board
     induces the character — the agent is a mirror for the
     environment's strategic structure."* Falsifiable if reasoning
     corpora are statistically indistinguishable across boards
     (which would itself be a calibration finding).

### Writing (~1 week)

4. **Reframe `\section{Introduction}` and `\section{Generalisation}`.**
   Position the contribution as a game-space exploration framework, cite
   Isaksen 2018 as conceptual ancestor, and explicitly own the
   DQN→parametric-pool pivot ("for design *feedback* a diverse
   parametric pool is more interpretable and ~100x cheaper to iterate
   than retraining a single agent").
   Trim `\section{Generalisation}` from 8 principles to 5 strong ones;
   cut #2 (worst-case term — well-known), #3 (run ablations — basic
   methodology), #5 (cross-eval — basic). Add three new principles:
   "Cross-source agreement structure is the trust signal,"
   "Game-space exploration is not single-objective optimization," and
   "The environment induces the agent's character — a fixed agent
   exhibits different reasoning under different environmental
   pressures, making the agent a mirror for the environment's
   strategic structure."
   Drop the proposal-era "What if Eye?" co-evolution framing.

5. **New `\section{Game-Space Exploration}`.** Inserts between
   `\section{Evaluation}` and `\section{Validation plan}`. Four
   subsections, each demonstrating a distinct game-space operation:
   - **5a. 1-D parameter sweep (Phase A promoted from `notes/`).**
     Pre-declared knobs (salary x2, Drop Brown, orange rent x2) on
     the 4x4 mini-board; RB and LLM curves; cross-class agreement
     structure annotated per knob.
   - **5b. Survival-analysis hazards.** Three hazard figures
     (game-end, cash-level, time-to-monopoly) with five board curves
     each. Predicted result to test: salary x2 produces a *decreasing*
     game-end hazard (Phase A draw-rate finding restated in
     survival-analysis terms).
   - **5c. Novelty search.** Three maximally-different non-degenerate
     boards plus per-metric profile.
   - **5d. LLM character induction by environment.** Fixed LLM
     (single Qwen2.5-1.5B, no personality scaffolding), self-play on
     each board, per-decision reasoning emission. Per-board reasoning
     corpus characterization plus cross-board character-divergence
     analysis. Headline finding shape: *the same instrument exhibits
     qualitatively different strategic mindsets across boards; the
     board induces the character.*

6. **Tighten `\section{Validation plan}`.** Stays prospective — no
   actual human data this round. Strengthen pre-commitment per the
   methodology block below so it reads as a ready-to-execute experiment
   with explicit falsification criteria, not aspiration.

7. **Final integration pass.** Top-to-bottom read for forward references,
   redundant paragraphs across `\section{Introduction}` /
   `\section{Generalisation}` / `\section{Validation plan}`, figure
   numbering, and a tightening pass on the abstract to reflect the new
   game-space framing. Add Isaksen 2018 to the `bib`.

## Pre-committed for the next round (do not execute now, but commit the methodology)

The human-validation phase is held but its design is locked in writing
so a future round can execute exactly the experiment the report
pre-declares.

### Boards under test

- **Default mini-board** (control)
- **Salary x2** — chosen to test the cross-source AGREEMENT signal.
  Phase A shows RB and LLM both predict shorter games and substantially
  more draws (RB: 71.9 -> 47.8 rounds, 50% -> 90% draws; LLM: 69.5 ->
  39.7 rounds, 50% -> 75% draws). If humans agree, the framework's
  strongest signal is validated.
- **Drop Brown** — chosen to test the cross-source DISAGREEMENT signal.
  Phase A shows RB and LLM disagree structurally and the gap *grows*
  with `n` (rules out noise): RB sees no-op, LLM sees a decisive effect
  on every metric. Whichever side humans land on becomes the "humans
  break the tie" finding.

### Sample design

- 3 testers x 3 boards x 2 games = 18 mini-board games.
- Mini-board target wall-clock: ~10 min/game = ~5 hr total human time
  (60 min/tester).
- Pre-declared per-game schema (matches Phase A): `rounds`, `draw`
  (boolean truncation flag), `bankruptcies` (count at game end),
  `winner_id`, `transfer_total` (sum of inter-player rent + trade
  cash deltas, recorded by referee on the score sheet).
- Optional 5-point Likert per game on perceived fairness, length,
  decisiveness.

### Cross-source decision matrix (pre-committed)

For each knob, the joint outcome `{RB direction, LLM direction, human
direction}` maps to one of:

- All three agree → the framework's prediction is validated for
  that knob.
- RB and LLM agree, human disagrees → calibration limit found;
  document as "this is the kind of question agents systematically
  miss."
- RB and LLM disagree, human breaks tie → headline methodological
  finding ("which agent class to trust on which dimension").
- All three disagree → "humans at `n=6` lack the statistical power to
  break this tie; here is the `n` that would be required."

All four outcomes are publishable. The experiment cannot embarrass us.

### Falsification criterion

The framework is judged decision-useful if, on Drop Brown, the human
direction matches at least one of {RB, LLM} on at least 3 of 5
pre-declared metrics. The framework is judged not decision-useful if
human direction contradicts both RB and LLM on a majority of metrics.

## Deferred / out of scope

- DQN agent: confirmed retired per advisor blessing of the parametric
  pool.
- 2D playability maps (Isaksen Fig 17-18): future work; doesn't pay for
  itself at 66 dims with stochastic outcomes.
- Co-evolutionary "What if Eye?" framing from the original proposal:
  scrubbed.
- Full canonical-board human playtests (60-90 min/game): not in scope
  for any near-term round.

## Definition of done per item

- **Hazard plots:** three PDF figures (game-end, cash-level,
  time-to-monopoly), five curves per figure, integrated into
  `\section{Game-Space Exploration}`, regenerable from
  `scripts/hazard_curves.py` with a recorded seed.
- **Novelty search:** three-board PDF figure (legacy + shrunk renders)
  plus per-metric profile table, regenerable from
  `scripts/novelty_search.py`.
- **LLM character induction:** raw outcome + per-decision reasoning
  JSONL per (board, game, turn, player) cell; per-board reasoning
  corpus summaries (sample of 20 reasoning blocks per board with
  qualitative characterization); concept-frequency bar chart figure
  (one panel per board, shared concept dictionary across panels);
  cross-board divergence figure or table identifying the maximum-
  divergence board pair; regenerable from `scripts/llm_character.py`
  with a recorded seed and the single neutral system prompt pinned
  in the script.
- **Reframe:** Isaksen citation present in `bib`; abstract reflects
  game-space framing; `\section{Generalisation}` has 7 principles, not
  8; "What if Eye?" reference removed.
- **New `\section{Game-Space Exploration}`:** ~1200-1600 words, 6-8
  figures (3 hazards + 1 novelty board grid + 1 sweep panel + 1-2 LLM
  personality figures), references all four code artifacts.
- **Pre-committed validation:** decision matrix above is in the report
  body; falsification criterion is explicit.
- **Report:** compiles cleanly on Overleaf; final integration pass done.

## Post-build compute observations (added 2026-04-27)

After Steps 0-7 of the LLM design-loop expansion shipped, the actual
wall-clock numbers on the dev host (Windows 11, single-thread, no GPU
work yet) diverged sharply from the original estimates. Updating the
plan so future-readers calibrate against measured reality, not the
pre-build guesses.

### What was measured

| What                                            | Estimated         | Measured           | Notes |
|-------------------------------------------------|-------------------|--------------------|-------|
| Step 0 transfer audit (5 designs × 2 boards × n=100 = 1000 games) | ~30 min       | **2.1 s**           | Each game with 2 ParametricPlayers + balanced seats runs at ~500 games/sec single-threaded. |
| Per-eval cell at n=100 on canonical             | implicit "high"   | **~0.2 s**          | Same throughput as the audit; the parametric pool eval is genuinely cheap. |
| Subprocess sandbox overhead (#5)                | ~unstated         | **~50-100 ms/cell** | Measured from the 20-test suite running 2 subprocess round-trips in 0.5 s. Per-cell cost is real but small relative to game time at n≥10. |
| Step 5 rule loop smoke (heuristic, 1 board, 3 trajectories: LLM/random/house, n=50, K=3) | a few min implicit | **~3 s**           | All three trajectory types complete inside one wall-clock second per iteration excluding subprocess startup. |
| Bootstrap CI (500 resamples) per iteration      | not estimated     | **~0.5-1 s**        | Cost is in the resampling, not the eval; comparable to one extra n=10 eval. |

### What is still unverified

Everything on the LLM-generation side. The smokes above all ran on the
**heuristic** backend (canned cycle for the design loop, fixed cycle
for the rule loop). No Qwen 1.5B forward pass and no Sonnet API call
was timed end-to-end as part of this build.

  - **#3 LLM character induction**: original 8-12 hr overnight estimate
    on a 1.5B-capable GPU stands. Untouched.
  - **#4 parametric loop (Qwen 1.5B)**: CEO plan estimated 1-3 hr
    wall-clock. The eval portion is now known to contribute negligibly
    (~24 s total across all 15 trajectories at K=8, n=100). The 1-3 hr
    estimate is therefore *almost entirely* Qwen generation latency,
    which depends on the available GPU. Worth re-confirming on the
    target host before committing to the full 15-trajectory run.
  - **#5 rule loop (Claude Sonnet)**: CEO plan estimated 1-2 hr wall-
    clock + $5-30 API. Eval portion now known to contribute ~minutes,
    so wall-clock is dominated by Sonnet round-trip latency (typically
    1-3 s/call) × K iterations × seeds × boards. At K=6, 3 seeds, 5
    boards, ~2 calls/iter (one for the patch + one retry budget) =
    ~180 calls × 2 s = ~6 min API time. The 1-2 hr estimate looks
    *very* generous; expect closer to 15-30 min wall-clock for v1
    barring rate limits.

### Implication for the plan

The eval-cost portion of the original CEO compute estimate over-
budgeted by ~1-2 orders of magnitude. The corrective move is **not**
to claim more iteration capacity (the LLM generation is still the
bottleneck and still unverified) but to relax some of the in-script
caps:

  - n_games per iteration could be raised from 100 → 250-500 for
    tighter improvement CIs (LOCK §5 demands 95% CI excludes prev
    score) without meaningfully changing wall-clock.
  - n_bootstrap on per-iteration CI could be raised from 500 → 2000
    for sub-percent precision on the improvement gate.
  - K could grow from 8 → 12 for the parametric loop if early seeds
    show non-convergence.

These are knobs the user can turn at run-time; the script defaults
remain at the original values so reproducing CEO-plan numbers exactly
is still possible.

## Risks

- **Hazards are shapeless.** Even a flat or noisy hazard is a finding
  ("monopoly game-end is approximately memoryless in turn count given
  these cash dynamics, contrary to dexterity-game intuition"). Each
  plot earns its keep regardless of shape. If all three hazards
  collapse to flat, drop one of them and use the saved figure budget
  for the personality matrix.
- **Novelty search returns degenerate boards.** Tighten the
  score-threshold gate; if still degenerate, fall back to `k=2` with
  manually seeded diversity targets (one shrinkage-heavy, one
  rent-multiplier-heavy).
- **LLM reasoning is generic / templated across boards.** Real risk at
  1.5B params: the model may produce reasoning that is qualitatively
  similar regardless of board state, washing out the cross-board
  signal. Mitigation: prompt the LLM to reason explicitly about
  *current cash, current property holdings, opponent state, and
  pending decision* — i.e., force it to ground reasoning in observable
  game state. If reasoning still collapses to similar text across
  boards, the finding inverts: "the LLM at this scale produces
  context-invariant strategic narration; environmental induction is
  not visible at this model size." Calibration finding, still
  publishable, motivates the same experiment at larger model size as
  obvious future work.
- **LLM compute budget overruns.** 8-12 hr is tractable as one
  overnight run but unforgiving. Reduce to 6 games/board (~4-6 hr),
  drop to 0.5B Qwen (faster, less interpretable text), or trim
  reasoning to ~50-token max per decision before pulling the
  experiment entirely.
- **Reasoning text is hard to analyze rigorously.** The qualitative
  characterization risks subjectivity. Mitigation: pre-commit the
  concept dictionary (e.g., {"cash", "risk", "monopoly", "trade",
  "rent", "bankrupt", "accumulate", "defensive"}) before looking at
  the corpora; report quantitative concept-frequency tables alongside
  the qualitative read; include 5-10 raw reasoning quotes per board
  in an appendix so the reader can audit the characterization.
- **Reframe makes existing prose feel out of place.** Keep the existing
  evaluation and approach sections intact; the reframe touches only the
  bookend sections (intro, generalisation, abstract) plus the new
  game-space section.
