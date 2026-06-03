# Mapping to the CS348K grading questions

> Removed from the main report (`report_with_playtest_shortlist.tex`,
> formerly Appendix B) on 2026-06-02. Kept here for reference.

The CS348K final-report guidelines ask two specific questions. We state both
verbatim and answer each in one paragraph; the body of the report is structured
around exactly these two questions, so this is a quick reader guide.

## (1) "What are the questions or goals your project aims to answer?"

We ask whether a diverse agent pool can serve as a beta-testing surrogate for
human playtesting in game design under two qualifications: (a) the agents are
imperfect and acknowledged as such, and (b) the unit of evidence is
*cross-class agreement*: two independent agent classes (a 30-strategy parametric
rule-based pool and a single-personality LLM with structured state validation)
ranking the same candidate boards in the same direction. The falsifiable
hypothesis is two-fold: (a) the directional ranking under the rule-based pool
matches the directional ranking under the LLM seats on the same boards; and
(b) the same directional ranking also holds in real human play, so the pool
acts as a proxy for human behaviour on the design axes the optimiser targets.
An N=10 two-player playtest pilot (Section: human playtest pilot) is the
empirical check on (b); a fuller multi-subject playtest is published as a
falsification plan (Appendix: playtest shortlist). The full statement is in the
Background and setup section.

## (2) "What experiments should be done to answer that question, and how will you know from the outcome of the experiment that you have succeeded?"

Nine experiments, each with an explicit success criterion:

1. **Default-board baseline at n=1000** (baseline section). *Success:*
   deterministic scores with 95% Wilson intervals on every metric.
2. **GA vs. random search at matched ~842-evaluation budget** (baseline
   section). *Success:* GA crosses the default reference within budget, beats
   random by a non-trivial margin, and is bit-reproducible. *Outcome:* GA wins
   by 13% in both player counts.
3. **Single-objective ablations** (ablation section). *Success:* visibly
   different per-aspect optima, confirming the composite is non-redundant.
   *Outcome:* confirmed (see the multipliers and board-grid figures).
4. **2p↔3p cross-evaluation at n=1000** (cross section). *Success:* improvement
   over default ≥ 20% in the off-regime; quantify the specialisation gap.
   *Outcome:* both designs improve over default by ~47% in the training regime;
   off-regime gap is 16–24% on the composite.
5. **Archetype WR reshuffle** (heatmap section). *Success:* the GA winner
   produces an interpretable, non-trivial change in which named archetypes are
   top-tier vs. middling vs. bottom-tier, with the systematic pattern of which
   matchups are immutable surfaced honestly. *Outcome:* the GA winner reshuffles
   middle-of-distribution archetypes (Δ up to ±0.18 in win-rate vs. the field,
   e.g. CashHoarder +0.18, HighCostOnly −0.18); the top-3 default strategies
   stay top-3 on the GA winner; Trader vs. RailroadKing (100/0) is genuinely
   immutable under any board configuration in the search space.
6. **LLM cross-class probe** (LLM section). *Success:* rounds and transfer move
   in the same direction the rule-based pool predicted on the same boards;
   per-decision hallucination rate below 1%. *Outcome:* 0/2,207 first-pass
   hallucinations under v2 ECHO validation; direction agrees on every measured
   metric at both 2p and 3p (e.g. rounds ~−40%, transfer per player-turn ~+60%
   at 2p).
7. **LLM-driven GA + cross-class verdict** (LLM-GA section). *Success:* the
   LLM-driven loop converges to a non-trivial design; cross-evaluating it
   against the rule-based GA winner under both evaluator classes resolves which
   is the better design. *Outcome:* both evaluator classes prefer the rule-based
   GA winner on 4 of 5 stable metrics (composite, F̄, F_max, length) at *both*
   2p and 3p (n=100 LLM seats each); the LLM-driven board wins only on transfer
   per player-turn, the metric its n=5 training overfit to. This is the central
   empirical claim of the paper.
8. **Human playtest pilot** (playtest pilot section). *Success:* the design
   constraints specified in the agent setting are satisfied in real human play,
   in the same direction and similar relative magnitude as under the agent
   evaluators, and agent feedback is conservative rather than optimistic.
   *Outcome:* N=10 two-player matched-seed pilot; ~50 vs. 131 rounds, $66 vs.
   $39 per round, 0% vs. 40% draws on the GA winner vs. the default, with humans
   showing a *larger* default-board failure than either simulator (40% draws
   vs. 0–7% in pool/LLM). The agent-driven predictions are conservative rather
   than optimistic.
9. **Playtest shortlist** (playtest-shortlist appendix). *Success:* a small,
   falsifiable selection of boards from the GA shortlist, each contrasting a
   different design lever against the default, with a stated pilot protocol.
   *Outcome:* 4 boards selected (default, GA-2p combined winner, GA-3p combined
   winner, money-only 2p optimum) with one-sentence design hypotheses and a
   proposed pilot (N=8, counterbalanced, Likert plus forced-choice). The fuller
   multi-subject playtest is published as a falsification plan, not a completed
   study.
