# Proposed experiment: fairness under heterogeneous (mixed-class) play

**Status:** proposed, not yet run. Documented as a future / nice-to-have
evaluation. Not required for the current results or the stated hypothesis.

## Question

All existing evaluations seat a *single* agent class per game: all-pool
games (30-strategy rule-based pool) or all-LLM games (Qwen2.5-1.5B in every
seat). We then compare the two classes' verdicts *across separate games*.

This experiment asks a different, complementary question:

> Does a board the rule-based pool deems fair stay fair under
> **heterogeneous** play, when agents of different classes sit at the same
> table?

Real player populations are heterogeneous (mixed skill, mixed style), so a
board that is only fair when everyone plays the same way is a weaker design
than one that is fair under a mix.

## Seating configurations

- **2 players:** LLM vs. rule-based-pool agent.
- **3 players:**
  - LLM vs. LLM vs. pool,
  - LLM vs. pool vs. pool.

For each configuration, the pool seat(s) are drawn from the existing
30-strategy pool using the same matchup-selection discipline as the
single-class evaluations.

## The confound to control (important)

Fairness is measured as win-rate spread between seats. In a same-class game
that spread reflects the **board**. In a mixed-class game it reflects
**board design + the skill gap between classes**. If the 1.5B LLM is simply
a weaker player than a scripted aggressive builder, *every* board will look
"unfair" in mixed games, which says nothing about the board.

**Control:** hold the seating mix fixed and use shared random numbers (CRN),
then compare metrics **across boards** rather than in absolute terms. The
constant cross-class skill gap cancels in the board-to-board comparison, so
any *difference* between boards is a real board effect. Absolute mixed-class
fairness is not interpreted as a board signal.

## Boards

At minimum: `default`, `ga_2p_winner`, `ga_3p_winner`. Optionally extend to
the single-objective ablation winners if budget allows.

## Metrics

Per board and per configuration, with CRN and Wilson CIs:
- fairness (win-rate spread across seats),
- mean rounds,
- inter-player money-transfer rate.

Reported as differences across boards (e.g. GA-winner vs. default under the
*same* mix), never as absolute cross-class fairness.

## Protocol

- Reuse `scripts/eval_llm_on_boards.py` style harness, extended to accept a
  per-seat class assignment (LLM vs. pool) instead of all-LLM.
- Same seeds across boards and across configurations (CRN), so an unchanged
  `(board, seating, seeds)` triple yields byte-identical metrics.
- Deterministic greedy LLM decoding with the existing
  STATE/ECHO/REASON/ANSWER validator, as in the current LLM eval.

## Cost

LLM decisions dominate runtime (~6--10 s/decision on a 12 GB GPU). Mixed
games have fewer LLM seats than all-LLM games (1--2 vs. all), so per-game
cost is lower, but the number of (board x configuration) cells multiplies.
Scope to start: 2p LLM-vs-pool on {default, ga_2p_winner} with ~20 seeds,
then add the 3p mixes only if the 2p result is informative. Rough estimate:
a few GPU-hours for the scoped 2p version; comparable to the existing
Phase C eval for the full set.

## Expected outcomes and interpretation

- If GA-winner boards keep their fairness advantage over default *under the
  same mix*, that strengthens the robustness claim: the design helps even
  under heterogeneous play.
- If the advantage disappears under mixing, that is an honest negative
  result worth reporting: same-class fairness does not transfer to mixed
  populations.
- The LLM may be exploited by scripted agents. That is a fine finding if
  framed as "the LLM is a useful design *probe* but not a strong
  *competitor*"; it must not be conflated with Task 1, where same-class
  seating already controls for skill.

## Relationship to existing experiments and the rubric

- Complements (does not replace) the same-class evaluations and the
  cross-class agreement / cross-evaluator-gap results.
- Answers a *robustness* question, not the stated hypothesis (cross-class
  agreement on board ranking). Frame as an add-on.
- Adds evaluation breadth, which is what CS348K grading criterion 2
  (thoroughness of evaluation) rewards, provided the skill-vs-design
  confound is controlled and the numbers are interpreted honestly.

## Limitations

- Cross-class skill gap is a constant nuisance term; only board-to-board
  differences are interpretable.
- One LLM (Qwen2.5-1.5B) only; a stronger model could change the picture.
- Larger configuration grid for 3p increases GPU cost quickly.
