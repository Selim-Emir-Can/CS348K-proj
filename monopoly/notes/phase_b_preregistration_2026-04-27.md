# Phase B preregistration — 2026-04-27

Pre-committing the falsification criteria, board priorities, and analysis
plan for the human playtest *before* any human data is collected. The
purpose is to make the eventual cross-comparison legitimate rather than
post-hoc cherry-picking.

## Boards to test

In order of priority (drop later ones if budget runs out):

1. **default mini board** — reference. Both players on the canonical 16-cell layout.
2. **drop Brown** — *highest-priority* knob because rule-based and LLM
   agent classes disagree on direction. Phase A: rule-based says no-op,
   LLM says decisive (bankruptcy 0.50→0.92, draws 50%→8%). Humans break
   the tie.
3. **salary ×2** — second priority. Phase A: cross-confirmed
   non-obvious finding (more draws, not fewer). Whether humans replicate
   this is the headline test of the project's central claim.
4. **orange rent ×2** — third priority. Phase A: weak/disagreed signal.
   Run only if budget permits.

## Sample size and protocol

- 3-5 testers per session. Same testers play multiple boards.
- 5-10 games per board, randomly ordered to control for fatigue/learning.
- Each game uses the printed score sheet in `phase_b_score_sheet.md`.
- 90-minute real-time cap per game; games hitting the cap are recorded as
  draws.

## Falsification criteria (committed in advance)

### Primary criterion: directional agreement on the salary-×2 finding

The agent loop is judged \emph{decision-useful} on this knob if humans
report the same direction of change as both agent classes on the
**draw rate** statistic — i.e., human draw rate on the salary-×2 board
is higher than human draw rate on the default board. Specifically:

- **Confirmed:** human salary-×2 draw rate > human default draw rate
  (one-sided binomial test, p < 0.1).
- **Falsified:** human salary-×2 draw rate < or ≈ human default draw rate.

If the binomial test is underpowered at small N (likely with ~5 games
per board), report the raw point estimates and the implied direction
without the p-value claim.

### Secondary criterion: drop-Brown tiebreaker

Whichever direction humans go on the drop-Brown board (RB-style no-op
or LLM-style decisive), report it honestly and update the
generalisation principles accordingly. Specifically:

- If humans look like RB → "rule-based agents alone would have
  predicted real human play correctly on this knob; the LLM was
  over-reactive."
- If humans look like LLM → "the LLM was the only agent class that
  predicted the human direction; rule-based agents alone would have
  missed it."
- If humans look like neither → "real play was outside the agent-class
  prediction range" (interesting and worth its own discussion).

### Tertiary criterion: subjective fairness

The combined-objective optimised board is judged perceptually fairer
than the default if the mean human Likert rating on \"fairness\" is
higher on the optimised board (paired t-test across testers, p < 0.1).
Underpowered with N=3-5; report point estimates if test fails to reach
significance.

## Statistics computed per board (eight, matching Phase A and the report's pre-declared list)

1. mean game length (turns)
2. std game length
3. draw rate
4. per-archetype win rate (using each tester's self-classified archetype)
5. time to first monopoly
6. mean rent paid per player per game
7. bankruptcies per game
8. mean transfer per round

These are the same eight that Phase A reported for both rule-based and
LLM agents, so the Phase A vs Phase B comparison is direct.

## Analysis plan

After all human data is collected, produce one summary table per board
with the eight statistics, plus the Phase A rule-based + LLM values for
comparison. Compute "directional agreement" (sign of (modified -
default)) for each statistic across all three sources, and the
agreement rate \(k/(8 \times \text{n\_boards})\) overall.

Headline figure: a small table showing, for each metric on each
modified board, whether the rule-based agent direction, the LLM agent
direction, and the human direction agree.

## What success and failure look like

- **Best case:** humans agree with the cross-confirmed Phase A finding
  on salary-×2 (more draws than default), and humans break the
  drop-Brown tie clearly in one direction. The project then makes a
  concrete claim of the form "agent feedback was directionally correct
  on N of M tested knobs in human play."
- **Worst case:** humans disagree with both agent classes on every
  knob. This would falsify the central claim. The honest writeup
  becomes "agent feedback in this implementation is not yet
  decision-useful for human play"; future work would explore richer
  agent populations or smaller design changes.
- **Middle case (most likely):** humans agree on the easy knobs and
  produce mixed signal on the harder ones. The writeup reports the
  partial confirmation honestly and discusses which conditions agent
  feedback is and isn't useful for.
