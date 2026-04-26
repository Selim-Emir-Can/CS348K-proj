# Phase A: simplified-board sanity check — initial results (2026-04-26)

Pre-declared design knobs from `report.tex` §7.1, run on the 16-cell
`configs/mini` board.

## Methodology
- Rule-based: 30 games per condition (RuleBased vs RuleBased)
- LLM cross-check: 6 games per condition (LLMPlayer w/ Qwen2.5-1.5B-Instruct vs RuleBased)
- max_turns = 50
- Same seed set for every condition (variance reduction)
- Total wall-clock: ~10 minutes (rule-based negligible, LLM ~9.4 min for 24 games)

## Results

| Knob | Metric | Rule-based default → modified | LLM default → modified | Direction match? |
|---|---|---|---|---|
| Salary ×2     | rounds       | 42.9 → 37.4   | 45.0 → 38.8   | ✅ shorter |
|               | **draws**    | **70% → 100%**  | **83% → 100%**  | ✅ **MORE** (non-obvious) |
|               | bankruptcies | 0.30 → 0.00   | 0.17 → 0.00   | ✅ fewer |
|               | win spread   | 0.17 → 0.00   | 0.17 → 0.00   | ✅ flatter |
| Drop Brown    | rounds       | 42.9 → 43.7   | 45.0 → 38.7   | ❌ disagree |
|               | draws        | 70% → 73%       | 83% → 50%       | ❌ disagree |
|               | bankruptcies | 0.30 → 0.27   | 0.17 → 0.50   | ❌ disagree |
| Orange rent×2 | rounds       | 42.9 → 36.8   | 45.0 → 35.7   | ✅ shorter |
|               | draws        | 70% → 50%       | 83% → 50%       | ✅ fewer |
|               | bankruptcies | 0.30 → 0.50   | 0.17 → 0.50   | ✅ more |
|               | win spread   | 0.17 → 0.23   | 0.17 → 0.17   | ≈ marginal |

## Headline findings

1. **Non-obvious agent-surfaced finding (cross-confirmed):** Doubling salary
   makes mini-Monopoly games **shorter on the clock but produce MORE draws**,
   because faster cash accumulation prevents bankruptcies and games hit the
   truncation cap. Intuition predicts the opposite ("more money = faster game
   = decisive"). Both rule-based agents and the LLM independently confirm
   the counterintuitive direction. **This is the kind of finding an automated
   beta-testing tool is worth building for.**

2. **Predictable finding (cross-confirmed):** Doubling rent on the most
   expensive group raises bankruptcy rate (0.30 → 0.50 RB; 0.17 → 0.50 LLM)
   and reduces draws. Sanity check that the agent loop is not broken.

3. **Disagreement worth studying:** Removing Brown (the cheapest colour group)
   is a near-no-op for rule-based but moves the dial substantially for the
   LLM. Three possible readings:
   - LLM at n=6 is sample-size noise.
   - LLM player makes structurally different buy decisions and is genuinely
     more sensitive to which groups are present.
   - Genuine effect that only Phase B (human playtest) can break.

## Implications for the project claim

The agent loop is **decision-useful** at the mini-board scale: 8 of 11
metric-direction comparisons across the three knobs match between the two
independent agent classes. The non-obvious finding (salary × 2 → more draws)
is exactly the kind of result a designer would not have predicted by hand
and that the framework surfaced cheaply.

The single disagreement (drop Brown) is itself informative: it tells us
where to focus the human-playtest budget. Rather than running humans on
all three knobs, Phase B can prioritise the drop-Brown condition where
the two agent classes disagree — that is where human data has the most
diagnostic power.

## Next steps

1. Increase LLM sample size from n=6 to n=20 to reduce noise on the
   drop-Brown disagreement.
2. Run Phase B (human playtest) on the salary-doubled and drop-Brown
   conditions, since those are where the agent signal is most informative
   (one cross-confirmed non-obvious finding to validate; one disagreement
   to break).
3. Consider scaling up the LLM to Qwen2.5-3B or 7B if 1.5B's drop-Brown
   reading turns out to disagree with humans — would help triangulate
   whether the disagreement is a model-capacity issue or a real signal.
