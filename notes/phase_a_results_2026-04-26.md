# Phase A: simplified-board sanity check — canonical results (2026-04-26)

Pre-declared design knobs from `report.tex` §7.1, run on the 16-cell
`configs/mini` board at the project-standard `max_turns=200`.

## Methodology

- Rule-based: 30 games per condition (RuleBased vs RuleBased)
- LLM cross-check: 12 games per condition (LLMPlayer w/ Qwen2.5-1.5B-Instruct vs RuleBased)
- max_turns = 200 (project standard, matches optimiser pipeline)
- Same seed set across every condition (variance reduction; identical dice/cards)
- Total wall-clock: ~17 minutes (rule-based 0.3 s; LLM ~17 min for 48 games)

## Results

| Knob | Metric | Rule-based default → modified | LLM default → modified | Match? |
|---|---|---|---|---|
| **Salary ×2**     | rounds       | 71.9 → 47.8   | 69.5 → 39.7   | ✅ shorter |
|                   | **draws**    | **50% → 90%**   | **50% → 75%**   | ✅ **MORE (non-obvious)** |
|                   | bankruptcies | 0.50 → 0.10   | 0.50 → 0.25   | ✅ fewer |
|                   | win spread   | 0.37 → 0.03   | 0.33 → 0.25   | ✅ flatter |
| **Drop Brown**    | rounds       | 71.9 → 67.7   | 69.5 → 46.1   | ❌ disagree |
|                   | draws        | 50% → 53%       | 50% → 8%        | ❌ disagree |
|                   | bankruptcies | 0.50 → 0.47   | 0.50 → 0.92   | ❌ disagree |
|                   | win spread   | 0.37 → 0.33   | 0.33 → 0.58   | ❌ disagree |
| **Orange rent×2** | rounds       | 71.9 → 51.6   | 69.5 → 54.2   | ✅ shorter |
|                   | draws        | 50% → 33%       | 50% → 50%       | ~ marginal |
|                   | bankruptcies | 0.50 → 0.67   | 0.50 → 0.50   | ~ marginal |
|                   | win spread   | 0.37 → 0.40   | 0.33 → 0.00   | ❌ disagree |

## Three findings, ranked by confidence

### 1. WORKSHOP HEADLINE: salary doubled produces MORE draws

Both rule-based agents and the LLM independently confirm the
counterintuitive direction across all four metrics:

- Game length **shrinks** by ~30% (RB: 71.9 → 47.8; LLM: 69.5 → 39.7)
- Draw rate **rises** sharply (RB: 50% → 90%; LLM: 50% → 75%)
- Bankruptcy rate **collapses** (RB: 0.50 → 0.10; LLM: 0.50 → 0.25)
- Win-rate spread **flattens** (RB: 0.37 → 0.03; LLM: 0.33 → 0.25)

Mechanism (same for both agent classes): doubling salary makes cash
accumulate faster than rents can drain it, so bankruptcy becomes rare,
and games end via the all-rich truncation rather than a decisive
bankruptcy. Intuition predicts "more salary → faster game → fewer
draws"; the agent loop falsifies that prediction cheaply, and the
falsification holds across two independent agent populations.

### 2. Drop Brown: the agent classes genuinely disagree (this is itself a finding)

Rule-based says Brown removal is essentially a no-op (rounds barely
change, draw rate moves 50% → 53%, bankruptcies move 0.50 → 0.47).
LLM says Brown removal is highly decisive (rounds 69.5 → 46.1, draw
rate collapses 50% → 8%, bankruptcies almost double 0.50 → 0.92).

At n=6 we wondered if this was sample-size noise. At n=12 the gap is
larger, not smaller, so it is NOT noise. The LLM player is making
structurally different decisions when Brown is absent (likely it diverts
the cash it would have spent on Brown into more aggressive spending on
remaining groups, which the rule-based always-buy-everything strategy
does not).

This disagreement is itself an important workshop finding: **which agent
population the designer chooses to validate against affects the design
conclusion**. A designer using only rule-based agents would conclude
"Brown is irrelevant, drop it freely". A designer using only an LLM
agent would conclude "Brown is critical to the game's pacing". Real
human play is the tiebreaker.

### 3. Orange rent×2: weaker signal at canonical fidelity

At n=6/cap=50 the rule-based and LLM agreed on direction; at n=12/cap=200
they only agree on rounds. RB sees the predicted bankruptcy increase;
the LLM does not. Suggests Orange-rent doubling is a weaker design
knob than the agents predicted, and probably less interesting for the
human study.

## Implications for the project claim

Phase A's combined result confirms the central claim with appropriate
nuance: agent feedback IS decision-useful for design, but the value
appears in the cross-class agreement structure rather than in any
single agent's predictions. Specifically:
- Where two independent agent classes agree (salary ×2), the prediction
  is robust enough to be trusted.
- Where they disagree (drop Brown), the disagreement itself is
  diagnostic — it identifies exactly the conditions where humans
  should be tested.
- Where the signal is weak in one class but strong in the other
  (Orange rent ×2), the human study can break the tie cheaply.

## Phase B priorities (informed by Phase A)

1. **Salary ×2** — confirm the non-obvious finding holds for humans.
   Predicted human result: shorter games, more draws.
2. **Drop Brown** — break the agent-class tie. Predicted: humans
   probably side with the LLM (real players DO redirect spending),
   but this is the hypothesis to test.
3. (Lower priority) **Orange rent ×2** — only run if Phase A budget
   has slack.
