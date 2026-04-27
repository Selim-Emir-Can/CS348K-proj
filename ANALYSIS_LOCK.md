# ANALYSIS_LOCK.md

Status: LOCKED 2026-04-27, before any LLM overnight run for #3, #4, or #5.
Source of truth: `ceo-plans/2026-04-27-llm-design-loop-expansion.md` (locked).

This file pins the analysis decisions for items #3, #4, #5 of the LLM
design-loop expansion BEFORE any data is collected at full scale. It exists
to defuse the multiple-comparisons cliff across five experiments (CEO C6).
Anything not pinned here is open to honest re-analysis after data lands;
anything pinned here is frozen for the duration of this round and may only
be revisited with an explicit "ANALYSIS_LOCK breach" note in the writeup.

---

## 1. Aligned 5-board set

Used by #1 (hazards), #3 (LLM character), #4 (parametric closed loop),
#5 (rule-mutation closed loop). Resolved by
`optimizer.board_sources.build_five_boards`:

  1. `default`     — canonical Hasbro-shaped config (`default_config.yaml`)
  2. `GA-2p`       — lowest-score entry from `logs/optimizer/ga_2p.jsonl`
  3. `GA-3p`       — lowest-score entry from `logs/optimizer/ga_3p.jsonl`
  4. `salary x2`   — `configs/mini` with `mechanics.salary` doubled
  5. `drop Brown`  — `configs/mini` with the entire Brown colour group removed

If a GA log is absent the corresponding curve is gracefully skipped, but
this lock requires both logs be present for the production runs of #3/#4/#5.

---

## 2. Concept dictionary (#3)

Frozen in `scripts/llm_character.py:CONCEPT_PATTERNS` and reproduced here
verbatim so any later change is forced to bump this lock:

```
cash       : \bcash\b | \bmoney\b | \bafford | \bliquid
risk       : \brisk | \bdanger | \brisky\b | \bunsafe
monopoly   : \bmonopoly\b | \bmonopolise | \bmonopolize | \bcomplete (?:the |my )?(?:group|set|monopoly)
trade      : \btrade | \bnegotiat | \bswap
rent       : \brent\b | \brents\b | \brent income | \brent return
bankrupt   : \bbankrupt
accumulate : \baccumulate | \bstockpile | \bhoard | \bsave
defensive  : \bdefensive | \bdefend | \bblock | \bdeny
opponent   : \bopponent | \bother player | \brival
group      : \bgroup\b | \bcolour group | \bcolor group | \bset\b
```

A reasoning hits a concept iff any pattern matches the lowercased text;
multiple matches within one block count once per concept.

---

## 3. Divergence threshold (#3)

Metric: pairwise L1 over the normalized concept-frequency distribution
(half-L1 form — see `l1_divergence` in `llm_character.py`).

A cross-board divergence pair `(X, Y)` is reported as MEANINGFUL iff:

  cross_L1(X, Y)  >  within_board_L1_mean  +  2 * within_board_L1_sigma

where `within_board_L1` is estimated by:

  - Take each board's reasoning corpus.
  - Split uniformly at random into two equal-sized halves.
  - Compute L1 between the two halves' concept frequencies.
  - Repeat with 100 random splits per board.
  - Pool across the 5 boards: `within_board_L1_mean` is the grand mean,
    `within_board_L1_sigma` is the grand std.

The 5×5 cross-board L1 matrix continues to be reported in full; the
threshold is what gates the "this pair is signal, not noise" annotation
on the figure and the headline claim in the writeup.

If NO pair clears `mean + 2 sigma`, the headline finding becomes:
  "the 1.5B LLM produces context-invariant strategic narration; cross-board
  character induction is not visible at this model size" (calibration
  finding, pre-declared as publishable).

This is enforced by `scripts/llm_character.py:within_board_noise_floor`
(added in Step 3).

---

## 4. Board-state-reference precondition (#3)

Before any cross-board divergence claim, report the rate at which each
reasoning block refers to OBSERVABLE GAME STATE. A block is counted as
"grounded" if it contains at least one of:

  - A property NAME present on the current board (matched against the
    cfg's `cells[*].name` list, lowercased substring match).
  - A rent-bearing INTEGER ($d{2,4} OR an integer >= 20 within reasonable
    bounds) — proxy for cash / rent reasoning.
  - An owned-list keyword: `owned`, `holdings`, `properties`, `portfolio`,
    `i have`, `i own`, `we own`, `they own`.

Per-board grounded rate is reported alongside the divergence matrix. If
the grounded rate falls below 50% on any board, divergence findings on
that board are flagged as "ungrounded" in the writeup.

This is enforced by `scripts/llm_character.py:per_board_grounded_rate`
(added in Step 3).

---

## 5. Per-iteration "improvement" definition (#4 / #5)

A score change at iteration k is counted as IMPROVEMENT iff:

  delta_score_relative >= 0.03   AND   95% CI on delta excludes zero.

`delta_score_relative` is computed over the SAME n_games-per-cell budget
(default n=100) using SAME seed structure as the prior iteration's eval.
The 95% CI is computed by bootstrap over per-game scores (n_resamples=500).

Implementation lives in the loop scripts (`llm_design_loop.py`,
`llm_rule_loop.py`); this lock pins the threshold before any loop runs.

A monotone-improvement claim in the writeup requires every reported
delta to clear this bar. A non-monotone trajectory is reported honestly
per the falsification table.

---

## 6. Comparator-gap definition (#5)

The LLM rule loop "beats random" iff:

  median_final_score(LLM)  <  25th_percentile_final_score(random_baseline)

over the same set of starting boards × seeds. The baseline budget is
matched: identical number of iterations, identical eval n_games. If the
LLM does not clear this bar, the writeup reports the calibration finding
("random rule mutation generates competitive designs at this scale; LLM
brings rationale but not score") rather than claiming a positive result.

The Monopoly house-rule static set (Free Parking jackpot, auction-unowned,
no-$200-on-first-pass, etc.) gives the static ceiling baseline. The LLM
"beats house rules" iff LLM median final score < house-rule mean score.

Both comparator gates are computed per-board and pooled.

---

## 7. Iteration board for #4 / #5 — DECISION

The CEO plan pre-committed to mini-board iteration with canonical
re-evaluation, gated by the C3 transfer audit. The audit ran at n=100
on 2026-04-27 (`report/figures/transfer_audit/`) and returned:

  Spearman rho(mini, canonical) = 0.300

This is below the pre-committed 0.4 cutoff. Per the locked plan, the
iteration board therefore becomes CANONICAL for both #4 and #5. The
pre-committed decision rule is honored; this lock records the outcome.

Wall-clock recomputation: per-design eval at n=100 on canonical is ~0.2s
on the development host (single thread), so the LLM call latency
dominates. Estimated wall-clock budgets are essentially unchanged from
the original CEO plan; the canonical-iteration choice does not invalidate
the rest of the plan.

The audit itself becomes a paper artifact. Surface in §5e (the parametric
closed loop section) under a 1-paragraph sub-bullet titled "Iteration
board choice: rho=0.30 audit". Five-design Spearman with rank-flips on
structural drops and rent inflation; pacing-only knobs (salary) transfer
cleanly. This is a finding, not a methodological wart.

---

## 8. Strategy pool (load-bearing across all five experiments)

Pinned: `optimizer/strategy_pool.json` (10 named + 20 sampled, seed=0)
already on disk. No regeneration is permitted between this lock and the
end of the round. Any pool edit BEFORE running #3/#4/#5 must be recorded
as an explicit lock breach in the writeup.

Recommended C5 audit (pool diagnostics: pairwise win-rate matrix +
archetype entropy) — out of scope for this lock, but should be added in
the appendix before submission.

---

## 9. Validation matrix LLM identity (C8)

The Phase A `LLMPlayer` (classification mode, BUY/PASS only) is the LLM
that appears in the Phase-A cross-source validation matrix.
The `CharacterLLMPlayer` (reasoning mode, REASON: + ANSWER:) used by
#3, #4 is a separate instrument and is labelled as such in the writeup.
They share the model weights (`Qwen2.5-1.5B-Instruct`) and base prompt
scaffolding but their outputs are not interchangeable for matrix
purposes.

---

## 10. Wall-clock cap on #5 v2 (C9)

`scripts/llm_rule_loop.py` is invoked with a hard `--max-wall-seconds`
default of 21600 (6 hours). The driver checks elapsed wall-clock before
each iteration and exits cleanly with the partial trajectory written to
disk. v2 is forbidden to start before v1 has completed and produced a
deliverable trajectory.

---

## What is NOT locked

- The qualitative read of the per-board reasoning corpora (one paragraph
  each) is post-hoc and labelled as such in the writeup.
- Selection of the top-3 rule diffs for #5 Goodhart audit is post-hoc;
  the audit itself is pre-committed but the rules audited are surfaced
  by the run.
- Choice of which #5 v2 trajectory to highlight in §5f, if multiple seeds
  produce qualitatively distinct designs.

---

## Lock breach protocol

If during analysis a problem is found with one of the items above
(e.g. concept dictionary turns out to have a regex bug), the
ANALYSIS_LOCK is updated with an explicit BREACH NOTE:

```
## BREACH 2026-04-XX: <one-line reason>
- Old rule: ...
- New rule: ...
- Affected sections of writeup: ...
```

The pre-breach result is reported alongside the post-breach result
in the writeup.
