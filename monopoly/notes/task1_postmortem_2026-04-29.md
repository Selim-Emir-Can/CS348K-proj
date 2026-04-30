# Task 1 (LLM-only eval) post-mortem — 2026-04-29

## Summary

Ran the 80-game LLM-only evaluation matrix:
- 2 player counts × 2 boards × 20 seeds × all-LLM seats × Qwen2.5-1.5B
- 2,304 total LLM buy decisions logged with full ECHO-validation traces
- Wall time: 2:17 (2p) + 3:54 (3p) ≈ **6 h 11 min** end-to-end on RTX
  3060-class GPU (no flash attention)

## Headline result

GA-optimised boards reproduce the rule-based GA's directional signal
when the seats are swapped to LLM agents:

| | rounds | $ transfer / round |
|---|---:|---:|
| 2p default | 60.5 | 50.1 |
| **2p GA-winner** | **46.8** | **78.6** |
| 3p default | 76.0 | 100.6 |
| **3p GA-winner** | **44.6** | **127.6** |

Shorter games + higher inter-player money flow on the GA boards in
both player counts. Same direction the rule-based pool produced —
this is the cross-class agreement the report was missing.

## Validator bug found and fixed

The original `_check_echo` parsed numeric STATE fields with `int()`.
`Player.money` ends up as a float in some games (rent×multiplier
rounding on the GA-winner boards leaks fractional dollars; tax/fine
interactions on default produce `.0`-valued floats too). The
validator threw `ValueError` on `int("411.79999999999995")` and
logged "echo unparseable", triggering 4 wasted retries.

100 / 2304 LLM calls were flagged this way (4.3%). The model itself
behaved correctly in all 100 — it copied STATE verbatim, including
the float drift.

**Fix (committed 2026-04-29):**
- `_build_buy_prompt` int-coerces `cash`, `cost`, `base_rent` before
  formatting, so STATE always shows whole dollars.
- `_check_echo` accepts float echoes within $0.50 of STATE as a
  defensive measure.

## True hallucination rate (post-reclassification)

After reclassifying float-drift "unparseable" flags as
validator-bug-spurious:

| board | flagged | **real first-pass hallucinations** |
|---|---:|---:|
| 2p default | 3 / 595 | **0** |
| 2p GA-winner | 15 / 341 | **0** |
| 3p default | 52 / 920 | **0** |
| 3p GA-winner | 30 / 448 | **0** |
| **all** | 100 / 2304 (4.3%) | **0 / 2304 (0.0%)** |

Zero. Across 2,304 LLM calls the ECHO-required prompt drove the
model to copy STATE correctly every time on the first attempt.

## Second-order finding: retries induce hallucinations

A more surprising result: 47 / 100 final-attempt issues (after the
4-retry budget was exhausted) were *real* numeric mismatches, even
though the first attempt was a clean float echo.

Inspecting the JSONLs, the failure mode is clear: when the validator
incorrectly rejected the model's correct first-pass echo, it sent
the model a "your previous response failed echo validation" message.
Four rounds of that feedback caused the model to give up trying to
copy STATE and start making up numbers — `"I have $1000, cost is
only $100"` when STATE said cash=$416.0, cost=$180.

In other words, the retry mechanism is a force multiplier: when the
validator is *right*, it nudges the model toward a clean answer; when
the validator is *wrong*, it nudges the model toward fabrication.
The fix above eliminates the validator's wrongness for the
float-drift case, so this second-order failure shouldn't recur in
future runs.

This is a reportable observation: feedback loops between automated
validators and the agent-under-test can create new failure modes
that didn't exist in single-shot evaluation. For the agent-as-design-
probe write-up, it's evidence that the retry policy is a part of the
"probe", not a neutral component.

## Buy-rate slices (qualitative health check)

From `analysis_default.md` (2p):
- buy rate by cash bucket: 200-500=0.85, 500-1000=0.74, 1000-1500=0.65, ≥1500=0.45
- buy rate by monopoly opportunity: fresh=0.74, partial_self=0.93, opponent_dominates=0.30

The model is correctly more cautious as cash falls and as opponents
accumulate group ownership. No saturation, no degeneracy.

## What does NOT need a re-run

- The 80 games are valid. The `parsed` answer field always reflects
  the model's actual decision (the parser locked onto the FIRST
  `ANSWER: BUY|PASS` regardless of validator outcome).
- Game results (rounds, transfer, winner) are unaffected by the
  validator bug — the bug only added latency from spurious retries.

## What DOES change for future runs (Task 2 and beyond)

- New validator + int-coerced prompt: 0 spurious flags, no retry
  noise. Estimate: ~10-20% wall-time reduction since the 100 spurious
  retries (each adding ~30s = 50min total) are gone.
- The `notes/revert_points.txt` baseline still applies; this fix sits
  on top of `c1b7e85`.
