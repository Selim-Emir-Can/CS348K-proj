# Meeting with Kayvon — 2026-04-24

Notes from a conversation about validating the agent-driven beta-testing
framework and tightening the project's core claim.

## The core thesis (refined)

> Building a tool that leverages agents whose tendencies I already know,
> running automated experiments with them, and then verifying that the
> design constraints I specified in the agent setting are reflected in
> real-world play with humans.

**The feedback from agents does not need to be literally accurate.**
The claim is weaker and stronger at the same time: the changes I made
had the property I desired in real play, and agent feedback was what
gave me the confidence to pick those changes.

## The central validation question

How can I gain confidence that feedback from agents is *somewhat*
predictive of real people on the dimensions I care about (fun, fair,
balanced, reasonable length)?

This is the bridge the project currently lacks: the optimiser produces
designs that score well under agent evaluation, but a reader has no
direct evidence that those designs will also feel fair / well-paced /
fun to real humans.

## Evaluation plan

### Phase A — Simplified-board sanity check
Run a smaller version of the setup (e.g. a 4x4 board) first, because:
- Fewer confounds and faster iteration.
- Clearer mapping between a single design knob and its effect.
- Easier to align agent-predicted and human-observed outcomes on
  something that takes minutes rather than an hour to play.

### Phase B — Human playtest on the optimised design
Actually sit down with real people and play the board the agents picked.
Compare real-world observations against the properties the agents said
the board would have:
- Did games actually end in the length the agent metric predicted?
- Did players who adopted archetype X in real play actually win at the
  rate the agent metric suggested?
- Did the subjective "fairness" / "decisiveness" feel match the metric?

### Phase C — Cross-validation with an LLM agent
Add a third data point beyond the rule-based agents and the real humans:
an **LLM agent** playing the same optimised boards. Useful because:
- It is a *different kind* of strategic intuition than the parametric
  rule-based pool — closer to human-style reasoning without being human.
- If rule-based agents, humans, and the LLM agent all agree on the
  direction of a design change, that is stronger evidence the change
  is real than any one signal alone.

## Statistics we want to nail down about the game

We should land on a clear, defensible set of summary statistics for
"what do games on this board look like" before running the human test.
Candidates (some already tracked, some to add):
- Mean and variance of game length
- Win-rate spread across agent archetypes
- Draw rate / truncation rate
- Per-player rent paid and rent received
- Time to first monopoly
- Number of bankruptcies per game

Having an agreed-upon statistic set makes it easy to frame the real-
human comparison as "these are the quantities I predicted; here is how
they came out in the human playtest".

## Experimental-design notes

### Sanity checks should go beyond the obvious
- Verifying that two cash-hoarders lengthen games is a trivial sanity
  check. Worth running but not the point.
- Aim for **new insights** from the agent evaluation: e.g. "boards that
  are shorter by X cells produce disproportionately fewer draws" —
  something a designer would not have obviously predicted before running
  the sweep.

### The success criterion
The project succeeds when I can say:
> "In the agent setting I specified a design constraint (e.g. shorter
> games, fairer strategy matchups); in the real-world playtest I
> observed that constraint being satisfied; therefore the agent-driven
> tool enabled a real design decision."

Note what this *does not* require: it does not require the agents'
predicted win rates to match human win rates exactly. It only requires
that the **direction** and **relative magnitude** of the agent-predicted
effect hold up in real play.

## Next actions

1. Pick two or three specific design knobs to validate (e.g. shorter
   board shortens games; removing high-rent groups evens out fairness;
   raising salary reduces draws).
2. Decide on the simplified-board variant for Phase A.
3. Prepare the fixed set of summary statistics we will report from each
   setting (rule-based pool, LLM agent, humans).
4. Recruit real-human playtesters for Phase B.
5. Run the LLM-agent playthroughs in parallel so Phase C is ready when
   the human data arrives.
