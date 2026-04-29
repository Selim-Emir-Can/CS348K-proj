# LLM prompts used by `LLMPlayer`

This folder is the canonical record of how the LLM-based player in
`agents.LLMPlayer` is prompted. The files here are documentation /
artefacts, not code: changing them does **not** change runtime
behaviour. The runtime values live in
[`monopoly/agents.py`](../agents.py)
(`LLMPlayer._SYSTEM_PROMPT`, `LLMPlayer._FEW_SHOT`,
`LLMPlayer._build_buy_prompt`).

## Files

| File | Format | Audience |
|---|---|---|
| `llm_player_prompts.txt` | Human-readable transcript laid out as a chat | Reviewers, paper readers, anyone wanting to see "what the LLM sees" at a glance |
| `llm_player_prompts.json` | Structured (system / few-shot list / template / settings) | Re-implementations, downstream tooling, automated extraction |

Both files describe the **same** prompt; the JSON is more machine-friendly,
the TXT is laid out the way a chat-template renderer would produce it.

## What the LLM is asked

A single binary BUY/PASS decision when the player lands on an unowned
property. The prompt includes:

- the player's cash
- the property's name, colour group, cost, base rent
- how many properties the player and opponent own (total + in this group)
- the size of the colour group on the current board

The LLM is asked to reply in a fixed two-line format
(`REASON: <one sentence>` then `ANSWER: BUY` or `ANSWER: PASS`).
A four-shot prompt covers the obvious cases (BUY when cheap and
unopposed, PASS when cash-tight or opponent-dominated, BUY to complete
a monopoly, PASS when liquidity matters).

## What the LLM is **not** asked

- House / hotel building decisions
- Trade decisions
- Jail-exit decisions

These keep the base `Player` logic. Routing every micro-decision through
the LLM would make games take minutes each rather than seconds, and the
buy decision is the strategically dominant one anyway.

## Why this format

We tried two model sizes:

- **Qwen2.5-0.5B-Instruct**: saturates to BUY on 100% of decisions
  even with the rich prompt. Insufficient capacity for the cash /
  opponent / monopoly-completion reasoning the prompt asks for.
- **Qwen2.5-1.5B-Instruct** (current default): differentiates correctly
  on the three diagnostic axes (cash level, opponent ownership in
  group, monopoly-completion opportunity). Roughly $1$s per call on
  a single-GPU machine; ~3.4s on CPU.

See `notes/phase_a_results_2026-04-26.md` for the validation findings
that motivated this choice and `scripts/smoke_llm_player.py` for the
diagnostic that uncovered the 0.5B model's saturation.
