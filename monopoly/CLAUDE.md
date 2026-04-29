# Project context for Claude Code

CS348K class project. Authors: Selim Emir Can (emirc) + Alaz Cig.
Monopoly is the substrate; the contribution is a closed-loop
**game-design optimisation** workflow that uses imperfect agents as
design probes.

The core idea is *not* RL: a genetic algorithm searches a 66-dim board
design space (per-property cost/rent multipliers + a 22-bit keep-mask
that shrinks the lap) against a composite objective (fairness, length,
draw rate, money transfer). Candidate boards are evaluated against a
fixed pool of 30 parametric rule-based strategies; on top of that we
have an LLM-as-decision-probe layer that re-evaluates the GA winner
with all-LLM seats so the report can show two independent agent classes
generalising in the same direction.

## Where we are right now (2026-04-28)

- **Done:** rule-based GA finished (logs/optimizer/ga_{2p,3p}_mask.*),
  LaTeX report draft sits in `report/report.tex` *(do NOT edit it
  during experimental work — see `feedback_no_report_edits_for_now`)*.
- **In progress: Task 1 (LLM-only eval).** Pipeline is built, all five
  smoke runs (smoke2..smoke6) confirm the ECHO validator drives
  hallucinations to 0% on first attempt. Full 80-game eval not yet
  launched (waiting for Sherlock 24 GB GPU; local 11 GB worked but
  slow at ~8 s/call).
- **Pending: Task 2 (LLM-driven GA).** Plan: 2p only, pop=8 ×
  generations=5 = 40 evals × 5 seeds. Not started.

The pre-LLM-eval revert point is **commit `8c1cbaf`** (2026-04-28T16:19
PT). See `notes/revert_points.txt` for rollback instructions.

## Run book (Sherlock)

Run from `monopoly/` with the conda env activated:
```
conda activate cs224r-proj   # local; on Sherlock create equivalent env
```

**Get the LLM weights** (one-time, ~3 GB):
```
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir models/qwen2.5-1.5B
```

**Smoke (1 game, ~5 min):**
```
python scripts/eval_llm_on_boards.py --boards default --n-players 2 \
    --n-seeds 1 --out-dir logs/llm_eval/smoke \
    --model-name models/qwen2.5-1.5B
```

**Task 1 full eval** (~6 h sequential on a single 24 GB GPU; both
commands write under different `--out-dir`):
```
python scripts/eval_llm_on_boards.py --boards default ga_2p_winner \
    --n-players 2 --n-seeds 20 --out-dir logs/llm_eval/2p \
    --model-name models/qwen2.5-1.5B

python scripts/eval_llm_on_boards.py --boards default ga_3p_winner \
    --n-players 3 --n-seeds 20 --out-dir logs/llm_eval/3p \
    --model-name models/qwen2.5-1.5B

python scripts/analyze_llm_decisions.py --in logs/llm_eval/2p
python scripts/analyze_llm_decisions.py --in logs/llm_eval/3p
```

The driver warms up the model first (`[warmup] OK; sample response: ...`)
and refuses to start on a partial HF download. tqdm shows current
turn + transfer total so you have liveness even on long games.

**Task 2 (not yet implemented).** Will be a thin wrapper around the
existing GA in `optimizer/search.py` that swaps the player factory to
all-LLM and tightens defaults (pop=8, gens=5, n_seeds=5). Naming
convention: `scripts/optimize_board_llm.py`, output under
`logs/optimizer_llm/2p/<run_id>/`.

## Code map

```
monopoly/
├── agents.py                       # all player classes, including LLMPlayer
├── config.py                       # GameConfig + YAML round-trip
├── player_settings.py              # dataclasses for player knobs
├── monopoly/                       # core game engine (vendored)
│   └── core/{board,player,...}.py
├── optimizer/                      # GA + objective + design-space encoder
│   ├── design_space.py             # 66-dim vec ↔ GameConfig
│   ├── simulate.py                 # run_single_game (player_kwargs_list!)
│   ├── objectives.py               # fairness/length/draw/money
│   ├── search.py                   # random search + GA
│   └── strategy_pool.py            # 30 parametric rule-based strategies
├── scripts/
│   ├── eval_llm_on_boards.py       # Task 1 driver (this branch)
│   ├── analyze_llm_decisions.py    # Task 1 post-hoc analyser
│   ├── optimize_board.py           # rule-based GA driver (already run)
│   ├── cross_eval.py               # 2p↔3p generalisation table
│   ├── strategy_heatmap.py         # 30×30 win-rate matrix
│   └── ...                         # render, plots, phase-A validation
├── prompts/                        # canonical record of the LLM prompt
│   ├── llm_player_prompts.json     # machine-readable
│   └── llm_player_prompts.txt      # human-readable transcript
├── notes/
│   ├── revert_points.txt           # known-good git commits
│   ├── phase_a_results_2026-04-26.md
│   └── ...
├── report/                         # LaTeX report (DO NOT EDIT during expt)
└── logs/                           # gitignored; recreated on Sherlock
```

## LLMPlayer architecture (the part most likely to need changes)

- **Greedy decoding** with Qwen2.5-1.5B-Instruct (Qwen2.5-0.5B saturates;
  see `notes/phase_a_results_2026-04-26.md`).
- **Pre-filter** (no LLM call) for cant_afford / cash_floor /
  ignore_group → ~60-80% of would-be calls short-circuited.
- **Prompt**: system + 4 few-shot exemplars + STATE block. Required
  output is `ECHO: <10 fields> / REASON: ... / ANSWER: BUY|PASS`.
- **Echo validator** (`_check_echo`) parses the model's ECHO block
  with a MULTILINE regex and equality-checks each of the 10 fields
  against ground-truth STATE.
- **Retry loop**: up to **MAX_RETRIES = 4** retries on any echo
  mismatch. Each retry's tail replays prior assistant turns plus a
  corrective user message naming the mismatched fields. Final
  attempt's parsed answer is the decision.
- **Decision parser** locks onto the FIRST `ANSWER: BUY|PASS` (case-
  insensitive). ECHO/REASON contain no `ANSWER:` so this is robust.
- **Per-decision JSONL** logged via `decision_log_path` kwarg —
  records prompt, raw_response, parse_path, ms_elapsed, gen_meta,
  echo_mismatches, full echo_attempts list. Old `hallucination_*` and
  `retry_*` fields kept for back-compat with earlier schema.

## Conventions / preferences (auto-memory pointers)

These are saved in user memory so future sessions inherit them, but
worth knowing here too:

- **Backwards-compat or revert point.** When extending an existing
  function, either keep the signature back-compat (add kwargs with
  defaults) or surface a `git rev-parse HEAD` + ISO date in the
  response so the user can `git reset --hard <sha>` if things break.
- **Step-by-step code changes.** Explain each change before writing
  it; never batch multiple file rewrites silently.
- **Preserve comments and formatting.** Use Edit (not Write) on
  existing files. Keep all comments, docstrings, backslash
  continuations.
- **User runs training/eval commands themselves.** Don't execute
  long-running python via the Bash tool — output the command and let
  the user launch it.
- **Don't edit `report/report.tex` during experimental work.** The
  user explicitly said "we will change it after."
- **Conda env**: `cs224r-proj` locally. On Sherlock create an
  equivalent env (transformers + torch + tqdm + pyyaml + numpy).
- **Shell**: bash on Windows here, but the user runs cmd.exe so
  shared-machine commands use `set VAR=.. &&` syntax. On Sherlock
  use bash.

## Git hygiene

- `models/`, `logs/`, `wandb/` are gitignored. Models re-download via
  `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`.
- `*.txt` is gitignored EXCEPT `prompts/*.txt` and `notes/*.txt`
  (curated documentation).
- Don't push the HF cache or experiment logs.

## Quick reference: Task 1 success criteria

When the full eval finishes, look at `logs/llm_eval/2p/analysis.md` for:

1. **First-pass echo mismatch rate < 5%** (smoke5 had 0%; 80 games will
   surface tail cases).
2. **Decisions still flagged after 4 retries < 1%.** Anything higher
   means the prompt is brittle and we should iterate.
3. **Buy rate when `opp_dominates` is < 30%.** Smoke5 was 21%.
4. **Hallucinated reason examples** look like real strategic
   judgements, not regurgitated few-shot phrasings.

If those hold, the report can claim "LLM-as-design-probe agrees
directionally with rule-based GA winner on N matchups out of M, with
a hallucination rate quantified by deterministic per-field validation."
