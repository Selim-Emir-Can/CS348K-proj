# Project context — CS348K Monopoly

**Last updated: 2026-05-06**

> **This file is the authoritative project context. If `CLAUDE.md`, any
> `notes/*.md`, README, or older comment disagrees with what is written
> here, *this file wins*.** When state changes, update this file in the
> same commit; never leave stale claims to be re-discovered later.

For numerical results, see `RESULTS.md` (which is the corresponding
override for values).

---

## What this project is

CS348K class project. Authors: **Selim Emir Can** (selimcan@stanford.edu),
**Alaz Cig** (alaz@stanford.edu).

Monopoly is the substrate; the contribution is a closed-loop
**game-design optimisation** workflow that uses imperfect agents as
*design probes*. The reusable artefact is the workflow, not the
optimised board.

The pipeline:

```
Parameterise environment   →   evaluate against a diverse agent
                                population
                           →   compare directional agreement across
                                agent classes
                           →   shortlist candidate designs
                           →   send only the most informative cases
                                to human playtest.
```

Two agent classes evaluate every candidate board:
1. A 30-strategy parametric rule-based pool (the "diverse population").
2. A single-personality LLM probe (Qwen2.5-1.5B with a structured
   `STATE/ECHO/REASON/ANSWER` prompt and deterministic per-field
   validation).

The headline empirical finding is that a single-personality evaluator
*systematically misses worst-case fairness* that the diverse pool
catches; the cross-evaluator gap is the diagnostic.

---

## Where things stand (2026-05-06)

| Block | State | Notes |
|---|---|---|
| Rule-based GA + ablations + cross-eval | **PENDING RE-RUN** | Apr 24 run had GA tied with random search at 2p. Re-run scheduled; user will issue the command. |
| Task 1 — LLM-only evaluation | **DONE** | 80 games, 0 first-pass hallucinations across 2,288 calls. |
| Task 2 — LLM-driven GA | **DONE** | 32 evals, winner at iter 15, score 0.465. |
| LLM-GA winner cross-eval under rule-based pool | **DONE** | Headline cross-evaluator gap ($+0.18$ at 2p, $+0.35$ at 3p). |
| Report — `monopoly/report/report_cs348k.tex` | **DRAFT** | Aligned with CS348K guidelines (Background, Approach, Evaluation and Results, Team responsibilities, References). Numbers in the rule-based-GA-dependent sections will need a refresh after the re-run. |
| Report — `monopoly/report/report.tex` | **OLDER DRAFT** | CVPR-style; superseded by `report_cs348k.tex` for submission. Kept for reference only. |
| Phase B — human playtest | **NOT RUN** | Pre-declared in the report's validation plan; not part of this submission. |

The single canonical numerical record is `monopoly/RESULTS.md`.

---

## Code map

Run from `monopoly/` with the `cs224r-proj` conda env active:

```
monopoly/
├── agents.py                       # all player classes, including LLMPlayer
├── config.py                       # GameConfig + YAML round-trip
├── player_settings.py              # frozen dataclasses for player knobs
├── monopoly/                       # core game engine (vendored)
│   └── core/{board,player,...}.py
├── optimizer/                      # GA, objective, design-space encoder
│   ├── design_space.py             # 66-dim vec ↔ GameConfig (cost + rent + 22-bit keep-mask)
│   ├── simulate.py                 # run_single_game with player_kwargs_list
│   ├── objectives.py               # fairness / length / draw / money
│   ├── search.py                   # random search + GA
│   └── strategy_pool.py            # 30 parametric rule-based strategies (saved JSON)
├── scripts/
│   ├── optimize_board.py           # rule-based GA driver (re-run target)
│   ├── cross_eval.py               # cross-evaluation harness
│   ├── strategy_heatmap.py         # 30×30 win-rate matrices
│   ├── eval_llm_on_boards.py       # Task 1 driver (DONE)
│   ├── analyze_llm_decisions.py    # Task 1 post-hoc analyser (DONE)
│   ├── optimize_board_llm.py       # Task 2 driver (DONE)
│   ├── plot_llm_results.py         # 7 LLM figures
│   ├── render_board.py             # board visualisation
│   └── ...
├── prompts/                        # canonical record of the LLM prompt
├── notes/                          # session notes (specific dates)
├── report/                         # LaTeX
│   ├── report_cs348k.tex           # canonical submission draft
│   ├── report.tex                  # older CVPR-style draft (kept for reference)
│   └── figures/                    # PNGs cited by the report
├── logs/                           # gitignored except the curated artefacts
│   ├── optimizer/                  # rule-based GA outputs (PENDING RE-RUN)
│   ├── optimizer_old/              # earliest 44-dim runs (do not cite)
│   ├── optimizer_llm/llm_ga_2p/    # Task 2 outputs (DONE)
│   └── llm_eval/{2p_v2,3p_v2,...}  # Task 1 outputs (DONE)
├── RESULTS.md                      # canonical numerical results (this file's sibling)
└── context.md                      # this file
```

---

## LLMPlayer architecture

- **Model:** Qwen2.5-1.5B-Instruct (greedy decoding). Local weights at
  `models/qwen2.5-1.5B/`. Re-download with
  `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5B`.
- **Pre-filter:** no LLM call when affordability or cash floor decides
  the action; ~60–80% of would-be calls short-circuit.
- **Prompt:** system + 4 few-shot exemplars + `STATE` block. Output
  format is `ECHO: <10 fields> / REASON: ... / ANSWER: BUY|PASS`.
- **Validator (`_check_echo`):** parses the model's `ECHO` block with a
  multiline regex and equality-checks each of the 10 fields against
  ground-truth state, with $0.50 tolerance on cash to absorb float drift.
- **Retry loop:** `MAX_RETRIES=4` on echo mismatch. Each retry replays
  prior assistant turns plus a corrective user message naming the
  mismatched fields.
- **Decision parser:** locks onto the **first** `ANSWER: BUY|PASS`
  (case-insensitive). `ECHO`/`REASON` are guaranteed not to contain
  `ANSWER:`, so this is robust.
- **Per-decision JSONL:** logged via the `decision_log_path` kwarg.
  Each record has `prompt`, `raw_response`, `parse_path`, `ms_elapsed`,
  `gen_meta`, `echo_mismatches`, full `echo_attempts` list.

The LLM's "validator-and-retry stack" is treated as part of the probe,
not infrastructure around it. v1 had a validator bug (int/float type
mismatch on cash); v2 fixed it. Both versions are reported.

---

## Run book

**Activate env once:**
```
conda activate cs224r-proj
```

**LLM weights (one-time, ~3 GB):**
```
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5B
```

**Task 1 full eval** (~6 h sequential on a single 24 GB GPU):
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

**Task 2 (LLM-driven GA, 2p only, ~5h on 12 GB GPU):**
```
python scripts/optimize_board_llm.py --n-players 2 --pop 8 --gens 5 \
    --n-seeds 5 --out-dir logs/optimizer_llm/llm_ga_2p \
    --model-name models/qwen2.5-1.5B
```

**Rule-based GA + ablations + cross-eval** (re-run command will be
issued by the user separately; once finalised, the canonical command
will be recorded here and `RESULTS.md` will be updated).

---

## Conventions

These were saved as auto-memory from prior sessions and remain in force:

- **Step-by-step code changes.** Explain each change before writing it;
  never batch multiple file rewrites silently.
- **Audit results require review before fixing.** When an audit (mine
  or a sub-agent's) finds multiple issues, present the full list to the
  user before applying any fixes.
- **Preserve comments and formatting.** Use `Edit` (not `Write`) on
  existing files; keep all comments, docstrings, backslash continuations.
- **Use existing abstractions.** Hold instances of `GameMechanics` /
  `StandardPlayerSettings` rather than redefining their fields.
- **Add new variants, don't modify existing.** New
  config-parameterised behaviour goes in new functions alongside the
  existing ones.
- **Player settings live in `player_settings.py`.** Not in
  `settings.py`, not in `agents.py`.
- **User runs training/eval commands themselves.** Output the command;
  don't execute long-running python via the Bash tool.
- **Backwards-compat or revert point.** When extending an existing
  function, either keep the signature back-compat OR surface
  `git rev-parse HEAD` + ISO date before risky edits.
- **No em-dashes in the report.** En-dashes for ranges (`6--10 s/decision`)
  are fine; `---` and U+2014 are not.

---

## Git hygiene

- `models/`, `logs/`, `wandb/` are gitignored. Models re-download via
  `huggingface-cli`.
- `*.txt` is gitignored except `prompts/*.txt` and `notes/*.txt`
  (curated documentation).
- Don't push the HF cache or experiment logs; do push curated outputs
  (CSVs, `summary.csv`, `analysis_*.md`, `evals.jsonl`, `best_design.json`)
  needed for reviewer reproducibility.
- The repo lives at https://github.com/Selim-Emir-Can/CS348K-proj.
  Active branch: `master`. Alaz has parallel work on `main` and
  `round1-phase1`; merging is a follow-up, not part of this submission.

---

## Pointers

- **Numbers:** `monopoly/RESULTS.md`
- **Submission draft:** `monopoly/report/report_cs348k.tex`
- **Older CVPR draft (reference only):** `monopoly/report/report.tex`
- **Task 1 postmortem:** `monopoly/notes/task1_postmortem_2026-04-29.md`
- **Task 2 results:** `monopoly/notes/task2_results_2026-04-30.md`
- **Cross-eval (LLM-GA):** `monopoly/notes/cross_eval_llm_ga_2026-04-30.md`
- **Revert points:** `monopoly/notes/revert_points.txt`
