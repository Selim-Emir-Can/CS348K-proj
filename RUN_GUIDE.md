# Round 1 Run Guide

This guide is the operator's manual for the round-1 LLM design-loop
experiments. It assumes the implementation in `ROUND1_ACTION_PLAN.md`
has shipped (Phase 1 complete).

## TL;DR

```bash
# 1. Calibrate (~30 min, gates the overnight)
python scripts/run_round1_overnight.py --calibrate-only \
    --backend local --model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir report/figures/round1

# 2. Overnight (~6-9 hr, hard cap 10 hr)
python scripts/run_round1_overnight.py \
    --backend local --model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir report/figures/round1 \
    2>&1 | tee logs/round1_overnight.log

# 3. Status while running
cat report/figures/round1/STATUS.md

# 4. If interrupted: resume picks up from the last completed phase
python scripts/run_round1_overnight.py --resume \
    --out-dir report/figures/round1
```

## Cluster setup (recommended for production)

The overnight runs every phase as a subprocess and would re-load model
weights each time. On a cluster, the recommended pattern is to start
**vLLM as a persistent OpenAI-compatible server** so the model loads
once and every script hits it via HTTP. The existing `--backend openai`
path is exactly the right plumbing.

### One-time setup

```bash
pip install -r requirements.txt
pip install vllm                # use the version pinned for your CUDA build

# Pre-download Qwen weights to shared scratch so jobs don't refetch.
export LLM_CACHE_DIR=/shared/scratch/$USER/hf_cache
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='$LLM_CACHE_DIR')"
```

### Start the vLLM server (one node)

```bash
# bf16 + flash attention + greedy + deterministic. Seed locks token-level
# determinism even with batched concurrent requests.
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --dtype bfloat16 \
    --enforce-eager \
    --seed 42 \
    --download-dir $LLM_CACHE_DIR \
    --port 8000 \
    --max-model-len 4096 \
    --disable-log-requests
```

`--enforce-eager` disables CUDA graphs, which matters for determinism;
the perf cost is small at this model size. `--seed` fixes the sampler's
RNG even though we use temperature=0 (some vLLM versions still consult
the RNG for tie-breaking).

### Point the overnight at it

```bash
export LLM_OPENAI_BASE_URL=http://localhost:8000/v1
export LLM_OPENAI_API_KEY=no-key                       # vLLM ignores this
export LLM_OPENAI_MODEL=Qwen/Qwen2.5-1.5B-Instruct

python scripts/run_round1_overnight.py \
    --backend openai \
    --out-dir report/figures/round1 \
    2>&1 | tee logs/round1_overnight.log
```

The model loads once when vLLM starts, then T-SANITY → PHASE-A-ROBUSTNESS
→ MIRROR-H → all 5 TUNER conditions hit it via HTTP. No reload between
phases, no GPU memory churn.

### Local-backend optimizations (if not using vLLM)

If you're running `--backend local` (transformers in-process), opt into
flash-attn-2 and bf16 via env vars before launching:

```bash
export LLM_DTYPE=bfloat16          # default 'float16'
export LLM_ATTN_IMPL=flash_attention_2     # default unset
export LLM_CACHE_DIR=/shared/scratch/$USER/hf_cache
```

These are read by `agents.py:LLMPlayer._get_local_model` and
`scripts/llm_design_loop.py:DesignerLLM._get_local`. They're recorded
into every JSONL record's `gen_cfg.dtype` / `gen_cfg.attn_impl` so a
post-run audit can verify the entire run used one configuration.

### Scientific validity of these optimizations

Each LLM call's "instrument" is the *combination* of (weights,
tokenizer, inference engine, precision, attention impl, decoding
strategy, generation params). The round-1 lock requires:

  - The instrument is fixed once *before* calibration and unchanged
    through the overnight (no engine swap mid-run, no precision
    change between conditions, no model snapshot bump).
  - Calibration runs on the same instrument as the overnight (the
    gates are only meaningful if calibration tested the same thing).
  - The instrument is recorded in `gen_cfg` for every JSONL record.

Determinism (`torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, both set by the overnight driver)
costs ~5-10% throughput but is what makes calibration→overnight gating
meaningful and lets reviewers reproduce a specific iteration from its
seed. Keep them on.

vLLM ≠ transformers bit-for-bit even with greedy + temperature=0. That's
fine — the experiment is internally valid as long as every condition uses
the same engine. **Don't switch engines mid-run.**

## What this round produces

Under `report/figures/round1/`:

- `STATUS.md` — live human-readable phase dashboard.
- `timing.jsonl` — append-only event log; source of truth for the phase
  state machine and the post-run timing report.
- `sanity/` — T-SANITY trajectory artefact (eval-pipeline correctness).
- `phase_a_robustness/` — PHASE-A-ROBUSTNESS results (NEUTRAL-PLAYER
  re-run of the original Phase A matrix).
- `mirror_h/` — MIRROR-H artefacts: `decisions.jsonl`,
  `data_quality.json` (with the format-pass-rate gate), concept
  frequency + divergence + noise-floor + grounded-rate figures, plus
  Empath and sentence-BERT robustness matrices.
- `tuner/<condition>/` — TUNER trajectories per condition, one JSONL
  per (board, seed); plus `trajectories_<cond>.{pdf,png}`,
  `rationales_<cond>.md`, `summary_<cond>.json`.
- `timing_report.md` — measured per-phase / per-LLM-call timings,
  written by `scripts/summarize_round1_timing.py`. Replaces the
  Apr-27 estimates.

## The three experiments

### PHASE-A-ROBUSTNESS

A re-run of the original Phase A matrix using NEUTRAL-PLAYER (no
strategic scaffolding, no few-shot). Original Phase A used
GUIDED-PLAYER, whose 4-shot examples reference real properties on the
canonical default board (G4 leak, see `prompts/README.md`). The
robustness re-run answers: was the cross-source-agreement signal driven
by the strategic scaffolding, or by the LLM's own induction from board
state?

### MIRROR-H

LLM character induction by environment, half-size (6 games × 5 boards =
30 self-play games). Hold the LLM completely fixed (NEUTRAL-PLAYER on
every seat, every board); vary only the board. Question: does the same
fixed instrument exhibit different strategic character in different
environments, and is that character visible in its reasoning?

Primary analysis: 10-concept dictionary L1 over per-board reasoning
corpora, with within-board noise floor as the meaningful-signal
threshold. Robustness: Empath-200 + sentence-BERT cosine, reported
alongside primary as agreement check. Format-pass-rate gate at 70%.

### TUNER

LLM-as-parametric-designer closed loop, with a 5-condition feedback
ablation:

  - **T-MUTE** — no diagnostic feed (control)
  - **T-HAZ** — hazards channel only
  - **T-MET** — metrics + per-group breakdown + strategy-pool
    exploit-resistance only
  - **T-FULL** — everything, GOAL-OPEN designer prompt
  - **T-BLIND** — everything, GOAL-CLOSED designer prompt

Plus three structural comparators:

  - **T-RAND** — uniform random parametric edits
  - **T-CANON** — no design intervention (variance floor)
  - **T-SANITY** — hardcoded "obviously good" trajectory (eval check)

5 boards × 3 seeds × K=8 iterations × 200 games per iteration per
condition. Headline comparisons (4 predeclared, no formal multiple-
comparison correction): T-FULL vs T-MUTE (does feedback help?), T-HAZ
vs T-MET (which signal helps more?), T-FULL vs T-RAND (does the LLM
beat random?), T-FULL vs T-BLIND (does goal disclosure help?).

## What is NOT in round 1

- **ARCHITECT** (LLM rule-mutation closed loop) — deferred to round 2.
- **MIRROR-F** (full-size MIRROR, 12 games/board) — gated on MIRROR-H
  signal being real.
- **Phase B** (human validation) — pre-committed in
  `notes/phase_b_preregistration_2026-04-27.md`; held until the LLM
  evidence lands.
- **GUIDED-PLAYER few-shot replacement** (synthetic property names) —
  blocks any future GUIDED-PLAYER experiment on the canonical board.

## How to run

### Calibration first

`--calibrate-only` runs four cheap probes and gates on:

  - MIRROR format-pass rate ≥ 70%
  - TUNER JSON parse rate ≥ 80%
  - T-SANITY iter 0→2 strictly monotone-decreasing
  - Calibration wall-clock under the cap

Failure exits non-zero; do not skip.

```bash
python scripts/run_round1_overnight.py --calibrate-only \
    --backend local --model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir report/figures/round1
```

### Overnight execution

```bash
python scripts/run_round1_overnight.py \
    --backend local --model Qwen/Qwen2.5-1.5B-Instruct \
    --out-dir report/figures/round1 \
    --max-wall-seconds 36000 \
    2>&1 | tee logs/round1_overnight.log
```

The driver runs phases in order: T-SANITY → PHASE-A-ROBUSTNESS →
MIRROR-H → TUNER (T-CANON, T-RAND, T-MUTE, T-HAZ, T-MET, T-FULL,
T-BLIND). GPU memory is released between phases. STATUS.md is rewritten
after every phase change.

### Resuming after a crash

```bash
python scripts/run_round1_overnight.py --resume \
    --out-dir report/figures/round1
```

`--resume` reads `timing.jsonl` and skips any phase already marked
complete or skipped. Failed phases are not skipped — they re-run.

## Output layout

```
report/figures/round1/
├── STATUS.md                          # live dashboard
├── timing.jsonl                       # phase/iter/llm_call event log
├── timing_report.md                   # post-run summary
├── sanity/
│   ├── sanity__default__seed42.jsonl
│   ├── trajectories_sanity.{pdf,png}
│   └── SANITY_RESULT.txt
├── phase_a_robustness/
│   └── ...
├── mirror_h/
│   ├── decisions.jsonl
│   ├── outcomes.json
│   ├── data_quality.json              # format-pass-rate gate
│   ├── concept_frequencies.{pdf,png}
│   ├── divergence_matrix.{pdf,png}
│   ├── noise_floor.{pdf,png}
│   ├── grounded_rate.{pdf,png}
│   ├── empath_divergence_matrix.{pdf,png}
│   ├── sbert_cosine_matrix.{pdf,png}
│   ├── empath_divergence.json
│   ├── sbert_cosine.json
│   └── corpus_samples.md
└── tuner/
    ├── canon/
    ├── rand/
    ├── mute/
    ├── haz/
    ├── met/
    ├── full/
    └── blind/
        ├── full__default__seed42.jsonl  (one per board × seed)
        ├── trajectories_full.{pdf,png}
        ├── rationales_full.md
        └── summary_full.json
```

## Reproducing one experiment in isolation

```bash
# MIRROR-H solo
python scripts/llm_character.py \
    --backend local --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-games 6 --max-turns 80 \
    --out-dir report/figures/round1/mirror_h

# TUNER one condition solo
python scripts/llm_design_loop.py \
    --backend local --ablation-condition full \
    --n-seeds 3 --K 8 --n-games 200 \
    --out-dir report/figures/round1/tuner/full

# Re-run analysis on existing decisions.jsonl (fast)
python scripts/llm_character.py --analyse-only \
    --decisions report/figures/round1/mirror_h/decisions.jsonl \
    --out-dir report/figures/round1/mirror_h

# Post-run timing summary
python scripts/summarize_round1_timing.py \
    --in report/figures/round1/timing.jsonl \
    --out report/figures/round1/timing_report.md
```

## How to interpret results — quick reference

**MIRROR-H.** Headline question: does cross-board reasoning divergence
exceed the within-board sampling noise? Read
`mirror_h/divergence.json`:

  - `cross_board_meaningful: true` — primary signal real;
  - Compare with `empath_divergence.json` and `sbert_cosine.json` —
    three independent dictionaries agreeing strengthens the claim;
  - If grounded-rate < 50% on any board, that board's divergence claim
    is flagged "ungrounded" in the writeup.

**TUNER.** Four headline comparisons; for each, check whether the
median final score difference passes the improvement gate (relative
drop ≥ 3% AND 95% bootstrap CI on Δ excludes zero, n_resamples=1500).
Use `summary_<cond>.json` and `rationales_<cond>.md` for each
condition. Inspect `trajectories_<cond>.pdf` for per-seed shape.

  - C3 (FULL) vs C0 (MUTE) — does feedback help?
  - C1 (HAZ) vs C2 (MET) — which signal helps more?
  - C3 (FULL) vs T-RAND — does LLM beat random?
  - C3 (FULL) vs T-BLIND — does goal disclosure help?

Goodhart audit: render the best final board per (condition × starting
board) and apply the visual check (≥3 colour groups present AND ≥1
monopoly path visible).

## Pre-registered analytical decisions

Frozen before any data lands. See `ANALYSIS_PLAN.md`:

  - §2 concept dictionary (10 concepts, regex)
  - §3 divergence threshold (mean + 2σ over within-board L1 noise)
  - §4 grounded-state precondition (≥ 50% per board)
  - §5 improvement gate (≥ 3% relative AND CI excludes zero, n=1500)
  - §6 comparator-gap (LLM beats random; exploit-resistance is parallel-
    track, NOT in score)
  - §7 iteration board = canonical (audit Spearman ρ = 0.30 < 0.4 cutoff)
  - §11 prior-iteration history redaction per condition
  - §12 generation-parameter lock (Qwen2.5-1.5B, greedy, T=0)

## Round-2 TODO

(Captured here so it doesn't get lost between rounds.)

  - **ARCHITECT**: rule-mutation closed loop with the same 5-condition
    matrix as TUNER; R-PILOT → R-FULL → R-OPUS pass with
    `claude-opus-4-7`.
  - **Neutral-LLM-as-player in eval pool**: add a non-Qwen player
    (Llama or Mistral) to the strategy pool to defuse the
    designer-pleases-its-mirror circularity in TUNER round 2.
  - **Phase B human validation**: 3 testers × 3 boards × 2 games per
    `notes/phase_b_preregistration_2026-04-27.md`.
  - **MIRROR-F**: full-size character induction once MIRROR-H signal
    is real.
  - **Player-model meta-study**: empirically compare {rule-based pool,
    parametric pool, LLM player, mixed pool, human pilot} as test-
    benches for the same design.
  - **GUIDED-PLAYER few-shot replacement**: synthetic property names
    before any future GUIDED-PLAYER experiment on canonical.
  - **Sparse-prompt robustness check**: re-run Phase A with Opus 4.7 +
    minimal prompt to test whether agreement structure depends on
    rich-prompt scaffolding.
