"""LLM-as-rule-designer closed loop, v1 (item #5 of the design-loop expansion).

Claude Sonnet 4.6 sees the same diagnostic feed as #4 plus a description of
the Step-2 structured-patch whitelist (salary, tax, jail, free-parking
jackpot, single card edits, per-group rent multiplier). It emits one or
more structured patches (capped at 2 per iteration in v1; uncapped in v2).
Each iteration applies the patches in the subprocess sandbox per Step 2,
records the resulting score and bootstrap CI, and logs to a per-trajectory
JSONL.

Two baselines (CEO plan + ANALYSIS_PLAN §6):
  - random rule menu: each iteration picks 1-2 patches uniformly from
    the fixed RANDOM_RULE_MENU below. Same iteration count as the LLM.
  - Monopoly house-rule static ceiling: a fixed bundle (Free Parking
    jackpot ON; jail fine $50; salary $200; etc.) applied once.

Goodhart audit (CEO #6): post-run, the top-3 rules per loop ranked by
score-improvement magnitude are surfaced into goodhart_audit.md with
reviewer-fillable notes fields.

Wall-clock cap: --max-wall-seconds (default 21600 = 6h). Checked before
each iteration; on overrun the loop exits cleanly with the partial
trajectory on disk.

Usage (from CS348K-proj/):
    # Heuristic smoke (no API; cycles a tiny canned patch list).
    set PYTHONPATH=. && python scripts/llm_rule_loop.py --backend heuristic \
        --boards default --n-seeds 1 --K 3 --n-games 20

    # Production (Claude Sonnet via anthropic SDK).
    set ANTHROPIC_API_KEY=...
    set PYTHONPATH=. && python scripts/llm_rule_loop.py --backend anthropic \
        --model claude-sonnet-4-6 --rule-cap 2 \
        --n-seeds 3 --K 6 --n-games 100

    # v2 (uncapped): same with --rule-cap 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import GameConfig
from optimizer.board_sources import build_five_boards, modify_salary, remove_group
from optimizer.group_design import bootstrap_score_ci, evaluate_config
from optimizer.rule_patches import (
    RejectedPatch, apply_patches, append_rejection, parse_llm_response,
    render_patch_as_diff, summarise_patch, validate_patch,
)
from optimizer.rule_sandbox import run_sandboxed
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool


# --------------------------------------------------------------------------- #
# Designer-LLM (Claude Sonnet via anthropic SDK)                                #
# --------------------------------------------------------------------------- #

DESIGNER_SYSTEM_PROMPT = """You are a tabletop-game designer mutating Monopoly rules to improve a combined score (lower is better; the score penalises unfairness, draw rate, length deviation from 60 rounds, and money-flow deviation from $100/round).

You may emit any number of patches per iteration up to the cap given in the user message. Each patch must be a JSON object whose `kind` is one of these whitelisted categories:

  salary_change         {"kind":"salary_change","value":<int 0..2000>}
  tax_change            {"kind":"tax_change","tax_kind":"luxury"|"income","value":<int 0..2000>}
  income_tax_pct        {"kind":"income_tax_pct","value":<float 0.0..0.5>}
  jail_fine_change      {"kind":"jail_fine_change","value":<int 0..1000>}
  free_parking_jackpot  {"kind":"free_parking_jackpot","enabled":<bool>}
  card_effect_change    {"kind":"card_effect_change","deck":"chance"|"chest","index":<int>,"new_text":<str>}
                        new_text MUST match one of these patterns:
                          "Pay (the )?bank $<int>"
                          "Collect $<int> from (the )?bank"
                          "Advance to Go|Boardwalk|Illinois Avenue|St Charles Place|nearest Railroad|nearest Utility"
                          "Go to Jail" / "Get out of Jail Free"
                          "Pay each player $<int>" / "Collect $<int> from each player"
  property_payout_mult  {"kind":"property_payout_mult","group":"<group>","rent_multiplier":<float 0.5..2.0>}

Reply with EXACTLY ONE JSON object inside a fenced ```json``` block:
{
  "rationale": "<one short sentence>",
  "patches":   [<patch>, ...],         // length 1..rule_cap
  "converged": <true|false>
}

If you cannot improve further, set "patches": [] and "converged": true.
"""


@dataclass
class RuleResponse:
    raw_text:        str
    parsed:          Optional[dict]
    patches:         Optional[List[dict]]      # validated against current cfg
    rationale:       str
    converged:       bool
    parser_status:   str    # 'ok' | 'parser_failure' | 'invalid_patch'
    error:           Optional[str] = None
    token_count:     Optional[int] = None
    rejected:        List[RejectedPatch] = field(default_factory=list)


def parse_rule_response(text: str, cfg: GameConfig,
                          rule_cap: int = 2) -> RuleResponse:
    """Pull a list of validated patches out of the LLM response. The cap of
    2 is the v1 constraint; v2 sets rule_cap=0 (interpreted as no cap)."""
    patches, parse_err = parse_llm_response(text)

    # Backwards-compat path: if the response is itself a wrapper {"rationale":...,
    # "patches":[...], "converged":...} (which is what the system prompt asks
    # for), unwrap it.
    rationale = ''
    converged = False
    if patches and len(patches) == 1 and 'patches' in patches[0]:
        wrapper = patches[0]
        rationale = str(wrapper.get('rationale', ''))
        converged = bool(wrapper.get('converged', False))
        inner = wrapper.get('patches', [])
        if not isinstance(inner, list):
            return RuleResponse(raw_text=text, parsed=wrapper, patches=None,
                                rationale=rationale, converged=converged,
                                parser_status='parser_failure',
                                error='wrapper.patches must be list')
        patches = inner

    if patches is None:
        return RuleResponse(raw_text=text, parsed=None, patches=None,
                            rationale='', converged=False,
                            parser_status='parser_failure', error=parse_err)

    # An empty list with converged=true is a valid {converged} signal.
    if not patches and converged:
        return RuleResponse(raw_text=text, parsed=patches, patches=[],
                            rationale=rationale, converged=True,
                            parser_status='ok')

    # Cap enforcement.
    if rule_cap and len(patches) > rule_cap:
        return RuleResponse(raw_text=text, parsed=patches, patches=None,
                            rationale=rationale, converged=converged,
                            parser_status='invalid_patch',
                            error=f'emitted {len(patches)} patches, cap is {rule_cap}')

    rejected: List[RejectedPatch] = []
    accepted: List[dict] = []
    cfg_running = cfg
    for i, p in enumerate(patches):
        err = validate_patch(p, cfg_running)
        if err is not None:
            rejected.append(RejectedPatch(
                raw_response=text, parsed_patch=p,
                reason=f'patch {i}: {err}'))
            continue
        try:
            cfg_running = apply_patches(cfg_running, [p])
            accepted.append(p)
        except Exception as ex:
            rejected.append(RejectedPatch(
                raw_response=text, parsed_patch=p,
                reason=f'patch {i} apply: {type(ex).__name__}: {ex}'))

    if not accepted and rejected:
        return RuleResponse(raw_text=text, parsed=patches, patches=[],
                            rationale=rationale, converged=converged,
                            parser_status='invalid_patch',
                            error=f'{len(rejected)} patches all rejected',
                            rejected=rejected)

    return RuleResponse(raw_text=text, parsed=patches, patches=accepted,
                        rationale=rationale, converged=converged,
                        parser_status='ok', rejected=rejected)


class RuleDesignerLLM:
    """Backend dispatch (anthropic | heuristic)."""

    _HEURISTIC_CYCLE: List[dict] = [
        {'rationale': 'speed up by raising salary',
         'patches': [{'kind': 'salary_change', 'value': 300}],
         'converged': False},
        {'rationale': 'tax adjusts cash flow',
         'patches': [{'kind': 'tax_change', 'tax_kind': 'luxury', 'value': 75}],
         'converged': False},
        {'rationale': 'no further improvement',
         'patches': [], 'converged': True},
    ]

    def __init__(self, backend: str = 'anthropic', model: Optional[str] = None,
                 max_tokens: int = 600,
                 system_prompt: str = DESIGNER_SYSTEM_PROMPT):
        self.backend = backend
        self.model = model or os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def query(self, user_prompt: str, iteration: int = 0,
              rule_cap: int = 2) -> Tuple[str, Optional[int]]:
        if self.backend == 'heuristic':
            payload = self._HEURISTIC_CYCLE[iteration % len(self._HEURISTIC_CYCLE)]
            return '```json\n' + json.dumps(payload) + '\n```', None
        return self._query_anthropic(user_prompt, rule_cap=rule_cap)

    def _query_anthropic(self, prompt: str, rule_cap: int) -> Tuple[str, Optional[int]]:
        # Imported lazily so heuristic-backend smoke runs don't require the SDK.
        try:
            import anthropic   # type: ignore
        except ImportError as ex:
            raise RuntimeError(
                'anthropic SDK not installed; pip install anthropic, or run '
                'with --backend heuristic for smoke tests') from ex
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise RuntimeError('ANTHROPIC_API_KEY not set')
        client = anthropic.Anthropic(api_key=api_key)
        cap_str = ('UNCAPPED' if not rule_cap else str(rule_cap))
        suffix = f'\n\nRULE CAP THIS ITERATION: {cap_str}'
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{'role': 'user', 'content': prompt + suffix}],
        )
        # Aggregate text blocks and pull the input/output token count from
        # the response usage if present.
        text = ''
        for block in msg.content:
            if getattr(block, 'type', '') == 'text':
                text += block.text
        ntok = getattr(getattr(msg, 'usage', None), 'output_tokens', None)
        return text, ntok


# --------------------------------------------------------------------------- #
# Random / house-rule baselines                                                  #
# --------------------------------------------------------------------------- #

# Fixed random rule menu: every entry is a structured patch the random
# baseline can emit. Picked for diversity across whitelisted kinds.
RANDOM_RULE_MENU: List[dict] = [
    {'kind': 'salary_change', 'value': 300},
    {'kind': 'salary_change', 'value': 100},
    {'kind': 'tax_change', 'tax_kind': 'luxury', 'value': 200},
    {'kind': 'tax_change', 'tax_kind': 'luxury', 'value': 0},
    {'kind': 'tax_change', 'tax_kind': 'income', 'value': 100},
    {'kind': 'tax_change', 'tax_kind': 'income', 'value': 400},
    {'kind': 'income_tax_pct', 'value': 0.05},
    {'kind': 'income_tax_pct', 'value': 0.20},
    {'kind': 'jail_fine_change', 'value': 25},
    {'kind': 'jail_fine_change', 'value': 200},
    {'kind': 'free_parking_jackpot', 'enabled': True},
    {'kind': 'free_parking_jackpot', 'enabled': False},
]


def random_baseline_step(rng: np.random.Generator, rule_cap: int) -> List[dict]:
    """Emit 1..rule_cap random patches from the menu."""
    cap = rule_cap if rule_cap else 2
    n = int(rng.integers(1, cap + 1))
    idx = rng.choice(len(RANDOM_RULE_MENU), size=n, replace=False)
    return [dict(RANDOM_RULE_MENU[int(i)]) for i in idx]


# Static house-rule ceiling. Five popular Monopoly house rules encoded as
# patches; applied as a single iteration. Picked to actually DEVIATE from
# canonical defaults so the ceiling is a meaningful comparator (a bundle
# whose patches are all canonical no-ops would be indistinguishable from
# baseline). Each entry is a folk-wisdom intervention people commonly use
# to "make Monopoly less brutal":
HOUSE_RULE_BUNDLE: List[dict] = [
    # Free Parking jackpot - the most popular house rule (default OFF).
    {'kind': 'free_parking_jackpot', 'enabled': True},
    # Easier jail escape (default 50).
    {'kind': 'jail_fine_change', 'value': 25},
    # Higher Go salary - shortens games (default 200).
    {'kind': 'salary_change', 'value': 300},
    # Lighter luxury tax (default 100).
    {'kind': 'tax_change', 'tax_kind': 'luxury', 'value': 50},
    # Lighter income tax bracket (default 0.10).
    {'kind': 'income_tax_pct', 'value': 0.05},
]


# --------------------------------------------------------------------------- #
# Diagnostic feed                                                                #
# --------------------------------------------------------------------------- #

def _per_group_breakdown_str(cfg: GameConfig) -> str:
    from monopoly.core.cell import Property
    rows: Dict[str, List[int]] = {}
    for c in cfg.cells:
        if not isinstance(c, Property): continue
        rows.setdefault(c.group, []).append((c.cost_base, c.rent_base))
    lines = []
    for g, lst in rows.items():
        n = len(lst)
        mc = sum(x[0] for x in lst) / max(n, 1)
        mr = sum(x[1] for x in lst) / max(n, 1)
        lines.append(f'  - {g:>10}: n={n} mean_cost=${mc:.0f} mean_rent=${mr:.0f}')
    return '\n'.join(lines)


def _patches_summary(patches: List[dict]) -> str:
    if not patches: return '  (no patches yet)'
    return '\n'.join(f'  - {summarise_patch(p)}' for p in patches)


def build_rule_feed(cfg: GameConfig, applied_patches: List[dict],
                    eval_out: dict, prior_eval: Optional[dict] = None,
                    hazard_summary: Optional[str] = None,
                    rb_llm_agreement: Optional[str] = None,
                    rule_cap: int = 2) -> str:
    metrics = eval_out.get('metrics', {})
    delta = ''
    if prior_eval is not None:
        delta = f'  delta_vs_prev: {eval_out["score"] - prior_eval["score"]:+.4f}\n'
    parts = []
    parts.append('## CURRENT EVAL')
    parts.append(f'  score: {eval_out["score"]:.4f}  (lower is better)')
    parts.append(f'  mean_rounds: {metrics.get("mean_rounds", 0):.1f}  (target 60)')
    parts.append(f'  mean_draw_rate: {metrics.get("mean_draw_rate", 0):.3f}')
    parts.append(f'  mean_fairness: {metrics.get("mean_fairness", 0):.3f}')
    parts.append(f'  mean_transfer_rate: {metrics.get("mean_transfer_rate", 0):.1f}/round  (target 100)')
    if delta: parts.append(delta.rstrip())
    parts.append('\n## RULES APPLIED SO FAR')
    parts.append(_patches_summary(applied_patches))
    parts.append('\n## CURRENT MECHANICS')
    m = cfg.settings.mechanics
    parts.append(f'  salary={m.salary}  luxury_tax={m.luxury_tax}  '
                 f'income_tax={m.income_tax}  income_tax_pct={m.income_tax_percentage:.3f}  '
                 f'jail_fine={m.exit_jail_fine}  free_parking_jackpot={m.free_parking_money}')
    parts.append('\n## PER-GROUP COST/RENT BREAKDOWN')
    parts.append(_per_group_breakdown_str(cfg))
    if hazard_summary:
        parts.append('\n## HAZARD SUMMARY (prior)')
        parts.append(hazard_summary)
    if rb_llm_agreement:
        parts.append('\n## RB vs LLM AGREEMENT')
        parts.append(rb_llm_agreement)
    parts.append('\n## YOUR JOB')
    parts.append(f'  Propose at most {rule_cap if rule_cap else "(uncapped)"} '
                 f'rule mutations. Reply only with the JSON wrapper described '
                 f'in the system prompt.')
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Trajectory runner                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class RuleIteration:
    iter:               int
    applied_patches:    List[dict]
    rationale:          str
    parser_status:      str
    parser_error:       Optional[str]
    converged_request:  bool
    score:              Optional[float]
    metrics:            Optional[dict]
    score_ci:           Optional[dict]
    delta_vs_prev:      Optional[float]
    improvement:        Optional[bool]
    n_games:            int
    sandbox_ok:         bool
    sandbox_failure:    Optional[str]
    token_count:        Optional[int]
    n_rejected:         int
    wall_seconds:       float
    raw_response:       str


def _is_improvement(prev_score: float, cur_score: float, ci: dict,
                    rel: float = 0.03) -> bool:
    if prev_score <= 0:
        return False
    return ((prev_score - cur_score) / prev_score >= rel
             and ci.get('ci_hi', cur_score) < prev_score)


def _eval_with_sandbox(cfg_yaml_path: str, patches: List[dict], pool_path: str,
                        matchups, n_games: int, base_seed: int, max_turns: int,
                        per_game_timeout: float = 10.0) -> dict:
    """Subprocess sandbox eval per Step 2 — returns a dict shaped like
    evaluate_config's output (score / metrics / per_game_records) when ok,
    else a sentinel dict with 'sandbox_failure'."""
    res = run_sandboxed(
        cfg_yaml_path=cfg_yaml_path, patches=patches, pool_path=pool_path,
        matchups=matchups, n_games=n_games, base_seed=base_seed,
        max_turns=max_turns, per_game_timeout_seconds=per_game_timeout,
    )
    if not res.ok:
        return {'sandbox_failure': res.failure_reason,
                'score': None, 'metrics': None, 'per_game_records': []}
    return {
        'score': res.aggregate_score, 'metrics': res.metrics,
        'per_game_records': res.per_game,
        'sandbox_failure': None,
    }


def run_rule_trajectory(cfg_yaml_path: str, board_label: str, seed: int,
                         pool_path: str, matchups, n_games: int, K: int,
                         designer: RuleDesignerLLM, max_turns: int,
                         out_path: Path, rule_cap: int,
                         hazard_summary: Optional[str] = None,
                         rb_llm_agreement: Optional[str] = None,
                         max_wall_seconds: float = 21600.0,
                         rejected_corpus_path: Optional[Path] = None,
                         ) -> List[RuleIteration]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    iterations: List[RuleIteration] = []
    applied: List[dict] = []
    base_cfg = GameConfig.from_yaml(cfg_yaml_path)

    # Iteration 0: baseline (no patches).
    t0 = time.perf_counter()
    base_ev = _eval_with_sandbox(cfg_yaml_path, [], pool_path,
                                  matchups, n_games, seed, max_turns)
    if base_ev.get('sandbox_failure'):
        rec = RuleIteration(0, [], '[baseline]', 'sandbox_failure',
                            base_ev['sandbox_failure'], False, None, None,
                            None, None, None, 0, False,
                            base_ev['sandbox_failure'], None, 0,
                            time.perf_counter() - t0, '')
        fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush(); fh.close()
        return [rec]
    base_ci = bootstrap_score_ci(base_ev['per_game_records'], seed=seed)
    rec0 = RuleIteration(0, [], '[baseline]', 'baseline', None, False,
                          base_ev['score'], base_ev['metrics'], base_ci,
                          None, None, n_games, True, None, None, 0,
                          time.perf_counter() - t0, '')
    iterations.append(rec0)
    fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()
    prev_score = base_ev['score']
    cfg_running = base_cfg

    t_start = time.perf_counter()

    for k in range(1, K + 1):
        elapsed = time.perf_counter() - t_start
        if elapsed > max_wall_seconds:
            print(f'[wall-clock cap] elapsed {elapsed:.0f}s > {max_wall_seconds:.0f}s; '
                  f'stopping trajectory before iter {k}')
            break

        t0 = time.perf_counter()
        feed = build_rule_feed(cfg_running, applied,
                                eval_out={'score': prev_score,
                                           'metrics': iterations[-1].metrics or {}},
                                prior_eval=({'score': iterations[-2].score}
                                              if len(iterations) >= 2 else None),
                                hazard_summary=hazard_summary,
                                rb_llm_agreement=rb_llm_agreement,
                                rule_cap=rule_cap)
        try:
            raw, ntok = designer.query(feed, iteration=k - 1, rule_cap=rule_cap)
        except Exception as ex:
            rec = RuleIteration(k, [], '', 'backend_failure',
                                f'{type(ex).__name__}: {ex}', False, None,
                                None, None, None, None, 0, False, None,
                                None, 0, time.perf_counter() - t0, '')
            iterations.append(rec)
            fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            break

        resp = parse_rule_response(raw, cfg_running, rule_cap=rule_cap)
        resp.token_count = ntok

        # Persist any rejected patches to the corpus.
        if rejected_corpus_path is not None:
            for rej in resp.rejected:
                rej.iteration = k
                rej.seed = seed
                rej.board_label = board_label
                append_rejection(rej, path=rejected_corpus_path)

        if resp.parser_status not in ('ok',):
            rec = RuleIteration(k, [], resp.rationale, resp.parser_status,
                                resp.error, resp.converged, None, None, None,
                                None, None, 0, False, None, resp.token_count,
                                len(resp.rejected),
                                time.perf_counter() - t0, raw)
            iterations.append(rec)
            fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            break

        if not resp.patches:
            # Empty patch list with converged=True is the {converged} signal.
            rec = RuleIteration(k, [], resp.rationale, 'ok', None, True,
                                prev_score, iterations[-1].metrics,
                                iterations[-1].score_ci, 0.0, False,
                                0, True, None, resp.token_count,
                                len(resp.rejected),
                                time.perf_counter() - t0, raw)
            iterations.append(rec)
            fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            break

        # Apply, eval in sandbox, log.
        new_applied = applied + resp.patches
        ev = _eval_with_sandbox(cfg_yaml_path, new_applied, pool_path,
                                  matchups, n_games, seed, max_turns)
        if ev.get('sandbox_failure'):
            rec = RuleIteration(k, resp.patches, resp.rationale,
                                'sandbox_failure', ev['sandbox_failure'],
                                resp.converged, None, None, None, None, None,
                                0, False, ev['sandbox_failure'],
                                resp.token_count, len(resp.rejected),
                                time.perf_counter() - t0, raw)
            iterations.append(rec)
            fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            break
        ci = bootstrap_score_ci(ev['per_game_records'], seed=seed + k)
        delta = ev['score'] - prev_score
        improvement = _is_improvement(prev_score, ev['score'], ci)

        rec = RuleIteration(k, resp.patches, resp.rationale, 'ok', None,
                            resp.converged, ev['score'], ev['metrics'], ci,
                            delta, improvement, n_games, True, None,
                            resp.token_count, len(resp.rejected),
                            time.perf_counter() - t0, raw)
        iterations.append(rec)
        fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()

        applied = new_applied
        prev_score = ev['score']
        cfg_running = apply_patches(base_cfg, applied)
        if resp.converged:
            break

    fh.close()
    return iterations


# --------------------------------------------------------------------------- #
# Random / house-rule baseline runners                                          #
# --------------------------------------------------------------------------- #

def run_random_baseline(cfg_yaml_path: str, board_label: str, seed: int,
                         pool_path: str, matchups, n_games: int, K: int,
                         max_turns: int, out_path: Path, rule_cap: int) -> List[RuleIteration]:
    """Same loop shape as the LLM but the designer is a fixed-menu sampler."""
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')

    base_cfg = GameConfig.from_yaml(cfg_yaml_path)
    base_ev = _eval_with_sandbox(cfg_yaml_path, [], pool_path, matchups,
                                   n_games, seed, max_turns)
    base_ci = bootstrap_score_ci(base_ev['per_game_records'], seed=seed)
    iterations = [RuleIteration(0, [], '[baseline]', 'baseline', None, False,
                                  base_ev['score'], base_ev['metrics'],
                                  base_ci, None, None, n_games, True, None,
                                  None, 0, 0.0, '')]
    fh.write(json.dumps(asdict(iterations[0])) + '\n'); fh.flush()

    applied: List[dict] = []
    prev_score = base_ev['score']
    for k in range(1, K + 1):
        t0 = time.perf_counter()
        candidate = random_baseline_step(rng, rule_cap or 2)
        # Validate against the running cfg; drop any rejected patch (so the
        # baseline is "random structured patches", not "random sequence
        # including invalid ones").
        new_applied = applied[:]
        cfg_running = apply_patches(base_cfg, applied)
        for p in candidate:
            err = validate_patch(p, cfg_running)
            if err is None:
                new_applied.append(p)
                cfg_running = apply_patches(base_cfg, new_applied)
        ev = _eval_with_sandbox(cfg_yaml_path, new_applied, pool_path,
                                 matchups, n_games, seed, max_turns)
        if ev.get('sandbox_failure'):
            break
        ci = bootstrap_score_ci(ev['per_game_records'], seed=seed + k)
        delta = ev['score'] - prev_score
        rec = RuleIteration(k, candidate, '[random]', 'ok', None, False,
                            ev['score'], ev['metrics'], ci, delta,
                            _is_improvement(prev_score, ev['score'], ci),
                            n_games, True, None, None, 0,
                            time.perf_counter() - t0, '')
        iterations.append(rec)
        fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
        applied = new_applied
        prev_score = ev['score']

    fh.close()
    return iterations


def run_house_rule_ceiling(cfg_yaml_path: str, board_label: str, seed: int,
                            pool_path: str, matchups, n_games: int,
                            max_turns: int, out_path: Path) -> dict:
    """Apply HOUSE_RULE_BUNDLE in one shot; return a single eval dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_cfg = GameConfig.from_yaml(cfg_yaml_path)

    base_ev = _eval_with_sandbox(cfg_yaml_path, [], pool_path, matchups,
                                  n_games, seed, max_turns)
    house_ev = _eval_with_sandbox(cfg_yaml_path, HOUSE_RULE_BUNDLE, pool_path,
                                   matchups, n_games, seed, max_turns)
    rec = {
        'board':            board_label,
        'seed':             seed,
        'baseline_score':   base_ev['score'],
        'house_score':      house_ev['score'],
        'house_rules':      HOUSE_RULE_BUNDLE,
        'baseline_metrics': base_ev['metrics'],
        'house_metrics':    house_ev['metrics'],
    }
    out_path.write_text(json.dumps(rec, indent=2))
    return rec


# --------------------------------------------------------------------------- #
# Goodhart audit                                                                 #
# --------------------------------------------------------------------------- #

def goodhart_audit(traj_dirs: List[Path], out_path: Path, top_k: int = 3) -> None:
    """Walk per-trajectory JSONL files, pick the top_k iterations per loop
    by score-improvement magnitude, write an editable markdown table."""
    lines = ['# Goodhart audit (LLM rule loop)',
             '',
             'For each loop, the top {} score-improving rule iterations are '
             'surfaced for human review. Fill in the `Verdict` column with '
             'GENUINE / GOODHART / UNCLEAR and a short note. Five-minute '
             'budget per loop.'.format(top_k),
             '',
             '| Loop | Iter | Score Δ | Patches (rendered) | Verdict | Reviewer note |',
             '|------|------|---------|--------------------|---------|---------------|']

    for d in traj_dirs:
        for path in sorted(d.glob('*.jsonl')):
            recs = []
            try:
                for line in path.read_text().splitlines():
                    if line.strip():
                        recs.append(json.loads(line))
            except Exception:
                continue
            improving = [r for r in recs
                          if r.get('parser_status') == 'ok'
                          and r.get('delta_vs_prev') is not None
                          and r['delta_vs_prev'] < 0]
            improving.sort(key=lambda r: r['delta_vs_prev'])
            for r in improving[:top_k]:
                patches = r.get('applied_patches', [])
                rendered = '<br/>'.join(summarise_patch(p) for p in patches) or '(none)'
                rendered = rendered.replace('|', '\\|')
                lines.append(f'| {path.stem} | {r["iter"]} | '
                             f'{r["delta_vs_prev"]:+.4f} | {rendered} | _____ | _____ |')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--mini-config',      default='configs/mini')
    ap.add_argument('--ga-2p', default='logs/optimizer/ga_2p.jsonl')
    ap.add_argument('--ga-3p', default='logs/optimizer/ga_3p.jsonl')
    ap.add_argument('--pool', default='optimizer/strategy_pool.json')
    ap.add_argument('--boards', default='all')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--n-games', type=int, default=100)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--rule-cap', type=int, default=2,
                    help='v1: 2; v2: 0 (uncapped). Per CEO plan two-stage release.')
    ap.add_argument('--backend', choices=('anthropic', 'heuristic'),
                    default='anthropic')
    ap.add_argument('--model', default=None,
                    help='ANTHROPIC_MODEL override (default: claude-sonnet-4-6)')
    ap.add_argument('--max-wall-seconds', type=float, default=21600.0,
                    help='Hard cap on per-trajectory wall-clock (CEO C9).')
    ap.add_argument('--out-dir', default='report/figures/llm_rule_loop')
    ap.add_argument('--variant', default='v1',
                    help='Subdir under --out-dir to keep v1 / v2 separate.')
    ap.add_argument('--skip-llm',     action='store_true', help='Run only the baselines.')
    ap.add_argument('--skip-random',  action='store_true')
    ap.add_argument('--skip-house',   action='store_true')
    ap.add_argument('--skip-audit',   action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Resolving boards...')
    sources = build_five_boards(
        canonical_config=args.canonical_config, mini_config=args.mini_config,
        ga_2p=args.ga_2p, ga_3p=args.ga_3p)
    canon_cfg = GameConfig.from_yaml(args.canonical_config)
    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        sources = [(l, c) for l, c in sources if l in wanted]

    # Each starting cfg gets serialised to a temp YAML so the sandbox
    # subprocess can load it. salary x2 / drop Brown are re-applied against
    # canonical to honour the canonical-iteration decision (LOCK §7).
    boards_for_loop: List[Tuple[str, str]] = []
    yaml_dir = out_dir / 'starting_cfgs'
    yaml_dir.mkdir(parents=True, exist_ok=True)
    for label, cfg in sources:
        if label == 'salary x2':
            cfg = modify_salary(canon_cfg, 2.0)
        elif label == 'drop Brown':
            cfg = remove_group(canon_cfg, 'Brown')
        tag = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')
        p = yaml_dir / f'{tag}.yaml'
        cfg.to_yaml(str(p))
        boards_for_loop.append((label, str(p)))
    print(f'  starting boards: {[l for l, _ in boards_for_loop]}')

    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                    n_matchups=args.n_matchups,
                                    seed=args.matchup_seed)

    rejected_corpus = Path('report/figures/llm_rules/rejected_corpus.jsonl')

    if not args.skip_llm:
        designer = RuleDesignerLLM(backend=args.backend, model=args.model)
        llm_dir = out_dir / 'llm'
        llm_dir.mkdir(parents=True, exist_ok=True)
        for board_label, cfg_path in boards_for_loop:
            for s in range(args.n_seeds):
                seed = args.seed_offset + s * 1000 + 42
                tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
                op = llm_dir / f'{tag}__seed{seed}.jsonl'
                print(f'\n[LLM][{board_label}] seed={seed}')
                run_rule_trajectory(
                    cfg_yaml_path=cfg_path, board_label=board_label,
                    seed=seed, pool_path=args.pool, matchups=matchups,
                    n_games=args.n_games, K=args.K, designer=designer,
                    max_turns=args.max_turns, out_path=op,
                    rule_cap=args.rule_cap,
                    max_wall_seconds=args.max_wall_seconds,
                    rejected_corpus_path=rejected_corpus)

    if not args.skip_random:
        rand_dir = out_dir / 'random'
        rand_dir.mkdir(parents=True, exist_ok=True)
        for board_label, cfg_path in boards_for_loop:
            for s in range(args.n_seeds):
                seed = args.seed_offset + s * 1000 + 42
                tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
                op = rand_dir / f'{tag}__seed{seed}.jsonl'
                print(f'[rand][{board_label}] seed={seed}')
                run_random_baseline(
                    cfg_yaml_path=cfg_path, board_label=board_label, seed=seed,
                    pool_path=args.pool, matchups=matchups,
                    n_games=args.n_games, K=args.K, max_turns=args.max_turns,
                    out_path=op, rule_cap=args.rule_cap)

    if not args.skip_house:
        house_dir = out_dir / 'house'
        house_dir.mkdir(parents=True, exist_ok=True)
        for board_label, cfg_path in boards_for_loop:
            seed = args.seed_offset + 42
            tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
            op = house_dir / f'{tag}.json'
            print(f'[house][{board_label}] seed={seed}')
            run_house_rule_ceiling(
                cfg_yaml_path=cfg_path, board_label=board_label, seed=seed,
                pool_path=args.pool, matchups=matchups,
                n_games=args.n_games, max_turns=args.max_turns, out_path=op)

    if not args.skip_audit:
        traj_dirs = [out_dir / 'llm']
        if (out_dir / 'random').exists(): traj_dirs.append(out_dir / 'random')
        goodhart_audit(traj_dirs, out_dir / 'goodhart_audit.md', top_k=3)
        print(f'  goodhart_audit.md -> {out_dir}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
