"""LLM-as-parametric-designer closed loop (#4 of the LLM design-loop expansion).

Qwen2.5-1.5B sees a diagnostic feed (hazards, RB-vs-LLM agreement, per-group
cost/rent breakdown, design diff from default) and emits a hierarchical
group-level intervention with rationale text. Loop runs up to K iterations
or until the LLM declares {converged}.

Iteration board: per Step 0 audit (Spearman rho=0.30 < 0.4 cutoff), the
iteration board is CANONICAL, not mini. ANALYSIS_PLAN §7.

Per-iteration JSONL logged to <out_dir>/<board>_seed<S>.jsonl with:
  - design vector and human-readable diff from default
  - rationale text
  - eval output (score + metrics) and per-iter CI on score delta
  - parser status (ok / parser_failure / invalid_patch)
  - LLM token count

Fixed across the run (per CEO plan + lock):
  - 5 starting boards from optimizer.board_sources.build_five_boards
  - 3 seeds per board => 15 trajectories
  - K=8 iterations max with LLM-declared {converged} action
  - Eval n_games per iteration; pool + matchups loaded once at start

Usage (from CS348K-proj/):
    # Heuristic smoke (no LLM); validates the loop end-to-end in a few sec.
    set PYTHONPATH=. && python scripts/llm_design_loop.py --backend heuristic \
        --boards default --n-seeds 1 --K 3 --n-games 20

    # Production:
    set PYTHONPATH=. && python scripts/llm_design_loop.py --backend local \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --n-seeds 3 --K 8 --n-games 100 \
        --out-dir report/figures/llm_design_loop
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
from monopoly.core.cell import Property
from optimizer.board_sources import build_five_boards
from optimizer.group_design import (GroupDesign, apply_design,
                                       bootstrap_score_ci, evaluate_config)
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool


# --------------------------------------------------------------------------- #
# Designer-LLM wrapper                                                          #
# --------------------------------------------------------------------------- #

DESIGNER_SYSTEM_PROMPT = (
    "You are a tabletop-game designer reasoning about a Monopoly variant. "
    "Each turn you receive a DIAGNOSTIC FEED describing the current board's "
    "behaviour (length, fairness, cash flow, hazard shape, per-group "
    "cost/rent breakdown, and the diff from the default board). Your job "
    "is to propose ONE small intervention to the parametric design that "
    "would improve the combined score (lower is better; the score "
    "penalises unfairness, draw rate, length deviation from 60 rounds, "
    "and money-flow deviation from $100/round).\n\n"
    "You must reply with EXACTLY one JSON object inside a fenced ```json``` "
    "block, with no prose outside the fences, in this schema:\n"
    "{\n"
    '  "rationale": "<one short sentence>",\n'
    '  "intervention": {\n'
    '     "salary_mult": <float in [0.5, 2.0]>,\n'
    '     "drop_groups": [<group_name strings>],\n'
    '     "group_cost_mult": {"<group>": <float in [0.5, 2.0]>, ...},\n'
    '     "group_rent_mult": {"<group>": <float in [0.5, 2.0]>, ...},\n'
    '     "prop_overrides": {"<prop_name>": {"cost_base": <int>, "rent_base": <int>}}\n'
    '  },\n'
    '  "converged": <true|false>\n'
    "}\n\n"
    "Set converged=true ONLY if you believe further intervention will not "
    "improve the score. Use the property-level escape hatch (prop_overrides) "
    "rarely; group-level levers are preferred. Group names you reference "
    "must appear in the per-group breakdown shown in the feed."
)


_RESPONSE_BLOCK = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)


@dataclass
class DesignerResponse:
    raw_text:       str
    parsed:         Optional[dict]      # full parsed payload
    design:         Optional[GroupDesign]
    rationale:      str
    converged:      bool
    parser_status:  str                  # 'ok' | 'parser_failure' | 'invalid_intervention'
    error:          Optional[str] = None
    token_count:    Optional[int] = None


def _coerce_design(intervention: dict, rationale: str) -> Optional[GroupDesign]:
    """Convert the LLM's intervention dict into a GroupDesign, returning None
    if any field has the wrong type (caller logs `parser_status=invalid`)."""
    try:
        return GroupDesign(
            salary_mult       = float(intervention.get('salary_mult', 1.0)),
            drop_groups       = list(intervention.get('drop_groups', [])),
            group_cost_mult   = {str(k): float(v) for k, v in
                                  (intervention.get('group_cost_mult', {}) or {}).items()},
            group_rent_mult   = {str(k): float(v) for k, v in
                                  (intervention.get('group_rent_mult', {}) or {}).items()},
            prop_overrides    = {str(k): dict(v) for k, v in
                                  (intervention.get('prop_overrides', {}) or {}).items()},
            label             = '',
            rationale         = rationale[:500],
        )
    except (TypeError, ValueError) as ex:
        return None


def parse_designer_response(text: str) -> DesignerResponse:
    """Extract a structured DesignerResponse from a raw LLM text."""
    candidates: List[str] = []
    for m in _RESPONSE_BLOCK.finditer(text):
        candidates.append(m.group(1))
    if not candidates:
        candidates.append(text.strip())

    last_err = 'no parseable JSON in response'
    parsed = None
    for c in candidates:
        try:
            parsed = json.loads(c)
            break
        except json.JSONDecodeError as ex:
            last_err = f'JSONDecodeError: {ex.msg}'
            parsed = None
    if parsed is None or not isinstance(parsed, dict):
        return DesignerResponse(raw_text=text, parsed=None, design=None,
                                rationale='', converged=False,
                                parser_status='parser_failure', error=last_err)

    rationale = str(parsed.get('rationale', '')).strip()
    converged = bool(parsed.get('converged', False))
    intervention = parsed.get('intervention', {}) or {}
    if not isinstance(intervention, dict):
        return DesignerResponse(raw_text=text, parsed=parsed, design=None,
                                rationale=rationale, converged=converged,
                                parser_status='invalid_intervention',
                                error='intervention must be an object')

    design = _coerce_design(intervention, rationale)
    if design is None:
        return DesignerResponse(raw_text=text, parsed=parsed, design=None,
                                rationale=rationale, converged=converged,
                                parser_status='invalid_intervention',
                                error='could not coerce intervention to GroupDesign')

    return DesignerResponse(raw_text=text, parsed=parsed, design=design,
                            rationale=rationale, converged=converged,
                            parser_status='ok')


class DesignerLLM:
    """Wraps backend dispatch (heuristic / local / openai). Heuristic backend
    cycles through a tiny library of canned interventions so the loop driver
    can be smoke-tested without any model."""

    _MODEL_CACHE: dict = {}

    _HEURISTIC_CYCLE: List[dict] = [
        {'rationale': 'pacing too long; raise salary',
         'intervention': {'salary_mult': 1.25}, 'converged': False},
        {'rationale': 'orange rents look weak; bump',
         'intervention': {'group_rent_mult': {'Orange': 1.25}}, 'converged': False},
        {'rationale': 'no improvement margin remaining',
         'intervention': {}, 'converged': True},
    ]

    def __init__(self, backend: str = 'local', model_name: Optional[str] = None,
                 max_new_tokens: int = 320, system_prompt: str = DESIGNER_SYSTEM_PROMPT):
        self.backend = backend
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def query(self, user_prompt: str, iteration: int = 0) -> Tuple[str, Optional[int]]:
        if self.backend == 'heuristic':
            payload = self._HEURISTIC_CYCLE[iteration % len(self._HEURISTIC_CYCLE)]
            return '```json\n' + json.dumps(payload) + '\n```', None
        if self.backend == 'openai':
            return self._query_openai(user_prompt), None
        return self._query_local(user_prompt)

    def _get_local(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        name = self.model_name or os.environ.get('LLM_MODEL',
                                                  'Qwen/Qwen2.5-1.5B-Instruct')
        if name in self._MODEL_CACHE:
            return self._MODEL_CACHE[name]
        cache_dir = os.environ.get('LLM_CACHE_DIR', 'models/hf_cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        is_local = Path(name).is_dir()
        kw = {} if is_local else {'cache_dir': cache_dir}
        tok = AutoTokenizer.from_pretrained(name, **kw)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype_kw = {'torch_dtype': torch.float16} if device == 'cuda' else {}
        model = AutoModelForCausalLM.from_pretrained(name, **kw, **dtype_kw).to(device)
        model.eval()
        self._MODEL_CACHE[name] = (tok, model, device)
        return self._MODEL_CACHE[name]

    def _query_local(self, prompt: str) -> Tuple[str, Optional[int]]:
        import torch
        tok, model, device = self._get_local()
        msgs = [{'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        gen_tokens = out[0, inputs['input_ids'].shape[1]:]
        gen = tok.decode(gen_tokens, skip_special_tokens=True)
        return gen.strip(), int(gen_tokens.shape[0])

    def _query_openai(self, prompt: str) -> str:
        import json, urllib.request
        base = os.environ.get('LLM_OPENAI_BASE_URL', 'http://localhost:11434/v1')
        key  = os.environ.get('LLM_OPENAI_API_KEY', 'no-key')
        model = self.model_name or os.environ.get('LLM_OPENAI_MODEL', 'qwen2.5:1.5b')
        body = {'model': model,
                'messages': [{'role': 'system', 'content': self.system_prompt},
                              {'role': 'user',   'content': prompt}],
                'max_tokens': self.max_new_tokens, 'temperature': 0.0}
        req = urllib.request.Request(f'{base}/chat/completions',
                                       data=json.dumps(body).encode('utf-8'),
                                       headers={'Content-Type': 'application/json',
                                                 'Authorization': f'Bearer {key}'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        return payload['choices'][0]['message']['content'].strip()


# --------------------------------------------------------------------------- #
# Diagnostic feed builder                                                       #
# --------------------------------------------------------------------------- #

def _per_group_breakdown(cfg: GameConfig) -> List[dict]:
    rows: Dict[str, dict] = {}
    for c in cfg.cells:
        if not isinstance(c, Property):
            continue
        r = rows.setdefault(c.group, {'group': c.group, 'n': 0,
                                        'mean_cost': 0.0, 'mean_rent': 0.0,
                                        'cost_sum': 0, 'rent_sum': 0})
        r['n'] += 1
        r['cost_sum'] += c.cost_base
        r['rent_sum'] += c.rent_base
    for g, r in rows.items():
        r['mean_cost'] = r['cost_sum'] / max(r['n'], 1)
        r['mean_rent'] = r['rent_sum'] / max(r['n'], 1)
    return [rows[g] for g in rows]


def _design_diff_summary(design: GroupDesign) -> str:
    parts = []
    if design.salary_mult != 1.0:
        parts.append(f'salary x{design.salary_mult:.2f}')
    for g in design.drop_groups:
        parts.append(f'dropped:{g}')
    for g, m in design.group_cost_mult.items():
        if m != 1.0: parts.append(f'cost {g} x{m:.2f}')
    for g, m in design.group_rent_mult.items():
        if m != 1.0: parts.append(f'rent {g} x{m:.2f}')
    for p, ov in design.prop_overrides.items():
        parts.append(f'override {p}={ov}')
    return '; '.join(parts) if parts else 'no diff from default'


def build_diagnostic_feed(cfg: GameConfig, design: GroupDesign,
                           eval_out: dict, prior_eval: Optional[dict] = None,
                           hazard_summary: Optional[str] = None,
                           rb_llm_agreement: Optional[str] = None) -> str:
    """Assemble the text the designer LLM sees per iteration."""
    metrics = eval_out.get('metrics', {})
    delta_line = ''
    if prior_eval is not None:
        delta = eval_out['score'] - prior_eval['score']
        delta_line = f'  delta_score_vs_prev: {delta:+.4f}\n'

    bd = _per_group_breakdown(cfg)
    bd_lines = [f'  - {r["group"]:>10}: n={r["n"]} mean_cost=${r["mean_cost"]:.0f} '
                f'mean_rent=${r["mean_rent"]:.0f}'
                for r in bd]

    parts = []
    parts.append('## CURRENT EVAL')
    parts.append(f'  score: {eval_out["score"]:.4f}  (lower is better)')
    parts.append(f'  mean_rounds: {metrics.get("mean_rounds", 0):.1f}  '
                 f'(target 60)')
    parts.append(f'  mean_draw_rate: {metrics.get("mean_draw_rate", 0):.3f}')
    parts.append(f'  mean_fairness: {metrics.get("mean_fairness", 0):.3f}')
    parts.append(f'  mean_transfer_rate: {metrics.get("mean_transfer_rate", 0):.1f}/round  '
                 f'(target 100)')
    if delta_line:
        parts.append(delta_line.rstrip())

    parts.append('\n## CURRENT DESIGN DIFF FROM DEFAULT')
    parts.append('  ' + _design_diff_summary(design))

    parts.append('\n## PER-GROUP COST/RENT BREAKDOWN')
    parts.extend(bd_lines)

    if hazard_summary:
        parts.append('\n## HAZARD SUMMARY (prior runs)')
        parts.append(hazard_summary)
    if rb_llm_agreement:
        parts.append('\n## RB vs LLM CROSS-SOURCE AGREEMENT')
        parts.append(rb_llm_agreement)

    parts.append('\n## YOUR JOB')
    parts.append('  Propose ONE small intervention. Reply only with the '
                 'JSON schema described in the system prompt.')
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Trajectory runner                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Iteration:
    iter:               int
    design:             dict
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
    token_count:        Optional[int]
    wall_seconds:       float
    raw_response:       str


def _is_improvement(prev_score: float, cur_score: float, ci: dict,
                    rel_threshold: float = 0.03) -> bool:
    """ANALYSIS_PLAN §5: improvement iff relative drop >= 3% AND 95% CI on
    score excludes prev_score (i.e., new-score CI doesn't reach prev)."""
    if prev_score <= 0:
        return False
    rel = (prev_score - cur_score) / prev_score
    return bool(rel >= rel_threshold and ci.get('ci_hi', cur_score) < prev_score)


def run_trajectory(starting_cfg: GameConfig, board_label: str, seed: int,
                   pool, matchups, n_games: int, K: int,
                   designer: DesignerLLM,
                   max_turns: int,
                   out_path: Path,
                   hazard_summary: Optional[str] = None,
                   rb_llm_agreement: Optional[str] = None,
                   ) -> List[Iteration]:
    """Run one (board, seed) trajectory of up to K iterations. Writes one
    JSON line per iteration to `out_path`."""
    cfg = starting_cfg
    cumulative_design = GroupDesign(label='cumulative')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    iterations: List[Iteration] = []
    prev_score: Optional[float] = None

    # Iteration 0: baseline eval (no LLM call).
    t0 = time.perf_counter()
    base_eval = evaluate_config(cfg, pool, matchups, n_games,
                                  base_seed=seed, max_turns=max_turns)
    base_ci = bootstrap_score_ci(base_eval['per_game_records'], seed=seed)
    rec0 = Iteration(
        iter=0, design=cumulative_design.to_dict(), rationale='[baseline]',
        parser_status='baseline', parser_error=None, converged_request=False,
        score=base_eval['score'], metrics=base_eval['metrics'],
        score_ci=base_ci, delta_vs_prev=None, improvement=None,
        n_games=base_eval['n_games_total'], token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='')
    iterations.append(rec0)
    fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()
    prev_score = base_eval['score']

    for k in range(1, K + 1):
        t0 = time.perf_counter()
        feed = build_diagnostic_feed(cfg, cumulative_design,
                                       eval_out=iterations[-1].__dict__,
                                       prior_eval=(iterations[-2].__dict__
                                                    if len(iterations) >= 2 else None),
                                       hazard_summary=hazard_summary,
                                       rb_llm_agreement=rb_llm_agreement)
        try:
            raw, ntok = designer.query(feed, iteration=k - 1)
        except Exception as ex:
            raw = ''
            ntok = None
            resp = DesignerResponse(raw_text='', parsed=None, design=None,
                                     rationale='', converged=False,
                                     parser_status='backend_failure',
                                     error=f'{type(ex).__name__}: {ex}')
        else:
            resp = parse_designer_response(raw)
            resp.token_count = ntok

        if resp.parser_status != 'ok' or resp.design is None:
            rec = Iteration(
                iter=k, design={}, rationale=resp.rationale,
                parser_status=resp.parser_status, parser_error=resp.error,
                converged_request=resp.converged, score=None, metrics=None,
                score_ci=None, delta_vs_prev=None, improvement=None,
                n_games=0, token_count=resp.token_count,
                wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text)
            iterations.append(rec)
            fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            break

        # Apply on top of cumulative_design. We do not blindly stack -
        # the LLM's intervention overwrites cumulative settings (so it can
        # walk back a previous patch).
        cumulative_design = _merge_designs(cumulative_design, resp.design)
        try:
            new_cfg = apply_design(starting_cfg, cumulative_design,
                                    strict_groups=False)
        except Exception as ex:
            rec = Iteration(
                iter=k, design=cumulative_design.to_dict(),
                rationale=resp.rationale, parser_status='apply_failure',
                parser_error=f'{type(ex).__name__}: {ex}',
                converged_request=resp.converged, score=None, metrics=None,
                score_ci=None, delta_vs_prev=None, improvement=None,
                n_games=0, token_count=resp.token_count,
                wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text)
            iterations.append(rec)
            fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            break

        cfg = new_cfg

        ev = evaluate_config(cfg, pool, matchups, n_games,
                               base_seed=seed, max_turns=max_turns)
        ci = bootstrap_score_ci(ev['per_game_records'], seed=seed + k)
        delta = (ev['score'] - prev_score) if prev_score is not None else None
        improvement = (_is_improvement(prev_score, ev['score'], ci)
                        if prev_score is not None else None)

        rec = Iteration(
            iter=k, design=cumulative_design.to_dict(),
            rationale=resp.rationale, parser_status='ok', parser_error=None,
            converged_request=resp.converged, score=ev['score'],
            metrics=ev['metrics'], score_ci=ci, delta_vs_prev=delta,
            improvement=improvement, n_games=ev['n_games_total'],
            token_count=resp.token_count,
            wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text)
        iterations.append(rec)
        fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
        prev_score = ev['score']
        if resp.converged:
            break

    fh.close()
    return iterations


def _merge_designs(prev: GroupDesign, new: GroupDesign) -> GroupDesign:
    """Merge new on top of prev: new fields overwrite prev where set."""
    return GroupDesign(
        salary_mult=(new.salary_mult if new.salary_mult != 1.0 else prev.salary_mult),
        drop_groups=list(set(prev.drop_groups) | set(new.drop_groups)),
        group_cost_mult={**prev.group_cost_mult, **new.group_cost_mult},
        group_rent_mult={**prev.group_rent_mult, **new.group_rent_mult},
        prop_overrides={**prev.prop_overrides, **new.prop_overrides},
        label='cumulative',
        rationale=new.rationale or prev.rationale,
    )


# --------------------------------------------------------------------------- #
# Plot driver                                                                   #
# --------------------------------------------------------------------------- #

def plot_trajectories(trajectories_by_board: Dict[str, List[List[dict]]],
                      out_path: Path) -> None:
    """One panel per board; lines are seeds, shaded inter-seed band (CEO #7)."""
    import matplotlib.pyplot as plt
    boards = list(trajectories_by_board.keys())
    n = len(boards)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, boards):
        seeds = trajectories_by_board[b]
        max_iters = max((len(s) for s in seeds), default=0)
        if max_iters == 0:
            continue
        iters_axis = np.arange(max_iters)
        for s_idx, traj in enumerate(seeds):
            ys = [it['score'] if it['score'] is not None else np.nan for it in traj]
            ys = ys + [np.nan] * (max_iters - len(ys))
            ax.plot(iters_axis, ys, label=f'seed{s_idx}', linewidth=1.4, alpha=0.85)
        # Inter-seed band (mean ± std at each iteration that has data).
        mat = np.full((len(seeds), max_iters), np.nan, dtype=float)
        for i, traj in enumerate(seeds):
            for j, it in enumerate(traj):
                if it['score'] is not None:
                    mat[i, j] = it['score']
        with np.errstate(all='ignore'):
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0)
            ax.fill_between(iters_axis, mean - std, mean + std,
                             alpha=0.15, color='#444444')
        ax.set_title(b, fontsize=10)
        ax.set_xlabel('iteration')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(fontsize=7, loc='best', framealpha=0.85)
    axes[0].set_ylabel('score (lower is better)')
    fig.suptitle('LLM parametric closed-loop trajectories', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def write_rationales_table(trajectories_by_board: Dict[str, List[List[dict]]],
                            out_path: Path, k: int = 4) -> None:
    """Markdown table of representative rationales (CEO §5e table)."""
    lines = ['| board | seed | iter | score_delta | rationale |',
             '|-------|------|------|-------------|-----------|']
    for board, seeds in trajectories_by_board.items():
        rows = 0
        for s_idx, traj in enumerate(seeds):
            for it in traj:
                if rows >= k: break
                if it.get('parser_status') != 'ok': continue
                d = it.get('delta_vs_prev')
                d_str = f'{d:+.3f}' if d is not None else '-'
                rat = (it.get('rationale') or '').replace('|', '/').strip()[:120]
                lines.append(f'| {board} | {s_idx} | {it["iter"]} | {d_str} | {rat} |')
                rows += 1
            if rows >= k: break
    out_path.write_text('\n'.join(lines))


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
    ap.add_argument('--boards', default='all',
                    help='Comma-separated board labels to run, or "all".')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-games', type=int, default=100,
                    help='Per-iteration game count for canonical eval.')
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--out-dir', default='report/figures/llm_design_loop')
    ap.add_argument('--hazard-summary-path', default=None,
                    help='Optional path to a text file pasted into the feed.')
    ap.add_argument('--rb-llm-agreement-path', default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Resolving boards (iteration board = canonical, per ANALYSIS_PLAN §7)...')
    sources = build_five_boards(
        canonical_config=args.canonical_config, mini_config=args.mini_config,
        ga_2p=args.ga_2p, ga_3p=args.ga_3p)
    # Per Step 0 audit (rho=0.30): iterate on canonical. We pass the canonical
    # config decoded to the corresponding starting design point. For boards
    # that are *defined* against mini (salary x2, drop Brown), we re-apply the
    # design transformation against canonical so the loop iterates against a
    # canonical-shaped board with the same intent.
    canon_cfg = GameConfig.from_yaml(args.canonical_config)
    starting_cfgs: List[Tuple[str, GameConfig]] = []
    for label, cfg in sources:
        if label == 'salary x2':
            from optimizer.board_sources import modify_salary
            starting_cfgs.append((label, modify_salary(canon_cfg, 2.0)))
        elif label == 'drop Brown':
            from optimizer.board_sources import remove_group
            starting_cfgs.append((label, remove_group(canon_cfg, 'Brown')))
        else:
            starting_cfgs.append((label, cfg))

    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        starting_cfgs = [(l, c) for l, c in starting_cfgs if l in wanted]
    print(f'  starting boards: {[l for l, _ in starting_cfgs]}')

    print('Loading strategy pool + matchups...')
    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                    n_matchups=args.n_matchups,
                                    seed=args.matchup_seed)

    designer = DesignerLLM(backend=args.backend, model_name=args.model)

    hazard_summary = (Path(args.hazard_summary_path).read_text()
                       if args.hazard_summary_path else None)
    rb_llm_agreement = (Path(args.rb_llm_agreement_path).read_text()
                          if args.rb_llm_agreement_path else None)

    trajectories_by_board: Dict[str, List[List[dict]]] = {}
    for board_label, cfg in starting_cfgs:
        trajectories_by_board[board_label] = []
        for s in range(args.n_seeds):
            seed = args.seed_offset + s * 1000 + 42
            print(f'\n[{board_label}] seed={seed} (s_idx={s})  K_max={args.K}')
            tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
            out_path = out_dir / f'{tag}__seed{seed}.jsonl'
            iters = run_trajectory(
                starting_cfg=cfg, board_label=board_label, seed=seed,
                pool=pool, matchups=matchups, n_games=args.n_games, K=args.K,
                designer=designer, max_turns=args.max_turns,
                out_path=out_path, hazard_summary=hazard_summary,
                rb_llm_agreement=rb_llm_agreement)
            for it in iters:
                ps = it.parser_status; sc = it.score
                cv = ' (converged)' if it.converged_request else ''
                print(f'    iter {it.iter}: parser={ps}  '
                      f'score={sc:.4f}' if sc is not None else
                      f'    iter {it.iter}: parser={ps}  score=N/A')
                if cv: print(f'      {cv.strip()}')
            trajectories_by_board[board_label].append([asdict(i) for i in iters])

    # Aggregate plots + tables.
    plot_trajectories(trajectories_by_board, out_dir / 'trajectories.pdf')
    plot_trajectories(trajectories_by_board, out_dir / 'trajectories.png')
    write_rationales_table(trajectories_by_board, out_dir / 'rationales.md')
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'trajectories_by_board': trajectories_by_board,
        }, f, indent=2, default=str)
    print(f'\nDone. Trajectories -> {out_dir}/{{trajectories.pdf,rationales.md,summary.json}}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
