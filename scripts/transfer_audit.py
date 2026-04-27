"""Step 0 (CEO plan, C3): mini→canonical transfer audit.

Question: if we iterate the LLM design loop on the 4x4 mini board, do its
rankings transfer to canonical? Answered by Spearman rho between mini-board
scores and canonical scores on a small set of designs spanning a diverse
slice of the design vocabulary.

Pre-committed thresholds:
  rho >= 0.7  ->  mini-board iteration justified for #4 / #5
  0.4 < rho < 0.7  ->  marginal; report and proceed with mini, document in writeup
  rho <= 0.4  ->  switch iteration board to canonical, recompute wall-clock budget

Designs and audit budget are FROZEN before running; see optimizer/group_design.py
for the AUDIT_DESIGNS list. Five designs at n=100 games each, on two boards
each => 1000 games total (~15-30 min on a modern laptop).

Outputs (under report/figures/transfer_audit/ by default):
  scores.json   -- per-(design, board) score, metric breakdown, n_games, seed
  rho.json      -- Spearman rho + interpretation against pre-committed thresholds
  scatter.pdf   -- mini score vs canonical score, one point per design, rho in title
  scatter.png   -- ditto, screen-resolution

Usage (from CS348K-proj/):
    set PYTHONPATH=. && python scripts/transfer_audit.py \
        --canonical-config default_config.yaml \
        --mini-config configs/mini \
        --n-games 100 \
        --out-dir report/figures/transfer_audit

    # Smoke check (~30 sec) before committing to the full run:
    set PYTHONPATH=. && python scripts/transfer_audit.py --n-games 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import GameConfig
from optimizer.group_design import AUDIT_DESIGNS, GroupDesign, apply_design
from optimizer.objectives import Targets, Weights, evaluate
from optimizer.simulate import run_matchup
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rho without scipy. Handles ties via average ranks."""
    n = len(x)
    if n != len(y) or n < 2:
        return float('nan')

    def _ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1.0   # 1-based average rank
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = sum((rx[i] - mx) ** 2 for i in range(n))
    den_y = sum((ry[i] - my) ** 2 for i in range(n))
    if den_x == 0 or den_y == 0:
        return float('nan')
    return num / (den_x ** 0.5 * den_y ** 0.5)


def _eval_design_on_board(cfg: GameConfig, pool, matchups, n_games: int,
                          base_seed: int, weights: Weights, targets: Targets,
                          max_turns: int) -> dict:
    """Run all matchups on cfg, return aggregate evaluate(...) output."""
    n_per = max(1, n_games // len(matchups))
    results_by_matchup = []
    for mi, idxs in enumerate(matchups):
        strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer') for i in idxs]
        seed = base_seed + mi * 10_000
        rs = run_matchup(cfg, strategies,
                          n_games=n_per, base_seed=seed,
                          max_turns=max_turns, balance_seats=True)
        results_by_matchup.append(rs)
    out = evaluate(results_by_matchup, weights=weights, targets=targets)
    out['n_games_total'] = sum(len(rs) for rs in results_by_matchup)
    return out


def _interpret_rho(rho: float) -> str:
    if np.isnan(rho):
        return 'undefined (constant scoreboard)'
    if rho >= 0.7:
        return 'mini-board iteration JUSTIFIED'
    if rho > 0.4:
        return 'marginal: proceed with mini and document caveat'
    return 'mini-board iteration NOT justified -- switch iteration board to canonical'


def _scatter_plot(designs: List[GroupDesign], mini: List[float],
                  canon: List[float], rho: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(mini, canon, c='#1f77b4', s=46, zorder=3)
    for d, mx, cx in zip(designs, mini, canon):
        ax.annotate(d.label, (mx, cx), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

    # Reference y=x line for visual ballast.
    lo = min(min(mini), min(canon))
    hi = max(max(mini), max(canon))
    pad = 0.1 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--',
            color='#888888', linewidth=0.8, label='y = x')

    ax.set_xlabel('mini-board score (lower = better)')
    ax.set_ylabel('canonical score (lower = better)')
    title = (f'Transfer audit: mini vs canonical scores\n'
             f'Spearman $\\rho$ = {rho:.3f}  ({_interpret_rho(rho)})')
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--mini-config',      default='configs/mini')
    ap.add_argument('--pool',             default='optimizer/strategy_pool.json')
    ap.add_argument('--n-games',  type=int, default=100,
                    help='Games per (design, board) cell. Pre-committed at 100.')
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--out-dir', default='report/figures/transfer_audit')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading configs and strategy pool...')
    canon_cfg = GameConfig.from_yaml(args.canonical_config)
    mini_cfg  = GameConfig.from_yaml(args.mini_config)
    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                   n_matchups=args.n_matchups,
                                   seed=args.matchup_seed)

    weights = Weights()
    targets = Targets()

    print(f'Designs under audit: {[d.label for d in AUDIT_DESIGNS]}')
    print(f'  per-cell budget: {args.n_games} games '
          f'(matchups={args.n_matchups}, n_per_matchup={args.n_games // args.n_matchups})')

    rows = []
    mini_scores: List[float] = []
    canon_scores: List[float] = []

    t0 = time.perf_counter()
    for di, design in enumerate(AUDIT_DESIGNS):
        print(f'\n[{di+1}/{len(AUDIT_DESIGNS)}] {design.label}')
        # Apply against each board. strict_groups=False so an audit design
        # naming a group present only on canonical (none here, but defensive)
        # would silently no-op on mini rather than crash the audit.
        mini_design_cfg  = apply_design(mini_cfg, design, strict_groups=False)
        canon_design_cfg = apply_design(canon_cfg, design, strict_groups=False)

        t_m = time.perf_counter()
        mini_eval = _eval_design_on_board(
            mini_design_cfg, pool, matchups, args.n_games,
            args.base_seed, weights, targets, args.max_turns)
        print(f'  mini      score={mini_eval["score"]:.4f}  '
              f'rounds={mini_eval["metrics"]["mean_rounds"]:.1f}  '
              f'wall={time.perf_counter() - t_m:.1f}s')

        t_c = time.perf_counter()
        canon_eval = _eval_design_on_board(
            canon_design_cfg, pool, matchups, args.n_games,
            args.base_seed, weights, targets, args.max_turns)
        print(f'  canonical score={canon_eval["score"]:.4f}  '
              f'rounds={canon_eval["metrics"]["mean_rounds"]:.1f}  '
              f'wall={time.perf_counter() - t_c:.1f}s')

        rows.append({
            'design':    design.to_dict(),
            'mini':      {'score': mini_eval['score'],
                          'metrics': mini_eval['metrics'],
                          'n_games': mini_eval['n_games_total']},
            'canonical': {'score': canon_eval['score'],
                          'metrics': canon_eval['metrics'],
                          'n_games': canon_eval['n_games_total']},
        })
        mini_scores.append(mini_eval['score'])
        canon_scores.append(canon_eval['score'])

    rho = _spearman(mini_scores, canon_scores)
    interp = _interpret_rho(rho)

    print(f'\nTotal wall-clock: {time.perf_counter() - t0:.1f}s')
    print(f'Spearman rho(mini, canonical) = {rho:.3f}  -> {interp}')

    scores_path = out_dir / 'scores.json'
    with open(scores_path, 'w') as f:
        json.dump({
            'config': {
                'canonical_config': args.canonical_config,
                'mini_config':      args.mini_config,
                'n_games_per_cell': args.n_games,
                'n_matchups':       args.n_matchups,
                'max_turns':        args.max_turns,
                'n_players':        args.n_players,
                'base_seed':        args.base_seed,
                'matchup_seed':     args.matchup_seed,
            },
            'rows': rows,
        }, f, indent=2)
    print(f'  scores -> {scores_path}')

    rho_path = out_dir / 'rho.json'
    with open(rho_path, 'w') as f:
        json.dump({
            'spearman_rho': float(rho) if not np.isnan(rho) else None,
            'mini_scores':  mini_scores,
            'canon_scores': canon_scores,
            'design_labels': [d.label for d in AUDIT_DESIGNS],
            'interpretation': interp,
            'thresholds': {'justified_at': 0.7, 'marginal_above': 0.4},
        }, f, indent=2)
    print(f'  rho    -> {rho_path}')

    _scatter_plot(AUDIT_DESIGNS, mini_scores, canon_scores, rho,
                  out_dir / 'scatter.pdf')
    _scatter_plot(AUDIT_DESIGNS, mini_scores, canon_scores, rho,
                  out_dir / 'scatter.png')
    print(f'  scatter -> {out_dir}/scatter.{{pdf,png}}')


if __name__ == '__main__':
    main()
