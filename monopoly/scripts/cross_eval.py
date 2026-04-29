"""Evaluate any design under 2p AND 3p harnesses for a cross-player-count comparison.

Reads the best design from one or more JSONL run logs (or a literal vec from
--vec) and runs it through both the 2-player and 3-player evaluation harness
at high game count for tight confidence intervals.

Usage (from monopoly/):
    # Best of GA-2p vs Best of GA-3p, 1000 games each, in both harnesses.
    python scripts/cross_eval.py \
        --runs logs/optimizer/ga_2p.jsonl logs/optimizer/ga_3p.jsonl \
        --n-games 1000 --out logs/optimizer/cross_eval.json

    # Or evaluate a specific design vector (e.g. the default identity).
    python scripts/cross_eval.py --identity --n-games 1000

Output JSON has one entry per (design × harness) cell so the reader sees
which numbers stay stable across player counts and which collapse.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import GameConfig
from optimizer.design_space import DesignSpace
from optimizer.objectives import (Targets, Weights, evaluate,
                                   per_strategy_win_rates)
from optimizer.simulate import run_matchup
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool


def _wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def best_design_from_run(run_path):
    """Pick the lowest-score entry from a JSONL run log."""
    best = None
    with open(run_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if best is None or e['score'] < best['score']:
                best = e
    return best


def evaluate_design(cfg, vec, pool, matchups, n_games_per_matchup,
                    base_seed, max_turns, weights, targets, label,
                    removal_direction='cheapest'):
    """Run the 10-matchup harness on a design and return aggregated metrics."""
    space = DesignSpace(cfg, removal_direction=removal_direction)
    decoded = space.decode(np.asarray(vec))
    results_by_matchup = []
    for mi, idxs in enumerate(tqdm(matchups, desc=label, leave=False, unit='mu')):
        strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer') for i in idxs]
        seed = base_seed + mi * 10_000
        results_by_matchup.append(run_matchup(
            decoded, strategies, n_games=n_games_per_matchup,
            base_seed=seed, max_turns=max_turns, balance_seats=True))
    out = evaluate(results_by_matchup, weights=weights, targets=targets)
    n_total = sum(len(rs) for rs in results_by_matchup)
    # Wins per matchup with CIs
    per_mu = []
    for rs, mu in zip(results_by_matchup, out['per_matchup']):
        wrs = per_strategy_win_rates(rs)
        per_mu.append({**mu,
                       'win_rates':    wrs,
                       'win_rates_ci': {k: list(_wilson_ci(v, len(rs)))
                                         for k, v in wrs.items()}})
    return {'score':       out['score'],
            'metrics':      out['metrics'],
            'per_matchup':  per_mu,
            'n_games':      n_total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[],
                    help='JSONL run logs; the best design from each is evaluated under both 2p and 3p.')
    ap.add_argument('--vec', type=str, default=None,
                    help='Comma-separated 45-dim vector (alternative to --runs).')
    ap.add_argument('--identity', action='store_true',
                    help='Also evaluate the identity vec (= default board) under both harnesses.')
    ap.add_argument('--config', default='default_config.yaml')
    ap.add_argument('--pool',   default='optimizer/strategy_pool.json')
    ap.add_argument('--n-games',    type=int, default=1000)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--base-seed',  type=int, default=42)
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--max-turns',  type=int, default=200)
    ap.add_argument('--removal-direction', choices=('cheapest', 'expensive', 'middle'),
                    default='cheapest',
                    help='Must match the optimisation-run setting for the design vec '
                         'to decode to the same GameConfig.')
    ap.add_argument('--w-fair',  type=float, default=1.0)
    ap.add_argument('--w-fmax',  type=float, default=0.5)
    ap.add_argument('--w-len',   type=float, default=0.5)
    ap.add_argument('--w-draw',  type=float, default=0.3)
    ap.add_argument('--w-money', type=float, default=0.3)
    ap.add_argument('--target-rounds', type=float, default=60.0)
    ap.add_argument('--target-transfer', type=float, default=100.0)
    ap.add_argument('--out', default='logs/optimizer/cross_eval.json')
    args = ap.parse_args()

    cfg = GameConfig.from_yaml(args.config)
    pool = load_strategy_pool(args.pool)
    weights = Weights(args.w_fair, args.w_fmax, args.w_len, args.w_draw, args.w_money)
    targets = Targets(args.target_rounds, args.target_transfer)
    n_per_mu = max(1, args.n_games // args.n_matchups)

    # Build the list of designs to evaluate.
    designs = []   # list of (label, vec, source)
    if args.identity:
        designs.append(('identity_default',
                        DesignSpace(cfg, removal_direction=args.removal_direction).identity_vec().tolist(),
                        'identity'))
    for run in args.runs:
        best = best_design_from_run(run)
        designs.append((Path(run).stem + '_best', best['vec'], run))
    if args.vec:
        v = [float(x) for x in args.vec.split(',')]
        designs.append(('cli_vec', v, '--vec'))

    if not designs:
        raise SystemExit('Pass --runs and/or --identity and/or --vec.')

    # Eval each design × {2p, 3p}.
    results = []
    for label, vec, source in designs:
        for n_players in (2, 3):
            matchups = load_eval_matchups(n_players, pool_size=len(pool),
                                           n_matchups=args.n_matchups,
                                           seed=args.matchup_seed)
            tqdm.write(f'\n[{label}] eval @ {n_players}p × {n_per_mu * args.n_matchups} games...')
            res = evaluate_design(cfg, vec, pool, matchups, n_per_mu,
                                  args.base_seed, args.max_turns,
                                  weights, targets,
                                  label=f'{label}@{n_players}p',
                                  removal_direction=args.removal_direction)
            results.append({'design':    label,
                            'source':    source,
                            'n_players': n_players,
                            **res})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'configs': vars(args), 'results': results}, f, indent=2)

    # Console summary table
    print('\n' + '=' * 96)
    print(f'{"design":<24}{"n_p":>5}{"score":>9}{"fair":>8}{"max_f":>8}'
          f'{"rounds":>9}{"draw":>8}{"transfer":>10}')
    print('-' * 96)
    for r in results:
        m = r['metrics']
        print(f'{r["design"]:<24}{r["n_players"]:>5}{r["score"]:>9.3f}'
              f'{m["mean_fairness"]:>8.3f}{m["max_fairness"]:>8.3f}'
              f'{m["mean_rounds"]:>9.1f}{m["mean_draw_rate"]:>8.3f}'
              f'{m["mean_transfer_rate"]:>10.1f}')
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
