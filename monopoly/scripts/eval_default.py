"""Evaluate the default board (identity design vector) through the same harness.

This produces the "before" reference point every optimisation run is compared
against. Emits one JSON file with metrics + a single-line JSONL so the reporter
can overlay it on convergence plots.

Usage (from monopoly/):
    python scripts/eval_default.py                                         # 2p
    python scripts/eval_default.py --n-players 3 --out logs/optimizer/default_3p.json
    python scripts/eval_default.py --n-games 1000                          # tight CI
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from config import GameConfig
from optimizer.design_space import DesignSpace
from optimizer.objectives import Targets, Weights, evaluate
from optimizer.simulate import run_matchup
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='default_config.yaml')
    ap.add_argument('--pool',       default='optimizer/strategy_pool.json')
    ap.add_argument('--n-players',  type=int, default=2, choices=(2, 3))
    ap.add_argument('--n-games',    type=int, default=100)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--base-seed',  type=int, default=42)
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--max-turns',  type=int, default=200)
    ap.add_argument('--w-fair',     type=float, default=1.0)
    ap.add_argument('--w-fmax',     type=float, default=0.5)
    ap.add_argument('--w-len',      type=float, default=0.5)
    ap.add_argument('--w-draw',     type=float, default=0.3)
    ap.add_argument('--w-money',    type=float, default=0.3)
    ap.add_argument('--target-rounds',   type=float, default=60.0)
    ap.add_argument('--target-transfer', type=float, default=100.0)
    ap.add_argument('--out', default=None,
                    help='Output JSON. Default: logs/optimizer/default_{n}p.json')
    args = ap.parse_args()

    cfg = GameConfig.from_yaml(args.config)
    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                   n_matchups=args.n_matchups,
                                   seed=args.matchup_seed)
    n_games_per_matchup = max(1, args.n_games // len(matchups))
    weights = Weights(args.w_fair, args.w_fmax, args.w_len, args.w_draw, args.w_money)
    targets = Targets(args.target_rounds, args.target_transfer)

    # Identity vec = default board (all multipliers = 1.0, N_props = 22)
    space = DesignSpace(cfg)
    vec = space.identity_vec()
    default_cfg = space.decode(vec)

    results_by_matchup = []
    for mi, idxs in enumerate(tqdm(matchups, desc='default', unit='mu')):
        strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer') for i in idxs]
        seed = args.base_seed + mi * 10_000
        rs = run_matchup(default_cfg, strategies,
                         n_games=n_games_per_matchup, base_seed=seed,
                         max_turns=args.max_turns, balance_seats=True)
        results_by_matchup.append(rs)

    out = evaluate(results_by_matchup, weights=weights, targets=targets)

    out_path = Path(args.out or f'logs/optimizer/default_{args.n_players}p.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'n_players':           args.n_players,
        'n_games':             args.n_games,
        'n_games_per_matchup': n_games_per_matchup,
        'matchups':            [list(m) for m in matchups],
        'weights':             vars(weights),
        'targets':             vars(targets),
        'max_turns':           args.max_turns,
        'score':               out['score'],
        'metrics':             out['metrics'],
        'per_matchup':         out['per_matchup'],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'\n=== Default board baseline ({args.n_players}p, {args.n_games} games) ===')
    print(f'  score:           {out["score"]:.4f}')
    for k, v in out['metrics'].items():
        print(f'  {k:<22}{v:>10.3f}')
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
