"""CLI for board-design optimisation over the ParametricPlayer strategy pool.

Usage (from monopoly/):
    # First build the pool (one-off):
    python scripts/build_strategy_pool.py

    # Random search, 2-player, default weights, 200 evals:
    python scripts/optimize_board.py --search random --iters 200 --n-players 2

    # Genetic algorithm, 3-player, fairness-only ablation:
    python scripts/optimize_board.py --search ga --n-players 3 \
        --w-fair 1 --w-fmax 0 --w-len 0 --w-draw 0 --w-money 0 \
        --run-name fairness_only_3p

    # Short smoke run to verify the pipeline end-to-end:
    python scripts/optimize_board.py --search random --iters 5 --n-games 20 --run-name smoke
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import GameConfig
from optimizer.design_space import DesignSpace
from optimizer.objectives import Targets, Weights, evaluate
from optimizer.search import genetic_algorithm, random_search
from optimizer.simulate import run_matchup
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool


def _build_eval_fn(base_cfg, pool, matchups, n_games_per_matchup, base_seed,
                   weights, targets, max_turns, removal_direction='cheapest'):
    """Closure: given a design vec, decode → run all matchups → evaluate → return (score, metrics).

    An inner tqdm bar is opened per-candidate so the user sees liveness during
    slow candidates (some random boards push games to the turn cap).
    """
    space = DesignSpace(base_cfg, removal_direction=removal_direction)

    def eval_fn(vec):
        cfg = space.decode(vec)
        inner = tqdm(total=len(matchups), desc='  matchups', unit='mu',
                     leave=False, dynamic_ncols=True, position=1)
        results_by_matchup = []
        for mi, idxs in enumerate(matchups):
            strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer') for i in idxs]
            seed = base_seed + mi * 10_000   # deterministic + well-separated across matchups
            results = run_matchup(
                cfg, strategies,
                n_games=n_games_per_matchup,
                base_seed=seed,
                max_turns=max_turns,
                balance_seats=True,
            )
            results_by_matchup.append(results)
            mean_rounds_so_far = (sum(r['rounds'] for rs in results_by_matchup for r in rs)
                                   / max(sum(len(rs) for rs in results_by_matchup), 1))
            inner.set_postfix(avg_rounds=f'{mean_rounds_so_far:.0f}')
            inner.update(1)
        inner.close()
        out = evaluate(results_by_matchup, weights=weights, targets=targets)
        return out['score'], out['metrics']

    return eval_fn, space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='default_config.yaml')
    ap.add_argument('--pool',       default='optimizer/strategy_pool.json',
                    help='Path to strategy pool JSON. Built by scripts/build_strategy_pool.py.')
    ap.add_argument('--n-players',  type=int, default=2, choices=(2, 3))
    ap.add_argument('--search',     choices=('random', 'ga'), default='ga')
    ap.add_argument('--iters',      type=int, default=200,
                    help='Random search: number of samples. GA: inferred from --pop * --generations.')
    ap.add_argument('--pop',        type=int, default=20)
    ap.add_argument('--generations',type=int, default=10)
    ap.add_argument('--elitism',    type=int, default=2,
                    help='Number of top individuals carried unchanged each generation. '
                         'Set to 0 to evaluate exactly pop*generations candidates.')
    ap.add_argument('--n-games',    type=int, default=100,
                    help='Total games per candidate (split across 10 matchups).')
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns',  type=int, default=200,
                    help='Hard turn cap per game. Games that hit this are counted as '
                         'truncated (no winner). Default 200 decides most Monopoly games; '
                         'SimulationSettings.n_moves (=1000) is way too slow for outer-loop use. '
                         'Lower to 150 if individual candidates feel slow.')
    ap.add_argument('--removal-direction', choices=('cheapest', 'expensive', 'middle'),
                    default='cheapest',
                    help='When N_props<22, which properties to drop. '
                         '"cheapest" (default) drops Brown/Lightblue first; '
                         '"expensive" drops Green/Indigo first (bigger impact on fairness); '
                         '"middle" drops from the middle of the cost distribution.')
    ap.add_argument('--base-seed',  type=int, default=42)
    ap.add_argument('--search-seed',type=int, default=0)
    ap.add_argument('--matchup-seed',type=int, default=1234)
    # Objective weights
    ap.add_argument('--w-fair',     type=float, default=1.0)
    ap.add_argument('--w-fmax',     type=float, default=0.5)
    ap.add_argument('--w-len',      type=float, default=0.5)
    ap.add_argument('--w-draw',     type=float, default=0.3)
    ap.add_argument('--w-money',    type=float, default=0.3)
    ap.add_argument('--target-rounds',   type=float, default=60.0)
    ap.add_argument('--target-transfer', type=float, default=100.0)
    # Output
    ap.add_argument('--run-name',   default=None)
    ap.add_argument('--out-dir',    default='logs/optimizer')
    args = ap.parse_args()

    # Load config and strategy pool
    cfg = GameConfig.from_yaml(args.config)
    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                   n_matchups=args.n_matchups, seed=args.matchup_seed)
    n_games_per_matchup = max(1, args.n_games // len(matchups))

    # Objective
    weights = Weights(args.w_fair, args.w_fmax, args.w_len, args.w_draw, args.w_money)
    targets = Targets(args.target_rounds, args.target_transfer)

    # Eval fn + space
    eval_fn, space = _build_eval_fn(
        cfg, pool, matchups, n_games_per_matchup, args.base_seed, weights, targets,
        max_turns=args.max_turns, removal_direction=args.removal_direction)

    # Output path
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f'{args.search}_{args.n_players}p_{ts}'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f'{run_name}.jsonl'
    meta_path = out_dir / f'{run_name}.meta.json'

    # Meta file with run config (so the grader can reproduce any number)
    with open(meta_path, 'w') as f:
        json.dump({
            'run_name':     run_name,
            'search':       args.search,
            'n_players':    args.n_players,
            'n_games':      args.n_games,
            'n_games_per_matchup': n_games_per_matchup,
            'max_turns':    args.max_turns,
            'matchups':     [list(m) for m in matchups],
            'pool_path':    args.pool,
            'pool_names':   [p[0] for p in pool],
            'weights':      vars(weights),
            'targets':      vars(targets),
            'base_seed':    args.base_seed,
            'search_seed':  args.search_seed,
            'matchup_seed': args.matchup_seed,
            'config':       args.config,
            'iters':        args.iters,
            'pop':          args.pop,
            'generations':  args.generations,
            'elitism':      args.elitism,
            'removal_direction': args.removal_direction,
        }, f, indent=2)
    print(f'Meta written to {meta_path}')

    # Progress bar with best-so-far. The GA only re-evaluates pop-elitism
    # individuals per generation after the first (elites carried unchanged),
    # so total evals = pop + (generations-1)*(pop-elitism), not pop*generations.
    n_total = (args.iters if args.search == 'random'
               else args.pop + (args.generations - 1) * (args.pop - args.elitism))
    pbar = tqdm(total=n_total, desc=args.search, dynamic_ncols=True, position=0)
    best = [float('inf')]
    fh = open(run_path, 'w')

    def on_iter(entry):
        fh.write(json.dumps(entry) + '\n')
        fh.flush()
        if entry['score'] < best[0]:
            best[0] = entry['score']
        pbar.set_postfix(best=f"{best[0]:.3f}", cur=f"{entry['score']:.3f}")
        pbar.update(1)

    # Run the optimiser
    if args.search == 'random':
        random_search(space, eval_fn, n_iters=args.iters,
                      seed=args.search_seed, on_iter=on_iter)
    else:
        genetic_algorithm(space, eval_fn, pop_size=args.pop,
                          generations=args.generations,
                          elitism=args.elitism,
                          seed=args.search_seed, on_iter=on_iter)

    pbar.close()
    fh.close()
    print(f'History written to {run_path}')
    print(f'Best score: {best[0]:.4f}')


if __name__ == '__main__':
    main()
