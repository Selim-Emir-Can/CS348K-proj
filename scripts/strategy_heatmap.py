"""Per-strategy 30×30 win-rate heatmap on a chosen board design.

Runs every (s_i, s_j) ordered pair from the strategy pool for N games on the
specified design. Aggregates by strategy name (not seat), so values are
position-independent. Produces:
  • <out>.npy        — 30×30 win-rate matrix W where W[i,j] = P(s_i wins | s_i vs s_j)
  • <out>.png        — heatmap (red/blue diverging colormap centred at 0.5)
  • <out>.diff.npy   — if --baseline-design is given, W_design − W_baseline
  • <out>.diff.png   — diff heatmap (centred at 0.0)
  • <out>.json       — summary stats (most-asymmetric pairs, mean abs gap, etc.)

Usage (from monopoly/):
    # Heatmap for the GA-2p winner
    python scripts/strategy_heatmap.py --runs logs/optimizer/ga_2p.jsonl \\
           --n-players 2 --n-games 20 --out logs/optimizer/heatmap_ga2p

    # Same but with the default board as baseline → also writes diff heatmap
    python scripts/strategy_heatmap.py --runs logs/optimizer/ga_2p.jsonl \\
           --identity-baseline --n-players 2 --n-games 20 --out logs/optimizer/heatmap_ga2p
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 3x bump, same scale as the convergence/Pareto plots.
plt.rcParams.update({
    'font.size':         39,
    'axes.titlesize':    48,
    'axes.labelsize':    45,
    'xtick.labelsize':   33,
    'ytick.labelsize':   33,
    'figure.titlesize':  48,
})

from config import GameConfig
from optimizer.design_space import DesignSpace
from optimizer.objectives import per_strategy_win_rates
from optimizer.simulate import run_matchup
from optimizer.strategy_pool import load_strategy_pool


def best_design_from_run(run_path):
    best = None
    with open(run_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if best is None or e['score'] < best['score']:
                best = e
    return best


def compute_matrix(cfg, vec, pool, n_players, n_games, base_seed, max_turns,
                   label='heatmap'):
    """Return an N×N matrix of P(row strategy wins vs column strategy).

    Diagonal is left at 0.5 (a strategy vs itself). For 2p: matrix is symmetric
    by construction (W[i,j] + W[j,i] + draw = 1), so we only run the upper
    triangle. For 3p: we still pair just two strategies at a time and slot a
    rotating third from the pool — this keeps the matrix size at N×N.
    """
    N = len(pool)
    W = np.full((N, N), np.nan, dtype=np.float64)
    np.fill_diagonal(W, 0.5)

    space = DesignSpace(cfg)
    decoded = space.decode(np.asarray(vec))

    # 2p case: simple pairwise.
    if n_players == 2:
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        for k, (i, j) in enumerate(tqdm(pairs, desc=label, unit='pair')):
            strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer'),
                          (pool[j][0], pool[j][1], 'ParametricPlayer')]
            seed = base_seed + k * 100
            results = run_matchup(decoded, strategies, n_games=n_games,
                                  base_seed=seed, max_turns=max_turns,
                                  balance_seats=True)
            wrs = per_strategy_win_rates(results)
            W[i, j] = wrs.get(pool[i][0], 0.0)
            W[j, i] = wrs.get(pool[j][0], 0.0)
        return W

    # 3p case: include each pair (i, j) plus a rotating "third" so the matrix
    # is still N×N. The third is chosen by a fixed rule (k mod N skipping i, j)
    # so the rotation is reproducible.
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    for k, (i, j) in enumerate(tqdm(pairs, desc=label, unit='triple')):
        # third = first index ≠ i, j in deterministic rotation
        third = (k + 7) % N
        while third == i or third == j:
            third = (third + 1) % N
        strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer'),
                      (pool[j][0], pool[j][1], 'ParametricPlayer'),
                      (pool[third][0], pool[third][1], 'ParametricPlayer')]
        seed = base_seed + k * 100
        results = run_matchup(decoded, strategies, n_games=n_games,
                              base_seed=seed, max_turns=max_turns,
                              balance_seats=True)
        wrs = per_strategy_win_rates(results)
        W[i, j] = wrs.get(pool[i][0], 0.0)
        W[j, i] = wrs.get(pool[j][0], 0.0)
    return W


def plot_matrix(W, names, title, out_path, diverging_centre=0.5,
                vmin=None, vmax=None, cmap='RdBu_r'):
    # Canvas scaled 3x so 30 tick labels at the bumped font size stay legible.
    fig, ax = plt.subplots(figsize=(32, 28))
    if vmin is None: vmin = 0.0 if diverging_centre == 0.5 else -0.5
    if vmax is None: vmax = 1.0 if diverging_centre == 0.5 else  0.5
    im = ax.imshow(W, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.set_title(title)
    ax.set_xlabel('opponent (column)')
    ax.set_ylabel('row strategy')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=36)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def summarise(W, names):
    """Most asymmetric off-diagonal pairs and global stats."""
    N = len(names)
    asym = []
    for i in range(N):
        for j in range(i + 1, N):
            wi, wj = W[i, j], W[j, i]
            if not (np.isnan(wi) or np.isnan(wj)):
                gap = abs(wi - wj)
                asym.append((gap, names[i], names[j], wi, wj))
    asym.sort(reverse=True)

    finite = W[~np.isnan(W) & (np.eye(N) == 0)]
    return {
        'mean_abs_diff':   float(np.mean(np.abs(W - 0.5)[~np.isnan(W)])),
        'top_asymmetric':  [
            {'A': a, 'B': b, 'P_A_wins': float(wi), 'P_B_wins': float(wj),
             'abs_gap': float(gap)}
            for gap, a, b, wi, wj in asym[:10]
        ],
        'matrix_min':      float(np.nanmin(finite)),
        'matrix_max':      float(np.nanmax(finite)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=str, default=None,
                    help='JSONL run log; the best design is used.')
    ap.add_argument('--vec',  type=str, default=None,
                    help='Comma-separated 45-dim vector (alternative to --runs).')
    ap.add_argument('--identity', action='store_true',
                    help='Use the identity vec (= default board).')
    ap.add_argument('--identity-baseline', action='store_true',
                    help='Also compute the identity matrix and emit a diff heatmap.')
    ap.add_argument('--config', default='default_config.yaml')
    ap.add_argument('--pool',   default='optimizer/strategy_pool.json')
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--n-games',   type=int, default=20)
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--out',       default='logs/optimizer/heatmap')
    args = ap.parse_args()

    cfg  = GameConfig.from_yaml(args.config)
    pool = load_strategy_pool(args.pool)
    names = [n for n, _ in pool]
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the design vector
    if args.identity:
        vec = DesignSpace(cfg).identity_vec()
        design_label = 'identity'
    elif args.vec:
        vec = np.array([float(x) for x in args.vec.split(',')])
        design_label = 'cli_vec'
    elif args.runs:
        best = best_design_from_run(args.runs)
        vec = np.array(best['vec'])
        design_label = Path(args.runs).stem + '_best'
    else:
        raise SystemExit('Pass --runs / --vec / --identity.')

    print(f'Computing {len(pool)}×{len(pool)} matrix on design "{design_label}" '
          f'({args.n_players}p, {args.n_games} games/pair)...')
    W = compute_matrix(cfg, vec, pool, args.n_players, args.n_games,
                       args.base_seed, args.max_turns, label=design_label)

    np.save(str(out_base) + '.npy', W)
    plot_matrix(W, names,
                title=f'P(row wins) — {design_label} ({args.n_players}p, n={args.n_games}/pair)',
                out_path=str(out_base) + '.png')
    summary = {'design': design_label, 'n_players': args.n_players,
               'n_games_per_pair': args.n_games, 'main': summarise(W, names)}

    if args.identity_baseline:
        print('Computing baseline (identity) matrix for diff...')
        baseline_vec = DesignSpace(cfg).identity_vec()
        W_base = compute_matrix(cfg, baseline_vec, pool, args.n_players,
                                args.n_games, args.base_seed, args.max_turns,
                                label='identity_baseline')
        diff = W - W_base
        np.save(str(out_base) + '.diff.npy', diff)
        plot_matrix(diff, names,
                    title=f'Δ P(row wins): {design_label} − default '
                          f'({args.n_players}p, n={args.n_games}/pair)',
                    out_path=str(out_base) + '.diff.png',
                    diverging_centre=0.0)
        summary['baseline'] = summarise(W_base, names)
        summary['diff']     = summarise(diff,    names)

    with open(str(out_base) + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDesign mean |W - 0.5| = {summary["main"]["mean_abs_diff"]:.3f}')
    print(f'Top 5 most asymmetric pairs on this design:')
    for entry in summary['main']['top_asymmetric'][:5]:
        print(f'  {entry["A"]:<22} vs {entry["B"]:<22}  '
              f'{100*entry["P_A_wins"]:>5.1f}% / {100*entry["P_B_wins"]:>5.1f}%  '
              f'(gap {100*entry["abs_gap"]:.1f})')
    print(f'\nFiles written under {out_base}*')


if __name__ == '__main__':
    main()
