"""Compute per-strategy 'WR vs.\\ the field' for the default vs.\\ GA-2p winner
board and print both a markdown table and a LaTeX rows snippet suitable for
pasting into slides_final.tex.

Source data (already on disk; no re-simulation required):
  logs/optimizer_v3/heatmap_ga2p_mask.npy       — 30x30 win-rate matrix W_GA
  logs/optimizer_v3/heatmap_ga2p_mask.diff.npy  — W_GA - W_default

For each strategy i,
    WR(B)_i = mean over j != i of W_B[i, j],
i.e. the empirical win rate of strategy i against the rest of the 30-strategy
pool on board B (20 games per pair, common-random-numbers within a pair).

The 10 named archetypes are shown individually; the 20 'Sampled_*' strategies
are summarised as one aggregate row at the bottom, sorted by delta descending.

Run from monopoly/:
    PYTHONIOENCODING=utf-8 python scripts/compute_archetype_table.py
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# One-line identity tags for the 10 named archetypes. Kept short for the slide.
IDENTITY = {
    'AggressiveBuilder': r'build at zero floor, no trades',
    'CashHoarder':       r'\$1k$+$ unspendable, no trades',
    'Trader':            r'aggressive build $+$ trade',
    'RailroadKing':      r'rails only, ignore colour groups',
    'LowCostOnly':       r'skip 5 expensive groups',
    'HighCostOnly':      r'skip 3 cheap groups',
    'Balanced':          r'balanced thresholds, trades',
    'Passive':           r'no trades, no rails/utils',
    'Bully':             r'ultra-aggressive build $+$ trade',
    'RiskAverse':        r'hoard, skip expensive groups',
}


def wr_vs_field(W: np.ndarray) -> np.ndarray:
    """Mean win rate of strategy i against all other strategies in the pool."""
    n = W.shape[0]
    out = np.empty(n)
    for i in range(n):
        ix = [j for j in range(n) if j != i]
        out[i] = W[i, ix].mean()
    return out


def main():
    sys.path.insert(0, str(_ROOT))
    from optimizer.strategy_pool import load_strategy_pool

    W_ga = np.load(_ROOT / 'logs/optimizer_v3/heatmap_ga2p_mask.npy')
    diff = np.load(_ROOT / 'logs/optimizer_v3/heatmap_ga2p_mask.diff.npy')
    W_default = W_ga - diff

    names = [n for n, _ in load_strategy_pool()]
    wr_d = wr_vs_field(W_default)
    wr_g = wr_vs_field(W_ga)
    delta = wr_g - wr_d

    # Named archetypes only (first 10), sorted by delta descending.
    named = [(names[i], wr_d[i], wr_g[i], delta[i]) for i in range(10)]
    named.sort(key=lambda r: -r[3])

    # 20 sampled strategies aggregated.
    sampled_d = wr_d[10:].mean()
    sampled_g = wr_g[10:].mean()
    sampled_delta = sampled_g - sampled_d

    # ------------------------------------------------------------------
    # Markdown (for quick reading)
    # ------------------------------------------------------------------
    print('## Archetype win-rate vs.\\ the 30-strategy pool (2p, 20 games/pair)\n')
    print(f'| {"Archetype":<20} | WR(default) | WR(GA-2p) | Delta  |')
    print(f'| {"-"*20} | {"-"*11} | {"-"*9} | {"-"*6} |')
    for n, d, g, dl in named:
        print(f'| {n:<20} | {d:>11.2f} | {g:>9.2f} | {dl:>+6.2f} |')
    print(f'| {"Random sampled (n=20)":<20} | {sampled_d:>11.2f} '
          f'| {sampled_g:>9.2f} | {sampled_delta:>+6.2f} |')
    print()
    print(f'spread (max - min) WR vs.\\ field, default = '
          f'{wr_d[:10].max() - wr_d[:10].min():.2f}, '
          f'GA-2p = {wr_g[:10].max() - wr_g[:10].min():.2f}')

    # ------------------------------------------------------------------
    # LaTeX rows snippet (paste into the slide tabular)
    # ------------------------------------------------------------------
    print('\n% --- LaTeX rows (paste into slides_final.tex) ---')
    n_named = len(named)
    for k, (n, d, g, dl) in enumerate(named):
        bold_d, bold_g, bold_dl = '{:.2f}', '{:.2f}', '${:+.2f}$'
        # Bold the largest positive and largest negative delta rows.
        if k == 0 or k == n_named - 1:
            bold_dl = r'$\mathbf{{{:+.2f}}}$'
        sign_dl = bold_dl.format(dl)
        # Bold the highest WR on each board (for the row that holds it).
        top_def = wr_d[:10].argmax()
        top_ga  = wr_g[:10].argmax()
        d_str = (r'\textbf{' + bold_d.format(d) + r'}'
                 if names.index(n) == top_def else bold_d.format(d))
        g_str = (r'\textbf{' + bold_g.format(g) + r'}'
                 if names.index(n) == top_ga else bold_g.format(g))
        # Visual separator before the negative-delta block.
        if k > 0 and named[k - 1][3] >= 0 and dl < 0:
            print(r'    \addlinespace')
        print(f'    {n:<18} & {d_str} & {g_str} & {sign_dl} & '
              f'{IDENTITY[n]} \\\\')
    print(r'    \addlinespace')
    print(f'    {"Random sampled (n=20)":<18} & {sampled_d:.2f} & '
          f'{sampled_g:.2f} & ${sampled_delta:+.2f}$ & '
          f'$20$ random draws from the parameter space \\\\')


if __name__ == '__main__':
    main()
