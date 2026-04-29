"""Dedicated per-aspect design-space visualisations.

The board design vector has three distinguishable aspects:
  (1) 22 cost multipliers    c_i in [0.5, 2.0] scaling cost_base / cost_house
  (2) 22 rent multipliers    r_i in [0.5, 2.0] scaling rent_base / rent_house
  (3) 22 keep-mask bits      m_i in {0, 1}    deciding which properties survive

The board renders (render_all_boards*.py) encode all three per-cell, but
readers have to squint at annotations. This script emits one high-level
figure per aspect -- a colour-group-by-run heatmap -- so the aggregate
pattern across ablations is visible at a glance.

Usage (from monopoly/):
    python scripts/multiplier_plots.py                                   # default: uses _mask run names
    python scripts/multiplier_plots.py --suffix '' --out-dir ../report/figures/multipliers_legacy

Produces six PNGs in out_dir (three aspects x two player counts):
    cost_multipliers_{2p,3p}.png
    rent_multipliers_{2p,3p}.png
    keep_mask_{2p,3p}.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  11,
    'ytick.labelsize':  12,
    'figure.titlesize': 16,
})


DEFAULT_RUNS_2P = [
    ('ga_2p',        'Combined'),
    ('abl_fair_2p',  'Fair-only'),
    ('abl_len_2p',   'Length-only'),
    ('abl_draw_2p',  'Draw-only'),
    ('abl_money_2p', 'Money-only'),
]
DEFAULT_RUNS_3P = [
    ('ga_3p',        'Combined'),
    ('abl_fair_3p',  'Fair-only'),
    ('abl_len_3p',   'Length-only'),
    ('abl_draw_3p',  'Draw-only'),
    ('abl_money_3p', 'Money-only'),
]


def _best_of_run(path: Path) -> dict:
    best = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if best is None or e['score'] < best['score']:
                best = e
    return best


def _load_property_names_and_groups():
    """Return parallel lists (names, groups) for the 22 colour-group properties
    in board-index order, from the default config."""
    # Lazy imports: this script can run standalone.
    sys.path.insert(0, '.')
    from config import GameConfig
    from optimizer.design_space import DesignSpace

    cfg = GameConfig.from_yaml('default_config.yaml')
    space = DesignSpace(cfg)
    names = []
    groups = []
    for pos, bi in enumerate(space._cg_indices):
        d = space._defaults[pos]
        # Trim leading prefix like "A1 " / "E3 " for cleaner x-axis labels.
        name = d['name']
        short = name.split(' ', 1)[1] if ' ' in name else name
        names.append(short)
        groups.append(d['group'])
    return names, groups


# Group palette consistent with render_board.GROUP_COLOURS
GROUP_COLOURS = {
    'Brown':     '#8B4513',
    'Lightblue': '#87CEEB',
    'Pink':      '#FF69B4',
    'Orange':    '#FFA500',
    'Red':       '#DC143C',
    'Yellow':    '#FFD700',
    'Green':     '#228B22',
    'Indigo':    '#191970',
}


def _plot_multiplier_heatmap(mat, row_labels, col_labels, col_groups,
                             title, out_path, vmin=0.5, vmax=2.0,
                             cmap='RdBu_r', centre=1.0):
    """Heatmap of multipliers: rows = runs, cols = properties."""
    fig, ax = plt.subplots(figsize=(18, 4 + 0.4 * len(row_labels)))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    # Colour the x-tick labels by colour group so the reader sees group structure.
    for tick, group in zip(ax.get_xticklabels(), col_groups):
        tick.set_color(GROUP_COLOURS.get(group, 'black'))

    # Annotate each cell with its numeric value.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_col = 'white' if abs(val - centre) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=txt_col)

    # Reference line below the plot: central value (= default board).
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(f'multiplier (1.0 = default)')
    cbar.ax.axhline(centre, color='black', linewidth=1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_keep_mask(mat, row_labels, col_labels, col_groups,
                    title, out_path):
    """Binary heatmap: rows = runs, cols = properties. 1 = kept, 0 = removed."""
    fig, ax = plt.subplots(figsize=(18, 4 + 0.4 * len(row_labels)))
    # Use a two-colour map: removed (dark grey) vs kept (pale green)
    cmap = matplotlib.colors.ListedColormap(['#3A3A3A', '#A7D99D'])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for tick, group in zip(ax.get_xticklabels(), col_groups):
        tick.set_color(GROUP_COLOURS.get(group, 'black'))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat[i, j])
            ax.text(j, i, 'keep' if val else 'drop',
                    ha='center', va='center',
                    fontsize=8,
                    color='black' if val else 'white')

    # Minimal legend via colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 1])
    cbar.ax.set_yticklabels(['removed', 'kept'])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_matrices(runs, log_dir: Path, suffix: str):
    """For each run, return best vec split into (cost, rent, mask) 22-vectors."""
    cost = []
    rent = []
    mask = []
    row_labels = []
    for stem, label in runs:
        p = log_dir / f'{stem}{suffix}.jsonl'
        if not p.exists():
            print(f'  skip (missing): {p}')
            continue
        best = _best_of_run(p)
        vec = np.asarray(best['vec'], dtype=np.float64)
        if len(vec) == 66:
            cost.append(vec[:22])
            rent.append(vec[22:44])
            mask.append(np.round(vec[44:]).astype(int))
        elif len(vec) == 45:
            # legacy encoding: no explicit mask; treat N_props dim as popcount
            cost.append(vec[:22])
            rent.append(vec[22:44])
            nprops = int(round(vec[-1]))
            # Mark the (22 - nprops) cheapest as removed — that matches
            # the legacy decoder's removal policy.
            sys.path.insert(0, '.')
            from config import GameConfig
            from optimizer.design_space import DesignSpace
            cfg = GameConfig.from_yaml('default_config.yaml')
            space = DesignSpace(cfg)
            m = np.ones(22, dtype=int)
            for pos in space._cost_rank[:22 - nprops]:
                m[pos] = 0
            mask.append(m)
        else:
            print(f'  skip (bad vec len): {p}')
            continue
        row_labels.append(label)
    return (np.vstack(cost) if cost else np.zeros((0, 22)),
            np.vstack(rent) if rent else np.zeros((0, 22)),
            np.vstack(mask) if mask else np.zeros((0, 22)),
            row_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='logs/optimizer')
    ap.add_argument('--out-dir', default='../report/figures/multipliers')
    ap.add_argument('--suffix',  default='_mask',
                    help='Run-name suffix (default "_mask" for the mask-based runs).')
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names, groups = _load_property_names_and_groups()

    for n_players, runs in [(2, DEFAULT_RUNS_2P), (3, DEFAULT_RUNS_3P)]:
        cost, rent, mask, row_labels = build_matrices(runs, log_dir, args.suffix)
        if not row_labels:
            print(f'  [{n_players}p] no runs found, skipping')
            continue

        _plot_multiplier_heatmap(
            cost, row_labels, names, groups,
            title=f'Cost multipliers (best design per run) — {n_players}-player',
            out_path=out_dir / f'cost_multipliers_{n_players}p.png',
        )
        _plot_multiplier_heatmap(
            rent, row_labels, names, groups,
            title=f'Rent multipliers (best design per run) — {n_players}-player',
            out_path=out_dir / f'rent_multipliers_{n_players}p.png',
        )
        _plot_keep_mask(
            mask, row_labels, names, groups,
            title=f'Keep mask (best design per run) — {n_players}-player',
            out_path=out_dir / f'keep_mask_{n_players}p.png',
        )
        print(f'  [{n_players}p] wrote 3 PNGs into {out_dir}')

    print(f'\nDone. Outputs in {out_dir}/')


if __name__ == '__main__':
    main()
