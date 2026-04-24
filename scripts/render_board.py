"""Render a GameConfig as a canonical 11×11 Monopoly board PNG.

Cells are drawn on the perimeter in the standard Monopoly layout:
  index 0  = GO                        (bottom-right corner)
  index 10 = Jail                      (bottom-left corner)
  index 20 = Free Parking              (top-left corner)
  index 30 = Go To Jail                (top-right corner)

The 9 cells of each non-corner side are drawn with their text oriented
readably (rotated so the long axis of each cell points inward). Properties
get a coloured strip at the outer edge; cost and rent are shown below the
name. Cells substituted with FreeParking by the DesignSpace shrinkage show
a "REMOVED" label so the reader can see which properties were dropped.

Usage (from monopoly/):
    # Default board
    python scripts/render_board.py --identity --out figures/board_default.png

    # Best design from a run
    python scripts/render_board.py --runs logs/optimizer/ga_2p.jsonl ^
        --out figures/board_ga2p.png --title "GA-2p winner"

    # Side-by-side default vs optimised, annotated with differences
    python scripts/render_board.py --runs logs/optimizer/ga_2p.jsonl ^
        --compare-identity --out figures/board_compare.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import GameConfig
from optimizer.design_space import DesignSpace
from monopoly.core.cell import (Cell, Chance, CommunityChest, FreeParking,
                                 GoToJail, IncomeTax, LuxuryTax, Property)


# --- Colour palette for property groups ------------------------------------ #

GROUP_COLOURS = {
    'Brown':     '#8B4513',
    'Lightblue': '#87CEEB',
    'Pink':      '#FF69B4',
    'Orange':    '#FFA500',
    'Red':       '#DC143C',
    'Yellow':    '#FFD700',
    'Green':     '#228B22',
    'Indigo':    '#191970',
    'Railroads': '#202020',
    'Utilities': '#B0B0B0',
}

CORNER_COLOURS = {
    'GO':           '#F0E68C',
    'JL Jail':      '#D3D3D3',
    'FP Free Parking': '#F5DEB3',
    'GJ Go To Jail':'#E0B0B0',
}


# --- Position mapping ------------------------------------------------------- #

def _cell_rect(index: int, n_cells: int = 40):
    """Return (x, y, w, h, side, is_corner) for the given cell position.

    Board is drawn on the unit square; corner cells are size 1.5 on a side,
    intermediate cells 1.0 × 1.5. The board side length is 11 * 1 + 2 * 0.5
    = 11 on a side (slightly oversized to highlight corners); we use 12 to
    give corners more breathing room.

    Side encoding (used for text rotation):
        0 = bottom  (indices 0..10), 1 = left (10..20),
        2 = top     (20..30),        3 = right (30..39, 0).
    """
    S = 12.0              # total board side length (display units)
    cs = 1.5              # corner cell size
    ic = (S - 2 * cs) / 9 # intermediate cell size  (= 1.0)

    # Bottom row: GO at (S - cs, 0), 9 intermediates going left, Jail at (0, 0)
    if index == 0:
        return S - cs, 0, cs, cs, 0, True
    if 1 <= index <= 9:
        # spans from Jail corner toward GO corner
        x = cs + (9 - index) * ic
        return x, 0, ic, cs, 0, False
    if index == 10:
        return 0, 0, cs, cs, 1, True
    if 11 <= index <= 19:
        # left column, going up
        y = cs + (index - 11) * ic
        return 0, y, cs, ic, 1, False
    if index == 20:
        return 0, S - cs, cs, cs, 2, True
    if 21 <= index <= 29:
        # top row, left→right
        x = cs + (index - 21) * ic
        return x, S - cs, ic, cs, 2, False
    if index == 30:
        return S - cs, S - cs, cs, cs, 3, True
    if 31 <= index <= 39:
        # right column, going down (index 31 near top, 39 near bottom)
        y = cs + (9 - (index - 30)) * ic
        return S - cs, y, cs, ic, 3, False
    raise ValueError(f'bad index {index}')


# --- Drawing --------------------------------------------------------------- #

def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars - 1] + '…'


def _draw_cell(ax, cfg_cell, default_cell, index, label_overrides=None,
               annotate_changes=False):
    x, y, w, h, side, is_corner = _cell_rect(index)
    base_color = 'white'
    name = cfg_cell.name

    # Corner background
    if is_corner:
        base_color = CORNER_COLOURS.get(name, '#F5F5F5')

    rect = mpatches.Rectangle((x, y), w, h,
                              facecolor=base_color, edgecolor='black', linewidth=1.0)
    ax.add_patch(rect)

    # Property colour strip (~20% of the perpendicular dimension, at outer edge)
    if isinstance(cfg_cell, Property):
        color = GROUP_COLOURS.get(cfg_cell.group, '#888888')
        if side == 0:      # bottom: strip at top of cell
            sx, sy, sw, sh = x, y + h - 0.25, w, 0.25
        elif side == 2:    # top: strip at bottom of cell
            sx, sy, sw, sh = x, y, w, 0.25
        elif side == 1:    # left: strip at right of cell
            sx, sy, sw, sh = x + w - 0.25, y, 0.25, h
        else:              # right: strip at left of cell
            sx, sy, sw, sh = x, y, 0.25, h
        ax.add_patch(mpatches.Rectangle((sx, sy), sw, sh,
                                         facecolor=color, edgecolor='black', linewidth=0.5))

    # Text orientation: rotate so the long axis of the cell points inward
    if side == 0:    rot = 0
    elif side == 2:  rot = 0
    elif side == 1:  rot = 90
    else:            rot = 270

    # Name + cost/rent label
    fontsize = 7 if not is_corner else 9
    lines = []
    short_name = _truncate(name.split(' ', 1)[-1] if ' ' in name else name, 14)
    lines.append(short_name)

    if isinstance(cfg_cell, Property):
        lines.append(f'${cfg_cell.cost_base}')
        lines.append(f'r${cfg_cell.rent_base}')
        if annotate_changes and default_cell is not None and isinstance(default_cell, Property):
            dc = default_cell.cost_base
            dr = default_cell.rent_base
            if dc != cfg_cell.cost_base:
                arrow = '▲' if cfg_cell.cost_base > dc else '▼'
                lines[-2] = f'${cfg_cell.cost_base} {arrow}{abs(cfg_cell.cost_base-dc)}'
            if dr != cfg_cell.rent_base:
                arrow = '▲' if cfg_cell.rent_base > dr else '▼'
                lines[-1] = f'r${cfg_cell.rent_base} {arrow}{abs(cfg_cell.rent_base-dr)}'
    elif isinstance(cfg_cell, FreeParking) and default_cell is not None \
            and isinstance(default_cell, Property) and index not in (20,):
        # A shrinkage-removed property (FreeParking in a non-corner slot)
        lines.append('(removed)')
        ax.add_patch(mpatches.Rectangle((x, y), w, h,
                                         facecolor='#EEEEEE', edgecolor='black',
                                         linewidth=1.0, hatch='//'))

    text = '\n'.join(lines)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center', rotation=rot,
            fontsize=fontsize, family='sans-serif')


def draw_board(ax, cfg, default_cfg=None, title: str = '',
               annotate_changes: bool = False):
    """Draw a full 40-cell board on the given axes."""
    for i in range(40):
        c = cfg.cells[i]
        d = default_cfg.cells[i] if default_cfg is not None else None
        _draw_cell(ax, c, d, i, annotate_changes=annotate_changes)

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=14, pad=10)


def _legend_patches():
    return [mpatches.Patch(color=c, label=g)
            for g, c in GROUP_COLOURS.items()]


# --- Load design ----------------------------------------------------------- #

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


def load_cfg(args):
    cfg = GameConfig.from_yaml(args.config)
    space = DesignSpace(cfg, removal_direction=getattr(args, 'removal_direction', 'cheapest'))
    if args.identity:
        return cfg, space.decode(space.identity_vec()), 'default board'
    if args.vec:
        vec = np.array([float(x) for x in args.vec.split(',')])
        return cfg, space.decode(vec), 'custom vec'
    if args.runs:
        best = best_design_from_run(args.runs)
        vec = np.array(best['vec'])
        label = Path(args.runs).stem + '_best'
        return cfg, space.decode(vec), label
    raise SystemExit('Pass --identity / --vec / --runs.')


# --- Main ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='default_config.yaml')
    ap.add_argument('--identity', action='store_true',
                    help='Render the canonical default board.')
    ap.add_argument('--vec', type=str, default=None,
                    help='Comma-separated 45-dim design vector.')
    ap.add_argument('--runs', type=str, default=None,
                    help='Path to a JSONL run log; best design is rendered.')
    ap.add_argument('--compare-identity', action='store_true',
                    help='Also render the default board alongside for comparison; '
                         'annotate the optimised board with ▲/▼ cost/rent changes.')
    ap.add_argument('--title', type=str, default=None)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--removal-direction', choices=('cheapest', 'expensive', 'middle'),
                    default='cheapest',
                    help='Must match the optimiser setting that produced the design vec.')
    args = ap.parse_args()

    base_cfg, decoded, label = load_cfg(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _space = DesignSpace(base_cfg, removal_direction=args.removal_direction)
    if args.compare_identity:
        default_cfg = _space.decode(_space.identity_vec())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
        draw_board(ax1, default_cfg, default_cfg=None,
                   title='Default board', annotate_changes=False)
        draw_board(ax2, decoded, default_cfg=default_cfg,
                   title=args.title or f'Optimised: {label}',
                   annotate_changes=True)
        legend = fig.legend(handles=_legend_patches(), loc='lower center',
                            ncol=10, frameon=False, fontsize=10)
        plt.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig, ax = plt.subplots(figsize=(12, 12))
        draw_board(ax, decoded,
                   default_cfg=_space.decode(_space.identity_vec())
                   if label != 'default board' else None,
                   title=args.title or label, annotate_changes=False)
        fig.legend(handles=_legend_patches(), loc='lower center',
                   ncol=10, frameon=False, fontsize=10)
        plt.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(str(out_path), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Board rendered to {out_path}')


if __name__ == '__main__':
    main()
