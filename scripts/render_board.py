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

def _cell_rect(index: int, corners: tuple, n_cells: int):
    """Return (x, y, w, h, side, is_corner) for the given cell position.

    `corners` is a 4-tuple of (go_idx, jail_idx, free_idx, gtj_idx) giving
    the board indices of the four corner cells in the current (possibly
    shrunk) board. Side lengths are derived from them so every side has a
    consistent cell count regardless of whether the board has 40 or 27 cells.

    Layout assumes corners appear in clockwise order when traversed along
    positive index direction (matching standard Monopoly):
        GO  --bottom-row-right-to-left-->  Jail
        Jail --left-column-bottom-to-top--> FreeParking
        FreeParking --top-row-left-to-right--> GoToJail
        GoToJail --right-column-top-to-bottom--> (wrap to GO)

    Side encoding (for text rotation):
        0 = bottom, 1 = left, 2 = top, 3 = right.
    """
    go_i, jail_i, free_i, gtj_i = corners

    # Number of non-corner cells per side
    bot_n   = (jail_i - go_i - 1) % n_cells
    left_n  = (free_i - jail_i - 1) % n_cells
    top_n   = (gtj_i - free_i - 1) % n_cells
    right_n = (go_i - gtj_i - 1) % n_cells

    # Layout geometry: pick the dominant side length for uniform cell size.
    max_side_n = max(bot_n, left_n, top_n, right_n)
    cs = 1.5                                      # corner square side
    ic = 1.0                                      # non-corner short side
    S  = 2 * cs + max_side_n * ic                 # total board edge length

    def _bot(i):   # non-corner index along bottom (i=0 nearest GO, larger toward Jail)
        x = S - cs - (i + 1) * ic
        return (x, 0.0, ic, cs, 0, False)

    def _left(i):  # non-corner index along left (i=0 nearest Jail, larger toward FreeParking)
        y = cs + i * ic
        return (0.0, y, cs, ic, 1, False)

    def _top(i):   # non-corner index along top (i=0 nearest FreeParking, larger toward GoToJail)
        x = cs + i * ic
        return (x, S - cs, ic, cs, 2, False)

    def _right(i): # non-corner index along right (i=0 nearest GoToJail, larger toward GO)
        y = S - cs - (i + 1) * ic
        return (S - cs, y, cs, ic, 3, False)

    # Corners:
    if index == go_i:
        return S - cs, 0.0, cs, cs, 0, True
    if index == jail_i:
        return 0.0, 0.0, cs, cs, 1, True
    if index == free_i:
        return 0.0, S - cs, cs, cs, 2, True
    if index == gtj_i:
        return S - cs, S - cs, cs, cs, 3, True

    # Non-corner: figure out which side this index falls on.
    def _offset_along(section_start_exclusive):
        return (index - section_start_exclusive - 1) % n_cells

    off_from_go   = (index - go_i  ) % n_cells
    off_from_jail = (index - jail_i) % n_cells
    off_from_free = (index - free_i) % n_cells
    off_from_gtj  = (index - gtj_i ) % n_cells

    if off_from_go   <= bot_n:     return _bot  (off_from_go   - 1)
    if off_from_jail <= left_n:    return _left (off_from_jail - 1)
    if off_from_free <= top_n:     return _top  (off_from_free - 1)
    if off_from_gtj  <= right_n:   return _right(off_from_gtj  - 1)
    raise ValueError(f'index {index} did not map to any side')


# --- Drawing --------------------------------------------------------------- #

def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars - 1] + '…'


def _draw_cell(ax, cfg_cell, default_cell, index, corners, n_cells,
               label_overrides=None, annotate_changes=False):
    x, y, w, h, side, is_corner = _cell_rect(index, corners, n_cells)
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


def _find_corners(cells):
    """Locate indices of the four corner cells (GO, Jail, FreeParking, GoToJail)
    in the cell list. Returns (go, jail, free, gtj). Assumes every config keeps
    all four special cells (which the optimiser never removes)."""
    go_i = jail_i = free_i = gtj_i = None
    for i, c in enumerate(cells):
        tn = type(c).__name__
        if tn == 'Cell':
            if c.name == 'GO':
                go_i = i
            elif 'Jail' in c.name:
                jail_i = i
        elif tn == 'FreeParking':
            if free_i is None:
                free_i = i
        elif tn == 'GoToJail':
            gtj_i = i
    # Fallback to canonical positions if something is missing (shouldn't happen
    # with the current optimiser, which never removes corner cells).
    if go_i   is None: go_i   = 0
    if jail_i is None: jail_i = 10 if len(cells) > 10 else 0
    if free_i is None: free_i = 20 if len(cells) > 20 else 0
    if gtj_i  is None: gtj_i  = 30 if len(cells) > 30 else 0
    return go_i, jail_i, free_i, gtj_i


def draw_board(ax, cfg, default_cfg=None, title: str = '',
               annotate_changes: bool = False):
    """Draw a full board (any cell count) on the given axes."""
    n = len(cfg.cells)
    corners = _find_corners(cfg.cells)

    # Default board mapping by cell name (so we annotate changes robustly
    # even when the two cfgs have different indices after shrinkage).
    default_by_name = {}
    if default_cfg is not None:
        for c in default_cfg.cells:
            default_by_name[c.name] = c

    for i in range(n):
        c = cfg.cells[i]
        d = default_by_name.get(c.name)
        _draw_cell(ax, c, d, i, corners, n, annotate_changes=annotate_changes)

    # Axis limits derived from the same geometry used by _cell_rect.
    bot_n   = (corners[1] - corners[0] - 1) % n
    left_n  = (corners[2] - corners[1] - 1) % n
    top_n   = (corners[3] - corners[2] - 1) % n
    right_n = (corners[0] - corners[3] - 1) % n
    max_side_n = max(bot_n, left_n, top_n, right_n)
    S = 2 * 1.5 + max_side_n * 1.0
    ax.set_xlim(-0.5, S + 0.5)
    ax.set_ylim(-0.5, S + 0.5)
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


def _decode_for_render(space, vec, style: str = 'shrunk'):
    """Decode a design vector in the chosen rendering style.

    style='shrunk'      → true structural shrinkage (board gets shorter).
    style='substituted' → keep board at 40 cells; replaced props → FreeParking.
    """
    if style == 'substituted':
        return space.decode_as_substituted(vec)
    return space.decode(vec)


def load_cfg(args):
    cfg = GameConfig.from_yaml(args.config)
    space = DesignSpace(cfg, removal_direction=getattr(args, 'removal_direction', 'cheapest'))
    style = getattr(args, 'render_style', 'shrunk')
    if args.identity:
        return cfg, _decode_for_render(space, space.identity_vec(), style), 'default board'
    if args.vec:
        vec = np.array([float(x) for x in args.vec.split(',')])
        return cfg, _decode_for_render(space, vec, style), 'custom vec'
    if args.runs:
        best = best_design_from_run(args.runs)
        vec = np.array(best['vec'])
        label = Path(args.runs).stem + '_best'
        return cfg, _decode_for_render(space, vec, style), label
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
    ap.add_argument('--render-style', choices=('shrunk', 'substituted'),
                    default='shrunk',
                    help='"shrunk" (default): removed properties are deleted, '
                         'board becomes shorter. "substituted": removed '
                         'properties become FreeParking cells, board stays 40 '
                         'cells (the "Free Parking style" used in the original renders).')
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
