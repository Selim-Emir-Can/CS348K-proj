"""Batch-render winning boards from all optimisation runs, LEGACY 40-cell layout.

Sibling of render_all_boards.py but uses render_board_legacy.draw_board
(fixed canonical 40-cell layout with FreeParking-substituted removed cells
and hatched "(removed)" overlays).

Usage (from monopoly/):
    python scripts/render_all_boards_legacy.py                                    # default output: ../report/figures/boards_legacy/
    python scripts/render_all_boards_legacy.py --suffix _mask --out-dir ../report/figures/boards_mask_legacy
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import GameConfig
from optimizer.design_space import DesignSpace

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from render_board_legacy import draw_board, _legend_patches, best_design_from_run


DEFAULT_RUNS_2P = [
    ('ga_2p',        'Combined objective (2p)'),
    ('abl_fair_2p',  'Fairness-only (2p)'),
    ('abl_len_2p',   'Length-only (2p)'),
    ('abl_draw_2p',  'Draw-rate only (2p)'),
    ('abl_money_2p', 'Money-transfer only (2p)'),
]
DEFAULT_RUNS_3P = [
    ('ga_3p',        'Combined objective (3p)'),
    ('abl_fair_3p',  'Fairness-only (3p)'),
    ('abl_len_3p',   'Length-only (3p)'),
    ('abl_draw_3p',  'Draw-rate only (3p)'),
    ('abl_money_3p', 'Money-transfer only (3p)'),
]


def _metric_summary(entry):
    m = entry['metrics']
    return (f'score={entry.get("score", 0):.3f}  '
            f'fair={m["mean_fairness"]:.2f}  '
            f'rounds={m["mean_rounds"]:.0f}  '
            f'draw={100*m["mean_draw_rate"]:.0f}%  '
            f'xfer={m["mean_transfer_rate"]:.0f}')


def render_pair(base_cfg, run_path, out_path, title_suffix,
                removal_direction='cheapest'):
    """Default vs best-of-run, side-by-side, legacy 40-cell layout."""
    space = DesignSpace(base_cfg, removal_direction=removal_direction)
    default_cfg = space.decode_as_substituted(space.identity_vec())
    best = best_design_from_run(run_path)
    decoded = space.decode_as_substituted(np.asarray(best['vec']))
    summary = _metric_summary(best)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))
    draw_board(ax1, default_cfg, default_cfg=None,
               title='Default board', annotate_changes=False)
    draw_board(ax2, decoded, default_cfg=default_cfg,
               title=f'{title_suffix}\n{summary}',
               annotate_changes=True)
    fig.legend(handles=_legend_patches(), loc='lower center',
               ncol=10, frameon=False, fontsize=10)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(str(out_path), dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_grid(base_cfg, runs, out_path, super_title,
                removal_direction='cheapest'):
    """Default + one winning board per ablation, legacy 40-cell layout."""
    space = DesignSpace(base_cfg, removal_direction=removal_direction)
    default_cfg = space.decode_as_substituted(space.identity_vec())

    n_panels = len(runs) + 1
    cols = 2
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))
    axes = axes.ravel()

    draw_board(axes[0], default_cfg, default_cfg=None,
               title='Default board', annotate_changes=False)
    for i, (stem, label) in enumerate(runs, start=1):
        run_path = Path('logs/optimizer') / f'{stem}.jsonl'
        if not run_path.exists():
            axes[i].set_visible(False)
            continue
        best = best_design_from_run(run_path)
        decoded = space.decode_as_substituted(np.asarray(best['vec']))
        summary = _metric_summary(best)
        draw_board(axes[i], decoded, default_cfg=default_cfg,
                   title=f'{label}\n{summary}',
                   annotate_changes=True)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(super_title, fontsize=18, y=0.995)
    fig.legend(handles=_legend_patches(), loc='lower center',
               ncol=10, frameon=False, fontsize=10)
    plt.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='default_config.yaml')
    ap.add_argument('--out-dir', default='../report/figures/boards_legacy')
    ap.add_argument('--removal-direction', choices=('cheapest', 'expensive', 'middle'),
                    default='cheapest',
                    help='Only affects legacy 45-dim vecs; ignored for 66-dim masks.')
    ap.add_argument('--suffix',  default='',
                    help='Appended to run-name stems and output filenames, e.g. "_mask".')
    args = ap.parse_args()

    base_cfg = GameConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix

    for stem, label in DEFAULT_RUNS_2P + DEFAULT_RUNS_3P:
        run_path = Path('logs/optimizer') / f'{stem}{sfx}.jsonl'
        if not run_path.exists():
            print(f'  skip (missing): {run_path}')
            continue
        out_path = out_dir / f'board_{stem}{sfx}.png'
        render_pair(base_cfg, run_path, out_path, label,
                    removal_direction=args.removal_direction)
        print(f'  wrote {out_path}')

    render_grid(base_cfg,
                [(s + sfx, l) for s, l in DEFAULT_RUNS_2P],
                out_dir / f'boards_grid_2p{sfx}.png',
                super_title='2-player winning boards - legacy 40-cell layout',
                removal_direction=args.removal_direction)
    print(f'  wrote {out_dir / f"boards_grid_2p{sfx}.png"}')
    render_grid(base_cfg,
                [(s + sfx, l) for s, l in DEFAULT_RUNS_3P],
                out_dir / f'boards_grid_3p{sfx}.png',
                super_title='3-player winning boards - legacy 40-cell layout',
                removal_direction=args.removal_direction)
    print(f'  wrote {out_dir / f"boards_grid_3p{sfx}.png"}')


if __name__ == '__main__':
    main()
