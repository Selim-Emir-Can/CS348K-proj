"""Combined Pareto plot: all evaluated candidates from GA + random +
ablations on a single fairness-vs-rounds scatter, with the GA optimum
highlighted.

Single-run Pareto plots (from report_runs.py) cluster densely around
the GA's converged optimum, producing a Gaussian blob without a clear
"knee". Combining all six runs (random + GA + 4 ablations) shows the
full design-space sweep, makes the lower-left frontier visible, and
puts the GA optimum on the knee.

Usage (from monopoly/):
    python scripts/combined_pareto.py --log-dir logs/optimizer_v3 \\
        --suffix _mask --out-dir report/figures/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RUN_STEMS = ['random', 'ga', 'abl_fair', 'abl_len', 'abl_draw', 'abl_money']


def load_run(path):
    if not path.exists():
        return None
    rows = []
    best = (float('inf'), None)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r.get('metrics') or {}
        f = m.get('mean_fairness')
        rd = m.get('mean_rounds')
        dr = m.get('mean_draw_rate', 0.0)
        if f is None or rd is None:
            continue
        rows.append((f, rd, dr, r['score']))
        if r['score'] < best[0]:
            best = (r['score'], (f, rd, dr))
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='logs/optimizer_v3')
    ap.add_argument('--suffix', default='_mask')
    ap.add_argument('--out-dir', default='report/figures/')
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n_p in (2, 3):
        all_pts = []
        ga_best_pt = None
        for stem in RUN_STEMS:
            run_path = log_dir / f'{stem}_{n_p}p{args.suffix}.jsonl'
            res = load_run(run_path)
            if res is None:
                print(f'  skip (missing): {run_path}')
                continue
            rows, best = res
            all_pts.extend(rows)
            if stem == 'ga':
                ga_best_pt = best[1]
                ga_best_score = best[0]

        if not all_pts:
            continue

        fs = np.array([p[0] for p in all_pts])
        rds = np.array([p[1] for p in all_pts])
        drs = np.array([p[2] for p in all_pts])

        plt.figure(figsize=(7.5, 5.0))
        sc = plt.scatter(fs, rds, c=drs, s=12, alpha=0.55,
                         cmap='viridis', vmin=0.0,
                         vmax=max(0.05, np.percentile(drs, 95)))
        if ga_best_pt is not None:
            plt.scatter([ga_best_pt[0]], [ga_best_pt[1]],
                        s=180, marker='*', c='red',
                        edgecolors='black', linewidths=1.2, zorder=5,
                        label=f'GA optimum (score={ga_best_score:.3f})')
            plt.legend(loc='upper right', fontsize=11)
        cbar = plt.colorbar(sc)
        cbar.set_label('draw rate', fontsize=12)
        plt.xlabel(r'mean pair-fairness $|\Delta\,\mathrm{win\ rate}|$',
                   fontsize=13)
        plt.ylabel('mean rounds / game', fontsize=13)
        plt.title(f'Design-space sweep: {len(all_pts)} candidates '
                  f'(GA + random + 4 ablations, {n_p}p)',
                  fontsize=12)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        out_path = out_dir / f'ga_{n_p}p_pareto.png'
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        plt.close()
        print(f'  wrote {out_path}  ({len(all_pts)} points)')


if __name__ == '__main__':
    main()
