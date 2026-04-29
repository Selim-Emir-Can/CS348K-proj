"""Post-hoc reporting for optimize_board.py runs.

Reads one or more JSONL history files and produces:
  - best-so-far convergence plot (overlaid if multiple runs given)
  - top-K designs table per run (metric breakdown)
  - Pareto scatter: fairness vs mean_rounds, coloured by draw rate

Usage (from monopoly/):
    # single run
    python scripts/report_runs.py logs/optimizer/ga_2p_*.jsonl

    # overlay several runs (e.g. five ablation runs) on one convergence plot
    python scripts/report_runs.py logs/optimizer/fair_only_2p.jsonl logs/optimizer/combined_2p.jsonl \
        --out-dir logs/optimizer/reports
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt
import numpy as np

# Larger fonts so the figures stay legible when scaled into a paper (3x bump).
plt.rcParams.update({
    'font.size':         42,
    'axes.titlesize':    48,
    'axes.labelsize':    45,
    'xtick.labelsize':   39,
    'ytick.labelsize':   39,
    'legend.fontsize':   36,
    'figure.titlesize':  48,
    'lines.linewidth':   6.0,
})


def load_history(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_meta(run_path):
    meta_path = Path(run_path).with_suffix('').as_posix() + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def best_so_far(entries):
    best, out = float('inf'), []
    for e in entries:
        best = min(best, e['score'])
        out.append(best)
    return out


def plot_convergence(runs, out_path):
    # Canvas scaled up to fit 3x-bigger text without clipping.
    plt.figure(figsize=(22, 14))
    for name, entries in runs:
        bsf = best_so_far(entries)
        plt.plot(bsf, label=name)
    plt.xlabel('Evaluation index')
    plt.ylabel('Best-so-far score (lower is better)')
    plt.title('Search convergence')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def plot_pareto(entries, out_path):
    xs = [e['metrics'].get('mean_fairness', np.nan) for e in entries]
    ys = [e['metrics'].get('mean_rounds', np.nan)   for e in entries]
    cs = [e['metrics'].get('mean_draw_rate', 0.0)    for e in entries]
    plt.figure(figsize=(20, 13))
    sc = plt.scatter(xs, ys, c=cs, cmap='viridis', s=120, alpha=0.85,
                     edgecolors='k', linewidths=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label('draw rate', fontsize=42)
    cbar.ax.tick_params(labelsize=36)
    plt.xlabel(r'mean pair-fairness $|\Delta\,\mathrm{win\ rate}|$')
    plt.ylabel('mean rounds / game')
    plt.title('Design-space sweep (all evaluated candidates)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def top_k_table(entries, k=5):
    ranked = sorted(entries, key=lambda e: e['score'])[:k]
    rows = []
    for e in ranked:
        m = e['metrics']
        rows.append({
            'iter':       e['iter'],
            'score':      e['score'],
            'fairness':   m.get('mean_fairness'),
            'max_fair':   m.get('max_fairness'),
            'rounds':     m.get('mean_rounds'),
            'draw_rate':  m.get('mean_draw_rate'),
            'transfer':   m.get('mean_transfer_rate'),
        })
    return rows


def format_top_k_table(rows):
    hdr = (f"{'iter':>5}{'score':>9}{'fair':>8}{'max_f':>8}"
           f"{'rounds':>9}{'draw':>8}{'transfer':>10}")
    lines = [hdr, '-' * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['iter']:>5}{r['score']:>9.3f}{r['fairness']:>8.3f}"
            f"{r['max_fair']:>8.3f}{r['rounds']:>9.1f}{r['draw_rate']:>8.3f}"
            f"{r['transfer']:>10.1f}"
        )
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+',
                    help='JSONL files (globs OK).')
    ap.add_argument('--out-dir', default='logs/optimizer/reports')
    ap.add_argument('--top-k', type=int, default=5)
    args = ap.parse_args()

    # Expand globs
    files = []
    for p in args.paths:
        matched = glob.glob(p)
        if matched:
            files.extend(matched)
        else:
            files.append(p)
    if not files:
        raise SystemExit('No input files matched.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for fp in files:
        name = Path(fp).stem
        entries = load_history(fp)
        if not entries:
            print(f'  skipping empty {fp}')
            continue
        runs.append((name, entries))

        print(f'\n=== {name} ({len(entries)} evals) ===')
        print(format_top_k_table(top_k_table(entries, k=args.top_k)))

        # Per-run Pareto
        plot_pareto(entries, out_dir / f'{name}_pareto.png')

    # Combined convergence across all given runs
    if runs:
        plot_convergence(runs, out_dir / 'convergence.png')
        print(f'\nReports written to {out_dir}/')


if __name__ == '__main__':
    main()
