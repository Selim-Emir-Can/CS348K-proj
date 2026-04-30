"""Generate figures for the Task 1 + Task 2 report sections.

Reads the JSON/JSONL outputs of the LLM-eval and LLM-GA pipelines and
emits PNG figures into ``monopoly/report/figures/llm/`` (and a copy to
``report/figures/llm/`` so both report-tex copies can use them).

Figures produced:
  fig_cross_class_agreement.png  — Task 1: side-by-side default vs
                                    GA-winner rounds and transfer-rate
                                    for 2p and 3p, all-LLM seats.
  fig_llm_ga_convergence.png     — Task 2: LLM-GA best-so-far score
                                    over evaluations, per-generation
                                    minima, and the winner's iteration.
  fig_llm_buy_rate_slices.png    — Task 1: buy-rate slices by cash
                                    bucket and by monopoly opportunity,
                                    contrasting default vs GA-winner.

Run from monopoly/:
  python scripts/plot_llm_results.py
"""
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')   # no GUI on Windows shell
import matplotlib.pyplot as plt


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_OUT_PRIMARY  = _ROOT / 'report' / 'figures' / 'llm'
_OUT_SECONDARY = _ROOT.parent / 'report' / 'figures' / 'llm'
_OUT_PRIMARY.mkdir(parents=True, exist_ok=True)
_OUT_SECONDARY.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    """Save figure to both report/figures/llm/ trees so either .tex compiles."""
    p1 = _OUT_PRIMARY / name
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    shutil.copy(p1, _OUT_SECONDARY / name)
    print(f'wrote {p1} (+ copy in project root)')


# --------------------------------------------------------------------------- #
# 1. Cross-class agreement (Task 1)                                              #
# --------------------------------------------------------------------------- #

def _read_summary(path: Path):
    """Return list[dict] from a summary.csv emitted by eval_llm_on_boards.py."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _agg(rows, board_tag):
    rows = [r for r in rows if r['board_tag'] == board_tag]
    if not rows:
        return {'rounds': 0.0, 'xfer_rate': 0.0, 'n': 0}
    rounds = [int(r['rounds']) for r in rows]
    xfer   = [float(r['transfer_rate']) for r in rows]
    return {'rounds': mean(rounds), 'xfer_rate': mean(xfer), 'n': len(rows)}


def fig_cross_class_agreement():
    rows_2p = _read_summary(_ROOT / 'logs/llm_eval/2p_v2/summary.csv')
    rows_3p = _read_summary(_ROOT / 'logs/llm_eval/3p_v2/summary.csv')
    cells = {
        '2p default':        _agg(rows_2p, 'default'),
        '2p GA-winner':      _agg(rows_2p, 'ga_2p_winner'),
        '3p default':        _agg(rows_3p, 'default'),
        '3p GA-winner':      _agg(rows_3p, 'ga_3p_winner'),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = list(cells.keys())
    rounds = [cells[k]['rounds'] for k in labels]
    xfer   = [cells[k]['xfer_rate'] for k in labels]
    colors = ['#888888', '#1f77b4', '#888888', '#d62728']
    bars1 = ax1.bar(labels, rounds, color=colors)
    ax1.set_ylabel('mean rounds per game')
    ax1.set_title('Game length: default vs GA-winner under LLM seats')
    ax1.tick_params(axis='x', labelrotation=15)
    for b, v in zip(bars1, rounds):
        ax1.text(b.get_x() + b.get_width()/2, v + 1.0, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=9)
    bars2 = ax2.bar(labels, xfer, color=colors)
    ax2.set_ylabel('mean transfer rate ($/round)')
    ax2.set_title('Inter-player money transfer: default vs GA-winner')
    ax2.tick_params(axis='x', labelrotation=15)
    for b, v in zip(bars2, xfer):
        ax2.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=9)
    fig.suptitle('Task 1: LLM seats reproduce the rule-based GA\'s '
                 'directional signal (rounds ↓, transfer ↑)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'fig_cross_class_agreement.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. LLM-GA convergence (Task 2)                                                 #
# --------------------------------------------------------------------------- #

def fig_llm_ga_convergence():
    evals = []
    with open(_ROOT / 'logs/optimizer_llm/llm_ga_2p/evals.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evals.append(json.loads(line))

    iters  = [e['iter']  for e in evals]
    scores = [e['score'] for e in evals]
    gens   = [e['gen']   for e in evals]

    # Best-so-far
    best_so_far = []
    cur = float('inf')
    for s in scores:
        cur = min(cur, s)
        best_so_far.append(cur)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(iters, scores, 'o-', alpha=0.45, color='#888888',
            label='per-eval score', markersize=4)
    ax.plot(iters, best_so_far, '-', linewidth=2.0, color='#1f77b4',
            label='best so far')
    # Generation boundaries
    last_gen = -1
    for it, g in zip(iters, gens):
        if g != last_gen:
            ax.axvline(it - 0.5, color='#cccccc', linestyle=':', linewidth=0.8)
            last_gen = g
            ax.text(it, max(scores) * 0.97, f'gen {g}',
                    fontsize=8, color='#888888', va='top')
    # Mark winner
    winner_iter = int(min(range(len(scores)), key=lambda i: scores[i]))
    ax.scatter([iters[winner_iter]], [scores[winner_iter]], s=80,
               color='#d62728', marker='*', zorder=5,
               label=f'winner (iter={iters[winner_iter]}, score={scores[winner_iter]:.3f})')
    # Reference: rule-based GA-2p winner score (from ga_2p_mask.jsonl)
    rb_evals = [json.loads(l) for l in
                open(_ROOT / 'logs/optimizer/ga_2p_mask.jsonl')]
    rb_best = min(rb_evals, key=lambda e: e['score'])['score']
    ax.axhline(rb_best, color='#888888', linestyle='--', linewidth=1.0,
               label=f'rule-based GA-2p winner score ({rb_best:.3f})')
    ax.set_xlabel('GA evaluation index')
    ax.set_ylabel('composite score (lower is better)')
    ax.set_title('Task 2: LLM-driven GA convergence (pop=8, gens=5, n_seeds=5)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig_llm_ga_convergence.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Buy-rate slices (Task 1)                                                    #
# --------------------------------------------------------------------------- #

_CASH_BUCKETS = [
    ('<200',      lambda c: c < 200),
    ('200-500',   lambda c: 200 <= c < 500),
    ('500-1000',  lambda c: 500 <= c < 1000),
    ('1000-1500', lambda c: 1000 <= c < 1500),
    ('>=1500',    lambda c: c >= 1500),
]


def _cash_bucket(c):
    for name, pred in _CASH_BUCKETS:
        if pred(c):
            return name
    return 'unknown'


def _monopoly_op(rec):
    g = rec.get('group_size') or 0
    own = rec.get('same_group_self') or 0
    opp = rec.get('opp_in_group') or 0
    if g == 0:
        return 'unknown'
    if opp >= max(1, g // 2) and opp >= own:
        return 'opp_dominates'
    if own > 0:
        return 'partial_self'
    return 'fresh'


def _read_decisions(decisions_dir: Path):
    out = []
    if not decisions_dir.exists():
        return out
    for log in sorted(decisions_dir.glob('seed_*.jsonl')):
        with open(log, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _slice(records, key_fn):
    by = defaultdict(lambda: [0, 0])
    for r in records:
        if r.get('prefilter') != 'sent_to_llm':
            continue
        k = key_fn(r)
        by[k][0] += 1
        if r.get('parsed') == 'BUY':
            by[k][1] += 1
    return {k: (n, b, b / n if n else 0.0) for k, (n, b) in by.items()}


def fig_llm_buy_rate_slices():
    default_2p = _read_decisions(_ROOT / 'logs/llm_eval/2p_v2/decisions/default')
    ga_2p     = _read_decisions(_ROOT / 'logs/llm_eval/2p_v2/decisions/ga_2p_winner')

    by_cash_d = _slice(default_2p, lambda r: _cash_bucket(r.get('cash', 0)))
    by_cash_g = _slice(ga_2p,     lambda r: _cash_bucket(r.get('cash', 0)))
    by_op_d   = _slice(default_2p, _monopoly_op)
    by_op_g   = _slice(ga_2p,     _monopoly_op)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: by cash bucket
    cash_order = [b for b, _ in _CASH_BUCKETS]
    x = list(range(len(cash_order)))
    w = 0.4
    rd = [by_cash_d.get(b, (0, 0, 0))[2] for b in cash_order]
    rg = [by_cash_g.get(b, (0, 0, 0))[2] for b in cash_order]
    axL.bar([i - w/2 for i in x], rd, w, label='default', color='#888888')
    axL.bar([i + w/2 for i in x], rg, w, label='GA-winner', color='#1f77b4')
    axL.set_xticks(x)
    axL.set_xticklabels(cash_order, rotation=15)
    axL.set_ylabel('buy rate (LLM calls)')
    axL.set_title('LLM buy rate by cash bucket (2p)')
    axL.legend()
    axL.set_ylim(0, 1.05)

    # Right: by monopoly opportunity
    op_order = ['fresh', 'partial_self', 'opp_dominates']
    x = list(range(len(op_order)))
    rd = [by_op_d.get(b, (0, 0, 0))[2] for b in op_order]
    rg = [by_op_g.get(b, (0, 0, 0))[2] for b in op_order]
    axR.bar([i - w/2 for i in x], rd, w, label='default', color='#888888')
    axR.bar([i + w/2 for i in x], rg, w, label='GA-winner', color='#1f77b4')
    axR.set_xticks(x)
    axR.set_xticklabels(op_order)
    axR.set_ylabel('buy rate (LLM calls)')
    axR.set_title('LLM buy rate by monopoly opportunity (2p)')
    axR.legend()
    axR.set_ylim(0, 1.05)

    fig.suptitle('Task 1: LLM behaviour shifts under GA-winner board '
                 '(cash-aggressive, opp_dominates → strict PASS)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'fig_llm_buy_rate_slices.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    fig_cross_class_agreement()
    fig_llm_ga_convergence()
    fig_llm_buy_rate_slices()
    print('done')


if __name__ == '__main__':
    main()
