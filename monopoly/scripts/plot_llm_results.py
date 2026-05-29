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
  fig_cross_evaluator_gap.png    — Cross-eval: each board scored under
                                    BOTH the LLM evaluator and the
                                    rule-based pool. Shows the
                                    "partial generalisation" finding.
  fig_fairness_asymmetry.png     — Cross-eval: fairness under LLM eval
                                    (~0.20 seat-position bias) vs pool
                                    eval (~0.38 strategic asymmetry)
                                    for the LLM-GA winner. The
                                    diagnostic finding.
  fig_llm_ga_score_distribution.png — Task 2: per-generation score
                                    distributions (box plot) showing
                                    selection pressure + n_seeds=5
                                    noise.
  fig_v1_vs_v2_hallucination.png — Task 1: stacked-bar comparing v1
                                    (validator-bug-induced fabrication)
                                    to v2 (post-fix, 0/2288 across the
                                    board).

Run from monopoly/:
  python scripts/plot_llm_results.py
"""
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')   # no GUI on Windows shell
import matplotlib.pyplot as plt
import numpy as np


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
    """Six-panel pool-vs-LLM comparison for the 4 cells (2p/3p x default/GA-winner).

    Each panel = one metric from the report's tab:cross; each cell shows two
    grouped bars (strategy pool n=1000 vs LLM seats n=20). The 'composite'
    score uses the cross-count-comparable normalised form (transfer in
    $/player-turn vs target 50) for consistency with the rest of the report.
    """
    sys.path.insert(0, str(_ROOT))
    from optimizer.objectives import evaluate

    def _llm_metrics(board_tag, n_players, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        games = [{
            'rounds':         int(r['rounds']),
            'truncated':      bool(int(r['truncated'])),
            'transfer_total': int(float(r['transfer_total'])),
            'winner':         r['winner'] or None,
            'strategy_names': [f'LLM_p{i}' for i in range(n_players)],
        } for r in rows]
        return evaluate([games])['metrics']

    # Pool eval results (canonical n=1000 from the GA cross-eval).
    pool_data = json.load(open(_ROOT / 'logs/optimizer_v3/cross_eval_mask.json',
                                encoding='utf-8'))
    pool_idx = {(r['design'], r['n_players']): r['metrics']
                for r in pool_data['results']}
    pool = {
        ('default', 2):   pool_idx[('identity_default', 2)],
        ('ga_winner', 2): pool_idx[('ga_2p_mask_best',  2)],
        ('default', 3):   pool_idx[('identity_default', 3)],
        ('ga_winner', 3): pool_idx[('ga_3p_mask_best',  3)],
    }
    # LLM eval metrics, recomputed via objectives.evaluate() for apples-to-apples.
    llm = {
        ('default', 2):   _llm_metrics('default',      2, '2p_n100'),
        ('ga_winner', 2): _llm_metrics('ga_2p_winner', 2, '2p_n100'),
        ('default', 3):   _llm_metrics('default',      3, '3p_n100'),
        ('ga_winner', 3): _llm_metrics('ga_3p_winner', 3, '3p_n100'),
    }

    def _norm_composite(m, n, t_pt=50.0, t_rounds=60.0):
        len_pen = abs(m['mean_rounds'] - t_rounds) / t_rounds
        t_pen   = abs(m['mean_transfer_rate'] / max(1, n) - t_pt) / t_pt
        return (1.0 * m['mean_fairness'] + 0.5 * m['max_fairness']
                + 0.5 * len_pen + 0.3 * m['mean_draw_rate'] + 0.3 * t_pen)

    cells  = [('2p default', 'default', 2),
              ('2p GA-winner', 'ga_winner', 2),
              ('3p default', 'default', 3),
              ('3p GA-winner', 'ga_winner', 3)]
    labels = [c[0] for c in cells]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6))
    ax_c, ax_f, ax_fm = axes[0]
    ax_r, ax_d, ax_t  = axes[1]
    x = np.arange(len(cells))
    w = 0.38
    C_POOL, C_LLM = '#888888', '#1f77b4'

    def panel(ax, mfn, title, ylabel, fmt, target=None, target_label=None):
        pool_v = [mfn(pool[(c[1], c[2])], c[2]) for c in cells]
        llm_v  = [mfn(llm[(c[1],  c[2])], c[2]) for c in cells]
        b1 = ax.bar(x - w/2, pool_v, w, label='strategy pool ($n{=}1000$)', color=C_POOL)
        b2 = ax.bar(x + w/2, llm_v,  w, label=r'LLM seats ($n{=}100$)',      color=C_LLM)
        ymax = max(max(pool_v), max(llm_v))
        if target is not None:
            ax.axhline(target, ls='--', lw=1, color='#555555')
            if target_label:
                ax.text(len(cells) - 0.5, target + ymax * 0.025, target_label,
                        fontsize=7.5, ha='right', va='bottom', color='#555555')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7.5, loc='upper right')
        for bars, vals in ((b1, pool_v), (b2, llm_v)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + ymax * 0.02,
                        fmt.format(v), ha='center', va='bottom', fontsize=7.5)

    panel(ax_c,  lambda m, n: _norm_composite(m, n),
          r'Composite (normalised, $\downarrow$)',  'composite score', '{:.2f}')
    panel(ax_f,  lambda m, n: m['mean_fairness'],
          r'Mean fairness $\overline{F}$ ($\downarrow$)',
          r'$\overline{F}$ (win-rate spread)',     '{:.2f}')
    panel(ax_fm, lambda m, n: m['max_fairness'],
          r'Worst-pair fairness $F_{\max}$ ($\downarrow$)',
          r'$F_{\max}$',                            '{:.2f}')
    panel(ax_r,  lambda m, n: m['mean_rounds'],
          r'Game length ($\downarrow$ = faster)',  'mean rounds per game',
          '{:.0f}', target=60, target_label='target 60')
    panel(ax_d,  lambda m, n: 100.0 * m['mean_draw_rate'],
          r'Draw rate ($\downarrow$)',             'draw / truncation (\\%)', '{:.1f}')
    panel(ax_t,  lambda m, n: m['mean_transfer_rate'] / max(1, n),
          r'Transfer per player-turn',             r'\$ / player-turn',
          '{:.1f}', target=50, target_label='target 50')

    fig.suptitle('Task 1: LLM seats and the 30-strategy pool give the same directional '
                 'verdict on the same boards, across all six evaluation metrics',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
                open(_ROOT / 'logs/optimizer_v3/ga_2p_mask.jsonl')]
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
    default_2p = _read_decisions(_ROOT / 'logs/llm_eval/2p_v3/decisions/default')
    ga_2p     = _read_decisions(_ROOT / 'logs/llm_eval/2p_v3/decisions/ga_2p_winner')

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
# 4. Cross-evaluator generalisation gap                                          #
# --------------------------------------------------------------------------- #

def fig_cross_evaluator_gap():
    """Each board × each evaluator → composite score. Shows partial
    generalisation of the LLM-GA winner.

    Numbers come from:
      - LLM eval: logs/llm_eval/2p_v3/{default, ga_2p_winner} summary +
                  logs/optimizer_llm/llm_ga_2p/best_design.json (LLM-GA winner
                  under LLM eval, n_seeds=5).
      - Rule-based pool eval: logs/optimizer_v3/cross_eval_mask.json
        (default + rule-based GA winners) and
        logs/optimizer/cross_eval_llm_ga_winner.json (LLM-GA winner).
    """
    # Pool eval results (already cached on disk).
    pool = {r['design'] + '_' + str(r['n_players']) + 'p': r
            for r in json.load(open(_ROOT / 'logs/optimizer_v3/cross_eval_mask.json',
                                     encoding='utf-8'))['results']}
    pool_llm = {r['design'] + '_' + str(r['n_players']) + 'p': r
                for r in json.load(open(_ROOT / 'logs/optimizer/cross_eval_llm_ga_winner.json',
                                         encoding='utf-8'))['results']}

    pool_default_2p     = pool['identity_default_2p']['score']
    pool_default_3p     = pool['identity_default_3p']['score']
    pool_rb_winner_2p   = pool['ga_2p_mask_best_2p']['score']
    pool_rb_winner_3p   = pool['ga_2p_mask_best_3p']['score']
    pool_llm_winner_2p  = pool_llm['evals_best_2p']['score']
    pool_llm_winner_3p  = pool_llm['evals_best_3p']['score']

    # LLM-eval scores (recompute from per-game summary using the same
    # evaluate() function would be more rigorous; here we use the cached
    # composite from logs/optimizer_llm/llm_ga_2p/best_design.json for the
    # LLM-GA winner and a "from-scratch" composite for default + rule-based
    # GA winner under LLM seats — those would require an extra harness run,
    # so we instead pull the per-game stats from logs/llm_eval/2p_v3 and
    # plug them through evaluate() to match.). We approximate the LLM-eval
    # score for the rule-based GA winner using the per-game stats from
    # logs/llm_eval/2p_v3/summary.csv (board_tag=ga_2p_winner) — this is
    # the same n=20-seed signal used in Task 1.
    sys.path.insert(0, str(_ROOT))
    from optimizer.objectives import Targets, Weights, evaluate

    def _llm_score(board_tag, n_players, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        # objectives.evaluate expects per-game dicts with 'rounds', 'truncated',
        # 'transfer_total', 'winner', 'strategy_names'. Reconstruct those.
        games = []
        for r in rows:
            seat_names = [f'LLM_p{i}' for i in range(n_players)]
            games.append({
                'rounds':         int(r['rounds']),
                'truncated':      bool(int(r['truncated'])),
                'transfer_total': int(float(r['transfer_total'])),
                'winner':         r['winner'] or None,
                'strategy_names': seat_names,
            })
        out = evaluate([games])
        return out['score']

    llm_default_2p = _llm_score('default',      2, '2p_v3')
    llm_rb_winner_2p = _llm_score('ga_2p_winner', 2, '2p_v3')
    llm_default_3p = _llm_score('default',      3, '3p_v3')
    llm_rb_winner_3p = _llm_score('ga_3p_winner', 3, '3p_v3')
    # LLM-GA winner under LLM eval at 2p = the GA's own composite (n_seeds=5).
    llm_ga_winner_score = json.load(
        open(_ROOT / 'logs/optimizer_llm/llm_ga_2p/best_design.json',
             encoding='utf-8'))['score']
    # LLM-GA winner under LLM eval at 3p — separate run (n_seeds=20), to
    # match the apples-to-apples sample size of the other 3p cells.
    llm_ga_winner_score_3p = _llm_score('llm_ga_2p_winner', 3,
                                         'llm_ga_winner_3p')

    # The cross-evaluator story is most legible on the per-component
    # metrics (mean rounds and mean fairness), not on the composite
    # score, because the composite at 3p cancels out (round penalty
    # for the default board offsets transfer-overshoot penalty for the
    # optimised boards, leaving the LLM evaluator nearly flat across
    # three very different boards). Two metric rows × two player counts
    # = 4 panels. Within each panel: 3 designs × 2 evaluators = 6 bars.
    pool_fair_default_2p = pool['identity_default_2p']['metrics']['mean_fairness']
    pool_fair_rb_2p      = pool['ga_2p_mask_best_2p']['metrics']['mean_fairness']
    pool_fair_llm_2p     = pool_llm['evals_best_2p']['metrics']['mean_fairness']
    pool_fair_default_3p = pool['identity_default_3p']['metrics']['mean_fairness']
    pool_fair_rb_3p      = pool['ga_3p_mask_best_3p']['metrics']['mean_fairness']
    pool_fair_llm_3p     = pool_llm['evals_best_3p']['metrics']['mean_fairness']
    pool_rounds_default_2p = pool['identity_default_2p']['metrics']['mean_rounds']
    pool_rounds_rb_2p      = pool['ga_2p_mask_best_2p']['metrics']['mean_rounds']
    pool_rounds_llm_2p     = pool_llm['evals_best_2p']['metrics']['mean_rounds']
    pool_rounds_default_3p = pool['identity_default_3p']['metrics']['mean_rounds']
    pool_rounds_rb_3p      = pool['ga_3p_mask_best_3p']['metrics']['mean_rounds']
    pool_rounds_llm_3p     = pool_llm['evals_best_3p']['metrics']['mean_rounds']

    # LLM-eval per-component (recompute via evaluate() so it's apples-to-apples).
    def _llm_metrics(board_tag, n_players, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        games = []
        for r in rows:
            games.append({
                'rounds':         int(r['rounds']),
                'truncated':      bool(int(r['truncated'])),
                'transfer_total': int(float(r['transfer_total'])),
                'winner':         r['winner'] or None,
                'strategy_names': [f'LLM_p{i}' for i in range(n_players)],
            })
        return evaluate([games])['metrics']

    llm_m_def_2p = _llm_metrics('default',          2, '2p_v3')
    llm_m_rb_2p  = _llm_metrics('ga_2p_winner',     2, '2p_v3')
    llm_ga_best  = json.load(open(_ROOT / 'logs/optimizer_llm/llm_ga_2p/best_design.json',
                                   encoding='utf-8'))['metrics']
    llm_m_def_3p = _llm_metrics('default',          3, '3p_v3')
    llm_m_rb_3p  = _llm_metrics('ga_3p_winner',     3, '3p_v3')
    llm_m_lga_3p = _llm_metrics('llm_ga_2p_winner', 3, 'llm_ga_winner_3p')

    # Pull the data into per-panel arrays.
    designs = ['default', 'rule-based\nGA winner', 'LLM-driven\nGA-2p winner']
    panels = [
        ('Mean rounds, 2p', 'rounds',
         [llm_m_def_2p['mean_rounds'], llm_m_rb_2p['mean_rounds'],
          llm_ga_best['mean_rounds']],
         [pool_rounds_default_2p, pool_rounds_rb_2p, pool_rounds_llm_2p]),
        ('Mean rounds, 3p', 'rounds',
         [llm_m_def_3p['mean_rounds'], llm_m_rb_3p['mean_rounds'],
          llm_m_lga_3p['mean_rounds']],
         [pool_rounds_default_3p, pool_rounds_rb_3p, pool_rounds_llm_3p]),
        ('Fairness ($|$WR$_{\\max}-$WR$_{\\min}|$), 2p', 'fairness',
         [_seat_fairness_2p('default', '2p_v3'),
          _seat_fairness_2p('ga_2p_winner', '2p_v3'),
          llm_ga_best['mean_fairness']],
         [pool_fair_default_2p, pool_fair_rb_2p, pool_fair_llm_2p]),
        ('Fairness, 3p', 'fairness',
         [llm_m_def_3p['mean_fairness'], llm_m_rb_3p['mean_fairness'],
          llm_m_lga_3p['mean_fairness']],
         [pool_fair_default_3p, pool_fair_rb_3p, pool_fair_llm_3p]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    x = np.arange(len(designs))
    w = 0.38
    for ax, (title, kind, llm_v, pool_v) in zip(flat, panels):
        b1 = ax.bar(x - w/2, llm_v, w, label='LLM evaluator',
                    color='#1f77b4')
        b2 = ax.bar(x + w/2, pool_v, w, label='rule-based pool',
                    color='#888888')
        ax.set_xticks(x)
        ax.set_xticklabels(designs)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('rounds' if kind == 'rounds' else 'fairness (max-min WR)')
        if kind == 'fairness':
            ax.set_ylim(0, max(max(pool_v), max(llm_v)) * 1.4)
        for bs in (b1, b2):
            for b in bs:
                v = b.get_height()
                fmt = f'{v:.0f}' if kind == 'rounds' else f'{v:.2f}'
                ax.text(b.get_x() + b.get_width()/2, v + (max(b1.datavalues)*0.02 if kind=='rounds' else 0.005),
                        fmt, ha='center', va='bottom', fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Cross-evaluator comparison: rounds-effect transfers across '
                 'evaluators; fairness-effect does not (single-personality '
                 'evaluator under-counts strategic asymmetry)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'fig_cross_evaluator_gap.png')
    plt.close(fig)


def _seat_fairness_2p(board_tag, run_dir):
    """Helper used by fig_cross_evaluator_gap to compute the 2p LLM-eval
    fairness as the seat-position bias on n=20 all-LLM games."""
    rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
    rows = [r for r in rows if r['board_tag'] == board_tag]
    if not rows:
        return 0.0
    wins = Counter(r['winner'] for r in rows)
    seats = [f'LLM_p{i}' for i in range(2)]
    wrs = [wins.get(s, 0) / len(rows) for s in seats]
    return max(wrs) - min(wrs)


# --------------------------------------------------------------------------- #
# 5. Fairness asymmetry across evaluators                                       #
# --------------------------------------------------------------------------- #

def fig_fairness_asymmetry():
    """The diagnostic finding: a single-personality evaluator
    under-counts fairness problems. The LLM-GA winner has fairness
    0.20 under LLM eval (just seat-position bias) but 0.379 under
    the diverse 30-strategy pool.
    """
    pool = {r['design'] + '_' + str(r['n_players']) + 'p': r
            for r in json.load(open(_ROOT / 'logs/optimizer_v3/cross_eval_mask.json',
                                     encoding='utf-8'))['results']}
    pool_llm = {r['design'] + '_' + str(r['n_players']) + 'p': r
                for r in json.load(open(_ROOT / 'logs/optimizer/cross_eval_llm_ga_winner.json',
                                         encoding='utf-8'))['results']}

    # Fairness numbers (mean_fairness)
    designs = ['default', 'rule-based\nGA-2p winner', 'LLM-driven\nGA-2p winner']
    # LLM-eval fairness is approximately the seat-position bias on n=20
    # all-LLM games. Compute it directly from summary.csv.
    def _seat_fairness(board_tag, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        if not rows:
            return 0.0
        wins = Counter(r['winner'] for r in rows)
        seat_names = [f'LLM_p{i}' for i in range(2)]
        wrs = [wins.get(s, 0) / len(rows) for s in seat_names]
        return max(wrs) - min(wrs)

    f_llm_eval = [
        _seat_fairness('default',      '2p_v3'),
        _seat_fairness('ga_2p_winner', '2p_v3'),
        # LLM-GA winner under LLM eval = mean_fairness from
        # best_design.json metrics (n_seeds=5 so binary at coarse resolution).
        json.load(open(_ROOT / 'logs/optimizer_llm/llm_ga_2p/best_design.json',
                       encoding='utf-8'))['metrics']['mean_fairness'],
    ]
    f_pool_eval = [
        pool['identity_default_2p']['metrics']['mean_fairness'],
        pool['ga_2p_mask_best_2p']['metrics']['mean_fairness'],
        pool_llm['evals_best_2p']['metrics']['mean_fairness'],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(designs))
    w = 0.36
    b1 = ax.bar(x - w/2, f_llm_eval, w, label='LLM evaluator (2 identical seats)',
                 color='#1f77b4')
    b2 = ax.bar(x + w/2, f_pool_eval, w, label='rule-based pool (30 archetypes)',
                 color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(designs)
    ax.set_ylabel('fairness ($|$WR$_{\\max}-$WR$_{\\min}|$)')
    ax.set_title('Fairness under LLM evaluator vs. rule-based pool. '
                 'The LLM-GA winner has the LARGEST gap — a board\n'
                 'that looks fair to identical agents but has '
                 'exploitable asymmetries diverse strategies surface.',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, max(f_pool_eval) * 1.25)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f'{b.get_height():.3f}', ha='center', va='bottom',
                     fontsize=8)
    # Highlight the LLM-GA winner gap with an annotation
    gap = f_pool_eval[2] - f_llm_eval[2]
    ax.annotate('', xy=(x[2] + w/2, f_pool_eval[2]),
                xytext=(x[2] - w/2, f_llm_eval[2]),
                arrowprops=dict(arrowstyle='<->', color='#888888', lw=1.0))
    ax.text(x[2] + 0.05, (f_llm_eval[2] + f_pool_eval[2]) / 2,
             f'gap: {gap:+.2f}', fontsize=9, color='#444444')
    fig.tight_layout()
    _save(fig, 'fig_fairness_asymmetry.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Per-generation score distribution (selection pressure)                     #
# --------------------------------------------------------------------------- #

def fig_llm_ga_score_distribution():
    """Box plot of per-generation score distributions for the LLM-driven GA.

    Visualises both selection pressure (gen-over-gen mean drops) and the
    per-eval noise that the n_seeds=5 protocol incurs.
    """
    evals = [json.loads(l) for l in
             open(_ROOT / 'logs/optimizer_llm/llm_ga_2p/evals.jsonl',
                  encoding='utf-8') if l.strip()]
    by_gen = defaultdict(list)
    for e in evals:
        by_gen[e['gen']].append(e['score'])
    gens = sorted(by_gen.keys())
    data = [by_gen[g] for g in gens]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bp = ax.boxplot(data, positions=gens, patch_artist=True,
                     boxprops=dict(facecolor='#cce0f4', color='#1f77b4'),
                     medianprops=dict(color='#d62728', linewidth=2),
                     whiskerprops=dict(color='#1f77b4'),
                     capprops=dict(color='#1f77b4'))
    # Overlay individual scores so the n_seeds=5 noise is visible
    for g, scores in zip(gens, data):
        ax.scatter([g] * len(scores), scores, alpha=0.4, s=22,
                   color='#1f77b4', zorder=3)
    # Mark the winner generation
    winner_score = min(min(data, key=min))
    winner_gen = next(g for g, scores in zip(gens, data)
                      if winner_score in scores)
    ax.scatter([winner_gen], [winner_score], marker='*', s=160,
               color='#d62728', zorder=5,
               label=f'overall winner (gen={winner_gen}, '
                     f'score={winner_score:.3f})')
    ax.set_xlabel('generation')
    ax.set_ylabel('candidate composite score')
    ax.set_title('LLM-driven GA: per-generation score distributions.\n'
                 'Selection pressure visible (median drops gen 0→2); '
                 'spread reflects n_seeds=5 evaluation noise.', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save(fig, 'fig_llm_ga_score_distribution.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. v1-vs-v2 hallucination accounting                                          #
# --------------------------------------------------------------------------- #

def fig_v1_vs_v2_hallucination():
    """Stacked-bar comparison: v1 (validator-bug-induced flags) vs
    v2 (post-fix, 0 across all boards). Visualises both:
    (a) the raw 4.3% headline rate that the broken validator produced,
    (b) the post-reclassification breakdown into spurious vs
        retry-induced, and
    (c) the v2 perfect run that confirms the methodology when correctly
        implemented.
    """
    # Per-board counts (computed by hand from analysis_*.md, double-checked
    # in notes/task1_postmortem_2026-04-29.md).
    boards = ['2p default', '2p GA-winner', '3p default', '3p GA-winner']
    n_calls = [595, 341, 920, 448]                 # v1 LLM calls
    v1_real_first    = [0,  0,  0,  0]              # 0/2304 first-pass real
    v1_spurious      = [3, 15, 52, 30]              # validator-bug flags
    v1_retry_induced = [2,  6, 19, 20]              # post-retry fabricated
    v2_flagged       = [0,  0,  0,  0]              # all clean post-fix

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                    gridspec_kw={'width_ratios': [1.1, 1]})

    # Left: v1 stacked by category, as percentages of n_calls
    x = np.arange(len(boards))
    w = 0.65
    spur_pct = [s / n * 100 for s, n in zip(v1_spurious, n_calls)]
    fab_pct  = [r / n * 100 for r, n in zip(v1_retry_induced, n_calls)]
    real_pct = [r / n * 100 for r, n in zip(v1_real_first, n_calls)]
    axL.bar(x, real_pct, w, color='#d62728', label='real first-pass (0/2304)')
    axL.bar(x, fab_pct, w, bottom=real_pct, color='#ff7f0e',
            label='retry-induced fabrication (47/2304)')
    axL.bar(x, spur_pct, w, bottom=[a + b for a, b in zip(real_pct, fab_pct)],
            color='#888888', label='spurious validator bug (100/2304)')
    axL.set_xticks(x)
    axL.set_xticklabels(boards, rotation=15)
    axL.set_ylabel('% of LLM calls (first-pass + retry artefacts)')
    axL.set_title('v1 (validator bug present): apparent halluc. 4.3%, '
                  'real 0%')
    axL.legend(loc='upper left', fontsize=8)
    axL.set_ylim(0, max([a + b + c for a, b, c in
                          zip(real_pct, fab_pct, spur_pct)]) * 1.4)

    # Right: v2 — single bar per board, all zero, stylised.
    axR.bar(x, v2_flagged, w, color='#888888')
    axR.set_xticks(x)
    axR.set_xticklabels(boards, rotation=15)
    axR.set_ylim(0, 7)   # match left-panel y for visual parity
    axR.set_ylabel('% of LLM calls flagged')
    axR.set_title('v2 (validator fixed): 0/2207 across the board')
    for i, _ in enumerate(boards):
        axR.text(i, 0.2, '0/{}'.format(['584', '281', '917', '425'][i]),
                  ha='center', fontsize=9, color='#444444')
    fig.suptitle('Hallucination accounting: the validator-and-retry stack '
                 'is part of the probe', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'fig_v1_vs_v2_hallucination.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def fig_evaluator_contrast():
    """One-chart visual: % improvement over default, per evaluator, per board.

    Tests the cross-class claim as a contrast:
      - rule-based GA winner: pool and LLM agree on the magnitude of improvement
        over default (validation).
      - LLM-driven GA winner:  pool and LLM disagree on it (diagnostic).

    Score is the cross-count-comparable normalised composite (transfer term
    in $/player-turn vs. target 50). Sample sizes:
      pool n=1000 for every cell;
      LLM n=20 for default + rule-based GA winner (2p_v3 summary);
      LLM n=5 (training metrics) for LLM-driven GA winner at 2p
              (best_design.json from the LLM-GA's own training run).
    """
    sys.path.insert(0, str(_ROOT))
    from optimizer.objectives import evaluate

    def _norm_composite(m, n, t_pt=50.0, t_rounds=60.0):
        len_pen = abs(m['mean_rounds'] - t_rounds) / t_rounds
        t_pen   = abs(m['mean_transfer_rate'] / max(1, n) - t_pt) / t_pt
        return (1.0 * m['mean_fairness'] + 0.5 * m['max_fairness']
                + 0.5 * len_pen + 0.3 * m['mean_draw_rate'] + 0.3 * t_pen)

    def _llm_metrics(board_tag, n_players, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        games = [{
            'rounds':         int(r['rounds']),
            'truncated':      bool(int(r['truncated'])),
            'transfer_total': int(float(r['transfer_total'])),
            'winner':         r['winner'] or None,
            'strategy_names': [f'LLM_p{i}' for i in range(n_players)],
        } for r in rows]
        return evaluate([games])['metrics']

    # Pool data (n=1000): default + rule-based GA winners from cross_eval_mask;
    # LLM-driven GA winner from a separate pool re-evaluation.
    pool_main = json.load(open(_ROOT / 'logs/optimizer_v3/cross_eval_mask.json',
                                encoding='utf-8'))
    pool_idx = {(r['design'], r['n_players']): r['metrics']
                for r in pool_main['results']}
    pool_lga = json.load(open(_ROOT / 'logs/optimizer/cross_eval_llm_ga_winner.json',
                               encoding='utf-8'))
    pool_lga_idx = {(r['design'], r['n_players']): r['metrics']
                    for r in pool_lga['results']}
    p_default = pool_idx[('identity_default', 2)]
    p_rb      = pool_idx[('ga_2p_mask_best',  2)]
    p_lga     = pool_lga_idx[('evals_best',   2)]

    # LLM data, all at n=100 (dedicated eval runs, not training metrics).
    l_default = _llm_metrics('default',          2, '2p_n100')
    l_rb      = _llm_metrics('ga_2p_winner',     2, '2p_n100')
    l_lga     = _llm_metrics('llm_ga_2p_winner', 2, 'llm_ga_winner_2p_n100')

    n = 2

    # Per-metric extractors (absolute values; both evaluators' raw metrics).
    metrics = [
        ('Composite (normalised)',  r'$\downarrow$ better',
         lambda m, n: _norm_composite(m, n), None, '{:.2f}'),
        (r'Mean fairness $\overline{F}$', r'$\downarrow$ better',
         lambda m, n: m['mean_fairness'], None, '{:.2f}'),
        (r'Worst-pair $F_{\max}$', r'$\downarrow$ better',
         lambda m, n: m['max_fairness'], None, '{:.2f}'),
        ('Game length',            r'$\downarrow$ = faster (target 60)',
         lambda m, n: m['mean_rounds'], 60, '{:.0f}'),
        ('Draw rate (%)',          r'$\downarrow$ better',
         lambda m, n: 100.0 * m['mean_draw_rate'], None, '{:.1f}'),
        ('Transfer per player-turn', r'(target 50)',
         lambda m, n: m['mean_transfer_rate'] / max(1, n), 50, '{:.1f}'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0))
    boards = ['rule-based\nGA winner', 'LLM-driven\nGA winner']
    x = np.arange(len(boards))
    w = 0.38
    C_POOL, C_LLM = '#888888', '#1f77b4'

    for ax, (title, subtitle, mfn, target, fmt) in zip(axes.flat, metrics):
        pool_v = [mfn(p_rb, n),  mfn(p_lga, n)]
        llm_v  = [mfn(l_rb, n),  mfn(l_lga, n)]
        b1 = ax.bar(x - w/2, pool_v, w,
                    label='pool eval (n=1000)' if ax is axes[0, 0] else None,
                    color=C_POOL)
        b2 = ax.bar(x + w/2, llm_v, w,
                    label='LLM eval (n=100)' if ax is axes[0, 0] else None,
                    color=C_LLM)
        if target is not None:
            ax.axhline(target, ls='--', lw=1, color='#555555')
            ax.text(len(boards) - 0.5, target,
                    f' target {target:g}', fontsize=8, va='center',
                    ha='right', color='#555555')
        ymax_bar = max(pool_v + llm_v)
        for bars, vals in ((b1, pool_v), (b2, llm_v)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + ymax_bar * 0.02,
                        fmt.format(v), ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(boards, fontsize=9.5)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=20)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=8,
                style='italic', color='#555555')
        ax.grid(axis='y', alpha=0.3)

    axes[0, 0].legend(fontsize=9, loc='upper right')

    fig.suptitle('Cross-class agreement: both evaluators prefer the rule-based GA winner '
                 'on the composite and on 5 of 6 per-metric verdicts '
                 '(LLM-driven wins only on transfer/turn, the metric its n=5 training optimized)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'fig_evaluator_contrast.png')
    plt.close(fig)


def fig_agreement_audit():
    """Per-metric agreement audit: % change default -> rule-based GA winner
    under the pool evaluator (n=1000) vs. under the LLM evaluator (n=100).

    Same-direction agreement <=> both bars on the same side of zero.
    Same-magnitude agreement <=> both bars at similar height.
    Direct visualization of the report's "agreement audit" table.
    """
    sys.path.insert(0, str(_ROOT))
    from optimizer.objectives import evaluate

    def _norm_composite(m, n, t_pt=50.0, t_rounds=60.0):
        len_pen = abs(m['mean_rounds'] - t_rounds) / t_rounds
        t_pen   = abs(m['mean_transfer_rate'] / max(1, n) - t_pt) / t_pt
        return (1.0 * m['mean_fairness'] + 0.5 * m['max_fairness']
                + 0.5 * len_pen + 0.3 * m['mean_draw_rate'] + 0.3 * t_pen)

    def _llm_metrics(board_tag, n_players, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        games = [{
            'rounds':         int(r['rounds']),
            'truncated':      bool(int(r['truncated'])),
            'transfer_total': int(float(r['transfer_total'])),
            'winner':         r['winner'] or None,
            'strategy_names': [f'LLM_p{i}' for i in range(n_players)],
        } for r in rows]
        return evaluate([games])['metrics']

    pool_main = json.load(open(_ROOT / 'logs/optimizer_v3/cross_eval_mask.json',
                                encoding='utf-8'))
    pool_idx = {(r['design'], r['n_players']): r['metrics']
                for r in pool_main['results']}
    p_def = pool_idx[('identity_default', 2)]
    p_rb  = pool_idx[('ga_2p_mask_best',  2)]
    l_def = _llm_metrics('default',      2, '2p_n100')
    l_rb  = _llm_metrics('ga_2p_winner', 2, '2p_n100')

    n = 2
    # (display label, direction-of-improvement label, value-getter)
    metrics = [
        ('Composite',                r'$\downarrow$ better',
         lambda m: _norm_composite(m, n)),
        (r'Fairness $\overline{F}$', r'$\downarrow$ better',
         lambda m: m['mean_fairness']),
        (r'Worst-pair $F_{\max}$',   r'$\downarrow$ better',
         lambda m: m['max_fairness']),
        ('Game length',              r'$\rightarrow$ target 60',
         lambda m: m['mean_rounds']),
        ('Draw rate',                r'$\downarrow$ better',
         lambda m: m['mean_draw_rate']),
        ('Transfer/turn',            r'$\rightarrow$ target 50',
         lambda m: m['mean_transfer_rate'] / n),
    ]

    def _pct(d, g):
        return None if d == 0 else 100.0 * (g - d) / d

    pct_pool, pct_llm = [], []
    for _, _, fn in metrics:
        pct_pool.append(_pct(fn(p_def), fn(p_rb)))
        pct_llm.append(_pct(fn(l_def), fn(l_rb)))

    labels = [f'{m[0]}\n({m[1]})' for m in metrics]
    x = np.arange(len(labels))
    w = 0.38
    # Grey/blue/orange reserved for boards; evaluators use green + purple.
    C_POOL, C_LLM = '#2ca02c', '#9467bd'

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    # Use 0 for missing bars (LLM draw rate is 0->0); we annotate "n/a" instead.
    bp_vals = [v if v is not None else 0.0 for v in pct_pool]
    bl_vals = [v if v is not None else 0.0 for v in pct_llm]
    b1 = ax.bar(x - w/2, bp_vals, w, label='pool eval (n=1000)', color=C_POOL)
    b2 = ax.bar(x + w/2, bl_vals, w, label='LLM eval (n=100)',  color=C_LLM)
    ax.axhline(0, color='black', lw=0.8)

    ymax = max([abs(v) for v in bp_vals + bl_vals] + [1.0])
    pad = ymax * 0.04
    for bars, raw in ((b1, pct_pool), (b2, pct_llm)):
        for b, v in zip(bars, raw):
            if v is None:
                ax.text(b.get_x() + b.get_width()/2, pad,
                        'n/a\n(no draws)', ha='center', va='bottom',
                        fontsize=8, color='#666666', style='italic')
                continue
            y_off = pad if v >= 0 else -pad
            va = 'bottom' if v >= 0 else 'top'
            ax.text(b.get_x() + b.get_width()/2, v + y_off,
                    f'{v:+.0f}%', ha='center', va=va,
                    fontsize=9.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'% change from default $\rightarrow$ rule-based GA winner',
                  fontsize=11)
    ax.set_ylim(-ymax * 1.20, ymax * 0.55)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)

    fig.suptitle('Cross-class agreement: 6/6 on direction, '
                 '3/6 on magnitude (composite, length, transfer)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'fig_agreement_audit.png')
    plt.close(fig)


def fig_verdict_audit():
    """Per-metric verdict audit: which board (rule-based GA winner vs LLM-driven
    GA winner) does each evaluator class prefer, on each metric?

    Bar value = "rule-based advantage over LLM-driven" as a metric-appropriate %.
      - "lower better" metrics (composite, F-bar, F_max): (llm_d - rb) / llm_d.
      - target metrics (length->60, transfer->50): % closer-to-target.
    Positive bar => rule-based wins; negative => LLM-driven wins. Same-side bars
    across pool and LLM => the two evaluator classes agree on that metric's verdict.
    Draw rate is excluded from the main panel because both rates are <2% under
    every cell; the absolute draw rates are noted in the figure footer instead.
    """
    sys.path.insert(0, str(_ROOT))
    from optimizer.objectives import evaluate

    def _llm_metrics(board_tag, n_players, run_dir):
        rows = _read_summary(_ROOT / f'logs/llm_eval/{run_dir}/summary.csv')
        rows = [r for r in rows if r['board_tag'] == board_tag]
        games = [{
            'rounds':         int(r['rounds']),
            'truncated':      bool(int(r['truncated'])),
            'transfer_total': int(float(r['transfer_total'])),
            'winner':         r['winner'] or None,
            'strategy_names': [f'LLM_p{i}' for i in range(n_players)],
        } for r in rows]
        return evaluate([games])['metrics']

    pool_main = json.load(open(_ROOT / 'logs/optimizer_v3/cross_eval_mask.json',
                                encoding='utf-8'))
    pool_idx = {(r['design'], r['n_players']): r['metrics']
                for r in pool_main['results']}
    pool_lga = json.load(open(_ROOT / 'logs/optimizer/cross_eval_llm_ga_winner.json',
                               encoding='utf-8'))
    pool_lga_idx = {(r['design'], r['n_players']): r['metrics']
                    for r in pool_lga['results']}
    p_rb  = pool_idx[('ga_2p_mask_best',  2)]
    p_lga = pool_lga_idx[('evals_best',   2)]
    l_rb  = _llm_metrics('ga_2p_winner',     2, '2p_n100')
    l_lga = _llm_metrics('llm_ga_2p_winner', 2, 'llm_ga_winner_2p_n100')

    n = 2

    def _adv_lower(ref, val):
        # "+" => rb is lower (better) than ref; ref = llm-driven.
        return 100.0 * (ref - val) / ref if ref else 0.0

    def _adv_target(ref, val, target):
        # "+" => rb is closer to target than ref.
        d_ref = abs(target - ref)
        d_val = abs(target - val)
        return 100.0 * (d_ref - d_val) / d_ref if d_ref else 0.0

    metrics = [
        ('Composite',                r'$\downarrow$ better',
         lambda m: 1.0 * m['mean_fairness'] + 0.5 * m['max_fairness']
                   + 0.5 * abs(m['mean_rounds'] - 60) / 60
                   + 0.3 * m['mean_draw_rate']
                   + 0.3 * abs(m['mean_transfer_rate'] / n - 50) / 50,
         _adv_lower, None),
        (r'Fairness $\overline{F}$', r'$\downarrow$ better',
         lambda m: m['mean_fairness'], _adv_lower, None),
        (r'Worst-pair $F_{\max}$',   r'$\downarrow$ better',
         lambda m: m['max_fairness'], _adv_lower, None),
        ('Game length',              r'$\rightarrow$ target 60',
         lambda m: m['mean_rounds'],  _adv_target, 60),
        ('Transfer/turn',            r'$\rightarrow$ target 50',
         lambda m: m['mean_transfer_rate'] / n, _adv_target, 50),
    ]

    adv_pool, adv_llm = [], []
    for _, _, get, adv_fn, target in metrics:
        if target is None:
            adv_pool.append(adv_fn(get(p_lga), get(p_rb)))
            adv_llm.append(adv_fn(get(l_lga), get(l_rb)))
        else:
            adv_pool.append(adv_fn(get(p_lga), get(p_rb), target))
            adv_llm.append(adv_fn(get(l_lga), get(l_rb), target))

    labels = [f'{m[0]}\n({m[1]})' for m in metrics]
    x = np.arange(len(labels))
    w = 0.38
    # Evaluator bar colors. Grey/blue/orange are reserved for boards
    # (default / rule-based GA winner / LLM-driven GA winner), so evaluators
    # get green + purple.
    C_POOL, C_LLM = '#2ca02c', '#9467bd'

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    # Clamp draw values so a single off-chart bar (e.g., LLM-eval transfer at
    # -258%) doesn't squash the visible range. Labels keep the true value and
    # gain an arrow marker so the reader knows the bar is truncated.
    CLAMP = 150.0
    adv_pool_draw = [max(-CLAMP, min(CLAMP, v)) for v in adv_pool]
    adv_llm_draw  = [max(-CLAMP, min(CLAMP, v)) for v in adv_llm]
    b1 = ax.bar(x - w/2, adv_pool_draw, w, label='pool eval (n=1000)', color=C_POOL)
    b2 = ax.bar(x + w/2, adv_llm_draw,  w, label='LLM eval (n=100)',  color=C_LLM)
    ax.axhline(0, color='black', lw=0.8)

    ymax = CLAMP
    pad = ymax * 0.04
    for bars, raw, drawn in ((b1, adv_pool, adv_pool_draw),
                             (b2, adv_llm,  adv_llm_draw)):
        for b, v_raw, v_draw in zip(bars, raw, drawn):
            clamped = abs(v_raw) > CLAMP
            y_off = pad if v_draw >= 0 else -pad
            va = 'bottom' if v_draw >= 0 else 'top'
            arrow = ''
            if clamped:
                arrow = r' $\downarrow$' if v_raw < 0 else r' $\uparrow$'
            ax.text(b.get_x() + b.get_width()/2, v_draw + y_off,
                    f'{v_raw:+.0f}%{arrow}', ha='center', va=va,
                    fontsize=9.5, fontweight='bold')

    # Shaded background hints: zones tinted with the BOARD color that wins.
    # rule-based board is blue; LLM-driven board is orange.
    ax.axhspan(0, ymax * 1.30, color='#1f77b4', alpha=0.06, zorder=0)
    ax.axhspan(-ymax * 1.30, 0, color='#ff7f0e', alpha=0.06, zorder=0)
    ax.text(0.005, 0.98, 'rule-based wins',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, style='italic', color='#1f4d6f')
    ax.text(0.005, 0.02, 'LLM-driven wins',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, style='italic', color='#7f3d00')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('rule-based advantage over LLM-driven (%)', fontsize=11)
    ax.set_ylim(-ymax * 1.30, ymax * 1.30)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)

    # Draw rate footer (both rates are <2%, % unstable; report absolutes).
    pool_dr_rb  = 100.0 * p_rb['mean_draw_rate']
    pool_dr_lga = 100.0 * p_lga['mean_draw_rate']
    llm_dr_rb   = 100.0 * l_rb['mean_draw_rate']
    llm_dr_lga  = 100.0 * l_lga['mean_draw_rate']
    fig.text(0.5, 0.005,
             f'Draw rate (not on bar chart, both <2%): '
             f'pool — rule-based {pool_dr_rb:.1f}% vs LLM-driven {pool_dr_lga:.1f}%; '
             f'LLM — rule-based {llm_dr_rb:.1f}% vs LLM-driven {llm_dr_lga:.1f}%.',
             ha='center', va='bottom', fontsize=8.5, style='italic', color='#444444')

    fig.suptitle('Rule-based vs LLM-driven (cross-class verdict): '
                 '4/5 favor rule-based, 1/5 favors LLM-driven (transfer)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    _save(fig, 'fig_verdict_audit.png')
    plt.close(fig)


def main():
    fig_cross_class_agreement()
    fig_llm_ga_convergence()
    fig_llm_buy_rate_slices()
    fig_cross_evaluator_gap()
    # fig_evaluator_contrast()  # deprecated: moved to monopoly/outdated_results/
    fig_agreement_audit()
    fig_verdict_audit()
    fig_fairness_asymmetry()
    fig_llm_ga_score_distribution()
    fig_v1_vs_v2_hallucination()
    print('done')


if __name__ == '__main__':
    main()
