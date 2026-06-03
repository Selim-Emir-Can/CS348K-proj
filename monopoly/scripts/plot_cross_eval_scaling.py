"""Scaling figure: how default vs GA-winner boards behave as player count grows.

Reads a cross_eval JSON that contains results across several player counts
(produced by scripts/cross_eval.py --player-counts 2,3,4,5) and plots three
panels vs player count, one line per board:
  - composite score (lower is better)
  - mean rounds (length; target 60)
  - draw / truncation rate (%)

Fairness is deliberately NOT plotted: the win-rate spread compresses as seats
grow (baseline ~1/n per seat), so cross-count fairness is not comparable.

Run from monopoly/:
  python scripts/plot_cross_eval_scaling.py \
      --in logs/optimizer_v3/cross_eval_scaling.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUT = Path(__file__).resolve().parent.parent / "report" / "figures"

# raw design label -> (display name, colour, marker)
STYLE = {
    "identity_default": ("default",       "#888888", "o"),
    "ga_2p_mask_best":  ("GA-2p winner",  "#1f77b4", "s"),
    "ga_3p_mask_best":  ("GA-3p winner",  "#2ca02c", "^"),
}


def _normalized_composite(m, n_players, target_transfer_pt=50.0,
                          target_rounds=60.0,
                          w_fair=1.0, w_fmax=0.5, w_len=0.5,
                          w_draw=0.3, w_money=0.3):
    """Composite score with normalized transfer (=$/player-turn) for cross-count eval.

    Same weights and same other-term penalties as the training composite, but the
    transfer term uses $/player-turn against a fixed target (default 50, which
    equals the original 100 $/round target evaluated at the 2p training regime).
    The training composite (with target=100 $/round) is still in r["score"];
    this recomputation is presented as a secondary cross-count-comparable score.
    """
    F = m["mean_fairness"]
    Fmax = m["max_fairness"]
    len_pen = abs(m["mean_rounds"] - target_rounds) / target_rounds
    draw = m["mean_draw_rate"]
    t_pt = m["mean_transfer_rate"] / max(1, n_players)
    t_pen = abs(t_pt - target_transfer_pt) / target_transfer_pt
    return (w_fair * F + w_fmax * Fmax + w_len * len_pen
            + w_draw * draw + w_money * t_pen)


def _load_series(path):
    """Returns {design: sorted [(n_players, score_norm, rounds, finished%,
                                  transfer/round, rel_fair%, transfer/player-turn)]}.

    score_norm is the cross-count-comparable composite (transfer term uses
    $/player-turn with a fixed target of 50). The training composite using
    transfer in $/round vs target 100 is in r["score"] (left unused here; that
    one is the optimization objective, this one is the cross-count comparator).

    rel_fair% = 100 * (F_design - F_default) / F_default at the same player count,
    to factor out the mechanical compression of win-rate spread with seats.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    default_fair = {r["n_players"]: r["metrics"]["mean_fairness"]
                    for r in data["results"] if r["design"] == "identity_default"}
    series = {}
    for r in data["results"]:
        d = r["design"]
        if d not in STYLE:
            continue
        m = r["metrics"]
        df = default_fair.get(r["n_players"])
        rel_fair = 100.0 * (m["mean_fairness"] - df) / df if df else 0.0
        n = r["n_players"]
        series.setdefault(d, []).append(
            (n, _normalized_composite(m, n), m["mean_rounds"],
             100.0 * (1.0 - m["mean_draw_rate"]),   # % games that FINISH
             m["mean_transfer_rate"], rel_fair,
             m["mean_transfer_rate"] / max(1, n),   # $/player-turn
             m["mean_fairness"], m["max_fairness"],  # F̄, F_max (raw)
             100.0 * m["mean_draw_rate"]))           # draw / truncation %
    for d in series:
        series[d].sort()
    return series


def _plot_series(series, idx, ylabel, title, subtitle, target=None,
                 target_label=None, ax=None, fontsize=12):
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for d, rows in series.items():
        name, color, marker = STYLE[d]
        xs = [r[0] for r in rows]
        ys = [r[idx] for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, label=name,
                linewidth=2.4, markersize=8)
    if target is not None:
        ax.axhline(target, ls="--", lw=1.2, color="#555555")
        if target_label:
            ax.text(ax.get_xlim()[1], target, "  " + target_label,
                    color="#555555", fontsize=fontsize - 3, va="center")
    ax.set_xlabel("players", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=24)
    if subtitle:
        ax.text(0.5, 1.012, subtitle, ha="center", va="bottom",
                transform=ax.transAxes, fontsize=fontsize - 2,
                color="#444444", style="italic")
    ax.set_xticks(sorted({r[0] for rows in series.values() for r in rows}))
    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=fontsize - 2, loc="best", framealpha=0.92)
    if new_fig:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="cross_eval JSON with multiple player counts.")
    ap.add_argument("--out-combined",
                    default=str(_OUT / "fig_cross_eval_scaling.png"),
                    help="Combined 3-panel figure path.")
    args = ap.parse_args()

    series = _load_series(args.inp)
    _OUT.mkdir(parents=True, exist_ok=True)

    # --- combined 2x3 overview (mirrors the report's tab:cross columns) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    axc, axf, axfm = axes[0]
    axr, axd, axt  = axes[1]
    _plot_series(series, 1, "composite (normalised)", "Composite",
                 r"$\downarrow$ is better", ax=axc, fontsize=10)
    _plot_series(series, 7, r"mean fairness $\overline{F}$",
                 r"Mean fairness $\overline{F}$",
                 r"$\downarrow$ is better (compresses with $n$; baseline $1/n$)",
                 ax=axf, fontsize=10)
    _plot_series(series, 8, r"worst-pair $F_{\max}$",
                 r"Worst-pair fairness $F_{\max}$",
                 r"$\downarrow$ is better",
                 ax=axfm, fontsize=10)
    _plot_series(series, 2, "mean rounds", "Game length",
                 r"$\downarrow$ = faster (target 60)", target=60,
                 ax=axr, fontsize=10)
    _plot_series(series, 9, "draw / truncation (%)", "Draw rate",
                 r"$\downarrow$ is better", ax=axd, fontsize=10)
    _plot_series(series, 6, r"transfer (\$ / player-turn)",
                 "Money transfer (normalised)",
                 r"target 50 = original 100 \$/round at 2p",
                 target=50, target_label="target 50",
                 ax=axt, fontsize=10)
    fig.suptitle("Generalization across player count (trained only on 2p / 3p)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_combined = Path(args.out_combined)
    fig.savefig(out_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cross_eval_scaling] wrote {out_combined}")

    # --- standalone per-metric figures (presentation-quality) ---
    specs = [
        (1, "composite score (normalised)",
         "Composite score across player counts (cross-count comparable)",
         r"$\downarrow$ is better; transfer term uses \$/player-turn vs target 50",
         None, None, "fig_scaling_composite.png"),
        (2, "mean rounds", "Game length across player counts",
         "the optimized boards stay near target 60 at every player count",
         60, "target 60", "fig_scaling_length.png"),
        (3, "% games that finish", "Decisiveness across player counts",
         "the default Monopoly board only finishes 29 % of games at 8 players",
         None, None, "fig_scaling_finished.png"),
        (4, "inter-player transfer ($ / round)",
         "Money transfer across player counts",
         "all boards overshoot target 100 at high counts; winners overshoot most",
         100, "target 100", "fig_scaling_transfer.png"),
        (5, "fairness vs default (%)",
         "Fairness across player counts (relative to default)",
         r"$\uparrow$ = less fair than default; winners are fairer in-distribution, less fair at high counts",
         0, "= default", "fig_scaling_fairness.png"),
        (6, "transfer (\\$ / player-turn)",
         "Money transfer per player-turn (normalised)",
         "= (\\$/round) / n_players; removes the trivial linear scaling with seats",
         50, "target 50", "fig_scaling_transfer_per_pturn.png"),
    ]
    for idx, ylabel, title, subtitle, target, tlabel, fname in specs:
        fig = _plot_series(series, idx, ylabel, title, subtitle,
                           target=target, target_label=tlabel)
        out = _OUT / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_cross_eval_scaling] wrote {out}")


if __name__ == "__main__":
    main()
