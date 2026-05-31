"""Figure for the human-playtest section of the report/slides.

Reads three evaluation sources for the two playtest boards (default,
GA-2p-winner) and plots all four headline metrics with three evaluator
subsections side by side:

  - Human       : the per-game stats JSONs produced by
                  scripts/human_game_stats.py, under matched RNG seeds
                  (common random numbers across boards). Small N, so the
                  per-game points are overlaid as dots and the default's
                  turn-cap games are flagged.
  - Strat pool  : the n=1000 strategy-pool cross-eval at 2p
                  (logs/optimizer_v3/cross_eval_mask.json).
  - LLM seats   : the n=100 all-LLM 2p eval
                  (logs/llm_eval/2p_n100/summary.csv), re-aggregated
                  through optimizer.objectives.evaluate() so the four
                  metrics are computed the exact same way as the pool
                  and human ones.

Four panels: game length (rounds, target 60), money circulation
(transfer $/round, target 100), draw / truncation rate, win-rate
spread (fairness).

Run from monopoly/:
  python scripts/plot_human_playtest.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_TEST = _ROOT / "human_players_test"
_OUT = _ROOT / "report" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

# (folder, label, colour) in narrative order. Ablation boards (length-only,
# money-only) are dropped from the playtest figure: the headline comparison is
# default vs the combined GA winner; the ablations belong in the simulation
# section, not in the human-pilot panel.
BOARDS = [
    ("default_board", "default",   "#888888"),
    ("ga_2p_winner",  "GA winner", "#1f77b4"),
]

# Map the human-eval folder name to (a) the canonical design key used in
# the strategy-pool cross-eval JSON and (b) the board_tag used in the LLM
# summary CSV. n_players is always 2 here (the playtest is 2p only).
POOL_KEYS = {
    "default_board": ("identity_default", 2),
    "ga_2p_winner":  ("ga_2p_mask_best",  2),
}
LLM_TAGS = {
    "default_board": "default",
    "ga_2p_winner":  "ga_2p_winner",
}


def load_games(folder):
    games = []
    for p in sorted((_TEST / folder).glob("game*_stats.json")):
        games.append(json.loads(p.read_text(encoding="utf-8")))
    return games


def _human_metrics(folder):
    """Per-board metrics + per-game points from the human playtest JSONs."""
    g = load_games(folder)
    rs  = [x["rounds"] for x in g]
    xs  = [x["transfer_rate_per_round"] for x in g]
    cen = [x.get("terminated_by") == "turn_limit" for x in g]
    # Fairness = |WR(P1) - WR(P2)|, treating a truncated/no-winner game as
    # 0.5 to each player (same convention as the simulation evaluator).
    wr_p1 = sum(
        (1.0 if x["winner"] == "Player 1"
         else 0.0 if x["winner"] == "Player 2"
         else 0.5) for x in g
    ) / len(g)
    fair = abs(wr_p1 - (1.0 - wr_p1))
    draws = sum(1 for x in g if x.get("truncated", False))
    return {
        "n":         len(g),
        "rounds":    sum(rs) / len(rs),
        "xfer":      sum(xs) / len(xs),
        "draw":      100.0 * draws / len(g),
        "fair":      fair,
        "rounds_pts": rs,
        "xfer_pts":   xs,
        "censored":   cen,
    }


def _pool_metrics():
    """Strategy-pool (n=1000) metrics at 2p for both boards."""
    data = json.loads(
        (_ROOT / "logs/optimizer_v3/cross_eval_mask.json")
        .read_text(encoding="utf-8")
    )
    idx = {(r["design"], r["n_players"]): r["metrics"] for r in data["results"]}
    out = {}
    for folder, key in POOL_KEYS.items():
        m = idx[key]
        out[folder] = {
            "n":      1000,
            "rounds": m["mean_rounds"],
            "xfer":   m["mean_transfer_rate"],
            "draw":   100.0 * m["mean_draw_rate"],
            "fair":   m["mean_fairness"],
        }
    return out


def _llm_metrics():
    """All-LLM-seats (n=100) metrics at 2p for both boards, recomputed via
    optimizer.objectives.evaluate() for apples-to-apples comparability."""
    sys.path.insert(0, str(_ROOT))
    from optimizer.objectives import evaluate

    with open(_ROOT / "logs/llm_eval/2p_n100/summary.csv",
              "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out = {}
    for folder, tag in LLM_TAGS.items():
        rs = [r for r in rows if r["board_tag"] == tag]
        games = [{
            "rounds":         int(r["rounds"]),
            "truncated":      bool(int(r["truncated"])),
            "transfer_total": int(float(r["transfer_total"])),
            "winner":         r["winner"] or None,
            "strategy_names": [f"LLM_p{i}" for i in range(2)],
        } for r in rs]
        m = evaluate([games])["metrics"]
        out[folder] = {
            "n":      len(games),
            "rounds": m["mean_rounds"],
            "xfer":   m["mean_transfer_rate"],
            "draw":   100.0 * m["mean_draw_rate"],
            "fair":   m["mean_fairness"],
        }
    return out


def main():
    # Load all three evaluation sources into the same shape (one dict per
    # board folder).
    human = {folder: _human_metrics(folder) for folder, _, _ in BOARDS}
    pool  = _pool_metrics()
    llm   = _llm_metrics()

    labels = [b[1] for b in BOARDS]
    colors = [b[2] for b in BOARDS]
    folders = [b[0] for b in BOARDS]

    # Three evaluator groups, two bars per group (default / GA winner).
    # Bar layout: per group, two bars of width 0.78 centred ±0.4 around the
    # group centre. Group centres are spaced by 2.2 so there is a clear
    # visual gap between evaluators.
    groups = [
        ("Human",      f"N={human[folders[0]]['n']}", human),
        ("Strat pool", f"n={pool [folders[0]]['n']}", pool),
        ("LLM seats",  f"n={llm  [folders[0]]['n']}", llm),
    ]
    bar_w = 0.78
    group_dx = 2.2
    group_centres = [i * group_dx for i in range(len(groups))]
    sep_xs = [(group_centres[i] + group_centres[i + 1]) / 2
              for i in range(len(groups) - 1)]
    bar_offsets = [-0.5 * bar_w * 1.05, +0.5 * bar_w * 1.05]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    def _bar_xs():
        """Yield (group_idx, board_idx, x, evaluator_dict[folder])."""
        for gi, (_, _, data) in enumerate(groups):
            cx = group_centres[gi]
            for bi, folder in enumerate(folders):
                yield gi, bi, cx + bar_offsets[bi], data[folder]

    def _draw_group_separators(ax):
        for sx in sep_xs:
            ax.axvline(sx, color="#dddddd", lw=0.9, zorder=0)
        ax.set_xticks(group_centres)
        ax.set_xticklabels([f"{g[0]}\n({g[1]})" for g in groups], fontsize=9)
        ax.tick_params(axis="x", length=0)

    # ---- top-left: rounds --------------------------------------------------
    for gi, bi, x, d in _bar_xs():
        ax1.bar([x], [d["rounds"]], color=colors[bi], alpha=0.85, width=bar_w)
        ax1.text(x, d["rounds"] + 4, f"{d['rounds']:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Per-game dots ONLY on the human bars: pool/LLM are big-N aggregates.
        if gi == 0:
            pts    = d["rounds_pts"]
            cflags = d["censored"]
            if len(pts) <= 1:
                offs = [0.0] * len(pts)
            else:
                offs = [-0.18 + 0.36 * j / (len(pts) - 1)
                        for j in range(len(pts))]
            for v, c, dx in zip(pts, cflags, offs):
                ax1.scatter([x + dx], [v],
                            color="red" if c else "#222222",
                            marker="x" if c else "o",
                            s=42, zorder=3, linewidths=1.6)
    ax1.axhline(60, ls="--", lw=1, color="#555555")
    ax1.text(group_centres[-1] + 0.55, 62, "target 60",
             fontsize=8, color="#555555", ha="right", va="bottom")
    # Flag the censored default human game (default is the first board, at
    # the leftmost group's leftmost offset).
    cap_x = group_centres[0] + bar_offsets[0]
    ax1.annotate("hit 200-turn cap\n(no winner)", xy=(cap_x, 195),
                 xytext=(cap_x + 0.55, 168),
                 fontsize=8, color="red", ha="left",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax1.set_ylabel("rounds per game")
    ax1.set_title(r"Game length ($\downarrow$ = faster)", fontsize=10)
    ax1.set_ylim(0, 215)
    _draw_group_separators(ax1)

    # ---- top-right: transfer rate -----------------------------------------
    xfer_max = max(d["xfer"] for *_, d in _bar_xs())
    for gi, bi, x, d in _bar_xs():
        ax2.bar([x], [d["xfer"]], color=colors[bi], alpha=0.85, width=bar_w)
        ax2.text(x, d["xfer"] + 2, f"{d['xfer']:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
        if gi == 0:
            pts = d["xfer_pts"]
            if len(pts) <= 1:
                offs = [0.0] * len(pts)
            else:
                offs = [-0.18 + 0.36 * j / (len(pts) - 1)
                        for j in range(len(pts))]
            for v, dx in zip(pts, offs):
                ax2.scatter([x + dx], [v], color="#222222",
                            marker="o", s=42, zorder=3)
    ax2.axhline(100, ls="--", lw=1, color="#555555")
    ax2.text(group_centres[-1] + 0.55, 101, "target 100",
             fontsize=8, color="#555555", ha="right", va="bottom")
    ax2.set_ylabel("inter-player transfer ($/round)")
    ax2.set_title(r"Money circulation ($\uparrow$ = livelier)", fontsize=10)
    ax2.set_ylim(0, max(115, xfer_max * 1.15))
    _draw_group_separators(ax2)

    # ---- bottom-left: draw rate -------------------------------------------
    draw_max = max(d["draw"] for *_, d in _bar_xs())
    for gi, bi, x, d in _bar_xs():
        ax3.bar([x], [d["draw"]], color=colors[bi], alpha=0.85, width=bar_w)
        ax3.text(x, d["draw"] + 1.2, f"{d['draw']:.0f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylabel("draw / truncation rate (%)")
    ax3.set_title(r"Decisiveness ($\downarrow$ = more games resolve)",
                  fontsize=10)
    ax3.set_ylim(0, max(55, draw_max + 10))
    _draw_group_separators(ax3)

    # ---- bottom-right: fairness (with caveat) -----------------------------
    fair_max = max(d["fair"] for *_, d in _bar_xs())
    for gi, bi, x, d in _bar_xs():
        ax4.bar([x], [d["fair"]], color=colors[bi], alpha=0.85, width=bar_w)
        ax4.text(x, d["fair"] + 0.02, f"{d['fair']:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_ylabel(r"win-rate spread $|WR_{P_1}-WR_{P_2}|$")
    ax4.set_title("Fairness  (caveat: small-N on the human side;\n"
                  "win-rate spread is coarse-discrete there)",
                  fontsize=9.5)
    ax4.set_ylim(0, max(1.15, fair_max * 1.25))
    _draw_group_separators(ax4)

    # Single legend at the top, identifying the two boards.
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l, alpha=0.85)
                      for c, l in zip(colors, labels)]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.985), ncol=2, fontsize=10,
               frameon=False)

    n_human = human[folders[0]]["n"]
    fig.suptitle(
        "The GA winner plays faster, circulates more money, and resolves more "
        "reliably than the default,\nacross all three evaluator classes "
        f"(human N={n_human}, strategy pool n=1000, all-LLM seats n=100; "
        "dots = individual human games)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = _OUT / "fig_human_playtest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_human_playtest] wrote {out}")
    for g_label, _, data in groups:
        rounds_m = {lbl: round(data[f]["rounds"], 1)
                    for f, lbl, _ in BOARDS}
        xfer_m   = {lbl: round(data[f]["xfer"],   1)
                    for f, lbl, _ in BOARDS}
        print(f"  [{g_label:>10}] rounds {rounds_m}  xfer {xfer_m}")


if __name__ == "__main__":
    main()
