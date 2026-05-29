"""Figure for the human-playtest section of the report/slides.

Reads the per-game stats JSONs produced by scripts/human_game_stats.py for the
two playtest boards (default, GA-winner) under matched RNG seeds (common
random numbers across boards), and plots:

  left  : mean rounds per board (the pacing headline) with both games as dots
          and the objective target (60) marked. The default's turn-cap game is
          flagged because it never resolved.
  right : mean inter-player transfer rate ($/round) with both games as dots and
          the objective target (100) marked.

n=2 per board is shown honestly via the overlaid per-game dots.

Run from monopoly/:
  python scripts/plot_human_playtest.py
"""
import json
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


def load_games(folder):
    games = []
    for p in sorted((_TEST / folder).glob("game*_stats.json")):
        games.append(json.loads(p.read_text(encoding="utf-8")))
    return games


def main():
    labels, colors = [], []
    rounds_mean, xfer_mean = [], []
    rounds_pts, xfer_pts = [], []      # per-board list of per-game values
    censored = []                      # per-board list of bool (turn-cap game?)
    fairness, draw_rate = [], []       # NEW: per-board scalars

    for folder, label, color in BOARDS:
        g = load_games(folder)
        labels.append(label)
        colors.append(color)
        rs = [x["rounds"] for x in g]
        xs = [x["transfer_rate_per_round"] for x in g]
        rounds_pts.append(rs)
        xfer_pts.append(xs)
        rounds_mean.append(sum(rs) / len(rs))
        xfer_mean.append(sum(xs) / len(xs))
        censored.append([x.get("terminated_by") == "turn_limit" for x in g])
        # Fairness = |WR(P1) - WR(P2)|, treating a truncated/no-winner game
        # as 0.5 to each player. At small N the spread is coarse-discrete; the
        # panel subtitle calls this out.
        wr_p1 = sum(
            (1.0 if x["winner"] == "Player 1"
             else 0.0 if x["winner"] == "Player 2"
             else 0.5) for x in g
        ) / len(g)
        fairness.append(abs(wr_p1 - (1.0 - wr_p1)))
        # Draw rate = fraction of games that hit the turn cap with no winner.
        draws = sum(1 for x in g if x.get("truncated", False))
        draw_rate.append(100.0 * draws / len(g))

    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # ---- left: rounds ------------------------------------------------------
    ax1.bar(x, rounds_mean, color=colors, alpha=0.85, width=0.62)
    for i, (pts, cflags) in enumerate(zip(rounds_pts, censored)):
        # Spread N dots evenly across [-0.14, +0.14] of the bar's x-centre.
        if len(pts) <= 1:
            offs = [0.0] * len(pts)
        else:
            offs = [-0.14 + 0.28 * i / (len(pts) - 1) for i in range(len(pts))]
        for v, c, dx in zip(pts, cflags, offs):
            ax1.scatter([i + dx], [v], color="red" if c else "#222222",
                        marker="x" if c else "o", s=42, zorder=3,
                        linewidths=1.6)
    ax1.axhline(60, ls="--", lw=1, color="#555555")
    ax1.text(len(labels) - 0.45, 62, "target 60", fontsize=8, color="#555555",
             ha="right", va="bottom")
    for i, m in enumerate(rounds_mean):
        ax1.text(i, m + 3, f"{m:.0f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    # flag the censored default game
    ax1.annotate("hit 200-turn cap\n(no winner)", xy=(0, 195), xytext=(0.7, 168),
                 fontsize=8, color="red", ha="left",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax1.set_ylabel("rounds per game")
    ax1.set_title(r"Game length ($\downarrow$ = faster)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 215)

    # ---- right: transfer rate ---------------------------------------------
    ax2.bar(x, xfer_mean, color=colors, alpha=0.85, width=0.62)
    for i, pts in enumerate(xfer_pts):
        # Spread N dots evenly across [-0.14, +0.14] of the bar's x-centre.
        if len(pts) <= 1:
            offs = [0.0] * len(pts)
        else:
            offs = [-0.14 + 0.28 * i / (len(pts) - 1) for i in range(len(pts))]
        for v, dx in zip(pts, offs):
            ax2.scatter([i + dx], [v], color="#222222", marker="o", s=42, zorder=3)
    ax2.axhline(100, ls="--", lw=1, color="#555555")
    ax2.text(len(labels) - 0.45, 101, "target 100", fontsize=8, color="#555555",
             ha="right", va="bottom")
    for i, m in enumerate(xfer_mean):
        ax2.text(i, m + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax2.set_ylabel("inter-player transfer ($/round)")
    ax2.set_title(r"Money circulation ($\uparrow$ = livelier)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 115)

    # ---- bottom-left: draw rate -------------------------------------------
    ax3.bar(x, draw_rate, color=colors, alpha=0.85, width=0.62)
    for i, v in enumerate(draw_rate):
        ax3.text(i, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax3.set_ylabel("draw / truncation rate (%)")
    ax3.set_title(r"Decisiveness ($\downarrow$ = more games resolve)", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylim(0, max(55, max(draw_rate) + 10))

    # ---- bottom-right: fairness (with caveat) -----------------------------
    ax4.bar(x, fairness, color=colors, alpha=0.85, width=0.62)
    for i, v in enumerate(fairness):
        ax4.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax4.set_ylabel(r"win-rate spread $|WR_{P_1}-WR_{P_2}|$")
    ax4.set_title("Fairness  (caveat: small-N; win-rate spread is\n"
                  "coarse-discrete at this sample size)",
                  fontsize=9.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylim(0, 1.15)

    fig.suptitle("Human playtest: the GA winner plays faster, circulates "
                 "more money, and resolves more reliably than the default\n"
                 "(N=3 per board, two human players, common-random-numbers "
                 "across boards; dots = individual games)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = _OUT / "fig_human_playtest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_human_playtest] wrote {out}")
    print(f"  rounds mean: {dict(zip(labels, [round(v,1) for v in rounds_mean]))}")
    print(f"  xfer   mean: {dict(zip(labels, [round(v,1) for v in xfer_mean]))}")


if __name__ == "__main__":
    main()
