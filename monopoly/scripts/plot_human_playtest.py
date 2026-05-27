"""Figure for the human-playtest section of the report/slides.

Reads the per-game stats JSONs produced by scripts/human_game_stats.py for the
four self-play boards (default, GA-winner, length-only, money-only), each run
under two matched seeds (0 and 42, common random numbers), and plots:

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

# (folder, label, colour) in narrative order
BOARDS = [
    ("default_board", "default",     "#888888"),
    ("ga_2p_winner",  "GA winner",   "#1f77b4"),
    ("abl_len_2p",    "length-only", "#ff7f0e"),
    ("abl_money_2p",  "money-only",  "#2ca02c"),
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

    x = list(range(len(labels)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- left: rounds ------------------------------------------------------
    ax1.bar(x, rounds_mean, color=colors, alpha=0.85, width=0.62)
    for i, (pts, cflags) in enumerate(zip(rounds_pts, censored)):
        offs = [-0.14, 0.14] if len(pts) == 2 else [0.0] * len(pts)
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
    ax1.set_title("Game length (lower = faster)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 215)

    # ---- right: transfer rate ---------------------------------------------
    ax2.bar(x, xfer_mean, color=colors, alpha=0.85, width=0.62)
    for i, pts in enumerate(xfer_pts):
        offs = [-0.14, 0.14] if len(pts) == 2 else [0.0] * len(pts)
        for v, dx in zip(pts, offs):
            ax2.scatter([i + dx], [v], color="#222222", marker="o", s=42, zorder=3)
    ax2.axhline(100, ls="--", lw=1, color="#555555")
    ax2.text(len(labels) - 0.45, 101, "target 100", fontsize=8, color="#555555",
             ha="right", va="bottom")
    for i, m in enumerate(xfer_mean):
        ax2.text(i, m + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax2.set_ylabel("inter-player transfer ($/round)")
    ax2.set_title("Money circulation (higher = livelier)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 115)

    fig.suptitle("Human playtest: optimized boards play faster and circulate "
                 "more money\n(n=2 per board, self-play, matched seeds 0 & 42; "
                 "dots = individual games)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = _OUT / "fig_human_playtest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_human_playtest] wrote {out}")
    print(f"  rounds mean: {dict(zip(labels, [round(v,1) for v in rounds_mean]))}")
    print(f"  xfer   mean: {dict(zip(labels, [round(v,1) for v in xfer_mean]))}")


if __name__ == "__main__":
    main()
